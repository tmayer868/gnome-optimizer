"""Modded-NanoGPT optimization Track 3, with portable MPS/CUDA execution.

    uv run -m experiments.transformers.modded_nanogpt --preset dev --synthetic
    uv run -m experiments.transformers.modded_nanogpt --optimizer muon --download-shards 20

See docs/modded_nanogpt.md for the reference protocol and Gnome's extra-pass
qualification. This runner uses one device, preserving global batch size by
gradient accumulation. It does not launch or rent cloud instances.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import platform
import time
import uuid
import zipfile

import torch
import torch.nn.functional as F

from experiments.baselines import SOAP
from experiments.common import DIVERGED_EXIT, RunLogger, diverged, pick_device
from experiments.transformers.fineweb_data import DATASET_REPO, SyntheticStream, TokenStream, download_shards
from experiments.transformers.modded_nanogpt_model import GPT, Muon, UPSTREAM_REVISION
from gnome import Gnome

EXPERIMENT = "modded_nanogpt"
REFERENCE = dict(vocab_size=50304, n_layer=12, n_embd=768, head_dim=128,
                 seq_len=1024, batch_tokens=524288, val_tokens=10485760)
DEV = dict(vocab_size=50304, n_layer=2, n_embd=128, head_dim=64,
           seq_len=64, batch_tokens=256, val_tokens=512)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=["benchmark", "dev"], default="benchmark")
    p.add_argument("--optimizer", choices=["muon", "gnome_hutchinson", "gnome_fisher", "soap", "adamw"],
                   default="gnome_hutchinson")
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    p.add_argument("--dtype", choices=["auto", "float32", "bfloat16"], default="auto")
    p.add_argument("--compile", action="store_true", help="Opt-in CUDA model compilation")
    p.add_argument("--head-init-std", type=float, default=0.0,
                   help="Output weight initialization std; 0 preserves the reference zero initialization")
    p.add_argument("--fp32-embedding", action="store_true",
                   help="Keep embedding weights/optimizer state in FP32 while retaining the chosen compute dtype")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int)
    for name in REFERENCE:
        p.add_argument("--" + name.replace("_", "-"), type=int)
    p.add_argument("--microbatch-size", type=int, default=4, help="Sequences per forward; global batch is unchanged")
    p.add_argument("--aux-batch-size", type=int, default=1, help="Gnome curvature sequences sampled from the main batch")
    p.add_argument("--lr", type=float, help="Gnome/SOAP/AdamW LR, or Muon hidden-matrix LR")
    p.add_argument("--weight-decay", type=float)
    p.add_argument("--beta1", type=float)
    p.add_argument("--beta2", type=float)
    p.add_argument("--eps", type=float)
    p.add_argument("--shampoo-beta", type=float, default=0.99)
    p.add_argument("--precondition-frequency", type=int, default=10)
    p.add_argument("--max-precond-dim", type=int, default=4096)
    p.add_argument("--trust-region", type=float, default=1.0,
                   help="Gnome relative L2 trust radius; 0 disables")
    p.add_argument("--max-grad-norm", type=float, default=0, help="0 disables clipping (reference default)")
    p.add_argument("--muon-momentum", type=float, default=0.95)
    p.add_argument("--embed-lr", type=float, default=0.7, help="Muon auxiliary AdamW only")
    p.add_argument("--head-lr", type=float, default=0.004, help="Muon auxiliary AdamW only")
    p.add_argument("--scalar-lr", type=float, default=0.015, help="Muon auxiliary AdamW only")
    p.add_argument("--aux-weight-decay", type=float, default=0.001, help="Muon auxiliary AdamW only")
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--cooldown-frac", type=float, default=0.7, help="Final fraction with linear LR decay")
    p.add_argument("--val-every", type=int, help="Default: 125, then 25 during the final 10%%")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--data-dir", type=Path, default=Path("experiments/data/fineweb10B"))
    p.add_argument("--download-shards", type=int, default=0, help="Opt-in download: N training shards, about 200 MB each")
    p.add_argument("--download-only", action="store_true")
    p.add_argument("--synthetic", action="store_true", help="Offline development check, requires --preset dev")
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--save-checkpoint", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    defaults = REFERENCE if args.preset == "benchmark" else DEV
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, 256 if args.synthetic and key == "vocab_size" else value)
    if args.steps is None:
        args.steps = 3250 if args.preset == "benchmark" else 5
    if args.lr is None:
        args.lr = {"muon": 0.025, "adamw": 0.0015, "soap": 0.003}.get(args.optimizer, 0.001)
    if args.weight_decay is None:
        args.weight_decay = 0.05 if args.optimizer == "muon" else 0.01
    if args.beta1 is None:
        args.beta1 = 0.8 if args.optimizer == "muon" else 0.9
    if args.beta2 is None:
        args.beta2 = 0.95 if args.optimizer in ("muon", "adamw") else 0.99
    if args.eps is None:
        args.eps = 1e-10 if args.optimizer == "muon" else 1e-8
    positive = [*REFERENCE, "steps", "microbatch_size", "aux_batch_size", "log_every",
                "precondition_frequency", "max_precond_dim"]
    for key in positive:
        if getattr(args, key) <= 0:
            p.error(f"--{key.replace('_', '-')} must be positive")
    if args.n_embd % args.head_dim or args.head_dim % 4:
        p.error("n-embd must be divisible by head-dim; head-dim must be divisible by 4")
    if args.batch_tokens % args.seq_len or args.val_tokens % args.seq_len:
        p.error("batch-tokens and val-tokens must be divisible by seq-len")
    if args.aux_batch_size > args.batch_tokens // args.seq_len:
        p.error("aux-batch-size cannot exceed the global batch's sequence count")
    if args.val_every is not None and args.val_every <= 0:
        p.error("val-every must be positive")
    if not 0 <= args.warmup_steps < args.steps or not 0 <= args.cooldown_frac <= 1:
        p.error("require 0 <= warmup-steps < steps and 0 <= cooldown-frac <= 1")
    for name in ("beta1", "beta2", "shampoo_beta", "muon_momentum"):
        if not 0 <= getattr(args, name) < 1:
            p.error(f"{name} must be in [0, 1)")
    for name in ("lr", "weight_decay", "eps", "trust_region", "max_grad_norm",
                 "embed_lr", "head_lr", "scalar_lr", "aux_weight_decay", "head_init_std"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            p.error(f"{name} must be finite and nonnegative")
    if args.eps == 0:
        p.error("eps must be positive")
    if not 0 <= args.download_shards <= 103:
        p.error("download-shards must be between 0 and 103")
    if args.download_only and not args.download_shards:
        p.error("download-only requires --download-shards N")
    if args.synthetic and (args.preset != "dev" or args.download_shards):
        p.error("synthetic data requires --preset dev and cannot be combined with downloads")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        p.error("This runner uses one device with accumulation; launch with python, not multi-rank torchrun")
    return args


def select_device(name):
    device = pick_device() if name == "auto" else torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable")
    return device


def select_dtype(name, device):
    if name == "auto":
        name = "bfloat16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "float32"
    if name == "bfloat16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("This runner supports BF16 on capable CUDA devices; use float32 for MPS/CPU")
    return torch.bfloat16 if name == "bfloat16" else torch.float32


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def schedule_multiplier(step, steps, cooldown_frac, warmup_steps=0):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if cooldown_frac == 0:
        return 1.0
    return min(1.0, (1 - step / steps) / cooldown_frac)


def build_optimizers(model, args, device):
    common = dict(lr=args.lr, weight_decay=args.weight_decay,
                  betas=(args.beta1, args.beta2), eps=args.eps)
    if args.optimizer.startswith("gnome"):
        optimizers = [Gnome(
            model.parameters(), **common,
            loss="cce_hutchinson" if args.optimizer == "gnome_hutchinson" else "cce",
            shampoo_beta=args.shampoo_beta, precondition_frequency=args.precondition_frequency,
            max_precond_dim=args.max_precond_dim,
            trust_radius=args.trust_region or None, max_grad_norm=args.max_grad_norm or None,
        )]
    elif args.optimizer == "soap":
        optimizers = [SOAP(model.parameters(), **common, shampoo_beta=args.shampoo_beta,
                           precondition_frequency=args.precondition_frequency,
                           max_precond_dim=args.max_precond_dim)]
    elif args.optimizer == "adamw":
        optimizers = [torch.optim.AdamW(model.parameters(), **common, fused=device.type == "cuda")]
    else:
        auxiliary = torch.optim.AdamW([
            dict(params=[model.embed.weight], lr=args.embed_lr),
            dict(params=[model.proj.weight], lr=args.head_lr),
            dict(params=[p for p in model.parameters() if p.ndim < 2], lr=args.scalar_lr),
        ], betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.aux_weight_decay,
            fused=device.type == "cuda")
        hidden = Muon([p for p in model.blocks.parameters() if p.ndim >= 2],
                      lr=args.lr, weight_decay=args.weight_decay, mu=args.muon_momentum)
        optimizers = [auxiliary, hidden]
    parameters = [p for opt in optimizers for group in opt.param_groups for p in group["params"]]
    assert len(parameters) == len(set(parameters)) == len(list(model.parameters()))
    assert set(parameters) == set(model.parameters())
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    return optimizers


def train_step(model, optimizers, batch, args, device, aux_generator):
    inputs, targets = batch  # CPU tokens; move only the current microbatch
    model.train()
    if args.optimizer.startswith("gnome"):
        def make_closure(x, y):
            def closure():
                logits = model(x.to(device))
                return logits.reshape(-1, logits.size(-1)), y.to(device).reshape(-1)
            return closure

        closures = [make_closure(inputs[i:i + args.microbatch_size], targets[i:i + args.microbatch_size])
                    for i in range(0, len(inputs), args.microbatch_size)]
        indices = torch.randperm(len(inputs), generator=aux_generator)[:args.aux_batch_size]
        auxiliary = make_closure(inputs[indices], targets[indices])
        # Gnome weights each closure by its token count, including partial microbatches.
        return float(optimizers[0].step(closures, auxiliary).item())

    model.zero_grad(set_to_none=True)
    loss_sum = torch.zeros((), device=device)
    for i in range(0, len(inputs), args.microbatch_size):
        logits = model(inputs[i:i + args.microbatch_size].to(device))
        y = targets[i:i + args.microbatch_size].to(device)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        loss_sum += loss.detach()
        # The tuned upstream Muon/aux AdamW run uses SUM gradients. Other
        # optimizer experiments use token means, matching Gnome's internal loss.
        (loss if args.optimizer == "muon" else loss / targets.numel()).backward()
    loss_value = float((loss_sum / targets.numel()).item())
    if diverged(loss_value):
        return loss_value
    if args.max_grad_norm:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    for opt in optimizers:
        opt.step()
    return loss_value


@torch.no_grad()
def evaluate(model, batch, microbatch_size, device):
    was_training = model.training
    model.eval()
    inputs, targets = batch
    total = 0.0
    try:
        for i in range(0, len(inputs), microbatch_size):
            logits = model(inputs[i:i + microbatch_size].to(device))
            y = targets[i:i + microbatch_size].to(device)
            total += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                     reduction="sum").item()
    finally:
        model.train(was_training)
    return total / targets.numel()


def source_bundle(path):
    """Capture dirty/local sources too; a git SHA alone cannot reproduce them."""
    root = Path(__file__).resolve().parents[2]
    sources = [Path(__file__), Path(__file__).with_name("modded_nanogpt_model.py"),
               Path(__file__).with_name("fineweb_data.py"), root / "pyproject.toml"]
    for folder in ("gnome", "experiments/baselines", "experiments/common",
                   "experiments/transformers/modded_nanogpt_reference"):
        sources += [p for p in (root / folder).rglob("*") if p.suffix == ".py" or p.name == "LICENSE"]
    hashes = {}
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(set(sources)):
            name = str(file.relative_to(root))
            contents = file.read_bytes()
            hashes[name] = hashlib.sha256(contents).hexdigest()
            archive.writestr(name, contents)
    return hashes


def run(args):
    if args.download_shards:
        download_shards(args.data_dir, args.download_shards)
    if args.download_only:
        return None
    device = select_device(args.device)
    dtype = select_dtype(args.dtype, device)
    if args.compile and device.type != "cuda":
        raise ValueError("--compile is CUDA-only; MPS/CPU run eagerly")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    if args.synthetic:
        train = SyntheticStream(args.batch_tokens, args.seq_len, args.vocab_size, args.seed + 1)
        validation = SyntheticStream(args.val_tokens, args.seq_len, args.vocab_size, args.seed + 2)
    else:
        train = TokenStream(args.data_dir / "fineweb_train_*.bin", args.batch_tokens,
                            args.seq_len, args.vocab_size)
        validation = TokenStream(args.data_dir / "fineweb_val_*.bin", args.val_tokens,
                                 args.seq_len, args.vocab_size)
    if train.capacity < args.steps:
        raise ValueError(f"Only {train.capacity} complete training batches available for {args.steps} steps; "
                         "download more shards (training never wraps)")
    val_batch = validation.next_batch()  # fixed first 10,485,760 tokens in benchmark mode
    model = GPT(args.vocab_size, args.n_layer, args.n_embd, args.head_dim, dtype,
                head_init_std=args.head_init_std, fp32_embedding=args.fp32_embedding).to(device)
    optimizers = build_optimizers(model, args, device)
    forward_model = torch.compile(model, dynamic=False) if args.compile else model
    auxiliary_rng = torch.Generator().manual_seed(args.seed + 3)
    extra_pass = args.optimizer.startswith("gnome")
    matches = not args.synthetic and all(getattr(args, k) == v for k, v in REFERENCE.items())
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    run_id = f"{EXPERIMENT}_{args.optimizer}_seed{args.seed}_{uuid.uuid4().hex[:12]}"
    output = args.runs_dir / EXPERIMENT
    output.mkdir(parents=True, exist_ok=True)
    bundle = output / f"{run_id}.sources.zip"
    hashes = source_bundle(bundle)
    config.update(
        device=str(device), compute_dtype=str(dtype), upstream_revision=UPSTREAM_REVISION,
        embedding_parameter_dtype=str(model.embed.weight.dtype),
        reference_initialization_matches=args.head_init_std == 0,
        n_params=sum(p.numel() for p in model.parameters()), dataset=DATASET_REPO if not args.synthetic else "synthetic",
        reference_configuration_matches=matches, extra_curvature_pass=extra_pass,
        forward_backward_token_ratio=1 + (args.aux_batch_size * args.seq_len / args.batch_tokens if extra_pass else 0),
        gradient_reduction="sum" if args.optimizer == "muon" else "token_mean",
        microbatches_per_step=math.ceil(args.batch_tokens / args.seq_len / args.microbatch_size),
        target_val_loss=3.28, source_bundle=str(bundle), source_sha256=hashes,
        platform=platform.platform(), cuda_version=torch.version.cuda,
        device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        submission_ready=False,  # requires upstream-format log packaging and statistical evidence
    )
    if not args.synthetic:
        config["data_shards"] = [{"name": p.name, "tokens": n} for stream in (train, validation)
                                 for p, n in zip(stream.files, stream.counts)]
    print(f"[{EXPERIMENT}] {args.optimizer} | {device} | {dtype} | {config['n_params']:,} parameters", flush=True)
    print(f"  {args.batch_tokens:,} tokens/step; {config['microbatches_per_step']} microbatches; "
          f"reference configuration matches: {matches}; extra curvature pass: {extra_pass}", flush=True)
    print(f"  head_init_std={args.head_init_std:g}; embedding_weights={model.embed.weight.dtype}", flush=True)
    with RunLogger(EXPERIMENT, args.optimizer, args.seed, config,
                   runs_dir=str(args.runs_dir), run_id=run_id) as log:
        training_seconds = 0.0
        completed_steps = 0
        final_val_loss = float("nan")
        for step in range(args.steps + 1):
            interval = args.val_every or (125 if step / args.steps < 0.9 else 25)
            if step == 0 or step == args.steps or step % interval == 0:
                final_val_loss = evaluate(forward_model, val_batch, args.microbatch_size, device)
                log.log_val(step, loss=final_val_loss,
                            ppl=math.exp(min(final_val_loss, 80)), training_seconds=training_seconds,
                            tokens_seen=step * args.batch_tokens)
                if not args.quiet:
                    print(f"  step {step}/{args.steps} val_loss={final_val_loss:.5f} "
                          f"training_seconds={training_seconds:.2f}", flush=True)
                if diverged(final_val_loss):
                    log.finish(completed=False, diverged=True, completed_steps=completed_steps,
                               training_seconds=training_seconds)
                    raise SystemExit(DIVERGED_EXIT)
            if step == args.steps:
                break
            multiplier = schedule_multiplier(step, args.steps, args.cooldown_frac, args.warmup_steps)
            for opt in optimizers:
                for group in opt.param_groups:
                    group["lr"] = group["initial_lr"] * multiplier
            synchronize(device)
            started = time.perf_counter()
            loss = train_step(forward_model, optimizers, train.next_batch(), args, device, auxiliary_rng)
            synchronize(device)
            elapsed = time.perf_counter() - started
            training_seconds += elapsed
            completed_steps = step + 1
            log.log_train(completed_steps, loss=loss, lr=args.lr * multiplier,
                          tokens_seen=completed_steps * args.batch_tokens, step_seconds=elapsed,
                          training_seconds=training_seconds, tokens_per_second=args.batch_tokens / elapsed,
                          auxiliary_tokens_seen=completed_steps * args.aux_batch_size * args.seq_len if extra_pass else 0)
            if diverged(loss):
                log.finish(completed=False, diverged=True, completed_steps=completed_steps,
                           training_seconds=training_seconds)
                raise SystemExit(DIVERGED_EXIT)
            if not args.quiet and (completed_steps % args.log_every == 0 or completed_steps == args.steps):
                print(f"  step {completed_steps}/{args.steps} loss={loss:.5f} "
                      f"lr={args.lr * multiplier:.3e} {elapsed:.2f}s", flush=True)
        checkpoint = None
        if args.save_checkpoint:
            checkpoint = str(output / f"{run_id}.pt")
            torch.save(dict(model=model.state_dict(), optimizers=[o.state_dict() for o in optimizers],
                            config=config, step=completed_steps), checkpoint)
        path = log.finish(completed=True, completed_steps=completed_steps, final_val_loss=final_val_loss,
                          target_reached=final_val_loss < 3.28, training_seconds=training_seconds,
                          tokens_seen=completed_steps * args.batch_tokens, checkpoint=checkpoint)
    print(f"[{EXPERIMENT}] saved → {path}", flush=True)
    return Path(path)


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
