"""Shared residual MLP, optimizer routing, and training loop for tabular tasks."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from experiments.baselines import SOAP
from experiments.common import DIVERGED_EXIT, RunLogger, cosine_with_warmup, diverged, pick_device
from experiments.transformers.modded_nanogpt_model import Muon
from gnome import Gnome

OPTIMIZERS = ("gnome", "gnome_hutchinson", "gnome_fisher", "adamw", "soap", "muon")


def split_and_standardize(x, y, split_seed=42, *, num_classes, num_continuous, log1p=False):
    """Stratified 70/15/15 split; fit numeric columns on train only.

    The split has its own RNG, independent of model/optimizer seeds. Returned
    indices and statistics are also stored in checkpoints for reconstruction.
    """
    if log1p:
        x = torch.log1p(x)
    rng = np.random.default_rng(split_seed)
    buckets = [[], [], []]
    for label in range(num_classes):
        indices = rng.permutation(np.flatnonzero(y.numpy() == label))
        n_train, n_val = int(0.70 * len(indices)), int(0.15 * len(indices))
        if min(n_train, n_val, len(indices) - n_train - n_val) == 0:
            raise ValueError("Each class needs enough examples for all three splits")
        for bucket, part in zip(buckets, np.split(indices, [n_train, n_train + n_val])):
            bucket.append(part)
    indices = {name: torch.from_numpy(rng.permutation(np.concatenate(parts)))
               for name, parts in zip(("train", "val", "test"), buckets)}
    continuous = x[indices["train"], :num_continuous].double()
    mean = continuous.mean(0).float()
    std = continuous.std(0, correction=0).float()
    std = torch.where(std > 0, std, torch.ones_like(std))
    normalized = x.clone()
    normalized[:, :num_continuous] = (normalized[:, :num_continuous] - mean) / std
    splits = {name: (normalized[idx], y[idx]) for name, idx in indices.items()}
    return splits, dict(indices=indices, mean=mean, std=std, log1p=log1p, num_continuous=num_continuous)


class ResidualBlock(nn.Module):
    def __init__(self, width, scale):
        super().__init__()
        self.layers = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
                                    nn.GELU(), nn.Linear(width, width))
        self.scale = scale

    def forward(self, x):
        return x + self.scale * self.layers(x)


class ResidualMLP(nn.Module):
    def __init__(self, width=256, blocks=3, *, num_features, num_classes):
        super().__init__()
        self.input = nn.Linear(num_features, width)
        self.blocks = nn.Sequential(*(ResidualBlock(width, blocks ** -0.5) for _ in range(blocks)))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, num_classes)

    def forward(self, x):
        return self.head(self.norm(self.blocks(F.gelu(self.input(x)))))


def build_optimizers(model, args):
    common = dict(lr=args.lr, weight_decay=args.weight_decay,
                  betas=(args.beta1, args.beta2), eps=args.eps)
    if args.optimizer.startswith("gnome"):
        return [Gnome(model.parameters(), **common,
                      use_soap_basis=args.use_soap_basis,
                      loss="cce" if args.optimizer == "gnome_fisher" else "cce_hutchinson",
                      trust_radius=args.trust_region or None, norm_free=args.norm_free,
                      max_grad_norm=args.max_grad_norm or None,
                      shampoo_beta=args.shampoo_beta,
                      precondition_frequency=args.precondition_frequency,
                      max_precond_dim=args.max_precond_dim, precondition_1d=args.precondition_1d)]
    if args.optimizer == "soap":
        return [SOAP(model.parameters(), **common, shampoo_beta=args.shampoo_beta,
                     precondition_frequency=args.precondition_frequency,
                     max_precond_dim=args.max_precond_dim, precondition_1d=args.precondition_1d)]
    if args.optimizer == "adamw":
        return [torch.optim.AdamW(model.parameters(), **common)]
    hidden = [p for p in model.blocks.parameters() if p.ndim == 2]
    hidden_ids = {id(p) for p in hidden}
    auxiliary = [p for p in model.parameters() if id(p) not in hidden_ids]
    return [Muon(hidden, lr=args.lr, weight_decay=args.weight_decay, mu=args.muon_momentum),
            torch.optim.AdamW(auxiliary, **{**common, "lr": args.aux_lr})]


def train_step(model, optimizers, x, y, args, aux_rng):
    if args.optimizer.startswith("gnome"):
        idx = torch.randperm(len(y), generator=aux_rng)[:args.aux_batch_size].to(x.device)
        return float(optimizers[0].step(lambda: (model(x), y),
                                       lambda: (model(x[idx]), y[idx])).item())
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    if args.max_grad_norm:
        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    for optimizer in optimizers:
        optimizer.step()
    return float(loss.detach().item())


@torch.no_grad()
def evaluate(model, x, y, batch_size, device):
    was_training = model.training
    model.eval()
    loss, correct = 0.0, 0
    try:
        for start in range(0, len(y), batch_size):
            xb, yb = x[start:start + batch_size].to(device), y[start:start + batch_size].to(device)
            logits = model(xb)
            loss += F.cross_entropy(logits, yb, reduction="sum").item()
            correct += (logits.argmax(-1) == yb).sum().item()
    finally:
        model.train(was_training)
    return dict(loss=loss / len(y), top1=correct / len(y))


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def train(args, *, experiment, load_data, split_data, build_model, dataset_metadata,
          step_fn=train_step):
    device = pick_device() if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    splits, preprocessing = split_data(*load_data(args.data_dir), args.split_seed)
    torch.manual_seed(args.seed)
    model = build_model(args.width, args.blocks).to(device)
    optimizers = build_optimizers(model, args)
    # Separate RNGs keep batch ordering identical even when Gnome consumes
    # random probes and auxiliary samples that the other optimizers do not.
    batch_rng = torch.Generator().manual_seed(args.seed)
    aux_rng = torch.Generator().manual_seed(args.seed + 1)
    x_train, y_train = splits["train"]
    steps_per_epoch = math.ceil(len(y_train) / args.batch_size)
    total_steps = min(args.epochs * steps_per_epoch, args.max_steps or math.inf)
    if args.warmup_steps >= total_steps:
        raise ValueError("warmup-steps must be less than the total training steps")
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(device=str(device), dtype="float32", total_steps=total_steps,
                  parameters=sum(p.numel() for p in model.parameters()),
                  split_protocol="stratified_70_15_15_v1", **dataset_metadata,
                  **{f"n_{name}": len(y) for name, (_, y) in splits.items()},
                  auxiliary_sampling="main_batch_subset_without_replacement",
                  muon_routing="hidden_matrices; AdamW input/head/biases/norms",
                  optimizer_groups=[[{k: v for k, v in group.items() if k != "params"}
                                     for group in optimizer.param_groups] for optimizer in optimizers])
    print(f"[{experiment}] {args.optimizer} | {device} | {config['parameters']:,} parameters\n"
          f"  train/val/test={config['n_train']:,}/{config['n_val']:,}/{config['n_test']:,}; "
          f"{total_steps:,} steps; lr={args.lr:g}", flush=True)
    initial_lrs = [[group["lr"] for group in opt.param_groups] for opt in optimizers]
    step, training_seconds, best_top1 = 0, 0.0, -math.inf
    best_val_loss = math.inf
    window_loss, window_samples = 0.0, 0
    with RunLogger(experiment, args.optimizer, args.seed, config, runs_dir=str(args.runs_dir)) as run:
        print(f"[{experiment}] logs: {run.path}", flush=True)
        for epoch in range(1, args.epochs + 1):
            model.train()
            permutation = torch.randperm(len(y_train), generator=batch_rng)
            for indices in permutation.split(args.batch_size):
                step += 1
                multiplier = cosine_with_warmup(step, args.warmup_steps, total_steps, args.min_lr_frac)
                for optimizer, lrs in zip(optimizers, initial_lrs):
                    for group, base_lr in zip(optimizer.param_groups, lrs):
                        group["lr"] = base_lr * multiplier
                synchronize(device)
                started = time.perf_counter()
                loss = step_fn(model, optimizers, x_train[indices].to(device),
                                  y_train[indices].to(device), args, aux_rng)
                synchronize(device)
                elapsed = time.perf_counter() - started
                training_seconds += elapsed
                if diverged(loss):
                    run.finish(completed=False, diverged=True, diverged_step=step)
                    raise SystemExit(DIVERGED_EXIT)
                run.log_train(step, loss=loss, lr=args.lr * multiplier,
                              step_time_ms=elapsed * 1000, training_seconds=training_seconds)
                window_loss += loss * len(indices)
                window_samples += len(indices)
                if step % args.log_every == 0 or step == total_steps:
                    if not args.quiet:
                        print(f"  step {step}/{total_steps} loss={window_loss / window_samples:.5f} "
                              f"lr={args.lr * multiplier:.3e} {elapsed * 1000:.1f}ms", flush=True)
                    window_loss, window_samples = 0.0, 0
                if step == total_steps:
                    break
            val = evaluate(model, *splits["val"], args.eval_batch_size, device)
            if diverged(val["loss"]):
                run.finish(completed=False, diverged=True, diverged_step=step)
                raise SystemExit(DIVERGED_EXIT)
            best_top1 = max(best_top1, val["top1"])
            best_val_loss = min(best_val_loss, val["loss"])
            run.log_val(step, epoch=epoch, **val, training_seconds=training_seconds)
            if not args.quiet:
                print(f"  epoch {epoch}: val_loss={val['loss']:.5f} top1={100 * val['top1']:.2f}%", flush=True)
            if step == total_steps:
                break
        # Test is evaluated once, after all updates; it never selects a model.
        test = evaluate(model, *splits["test"], args.eval_batch_size, device)
        if diverged(test["loss"]):
            run.finish(completed=False, diverged=True, diverged_step=step)
            raise SystemExit(DIVERGED_EXIT)
        run.log("test", step=step, **test)
        if args.save_checkpoint:
            torch.save(dict(model=model.state_dict(), optimizers=[o.state_dict() for o in optimizers],
                            config=config, preprocessing=preprocessing, step=step),
                       Path(run.path).with_suffix(".pt"))
        run.finish(final_val_loss=val["loss"], best_val_loss=best_val_loss,
                   final_top1=val["top1"], best_top1=best_top1,
                   test_loss=test["loss"], test_top1=test["top1"], training_seconds=training_seconds)
    print(f"[{experiment}] done: test_loss={test['loss']:.5f} top1={100 * test['top1']:.2f}%", flush=True)
    return run.path




def parse_args(p, argv=None):
    args = p.parse_args(argv)
    if args.lr is None:
        args.lr = {"soap": 0.003, "muon": 0.02}.get(args.optimizer, 0.001)
    if args.beta2 is None:
        args.beta2 = 0.999 if args.optimizer in ("adamw", "muon") else 0.99
    for key in ("epochs", "width", "blocks", "batch_size", "aux_batch_size", "eval_batch_size",
                "log_every", "precondition_frequency", "max_precond_dim", "max_steps"):
        value = getattr(args, key)
        if value is not None and value <= 0:
            p.error(f"{key.replace('_', '-')} must be positive")
    if args.aux_batch_size > args.batch_size:
        p.error("aux-batch-size cannot exceed batch-size (partial final batches use fewer auxiliary rows)")
    if min(args.seed, args.split_seed, args.warmup_steps) < 0:
        p.error("seeds and warmup-steps must be nonnegative")
    if args.max_steps is not None and args.warmup_steps >= args.max_steps:
        p.error("warmup-steps must be less than max-steps")
    for key in ("lr", "aux_lr", "weight_decay", "eps", "trust_region", "max_grad_norm"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            p.error(f"{key.replace('_', '-')} must be finite and nonnegative")
    if args.eps == 0:
        p.error("eps must be positive")
    for key in ("beta1", "beta2", "shampoo_beta", "muon_momentum"):
        if not 0 <= getattr(args, key) < 1:
            p.error(f"{key.replace('_', '-')} must be in [0, 1)")
    if not 0 <= args.min_lr_frac <= 1:
        p.error("min-lr-frac must be in [0, 1]")
    return args
