"""WikiText-103 NanoGPT — Gnome (Fisher) vs Gnome (Hutchinson) vs SOAP vs AdamW.

A small GPT-2-style transformer trained on WikiText-103 for next-token
prediction. This is the highest-K experiment in the suite: with
``vocab_size=50257``, the Fisher-sampling surrogate touches only
``aux_batch_size`` tokens × 1 sampled class per token out of 50k classes,
while the Hutchinson surrogate covers all 50k classes per (aux) token
simultaneously via Rademacher draws on the softmax-Hessian factorization
``diag(p) - p p^T = A A^T``. The variance-reduction argument predicts the
gap should be largest here.

Optimizer choices:

    * ``gnome_fisher``       — Gnome with ``loss="cce"``
    * ``gnome_hutchinson``   — Gnome with ``loss="cce_hutchinson"``
    * ``soap``               — empirical-Fisher SOAP baseline
    * ``adamw``              — first-order baseline

Unlike the MSE experiments (where Gnome runs a fixed lr), this CCE run gives
*every* optimizer — Gnome included — a shared linear-warmup + cosine-decay
schedule: cross-entropy gradients don't self-anneal the way an MSE residual
does. Gnome's internal warmup is disabled (``warmup=0``) so the two ramps don't
compound. ``--lr-min-frac`` is the cosine floor (final lr as a fraction of peak).

Requires ``datasets`` + ``transformers`` (the ``llm`` extra):

    uv pip install -e ".[llm]"

Usage:

    uv run -m experiments.wikitext_gpt --optimizer gnome_hutchinson --seed 0
    uv run -m experiments.wikitext_gpt --optimizer gnome_fisher     --seed 0
    uv run -m experiments.wikitext_gpt --optimizer soap             --seed 0
    uv run -m experiments.wikitext_gpt --optimizer adamw            --seed 0

The first run downloads ~500MB of WikiText-103 and tokenizes it (~5
minutes); HuggingFace caches the tokenized result so re-runs skip both.

Note on gradient accumulation: Gnome *does* support accumulation (pass a list
of main closures), but this script runs a single-batch step at ``--batch-size``
and scales the effective budget by raising batch size + step count. Default
``batch=48`` fits a 10-layer / 352-embd GPT on a mid-size GPU.
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gnome import Gnome
from experiments.baselines import SOAP
from experiments.common import RunLogger, pick_device


EXPERIMENT = "wikitext_gpt"


# ========================= Model =========================

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, bias: bool = False) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, bias: bool = False) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, bias)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=bias),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """Small GPT-2-style transformer with weight-tied token + output embeddings.

    Constructor args follow the nanoGPT conventions. ``c_proj`` and the
    second MLP linear get a scaled init to compensate for residual accumulation
    over ``n_layer`` blocks (same as the source script).
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(seq_len, n_embd)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, bias) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("mlp.2.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer)
                )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


# ========================= Data =========================

def load_wikitext103(seq_len: int, batch_size: int, seed: int):
    """Tokenize and chunk WikiText-103 into ``seq_len + 1`` blocks.

    First call downloads the raw text + GPT-2 tokenizer and runs the
    tokenize/group_texts map (~5 minutes). HuggingFace's ``datasets`` cache
    persists between runs so subsequent calls are fast.
    """
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast

    print(f"[{EXPERIMENT}] tokenizing WikiText-103 (GPT-2 BPE)...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    block_size = seq_len + 1

    def tokenize(element):
        return tokenizer(element["text"])

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        return {
            k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }

    grouped = tokenized.map(group_texts, batched=True)
    grouped.set_format(type="torch", columns=["input_ids"])

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        grouped["train"], batch_size=batch_size, shuffle=True, generator=g,
    )
    val_loader = DataLoader(
        grouped["validation"], batch_size=batch_size, shuffle=False,
    )
    test_loader = DataLoader(
        grouped["test"], batch_size=batch_size, shuffle=False,
    )
    print(
        f"[{EXPERIMENT}] train batches: {len(train_loader)}  "
        f"val batches: {len(val_loader)}  test batches: {len(test_loader)}"
    )
    return train_loader, val_loader, test_loader, tokenizer


# ========================= Optimizer factory =========================

def build_optimizer_and_config(name: str, params, lr: float, weight_decay: float):
    """Construct one of the four supported optimizers.

    Both Gnome variants share every hyperparameter except ``loss=`` so the
    A/B between Fisher sampling and Hutchinson is on the surrogate only.

    Gnome's internal warmup is disabled (warmup=0): this experiment owns the
    LR schedule via ``cosine_with_warmup``, applied to every optimizer's
    ``group["lr"]`` each step. Leaving Gnome's internal warmup on would compound
    the two ramps (Gnome multiplies its own warmup factor onto the already-
    cosined lr), handicapping it vs SOAP/AdamW which have no internal warmup.

    aux_batch_size is small (5) because the Hutchinson surrogate's per-token
    tensors are ``(aux*seq_len, vocab)`` ≈ ``(aux*seq_len, 50k)`` — at aux=10
    those temporaries are >1 GB. Each aux sequence already provides
    seq_len × vocab worth of Rademacher coverage so small aux is fine here.
    """
    common_gnome = dict(
        lr=lr, weight_decay=weight_decay,
        betas=(0.9, 0.95), shampoo_beta=0.95, eps=1e-4,
        precondition_frequency=10,
        clip=1.0, warmup=0,
        precondition_1d=False,
    )
    # aux_batch_size sizes the auxiliary batch the caller builds for
    # opt.step(...); it is not a Gnome constructor arg (see K below, and the
    # docstring above for why it is kept small).
    aux_k = 5
    if name == "gnome_fisher":
        cfg = dict(common_gnome, loss="cce")
        opt = Gnome(params, **cfg)
        cfg["aux_batch_size"] = aux_k
        return opt, cfg
    if name == "gnome_hutchinson":
        cfg = dict(common_gnome, loss="cce_hutchinson")
        opt = Gnome(params, **cfg)
        cfg["aux_batch_size"] = aux_k
        return opt, cfg
    if name == "soap":
        cfg = dict(
            lr=lr, weight_decay=weight_decay,
            betas=(0.9, 0.95), shampoo_beta=0.95, eps=1e-8,
            precondition_frequency=10, precondition_1d=False,
        )
        return SOAP(params, **cfg), cfg
    if name == "adamw":
        cfg = dict(
            lr=lr, weight_decay=weight_decay,
            betas=(0.9, 0.95), eps=1e-8,
        )
        return torch.optim.AdamW(params, **cfg), cfg
    raise ValueError(
        f"unknown optimizer {name!r}; expected one of "
        f"'gnome_fisher', 'gnome_hutchinson', 'soap', 'adamw'"
    )


# ========================= Evaluation =========================

@torch.no_grad()
def evaluate(model: nn.Module, dataloader, device: torch.device):
    """Return (avg cross-entropy per token, perplexity) over the loader."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in dataloader:
        data = batch["input_ids"].to(device)
        inputs, targets = data[:, :-1], data[:, 1:]
        logits = model(inputs)
        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(-1, V), targets.reshape(-1), reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += targets.numel()
    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss, math.exp(avg_loss)


# ========================= Training =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--optimizer", required=True,
        choices=["gnome_fisher", "gnome_hutchinson", "soap", "adamw"],
    )
    p.add_argument("--seed", type=int, default=0)
    # The transformer needs more steps than the CNN experiments; we cap by
    # total optimizer steps rather than epochs so the budget is comparable
    # across batch sizes.
    p.add_argument("--max-steps", type=int, default=5000,
                   help="Total optimizer steps to run.")
    p.add_argument("--batch-size", type=int, default=48,
                   help="Batch size per optimizer step. Raise this to scale "
                        "the effective batch (single-batch steps here).")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-min-frac", type=float, default=0.1,
                   help="Cosine schedule floor as a fraction of lr (final lr).")
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--max-grad-norm", type=float, default=1.0,
                   help="Global-norm gradient clipping for non-Gnome paths. "
                        "Gnome uses its own per-coordinate clip (clip=1.0).")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--n-layer", type=int, default=10)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--n-embd", type=int, default=352)
    p.add_argument("--val-every", type=int, default=500,
                   help="Run validation every N optimizer steps.")
    p.add_argument("--log-every", type=int, default=50,
                   help="Print a running-mean train loss every N steps.")
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def cosine_with_warmup(step: int, warmup: int, total: int, min_frac: float) -> float:
    """LR multiplier: linear warmup → cosine decay to ``min_frac``.

    Note: this CCE schedule ramps warmup from ``min_frac`` up to 1.0 (not from
    0), and is applied to every optimizer including Gnome — distinct from the
    MSE baselines' ``experiments.common.baseline_cosine_scheduler`` (warmup from
    0, applied only to SOAP/AdamW). Kept local to preserve the tuned LM run.
    """
    if step < warmup:
        return min_frac + (1.0 - min_frac) * step / max(warmup, 1)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(progress, 1.0)
    return min_frac + 0.5 * (1.0 - min_frac) * (1.0 + math.cos(math.pi * progress))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"[{EXPERIMENT}] {args.optimizer} | device={device}")

    train_loader, val_loader, test_loader, _ = load_wikitext103(
        args.seq_len, args.batch_size, args.seed,
    )

    vocab_size = 50257
    model = GPT(
        vocab_size=vocab_size, seq_len=args.seq_len,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, bias=False,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[{EXPERIMENT}] GPT  n_layer={args.n_layer}  n_head={args.n_head}  "
        f"n_embd={args.n_embd}  seq_len={args.seq_len}  params={n_params:,}"
    )

    opt, opt_cfg = build_optimizer_and_config(
        args.optimizer, model.parameters(), args.lr, args.weight_decay,
    )
    K = opt_cfg.get("aux_batch_size", 4) if args.optimizer.startswith("gnome") else 0

    hyperparameters = {
        "optimizer": args.optimizer,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "warmup_steps": args.warmup_steps,
        "lr": args.lr,
        "lr_min_frac": args.lr_min_frac,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "seq_len": args.seq_len,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "n_embd": args.n_embd,
        "vocab_size": vocab_size,
        "n_params": n_params,
        "device": str(device),
        **{f"opt.{k}": v for k, v in opt_cfg.items()},
    }
    run = RunLogger(
        experiment=EXPERIMENT,
        optimizer=args.optimizer,
        seed=args.seed,
        hyperparameters=hyperparameters,
        runs_dir=args.runs_dir,
    )

    # Manual LR schedule — applied each step before opt.step() so it works
    # for both the closure-based Gnome path and the standard backward path.
    def set_lr(step: int) -> float:
        mul = cosine_with_warmup(
            step, args.warmup_steps, args.max_steps, args.lr_min_frac,
        )
        new_lr = args.lr * mul
        for group in opt.param_groups:
            group["lr"] = new_lr
        return new_lr

    train_iter = iter(train_loader)
    step = 0
    window_loss_sum = 0.0
    window_n = 0
    last_val_loss = last_val_ppl = float("nan")
    best_val_ppl = float("inf")
    t_start = time.time()
    while step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        data = batch["input_ids"].to(device)
        if data.size(0) < 2:
            continue
        inputs, targets = data[:, :-1], data[:, 1:]
        B, T = inputs.shape

        lr_now = set_lr(step)
        model.train()

        if args.optimizer.startswith("gnome"):
            # Aux: take the first K sequences of the batch. Gnome's main and aux
            # gradients are computed in independent forward passes, so strict
            # disjointness isn't required — this mirrors the other experiments.
            k = min(K, max(1, B - 1))
            aux_inputs = inputs[:k]
            aux_targets = targets[:k]

            def main_closure():
                logits = model(inputs)
                V = logits.shape[-1]
                # Flatten to (N, V) and (N,) so Gnome's internal
                # F.cross_entropy and softmax-Hessian work without reshape.
                return logits.reshape(-1, V), targets.reshape(-1)

            def aux_closure():
                logits = model(aux_inputs)
                V = logits.shape[-1]
                return logits.reshape(-1, V), aux_targets.reshape(-1)

            loss = opt.step(main_closure, aux_closure)
        else:
            opt.zero_grad()
            logits = model(inputs)
            V = logits.shape[-1]
            loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()

        loss_val = float(loss.detach().item())
        run.log_train(step, loss=loss_val, lr=lr_now)
        window_loss_sum += loss_val
        window_n += 1
        step += 1

        if (not args.quiet) and args.log_every > 0 and step % args.log_every == 0:
            avg = window_loss_sum / max(window_n, 1)
            elapsed = time.time() - t_start
            ms = 1000.0 * elapsed / step
            print(
                f"  step {step:6d}/{args.max_steps}  "
                f"train_loss[last {window_n}]={avg:.4f}  ppl={math.exp(avg):.2f}  "
                f"lr={lr_now:.2e}  {ms:.1f}ms/step",
                flush=True,
            )
            window_loss_sum = 0.0
            window_n = 0

        if step % args.val_every == 0 or step == args.max_steps:
            val_loss, val_ppl = evaluate(model, val_loader, device)
            last_val_loss, last_val_ppl = val_loss, val_ppl
            best_val_ppl = min(best_val_ppl, val_ppl)
            run.log_val(step, loss=val_loss, ppl=val_ppl)
            if not args.quiet:
                print(
                    f"  [val @ step {step}] loss={val_loss:.4f}  ppl={val_ppl:.2f}",
                    flush=True,
                )

    test_loss, test_ppl = evaluate(model, test_loader, device)
    path = run.finish(
        completed=True,
        test_loss=test_loss, test_ppl=test_ppl,
        final_val_loss=last_val_loss, final_val_ppl=last_val_ppl,
        best_val_ppl=best_val_ppl,
    )
    print(f"[{EXPERIMENT}] saved → {path}")
    print(
        f"  test_loss={test_loss:.4f}  test_ppl={test_ppl:.2f}  "
        f"final val_ppl={last_val_ppl:.2f}  best val_ppl={best_val_ppl:.2f}"
    )


if __name__ == "__main__":
    main()
