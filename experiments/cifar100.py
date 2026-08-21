"""CIFAR-100 classification: Gnome (Fisher) vs Gnome (Hutchinson) vs SOAP vs AdamW.

A small, fast CCE benchmark to bookend ``wikitext_gpt``. That experiment runs
cross-entropy at ``vocab_size=50257``, where the tied embedding is more than
half the parameters and its 50257-wide axis exceeds ``max_precond_dim``, so it
gets no Kronecker factor at all. Here ``K=100``: every factor fits, every tensor
is fully rotated, and a run is minutes rather than hours. That makes this the
place to check CCE behaviour — surrogate variance, trust-region sizing, ``eps``
— before spending compute on the transformer.

Optimizer choices mirror ``wikitext_gpt`` so the surrogate A/B is the same:

    * ``gnome_fisher``      — Gnome with ``loss="cce"`` (Fisher sampling: one
                              class drawn per aux sample from softmax(logits))
    * ``gnome_hutchinson``  — Gnome with ``loss="cce_hutchinson"`` (Rademacher
                              over the analytic softmax-Hessian square root,
                              covering all K classes per aux sample at once)
    * ``soap``              — empirical-Fisher SOAP baseline
    * ``adamw``             — first-order baseline

The variance-reduction argument for Hutchinson predicts the gap between the two
Gnome variants should *narrow* here relative to wikitext: with K=100 rather than
50257, Fisher sampling's single drawn class already covers 1% of the classes per
sample instead of 0.002%. A null result here is informative — it would say the
Hutchinson advantage is specifically a large-K effect.

Every optimizer here — Gnome included — gets a shared linear-warmup +
cosine-decay schedule. Decay matters more here than on the MSE experiments,
where Gnome's step self-anneals as the residual shrinks: cross-entropy
gradients do not self-anneal, since the Fisher stays O(1) at the optimum rather
than going to zero.

    uv run python -m experiments.cifar100 --optimizer gnome_hutchinson --seed 0
    uv run python -m experiments.cifar100 --optimizer gnome_fisher     --seed 0
    uv run python -m experiments.cifar100 --optimizer soap             --seed 0
    uv run python -m experiments.cifar100 --optimizer adamw            --seed 0

CIFAR-100 downloads once via torchvision (~170MB) and is cached under
``experiments/data``.
"""

from __future__ import annotations

import argparse
import math
import warnings

import torch
import torch.nn.functional as F

from gnome import Gnome
from experiments.baselines import SOAP
from experiments.common import (
    DIVERGED_EXIT,
    diverged,
    RunLogger,
    pick_device,
    cosine_with_warmup,
)
from experiments.common.resnet import build_model, MODEL_NAMES


EXPERIMENT = "cifar100"
DEFAULT_DATA_DIR = "experiments/data"
NUM_CLASSES = 100

# Standard CIFAR-100 per-channel statistics.
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)

# Normalized value of a black (raw 0) pixel — what torchvision's
# RandomCrop(padding=4) border becomes once Normalize runs after the crop.
PAD_VALUE = torch.tensor(
    [-m / s for m, s in zip(CIFAR100_MEAN, CIFAR100_STD)]
).view(1, 3, 1, 1)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def load_cifar100_tensors(data_dir: str = DEFAULT_DATA_DIR):
    """Load CIFAR-100 train+test as ``([N,3,32,32] float32, [N] int64)`` pairs.

    Images are channel-normalized with the standard CIFAR-100 statistics.
    Downloads once via torchvision (cached under ``data_dir``). The unverified
    SSL context is a standard workaround for intermittent cert failures on the
    upstream CIFAR host — same as ``cifar_rotation``.
    """
    import ssl
    from torchvision import datasets, transforms

    ssl._create_default_https_context = ssl._create_unverified_context
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tfm = transforms.ToTensor()
        train = datasets.CIFAR100(data_dir, train=True, download=True, transform=tfm)
        test = datasets.CIFAR100(data_dir, train=False, download=True, transform=tfm)

    x_train = torch.stack([img for img, _ in train])
    y_train = torch.tensor([lbl for _, lbl in train], dtype=torch.long)
    x_val = torch.stack([img for img, _ in test])
    y_val = torch.tensor([lbl for _, lbl in test], dtype=torch.long)

    mean = torch.tensor(CIFAR100_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR100_STD).view(1, 3, 1, 1)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    return x_train, y_train, x_val, y_val


def augment(x: torch.Tensor) -> torch.Tensor:
    """The standard CIFAR recipe: ``RandomCrop(32, padding=4)`` + horizontal flip.

    Equivalent to torchvision's

        RandomCrop(32, padding=4); RandomHorizontalFlip(); ToTensor(); Normalize()

    including the padding value. That last part is easy to get wrong: torchvision
    crops the *raw* image and normalizes afterwards, so the 4-px border is black
    (raw 0), which lands at ``(0 - mean)/std`` once normalized. Here the tensors
    are normalized at load time, so padding with a plain 0 would make the border
    the dataset *mean* (gray) instead. ``PAD_VALUE`` restores black; the
    shift-pad-unshift trick applies it per channel, since ``F.pad`` takes only a
    scalar constant.

    Batched, and run on CPU before the host->device copy (see the train loop).
    Without augmentation a ResNet memorizes CIFAR-100 within a few epochs and the
    val curve stops distinguishing optimizers.
    """
    b = x.shape[0]
    flip = torch.rand(b, device=x.device) < 0.5
    x = torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)

    pv = PAD_VALUE.to(x.device, x.dtype)
    padded = F.pad(x - pv, (4, 4, 4, 4)) + pv
    # One random (dy, dx) offset per sample, gathered via advanced indexing.
    dy = torch.randint(0, 9, (b,), device=x.device)
    dx = torch.randint(0, 9, (b,), device=x.device)
    rows = dy.view(-1, 1) + torch.arange(32, device=x.device).view(1, -1)  # [B,32]
    cols = dx.view(-1, 1) + torch.arange(32, device=x.device).view(1, -1)  # [B,32]
    bi = torch.arange(b, device=x.device).view(-1, 1, 1)
    return padded[bi, :, rows.unsqueeze(-1), cols.unsqueeze(-2)].permute(0, 3, 1, 2)


# ----------------------------------------------------------------------
# Optimizer
# ----------------------------------------------------------------------

def build_optimizer(
    name: str, params, lr: float, weight_decay: float,
    beta1: float = 0.9, beta2: float = 0.99, eps: float = 1e-6,
    trust_region: float = 1.0, aux_batch_size: int = 32,
):
    """Return ``(optimizer, config)``.

    No scheduler is returned: this experiment owns the LR schedule itself and
    applies it to every optimizer's ``group["lr"]`` each step (see ``set_lr``),
    because CCE gradients don't self-anneal the way an MSE residual does.

    Both Gnome variants share every hyperparameter except ``loss=`` so the A/B
    is on the surrogate alone. ``precondition_1d`` is off — the 1D norm
    gamma/beta tensors carry no cross-coordinate structure worth a factor.
    """
    common_gnome = dict(
        lr=lr, weight_decay=weight_decay,
        betas=(beta1, beta2), shampoo_beta=beta2, eps=eps,
        precondition_frequency=10,
        trust_radius=(trust_region if trust_region > 0 else None),
        # CIFAR-100 moves 4.6 -> 2.6 loss inside ~200 steps, which is ~2 EMA
        # windows at beta2=0.99. Without this the curvature EMA is dominated by
        # the oldest (largest ||g_s||) entries in its window and tracks a scale
        # that is already stale. CCE has no residual-driven self-annealing to
        # lose here, unlike the MSE experiments.
        norm_free=False,
        precondition_1d=False,
    )
    if name in ("gnome_fisher", "gnome_hutchinson"):
        loss_mode = "cce" if name == "gnome_fisher" else "cce_hutchinson"
        cfg = dict(common_gnome, loss=loss_mode)
        opt = Gnome(params, **cfg)
        # aux_batch_size sizes the auxiliary batch the caller builds for
        # opt.step(...); it is not a Gnome constructor arg.
        cfg["aux_batch_size"] = aux_batch_size
        return opt, cfg
    if name == "soap":
        cfg = dict(
            lr=lr, weight_decay=weight_decay,
            betas=(beta1, beta2), shampoo_beta=beta2, eps=1e-8,
            precondition_frequency=10, precondition_1d=False,
        )
        return SOAP(params, **cfg), cfg
    if name == "adamw":
        cfg = dict(lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)
        return torch.optim.AdamW(params, **cfg), cfg
    raise ValueError(f"unknown optimizer: {name}")


# ----------------------------------------------------------------------
# Eval
# ----------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, x_val, y_val, batch_size: int):
    """Mean CCE loss and top-1 / top-5 accuracy over the val set."""
    was_training = model.training
    model.eval()
    total_loss, correct1, correct5, n = 0.0, 0, 0, 0
    for i in range(0, x_val.shape[0], batch_size):
        xb, yb = x_val[i:i + batch_size], y_val[i:i + batch_size]
        logits = model(xb)
        total_loss += F.cross_entropy(logits, yb, reduction="sum").item()
        top5 = logits.topk(5, dim=-1).indices
        correct1 += (top5[:, 0] == yb).sum().item()
        correct5 += (top5 == yb.unsqueeze(-1)).any(dim=-1).sum().item()
        n += yb.numel()
    if was_training:
        model.train()
    return total_loss / n, correct1 / n, correct5 / n


# ----------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device()

    print(f"[{EXPERIMENT}] {args.optimizer} | loading CIFAR-100...", flush=True)
    x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu = load_cifar100_tensors()
    n_train = int(x_train_cpu.shape[0])
    x_val, y_val = x_val_cpu.to(device), y_val_cpu.to(device)

    model = build_model(args.model, num_outputs=NUM_CLASSES, norm=args.norm).to(device)
    steps_per_epoch = math.ceil(n_train / args.batch_size)
    total_steps = args.epochs * steps_per_epoch

    is_gnome = args.optimizer.startswith("gnome")
    opt, opt_cfg = build_optimizer(
        args.optimizer, model.parameters(), args.lr, args.weight_decay,
        beta1=args.beta1, beta2=args.beta2, eps=args.eps,
        trust_region=args.trust_region, aux_batch_size=args.aux_batch_size,
    )
    K = opt_cfg.get("aux_batch_size", 0) if is_gnome else 0

    # Manual LR schedule — applied each step before opt.step() so it works for
    # both the closure-based Gnome path and the standard backward path.
    def set_lr(step: int) -> float:
        mul = cosine_with_warmup(
            step, args.warmup_steps, total_steps, args.lr_min_frac,
        )
        new_lr = args.lr * mul
        for group in opt.param_groups:
            group["lr"] = new_lr
        return new_lr

    hyperparameters = {
        "optimizer": args.optimizer,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "n_train": n_train,
        "n_val": int(x_val.shape[0]),
        "num_classes": NUM_CLASSES,
        "model": args.model,
        "norm": args.norm,
        "augment": not args.no_augment,
        "warmup_steps": args.warmup_steps,
        "lr_min_frac": args.lr_min_frac,
        "n_params": sum(p.numel() for p in model.parameters()),
        "device": str(device),
        **{f"opt.{k}": v for k, v in opt_cfg.items()},
    }

    if not args.quiet:
        print(f"[{EXPERIMENT}] {args.optimizer} | {args.model}/{args.norm} | "
              f"params={hyperparameters['n_params']:,} | device={device}\n"
              f"  epochs={args.epochs} batch={args.batch_size} "
              f"steps={total_steps} aux_K={K}", flush=True)

    with RunLogger(EXPERIMENT, args.optimizer, args.seed, hyperparameters,
                   runs_dir=args.runs_dir) as run:
        step = 0
        best_top1 = 0.0
        window_sum, window_n = 0.0, 0
        for epoch in range(args.epochs):
            model.train()
            perm = torch.randperm(n_train)
            for i in range(0, n_train, args.batch_size):
                idx = perm[i:i + args.batch_size]
                # Augment on CPU, then transfer. Two reasons, both load-bearing:
                # the pad temporary never reaches device memory, and — more
                # importantly — the source of the host->device copy is a live
                # named tensor rather than an unreferenced temporary.
                #
                # ``x_train_cpu[idx].to(device, non_blocking=True)`` is a real
                # bug: the indexed result is a temporary in pageable memory with
                # no surviving reference, so an async copy can read it after
                # it has been freed. It reproduces on MPS at a *new* batch size
                # (the allocator hands back a fresh block whose lifetime differs
                # from the cached one), which is why it fired only on the final
                # partial batch of an epoch and never at a batch size that
                # divides the dataset evenly. ``non_blocking`` stays off: the
                # very next op needs the data, so there is nothing to overlap.
                xb = x_train_cpu[idx]
                if not args.no_augment:
                    xb = augment(xb)
                xb = xb.to(device)
                yb = y_train_cpu[idx].to(device)
                lr_now = set_lr(step)

                if is_gnome:
                    b = xb.shape[0]
                    k = min(K, max(1, b - 1))
                    a_idx = torch.randperm(b, device=device)[:k]
                    x_aux, y_aux = xb[a_idx], yb[a_idx]

                    def main_closure():
                        return model(xb), yb

                    def aux_closure():
                        return model(x_aux), y_aux

                    loss = opt.step(main_closure, aux_closure)
                else:
                    opt.zero_grad()
                    loss = F.cross_entropy(model(xb), yb)
                    loss.backward()
                    opt.step()

                loss_val = float(loss.detach().item())
                if diverged(loss_val):
                    # Localize the failure before exiting. The fork that
                    # matters: are the *parameters* already non-finite (so the
                    # bad update happened on an earlier step and this loss is
                    # only where it surfaced), or are they clean and this one
                    # forward produced the NaN?
                    bad_p = [n for n, p in model.named_parameters()
                             if not torch.isfinite(p).all()]
                    n_p = sum(1 for _ in model.parameters())
                    with torch.no_grad():
                        logits = model(xb)
                    print(f"[{EXPERIMENT}] diverged at step {step} — stopping.\n"
                          f"  loss={loss_val!r}  batch_shape={tuple(xb.shape)}\n"
                          f"  non-finite params: {len(bad_p)}/{n_p}"
                          + (f"  first={bad_p[:4]}" if bad_p else "")
                          + f"\n  logits finite={bool(torch.isfinite(logits).all())}"
                            f"  |logits|max={float(logits.abs().max()):.3e}"
                            f"  x finite={bool(torch.isfinite(xb).all())}",
                          flush=True)
                    run.finish(completed=False, diverged=True, diverged_step=step)
                    raise SystemExit(DIVERGED_EXIT)
                run.log_train(step, loss=loss_val, lr=lr_now)
                window_sum += loss_val
                window_n += 1
                step += 1

                if (not args.quiet) and args.log_every > 0 and step % args.log_every == 0:
                    avg = window_sum / max(window_n, 1)
                    print(f"    step {step:6d}  epoch {epoch:3d}  "
                          f"train_loss[last {window_n}]={avg:.4f}  lr={lr_now:.2e}",
                          flush=True)
                    window_sum, window_n = 0.0, 0

            val_loss, top1, top5 = evaluate(model, x_val, y_val, args.batch_size)
            best_top1 = max(best_top1, top1)
            run.log_val(step, epoch=epoch, loss=val_loss, top1=top1, top5=top5,
                        lr=lr_now)
            if not args.quiet:
                print(f"  epoch {epoch:3d}/{args.epochs}  val_loss={val_loss:.4f}  "
                      f"top1={top1*100:.2f}%  top5={top5*100:.2f}%", flush=True)

        run.finish(completed=True, final_top1=top1, best_top1=best_top1,
                   final_val_loss=val_loss)
    print(f"[{EXPERIMENT}] done → final_top1={top1*100:.2f}%  "
          f"best_top1={best_top1*100:.2f}%", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--optimizer", required=True,
                   choices=["gnome_fisher", "gnome_hutchinson", "soap", "adamw"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--trust-region", type=float, default=1.0,
                   help="Gnome per-coordinate update bound: lambda is set to "
                        "the smallest value with max|m̂/(v̂+lambda)| <= this, "
                        "so no coordinate moves more than lr*trust_region in "
                        "a step. Larger -> weaker bound -> longer steps. "
                        "0 disables it, falling back to plain m̂/(v̂+eps) "
                        "damping.")
    p.add_argument("--eps", type=float, default=1e-6,
                   help="Gnome curvature-damping epsilon; with the trust region "
                        "active this is only the lower bound on lambda, so it "
                        "can go much smaller than the pre-trust-region default. "
                        "Gnome only; SOAP/AdamW keep their fixed eps=1e-8.")
    p.add_argument("--beta1", type=float, default=0.9,
                   help="First-moment (momentum) EMA for Gnome and SOAP.")
    p.add_argument("--beta2", type=float, default=0.99,
                   help="Second-moment / preconditioner EMA (also shampoo_beta) "
                        "for Gnome and SOAP.")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--aux-batch-size", type=int, default=10,
                   help="Aux batch K for the Gnome surrogate, drawn as a subset "
                        "of the main batch. Cheap here: the Hutchinson "
                        "surrogate's per-sample tensor is only (K, 100).")
    p.add_argument("--model", choices=MODEL_NAMES, default="resnet12",
                   help="Architecture (default resnet12).")
    p.add_argument("--norm", choices=["gn", "bn"], default="gn",
                   help="Normalization: GroupNorm (default) or BatchNorm. "
                        "GroupNorm keeps the per-sample block structure the "
                        "surrogate assumes (see docs/method.md 5.3).")
    p.add_argument("--no-augment", action="store_true",
                   help="Disable flip + random-crop augmentation.")
    p.add_argument("--warmup-steps", type=int, default=200,
                   help="LR warmup steps, applied to every optimizer including "
                        "Gnome (CCE does not self-anneal).")
    p.add_argument("--lr-min-frac", type=float, default=0.1,
                   help="Cosine floor: final lr as a fraction of peak.")
    p.add_argument("--log-every", type=int, default=50,
                   help="Print a running-mean train loss every N steps (0 "
                        "disables). Per-step train loss is logged regardless.")
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
