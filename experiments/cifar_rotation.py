"""CIFAR-10 rotation regression: AdamW vs SOAP vs Gnome on real images.

A real-image regression benchmark to bookend the synthetic OLS example.
Each CIFAR-10 image is rotated by a random angle sampled from U(-90°, +90°),
and the model must regress that angle back. The target is ``angle / 90``
(normalized to [-1, 1]); the loss is MSE on that target, and ``mae_deg``
un-normalizes the validation error into degrees for a human-readable readout.

Training angles are resampled fresh every batch, so the model never sees the
same (image, angle) pair twice — the rotation is its own on-the-fly
augmentation, which keeps the task well-conditioned. Validation angles are
frozen per seed, so val metrics are comparable across runs and epochs.

This is a genuine conv-net regression (Gnome runs on ``[C, C, 3, 3]`` conv
kernels, not just Linear layers) on real data, without a fragile external
dataset — CIFAR-10 downloads once via torchvision and is cached.

    uv run python -m experiments.cifar_rotation --optimizer gnome --seed 0
    uv run python -m experiments.cifar_rotation --optimizer soap  --seed 0 --cosine-decay 0
    uv run python -m experiments.cifar_rotation --optimizer adamw --seed 0 --cosine-decay 0
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
    baseline_cosine_scheduler,
    current_lr,
)
from experiments.common.resnet import build_model, MODEL_NAMES


EXPERIMENT = "cifar_rotation"
DEFAULT_DATA_DIR = "experiments/data"

# Rotations are sampled uniformly from [-90, +90] degrees; the regression
# target is angle / ANGLE_RANGE_DEG, so it lands in [-1, 1]. mae_deg
# un-normalizes the validation residual back into degrees.
ANGLE_RANGE_DEG = 90.0


# ----------------------------------------------------------------------
# Data + rotation
# ----------------------------------------------------------------------

def rotate_batch(x: torch.Tensor, angles_deg: torch.Tensor) -> torch.Tensor:
    """Per-sample rotation of an image batch about its center (one batched op).

    Builds a per-sample 2×3 affine matrix from the angles and applies it via
    ``affine_grid`` + ``grid_sample`` — no Python loop over the batch. Regions
    outside the source frame are zero-padded.
    """
    theta = angles_deg * (math.pi / 180.0)
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    zero = torch.zeros_like(cos_t)
    rot = torch.stack([
        torch.stack([cos_t, -sin_t, zero], dim=-1),
        torch.stack([sin_t,  cos_t, zero], dim=-1),
    ], dim=-2)  # [B, 2, 3]
    grid = F.affine_grid(rot, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


def load_cifar10_tensors(data_dir: str = DEFAULT_DATA_DIR):
    """Load CIFAR-10 train+test as float32 ``[N,3,32,32]`` tensors in [-1, 1].

    Downloads once via torchvision (cached under ``data_dir``). The unverified
    SSL context is a standard workaround for intermittent cert failures on the
    upstream CIFAR host.
    """
    import ssl
    from torchvision import datasets, transforms

    ssl._create_default_https_context = ssl._create_unverified_context
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tfm = transforms.ToTensor()
        train = datasets.CIFAR10(data_dir, train=True, download=True, transform=tfm)
        test = datasets.CIFAR10(data_dir, train=False, download=True, transform=tfm)
    x_train = torch.stack([img for img, _ in train])
    x_val = torch.stack([img for img, _ in test])
    x_train = (x_train - 0.5) / 0.5
    x_val = (x_val - 0.5) / 0.5
    return x_train, x_val


# ----------------------------------------------------------------------
# Optimizer + schedule
# ----------------------------------------------------------------------

def build_optimizer(name, params, lr, weight_decay, warmup, total_steps, cosine_decay,
                    eps=1e-6, beta1=0.9, beta2=0.99):
    """Return ``(optimizer, config, scheduler)`` with conv-friendly defaults.

    MSE regression, so the repo protocol applies: Gnome runs at a fixed
    learning rate (self-anneals; no scheduler), while SOAP/AdamW get warmup +
    cosine decay to a ``cosine_decay`` final-LR fraction. ``precondition_1d`` is
    off — the small 1D norm gamma/beta tensors carry no cross-coordinate
    structure worth a Kronecker factor.
    """
    if name == "gnome":
        cfg = dict(
            lr=lr, weight_decay=weight_decay,
            betas=(beta1, beta2), shampoo_beta=beta2, eps=eps,
            precondition_frequency=10, aux_batch_size=10,
            clip=1.0, warmup=warmup, loss="mse", precondition_1d=False,
        )
        return Gnome(params, **cfg), cfg, None
    if name == "soap":
        cfg = dict(
            lr=lr, weight_decay=weight_decay,
            betas=(beta1, beta2), shampoo_beta=beta2, eps=1e-8,
            precondition_frequency=10, precondition_1d=False,
        )
        opt = SOAP(params, **cfg)
    elif name == "adamw":
        cfg = dict(lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)
        opt = torch.optim.AdamW(params, **cfg)
    else:
        raise ValueError(f"unknown optimizer: {name}")
    cfg["warmup"] = warmup   # unified meta key across optimizers
    scheduler = baseline_cosine_scheduler(opt, warmup, total_steps, cosine_decay)
    return opt, cfg, scheduler


# ----------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device()

    print(f"[{EXPERIMENT}] {args.optimizer} | loading CIFAR-10...", flush=True)
    x_train_cpu, x_val_cpu = load_cifar10_tensors()
    n_train = int(x_train_cpu.shape[0])

    # Frozen val rotations (deterministic per seed) so val metrics are
    # comparable across runs, optimizers, and epochs.
    val_gen = torch.Generator().manual_seed(args.seed)
    val_angles = (torch.rand(x_val_cpu.shape[0], generator=val_gen) * 2 - 1) * ANGLE_RANGE_DEG
    with torch.no_grad():
        x_val = rotate_batch(x_val_cpu, val_angles).to(device)
    y_val = (val_angles / ANGLE_RANGE_DEG).unsqueeze(-1).to(device)

    model = build_model(args.model, num_outputs=1, norm=args.norm).to(device)
    steps_per_epoch = math.ceil(n_train / args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    opt, opt_cfg, scheduler = build_optimizer(
        args.optimizer, model.parameters(), args.lr, args.weight_decay,
        args.warmup_steps, total_steps, args.cosine_decay, eps=args.eps,
        beta1=args.beta1, beta2=args.beta2,
    )
    K = opt_cfg.get("aux_batch_size", 10) if args.optimizer == "gnome" else 0

    hyperparameters = {
        "optimizer": args.optimizer,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "n_train": n_train,
        "n_val": int(x_val.shape[0]),
        "angle_range_deg": ANGLE_RANGE_DEG,
        "model": args.model,
        "norm": args.norm,
        "warmup_steps": args.warmup_steps,
        "cosine_decay": args.cosine_decay,
        "n_params": sum(p.numel() for p in model.parameters()),
        "device": str(device),
        **{f"opt.{k}": v for k, v in opt_cfg.items()},
    }

    with RunLogger(EXPERIMENT, args.optimizer, args.seed, hyperparameters,
                   runs_dir=args.runs_dir) as run:
        step = 0
        best_mae_deg = float("inf")
        window_sum, window_n = 0.0, 0
        for epoch in range(args.epochs):
            model.train()
            perm = torch.randperm(n_train)
            for i in range(0, n_train, args.batch_size):
                idx = perm[i:i + args.batch_size]
                x_batch = x_train_cpu[idx].to(device, non_blocking=True)
                b = x_batch.shape[0]
                # Fresh random rotation for every image each time it's seen.
                angles = (torch.rand(b, device=device) * 2 - 1) * ANGLE_RANGE_DEG
                x_rot = rotate_batch(x_batch, angles)
                y_batch = (angles / ANGLE_RANGE_DEG).unsqueeze(-1)

                if args.optimizer == "gnome":
                    k = min(K, max(1, b - 1))
                    a_idx = torch.randperm(b, device=device)[:k]
                    x_main, y_main = x_rot, y_batch
                    x_aux, y_aux = x_rot[a_idx], y_batch[a_idx]

                    def main_closure():
                        return model(x_main), y_main

                    def aux_closure():
                        return model(x_aux), y_aux

                    loss = opt.step(main_closure, aux_closure)
                else:
                    opt.zero_grad()
                    loss = F.mse_loss(model(x_rot), y_batch)
                    loss.backward()
                    opt.step()
                if scheduler is not None:
                    scheduler.step()

                loss_val = float(loss.detach().item())
                if diverged(loss_val):
                    run.finish(completed=False, diverged=True, diverged_step=step)
                    print(f"[{EXPERIMENT}] diverged at step {step} — stopping.", flush=True)
                    raise SystemExit(DIVERGED_EXIT)
                run.log_train(step, loss=loss_val)
                window_sum += loss_val
                window_n += 1
                step += 1

                if (not args.quiet) and args.log_every > 0 and step % args.log_every == 0:
                    avg = window_sum / max(window_n, 1)
                    # loss is MSE on the [-1,1] angle target, so sqrt(avg)*90 is
                    # a human-readable train RMSE in degrees.
                    rmse_deg = (avg ** 0.5) * ANGLE_RANGE_DEG
                    print(f"    step {step:6d}  epoch {epoch:3d}  "
                          f"train_loss[last {window_n}]={avg:.4f}  "
                          f"(~{rmse_deg:.1f}° RMSE)", flush=True)
                    window_sum, window_n = 0.0, 0

            # ---- validation ----
            model.eval()
            with torch.no_grad():
                preds = [model(x_val[i:i + args.batch_size])
                         for i in range(0, x_val.shape[0], args.batch_size)]
                val_pred = torch.cat(preds)
                val_loss = F.mse_loss(val_pred, y_val).item()
                ss_res = ((val_pred - y_val) ** 2).sum().item()
                ss_tot = ((y_val - y_val.mean()) ** 2).sum().item()
                r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
                mae_deg = (val_pred - y_val).abs().mean().item() * ANGLE_RANGE_DEG
            best_mae_deg = min(best_mae_deg, mae_deg)
            run.log_val(step, epoch=epoch, loss=val_loss, r2=r2,
                        mae_deg=mae_deg, lr=current_lr(opt))
            if not args.quiet:
                print(f"  epoch {epoch:3d}/{args.epochs}  val_loss={val_loss:.4f}  "
                      f"r2={r2:.4f}  mae_deg={mae_deg:.2f}", flush=True)

        run.finish(completed=True, final_mae_deg=mae_deg,
                   best_mae_deg=best_mae_deg, final_val_loss=val_loss)
    print(f"[{EXPERIMENT}] done → final_mae_deg={mae_deg:.2f}  "
          f"best_mae_deg={best_mae_deg:.2f}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--optimizer", required=True, choices=["gnome", "soap", "adamw"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eps", type=float, default=1e-6,
                   help="Gnome curvature-damping epsilon in m̂/(v̂+eps): larger "
                        "-> more gradient-descent-like, smaller -> fuller Newton "
                        "step. Gnome only; SOAP/AdamW keep their fixed eps=1e-8.")
    p.add_argument("--beta1", type=float, default=0.9,
                   help="First-moment (momentum) EMA for Gnome and SOAP.")
    p.add_argument("--beta2", type=float, default=0.99,
                   help="Second-moment / preconditioner EMA (also shampoo_beta) for Gnome and SOAP.")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--model", choices=MODEL_NAMES, default="resnet12",
                   help="Architecture (default resnet12).")
    p.add_argument("--norm", choices=["gn", "bn"], default="gn",
                   help="Normalization: GroupNorm (default) or BatchNorm.")
    p.add_argument("--warmup-steps", type=int, default=100,
                   help="LR warmup steps. Baselines warmup then cosine-decay; "
                        "Gnome uses this as its internal warmup only.")
    p.add_argument("--cosine-decay", type=float, default=0.0,
                   help="Final-LR fraction for the SOAP/AdamW cosine decay: "
                        "0.0 decays to zero (default), 1.0 disables decay. "
                        "Gnome (MSE) never decays regardless.")
    p.add_argument("--log-every", type=int, default=50,
                   help="Print a running-mean train loss every N steps within "
                        "an epoch (0 disables). Per-step train loss is logged "
                        "to the artifact regardless.")
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
