"""UTKFace age regression: AdamW vs SOAP vs Gnome on a real-world dataset.

A natural-target regression benchmark to bookend the synthetic OLS example:
predict age (in years) from a face crop, with a small ResNet. The residual
distribution of real age data has fat tails (older faces, demographic
imbalance) — exactly the regime where dividing by curvature (Gnome/GGN)
rather than gradient RMS (SOAP/AdamW empirical-Fisher) is expected to pay
off. The whole point is the split between the MSE the optimizer trains on
and the **mean absolute error in years** you actually care about, so both
are logged every epoch.

Loss is MSE on age mapped to ``[-1, 1]`` via ``age / 60 - 1`` (a fixed
constant map, no dataset statistics); ``mae_years`` / ``mse_years``
un-normalize the validation residual back into years for a human-readable
readout.

Dataset: the full ``nu-delta/utkface`` (~23.7k images, columns
``image``/``age``/``gender``/``ethnicity``) is fetched once via HuggingFace
``datasets`` and cached under ``experiments/data/utkface_hf/`` (first call
downloads and decodes — a minute or two, then reused).

    uv run python -m experiments.utkface --optimizer gnome --seed 0
    uv run python -m experiments.utkface --optimizer soap  --seed 0 --cosine-decay 0
    uv run python -m experiments.utkface --optimizer adamw --seed 0 --cosine-decay 0
"""

from __future__ import annotations

import argparse
import math
import os
import warnings

import numpy as np
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


EXPERIMENT = "utkface"
# Full UTKFace (~23.7k) with structured image/age columns. (The prototype's
# Subh775/UTKFace_demographics_V1 was removed from the Hub; nu-delta/utkface
# is a drop-in full-dataset replacement with the same image + integer-age
# schema.) Swap this constant to use any UTKFace repo exposing `image`+`age`.
HF_DATASET = "nu-delta/utkface"
DEFAULT_CACHE_DIR = "experiments/data/utkface_hf"

# UTKFace ages run 0-116. Map with a round upper bound of 120 so every
# sample lands strictly inside [-1, 1]: target = age / 60 - 1. The
# ``mae_years`` readout un-normalizes by AGE_HALF_RANGE = 60.
AGE_RANGE_MAX = 120.0
AGE_HALF_RANGE = AGE_RANGE_MAX / 2.0


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def _decode_image(pil_img, image_size: int) -> np.ndarray:
    """Resize a PIL image to ``image_size`` and return ``[3, H, W]`` float32
    normalized to ``[-1, 1]``."""
    from PIL import Image

    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    if pil_img.size != (image_size, image_size):
        pil_img = pil_img.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def load_utkface_data(seed: int, val_frac: float, image_size: int,
                      cache_dir: str = DEFAULT_CACHE_DIR):
    """Load the full UTKFace dataset once; return a seeded train/val split.

    Returns ``(x_train, y_train, x_val, y_val)`` with images float32
    ``[N,3,H,W]`` in ``[-1,1]`` and targets ``age/60-1`` shaped ``[N,1]``.
    UTKFace has no canonical split, so validation is a seeded ``val_frac``
    holdout.
    """
    from datasets import load_dataset

    os.makedirs(cache_dir, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = load_dataset(HF_DATASET, cache_dir=cache_dir, split="train")

    n = len(ds)
    ages = np.asarray(ds["age"], dtype=np.float32)
    if ages.min() < 0.0 or ages.max() > AGE_RANGE_MAX:
        raise ValueError(
            f"UTKFace ages [{ages.min()}, {ages.max()}] outside "
            f"[0, {AGE_RANGE_MAX}] — bump AGE_RANGE_MAX."
        )

    imgs = np.empty((n, 3, image_size, image_size), dtype=np.float32)
    for i in range(n):
        imgs[i] = _decode_image(ds[i]["image"], image_size)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = int(round(n * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    y = ages / AGE_HALF_RANGE - 1.0
    return (
        torch.from_numpy(imgs[train_idx]),
        torch.from_numpy(y[train_idx]).unsqueeze(-1),
        torch.from_numpy(imgs[val_idx]),
        torch.from_numpy(y[val_idx]).unsqueeze(-1),
    )


def augment_batch(
        x: torch.Tensor,
        max_rotation_deg: float = 10.0,
        max_scale: float = 1.1,
        hflip_prob: float = 0.5,
        brightness: float = 0.15,
        contrast: float = 0.15,
        geo_prob: float = 0.5,
        photo_prob: float = 0.5,
        cutout_prob: float = 0.35,
        cutout_ratio: float = 0.10,
) -> torch.Tensor:
    """On-device batched train augmentation for face/age estimation.

    Assumes ``x`` is normalized to [-1, 1] (0 = mid-gray) and square (h == w).

    Blocks (each gated independently PER SAMPLE):
      - hflip:      applied with probability ``hflip_prob``
      - geometric:  rotation + zoom-in + translation, with prob ``geo_prob``
      - photometric: additive brightness + contrast about the grayscale mean,
                     with prob ``photo_prob``
      - cutout:     one square erased region, with prob ``cutout_prob``

    With independent gates, most samples receive only a subset of transforms,
    which keeps per-batch difficulty well-behaved instead of stacking every
    distortion on every image.
    """
    b, c, h, w = x.shape
    device = x.device
    assert h == w, "Rotation via affine_grid assumes square inputs (h == w)."
    assert max_scale >= 1.0, "Only zoom-in is supported (max_scale >= 1.0)."

    # ---- horizontal flip (per-sample gate) ----------------------------------
    if hflip_prob > 0:
        flip = torch.rand(b, device=device) < hflip_prob
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)

    # ---- geometric: rotation + zoom + translation (per-sample gate) ---------
    if geo_prob > 0 and (max_rotation_deg > 0 or max_scale > 1.0):
        apply_geo = (torch.rand(b, device=device) < geo_prob).float()

        # Zero out strengths for gated-off samples -> identity transform.
        s = 1.0 + (max_scale - 1.0) * torch.rand(b, device=device) * apply_geo
        inv_s = 1.0 / s

        theta = (
                (torch.rand(b, device=device) * 2 - 1)
                * max_rotation_deg
                * (math.pi / 180.0)
                * apply_geo
        )
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)

        # Translation bounded so the zoomed window stays inside the image.
        margin = 1.0 - inv_s
        tx = (torch.rand(b, device=device) * 2 - 1) * margin
        ty = (torch.rand(b, device=device) * 2 - 1) * margin

        aff = torch.stack(
            [
                torch.stack([inv_s * cos_t, -inv_s * sin_t, tx], dim=-1),
                torch.stack([inv_s * sin_t, inv_s * cos_t, ty], dim=-1),
            ],
            dim=-2,
        )
        grid = F.affine_grid(aff, list(x.shape), align_corners=False)
        # Reflection padding: no gray wedges in rotated corners.
        x = F.grid_sample(
            x, grid, align_corners=False, padding_mode="reflection"
        )

    # ---- photometric: brightness + contrast (per-sample gate) ---------------
    if photo_prob > 0 and (brightness > 0 or contrast > 0):
        apply_photo = (
                torch.rand(b, 1, 1, 1, device=device) < photo_prob
        ).float()

        # Additive brightness (correct for [-1, 1] normalization).
        if brightness > 0:
            delta = (
                    (torch.rand(b, 1, 1, 1, device=device) * 2 - 1)
                    * brightness
                    * apply_photo
            )
            x = x + delta

        # Contrast about the grayscale mean (all channels), matching
        # torchvision semantics; avoids per-channel color-balance drift.
        if contrast > 0:
            cf = 1.0 + (
                    (torch.rand(b, 1, 1, 1, device=device) * 2 - 1)
                    * contrast
                    * apply_photo
            )
            mean = x.mean(dim=[1, 2, 3], keepdim=True)
            x = (x - mean) * cf + mean

        x = torch.clamp(x, -1.0, 1.0)

    # ---- cutout / random erasing (per-sample gate) ---------------------------
    if cutout_prob > 0 and cutout_ratio > 0:
        mask = torch.rand(b, device=device) < cutout_prob
        bh, bw = max(1, int(h * cutout_ratio)), max(1, int(w * cutout_ratio))
        y1 = torch.randint(0, h - bh + 1, (b,), device=device)
        x1 = torch.randint(0, w - bw + 1, (b,), device=device)
        yg = torch.arange(h, device=device).view(1, h, 1)
        xg = torch.arange(w, device=device).view(1, 1, w)
        ym = (yg >= y1.view(b, 1, 1)) & (yg < (y1 + bh).view(b, 1, 1))
        xm = (xg >= x1.view(b, 1, 1)) & (xg < (x1 + bw).view(b, 1, 1))
        region = ym & xm & mask.view(b, 1, 1)
        x = torch.where(region.unsqueeze(1), torch.zeros_like(x), x)

    return x


# ----------------------------------------------------------------------
# Optimizer + schedule
# ----------------------------------------------------------------------

def build_optimizer(name, params, lr, weight_decay, warmup, total_steps, cosine_decay,
                    eps=1e-6, beta1=0.9, beta2=0.99):
    """Return ``(optimizer, config, scheduler)``.

    MSE regression, so the repo protocol applies: Gnome runs at a fixed
    learning rate (self-anneals; no scheduler), while SOAP/AdamW get warmup +
    cosine decay to a ``cosine_decay`` final-LR fraction.
    """
    if name == "gnome":
        cfg = dict(
            lr=lr, weight_decay=weight_decay,
            betas=(beta1, beta2), shampoo_beta=beta2, eps=eps,
            precondition_frequency=10,
            clip=1.0, warmup=warmup,
            loss="mse", precondition_1d=False,
        )
        opt = Gnome(params, **cfg)
        # aux_batch_size sizes the auxiliary batch the caller builds for
        # opt.step(...); it is not a Gnome constructor arg. Recorded in the
        # returned config for logging and to set K below.
        cfg["aux_batch_size"] = 10
        return opt, cfg, None
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

    print(f"[{EXPERIMENT}] {args.optimizer} | loading UTKFace ({HF_DATASET})...",
          flush=True)
    x_train_cpu, y_train_cpu, x_val, y_val = load_utkface_data(
        seed=args.seed, val_frac=args.val_frac, image_size=args.image_size,
    )
    n_train = int(x_train_cpu.shape[0])
    # Targets are tiny; pre-transfer the whole table to device once and index
    # on-device per batch (a per-batch host->device copy of a few floats can
    # race the forward pass on MPS). Images stay on CPU, sliced per batch.
    y_train_dev = y_train_cpu.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)

    model = build_model(args.model, num_outputs=1, norm=args.norm).to(device)
    steps_per_epoch = math.ceil(n_train / args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    opt, opt_cfg, scheduler = build_optimizer(
        args.optimizer, model.parameters(), args.lr, args.weight_decay,
        args.warmup_steps, total_steps, args.cosine_decay, eps=args.eps,
        beta1=args.beta1, beta2=args.beta2,
    )
    K = opt_cfg.get("aux_batch_size", 20) if args.optimizer == "gnome" else 0

    hyperparameters = {
        "optimizer": args.optimizer,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "n_train": n_train,
        "n_val": int(x_val.shape[0]),
        "image_size": args.image_size,
        "val_frac": args.val_frac,
        "model": args.model,
        "norm": args.norm,
        "age_range_max": AGE_RANGE_MAX,
        "augment": args.augment,
        "warmup_steps": args.warmup_steps,
        "cosine_decay": args.cosine_decay,
        "n_params": sum(p.numel() for p in model.parameters()),
        "device": str(device),
        "hf_dataset": HF_DATASET,
        **{f"opt.{k}": v for k, v in opt_cfg.items()},
    }

    with RunLogger(EXPERIMENT, args.optimizer, args.seed, hyperparameters,
                   runs_dir=args.runs_dir) as run:
        step = 0
        best_mae_years = float("inf")
        window_sum, window_n = 0.0, 0   # running-mean train loss for --log-every
        for epoch in range(args.epochs):
            model.train()
            perm = torch.randperm(n_train)
            for i in range(0, n_train, args.batch_size):
                idx = perm[i:i + args.batch_size]
                x_batch = x_train_cpu[idx].to(device, non_blocking=True)
                y_batch = y_train_dev[idx.to(device, non_blocking=True)]
                if args.augment:
                    x_batch = augment_batch(
                        x_batch, args.aug_max_rotation_deg, args.aug_max_scale,
                        args.aug_hflip_prob,
                    )
                b = x_batch.shape[0]

                if args.optimizer == "gnome":
                    k = min(K, max(1, b - 1))
                    a_idx = torch.randperm(b, device=device)[:k]
                    x_main, y_main = x_batch, y_batch
                    x_aux, y_aux = x_batch[a_idx], y_batch[a_idx]

                    def main_closure():
                        return model(x_main), y_main

                    def aux_closure():
                        return model(x_aux), y_aux

                    loss = opt.step(main_closure, aux_closure)
                else:
                    opt.zero_grad()
                    loss = F.mse_loss(model(x_batch), y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                    # loss is MSE on the [-1,1] age target, so sqrt(avg)*60 is a
                    # human-readable train RMSE in years.
                    rmse_years = (avg ** 0.5) * AGE_HALF_RANGE
                    print(f"    step {step:6d}  epoch {epoch:3d}  "
                          f"train_loss[last {window_n}]={avg:.4f}  "
                          f"(~{rmse_years:.1f} yr RMSE)", flush=True)
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
                resid_years = (val_pred - y_val) * AGE_HALF_RANGE
                mae_years = resid_years.abs().mean().item()
                mse_years = (resid_years ** 2).mean().item()
            best_mae_years = min(best_mae_years, mae_years)
            run.log_val(step, epoch=epoch, loss=val_loss, r2=r2,
                        mae_years=mae_years, mse_years=mse_years,
                        lr=current_lr(opt))
            if not args.quiet:
                print(f"  epoch {epoch:3d}/{args.epochs}  val_loss={val_loss:.4f}  "
                      f"r2={r2:.4f}  mae_years={mae_years:.2f}  "
                      f"mse_years={mse_years:.2f}", flush=True)

        run.finish(
            completed=True,
            final_mae_years=mae_years, best_mae_years=best_mae_years,
            final_val_loss=val_loss,
        )
    print(f"[{EXPERIMENT}] done → final_mae_years={mae_years:.2f}  "
          f"best_mae_years={best_mae_years:.2f}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--optimizer", required=True, choices=["gnome", "soap", "adamw"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
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
                   help="Architecture. resnet12 (default) is a custom net "
                        "between resnet8 and resnet18.")
    p.add_argument("--norm", choices=["gn", "bn"], default="gn",
                   help="Normalization: GroupNorm (default) or BatchNorm.")
    p.add_argument("--image-size", type=int, default=64,
                   help="Side length to resize the UTKFace crops to.")
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="Seeded validation holdout fraction (no canonical split).")
    p.add_argument("--warmup-steps", type=int, default=100,
                   help="LR warmup steps. Baselines warmup then cosine-decay; "
                        "Gnome uses this as its internal warmup only.")
    p.add_argument("--cosine-decay", type=float, default=0.0,
                   help="Final-LR fraction for the SOAP/AdamW cosine decay: "
                        "0.0 decays to zero (default), 1.0 disables decay. "
                        "Gnome (MSE) never decays regardless.")
    p.add_argument("--no-augment", dest="augment", action="store_false",
                   help="Disable train-time augmentation (on by default).")
    p.set_defaults(augment=True)
    p.add_argument("--aug-max-rotation-deg", type=float, default=10.0)
    p.add_argument("--aug-max-scale", type=float, default=1.1)
    p.add_argument("--aug-hflip-prob", type=float, default=0.5)
    p.add_argument("--log-every", type=int, default=50,
                   help="Print a running-mean train loss every N steps within "
                        "an epoch (0 disables). Per-step train loss is logged "
                        "to the artifact regardless.")
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
