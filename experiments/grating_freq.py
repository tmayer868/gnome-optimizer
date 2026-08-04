"""Grating-frequency regression: AdamW vs SOAP vs Gnome on real images.

A 2D sinusoidal grating is added on top of a CIFAR-100 image and the model must
regress the grating's spatial frequency in cycles per image. The target is
``freq`` mapped to [-1, 1] over ``[FREQ_MIN, FREQ_MAX]``; the loss is MSE on
that target, and ``mae_cycles`` un-normalizes the validation error back into
cycles for a human-readable readout.

Structurally this is a cleaner regression than ``cifar_rotation``: the target is
drawn independently of the image, so unlike a rotation angle -- which is only
defined relative to whatever orientation the dataset happened to store, and is
genuinely ambiguous for content with no canonical "up" -- the frequency label
carries no per-image irreducible error. The image is *clutter*, and the noise
floor is set by how well the grating stands out from it, which ``--amp-min/max``
controls directly.

Grating parameters are resampled fresh every batch, so the model never sees the
same (image, grating) pair twice. Validation gratings are frozen per seed, so
val metrics are comparable across runs and epochs.

Three things are randomized per sample, each closing a specific shortcut:

  phase        Without it the grating is a fixed function of frequency, so the
               value at any single pixel decodes the label outright.
  orientation  Forces an orientation-invariant frequency estimate rather than a
               readout along one image axis.
  amplitude    The important one. Total variation scales as A*f, so at fixed A
               it is a near-perfect proxy for the label: measured on real CIFAR
               clutter, the best 1-D predictor of TV explains R2 = 0.94 of the
               target variance, and a net can win without ever locating a
               spectral peak. Drawing A log-uniformly over a 20x range confounds
               the product and brings that to ~0.15, so most of the task now
               requires estimating *where* the energy sits, not how much.

    uv run python -m experiments.grating_freq --optimizer gnome --seed 0
    uv run python -m experiments.grating_freq --optimizer soap  --seed 0 --cosine-decay 0
    uv run python -m experiments.grating_freq --optimizer adamw --seed 0 --cosine-decay 0
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


EXPERIMENT = "grating_freq"
DEFAULT_DATA_DIR = "experiments/data"

# Frequency band in cycles per image. The 32x32 sampling lattice puts Nyquist
# at 16 cycles/image per axis; an oriented grating at f cycles has axis
# components |f*cos| , |f*sin| <= f, so FREQ_MAX = 8 keeps a 2x margin and no
# sample can alias to a *different* apparent frequency (which would make the
# label unrecoverable rather than merely hard). Below ~1 cycle the grating is
# barely a full period across the frame and degenerates into a DC shift.
FREQ_MIN = 1.0
FREQ_MAX = 8.0

# Per-sample grating amplitude, on the same [-1, 1] scale as the image, drawn
# log-uniformly. This range is the difficulty knob and the main defence against
# the energy-magnitude shortcut — see ``sample_grating_params``. Widening it
# lowers the shortcut but raises the clutter-limited noise floor, since a
# grating at amp 0.05 sits well under CIFAR's own contrast.
AMP_MIN = 0.05
AMP_MAX = 1.0


# ----------------------------------------------------------------------
# Data + grating
# ----------------------------------------------------------------------

def sample_grating_params(n, generator=None, device=None,
                          amp_min=AMP_MIN, amp_max=AMP_MAX):
    """Draw ``(freq, orient, phase, amp)``, each ``[n]``, for a batch.

    Amplitude is drawn *log*-uniformly. Measured on real CIFAR clutter, the
    variance of the target recoverable by the best 1-D predictor of total
    variation, as the amplitude range widens:

        range        dynamic   uniform   log-uniform
        [1.0, 1.0]      1.0x     0.940      --
        [0.30, 1.0]     3.3x     0.604     0.566
        [0.15, 1.0]     6.6x     0.442     0.324
        [0.08, 1.0]    12.5x     0.373     0.217
        [0.04, 1.0]    25.0x     0.339     0.136

    Log-uniform wins everywhere and the gap grows with the range, because
    uniform sampling piles mass near ``amp_max`` and so buys much less
    effective dynamic range than its endpoints suggest.
    """
    def u(lo, hi):
        r = torch.rand(n, generator=generator, device=device)
        return lo + (hi - lo) * r

    freq = u(FREQ_MIN, FREQ_MAX)
    orient = u(0.0, math.pi)        # a grating is pi-periodic in orientation
    phase = u(0.0, 2.0 * math.pi)
    amp = torch.exp(u(math.log(amp_min), math.log(amp_max)))
    return freq, orient, phase, amp


def add_grating(x, freq, orient, phase, amp):
    """Add a luminance grating ``amp * sin(2pi(fx*u + fy*v) + phase)`` to ``x``.

    ``x`` is ``[B,3,H,W]``; the four parameter tensors are ``[B]``. Coordinates
    ``u, v`` run over [-0.5, 0.5) at pixel centers, so ``freq`` reads directly
    as cycles per image and is independent of resolution.

    The same grating goes on all three channels (a luminance grating, not a
    chromatic one), and the result is deliberately *not* clipped back into
    range: clipping a sinusoid generates harmonics whose structure depends on
    amplitude and frequency together, which both distorts the target and hands
    the model an edge-density cue that scales with f.
    """
    h, w = x.shape[-2], x.shape[-1]
    dev, dt = x.device, x.dtype
    v = (torch.arange(h, device=dev, dtype=dt) + 0.5) / h - 0.5
    u = (torch.arange(w, device=dev, dtype=dt) + 0.5) / w - 0.5

    fx = (freq * torch.cos(orient)).view(-1, 1, 1)
    fy = (freq * torch.sin(orient)).view(-1, 1, 1)
    ph = 2.0 * math.pi * (fx * u[None, None, :] + fy * v[None, :, None])
    g = amp.view(-1, 1, 1) * torch.sin(ph + phase.view(-1, 1, 1))
    return x + g.unsqueeze(1)       # [B,1,H,W] broadcasts over channels


def freq_to_target(freq):
    """Map cycles-per-image onto the [-1, 1] regression target."""
    return 2.0 * (freq - FREQ_MIN) / (FREQ_MAX - FREQ_MIN) - 1.0


# ``mae_cycles = mae_target * TARGET_TO_CYCLES``. Always predicting the band
# center gives mae = (FREQ_MAX - FREQ_MIN) / 4 = 1.75 cycles, which is the
# chance baseline to read the metric against.
TARGET_TO_CYCLES = 0.5 * (FREQ_MAX - FREQ_MIN)


def load_cifar100_tensors(data_dir: str = DEFAULT_DATA_DIR):
    """Load CIFAR-100 train+test as float32 ``[N,3,32,32]`` tensors in [-1, 1].

    Downloads once via torchvision (cached under ``data_dir``). Labels are
    discarded -- only the pixels matter, since the target is the grating
    frequency. The unverified SSL context is a standard workaround for
    intermittent cert failures on the upstream CIFAR host.
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
    x_val = torch.stack([img for img, _ in test])
    x_train = (x_train - 0.5) / 0.5
    x_val = (x_val - 0.5) / 0.5
    return x_train, x_val


# ----------------------------------------------------------------------
# Optimizer + schedule
# ----------------------------------------------------------------------

def build_optimizer(name, params, lr, weight_decay, warmup, total_steps, cosine_decay,
                    eps=1e-6, beta1=0.9, beta2=0.99,
                    trust_region=1.0):
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
            precondition_frequency=10,
            clip=None, warmup=warmup, loss="mse", precondition_1d=False,
            trust_radius=(trust_region if trust_region > 0 else None),
            norm_free=False,
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

    print(f"[{EXPERIMENT}] {args.optimizer} | loading CIFAR-100...", flush=True)
    x_train_cpu, x_val_cpu = load_cifar100_tensors()
    n_train = int(x_train_cpu.shape[0])

    # Frozen val gratings (deterministic per seed) so val metrics are
    # comparable across runs, optimizers, and epochs.
    val_gen = torch.Generator().manual_seed(args.seed)
    v_freq, v_orient, v_phase, v_amp = sample_grating_params(
        x_val_cpu.shape[0], generator=val_gen,
        amp_min=args.amp_min, amp_max=args.amp_max,
    )
    with torch.no_grad():
        x_val = add_grating(x_val_cpu, v_freq, v_orient, v_phase, v_amp).to(device)
    y_val = freq_to_target(v_freq).unsqueeze(-1).to(device)

    model = build_model(args.model, num_outputs=1, norm=args.norm).to(device)
    steps_per_epoch = math.ceil(n_train / args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    opt, opt_cfg, scheduler = build_optimizer(
        args.optimizer, model.parameters(), args.lr, args.weight_decay,
        args.warmup_steps, total_steps, args.cosine_decay, eps=args.eps,
        beta1=args.beta1, beta2=args.beta2,
        trust_region=args.trust_region,
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
        "freq_min": FREQ_MIN,
        "freq_max": FREQ_MAX,
        "amp_min": args.amp_min,
        "amp_max": args.amp_max,
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
        best_mae_cycles = float("inf")
        window_sum, window_n = 0.0, 0
        for epoch in range(args.epochs):
            model.train()
            perm = torch.randperm(n_train)
            for i in range(0, n_train, args.batch_size):
                idx = perm[i:i + args.batch_size]
                # NOT ``non_blocking=True``: ``x_train_cpu[idx]`` is a temporary
                # in pageable memory with no surviving reference, so an async
                # copy can read it after it is freed, silently landing garbage
                # (observed as NaN on MPS, on the final partial batch of an
                # epoch — a fresh allocation size whose lifetime differs from
                # the cached one). Nothing here overlaps with the copy anyway.
                x_batch = x_train_cpu[idx].to(device)
                b = x_batch.shape[0]
                # Fresh random grating for every image each time it's seen.
                freq, orient, phase, amp = sample_grating_params(
                    b, device=device, amp_min=args.amp_min, amp_max=args.amp_max)
                x_grt = add_grating(x_batch, freq, orient, phase, amp)
                y_batch = freq_to_target(freq).unsqueeze(-1)

                if args.optimizer == "gnome":
                    k = min(K, max(1, b - 1))
                    a_idx = torch.randperm(b, device=device)[:k]
                    x_main, y_main = x_grt, y_batch
                    x_aux, y_aux = x_grt[a_idx], y_batch[a_idx]

                    def main_closure():
                        return model(x_main), y_main

                    def aux_closure():
                        return model(x_aux), y_aux

                    loss = opt.step(main_closure, aux_closure)
                else:
                    opt.zero_grad()
                    loss = F.mse_loss(model(x_grt), y_batch)
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
                    # loss is MSE on the [-1,1] frequency target, so
                    # sqrt(avg) * TARGET_TO_CYCLES is a train RMSE in cycles.
                    rmse_cyc = (avg ** 0.5) * TARGET_TO_CYCLES
                    print(f"    step {step:6d}  epoch {epoch:3d}  "
                          f"train_loss[last {window_n}]={avg:.4f}  "
                          f"(~{rmse_cyc:.2f} cyc RMSE)", flush=True)
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
                mae_cycles = (val_pred - y_val).abs().mean().item() * TARGET_TO_CYCLES
            best_mae_cycles = min(best_mae_cycles, mae_cycles)
            run.log_val(step, epoch=epoch, loss=val_loss, r2=r2,
                        mae_cycles=mae_cycles, lr=current_lr(opt))
            if not args.quiet:
                print(f"  epoch {epoch:3d}/{args.epochs}  val_loss={val_loss:.4f}  "
                      f"r2={r2:.4f}  mae_cycles={mae_cycles:.3f}", flush=True)

        run.finish(completed=True, final_mae_cycles=mae_cycles,
                   best_mae_cycles=best_mae_cycles, final_val_loss=val_loss)
    print(f"[{EXPERIMENT}] done → final_mae_cycles={mae_cycles:.3f}  "
          f"best_mae_cycles={best_mae_cycles:.3f}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--optimizer", required=True, choices=["gnome", "soap", "adamw"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--amp-min", type=float, default=AMP_MIN,
                   help="Lower end of the per-sample grating amplitude (drawn "
                        "log-uniformly), on the same [-1,1] scale as the image. "
                        "The amplitude range is the difficulty knob: widening it "
                        "confounds the energy-magnitude shortcut but lowers SNR "
                        "against image clutter. amp_min == amp_max makes the "
                        "task ~94%% solvable by total variation alone.")
    p.add_argument("--amp-max", type=float, default=AMP_MAX,
                   help="Upper end of the per-sample grating amplitude.")
    p.add_argument("--trust-region", type=float, default=1.0,
                   help="Gnome per-coordinate update bound: lambda is set to "
                        "the smallest value with max|m̂/(v̂+lambda)| <= this, "
                        "so no coordinate moves more than lr*trust_region in "
                        "a step. Larger -> weaker bound -> longer steps. "
                        "0 disables it, falling back to plain m̂/(v̂+eps) "
                        "damping.")
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
    p.add_argument("--cosine-decay", type=float, default=1.0,
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
