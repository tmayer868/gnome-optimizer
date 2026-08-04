"""Random band-limited regression: fitting a multi-scale function with an MLP.

The target is a random Fourier field on the box ``[-1, 1]^d``::

    y(x) = sum_k  a_k * sin(2*pi * f_k * (u_k . x) + phi_k)

with directions ``u_k`` uniform on the sphere, frequencies ``f_k`` log-uniform
over ``[FREQ_MIN, FREQ_MAX]``, and amplitudes ``a_k = f_k^-alpha``. The whole
field is then standardized to zero mean and unit variance, so ``val_loss`` is
MSE against a unit-variance target and ``r2 = 1 - val_loss`` — a model that
predicts the mean scores exactly 0.

This is deliberately an *optimization*-limited benchmark, not a
capacity-limited or a statistics-limited one. Two design choices enforce that:

**Inputs are resampled every batch.** The target is a closed-form function, so
there is no reason to hold a finite training set: every step sees fresh ``x``.
On a fixed 32k-point set at ``d=8``, validation r2 *peaked at 5k steps and then
fell* (0.767 -> 0.702 by 20k) — the net was memorizing, and the run was scoring
generalization gap rather than optimization. With fresh sampling the curves are
monotone, train and val loss measure the same thing, and any gap between them
is estimator noise. It also means weight decay has no regularization job here,
hence the 0.0 default.

**Capacity is not the binding constraint.** Measured at ``d=4``, 20k AdamW
steps, holding everything else fixed:

    width  depth   params    r2
      128      4    50,305   0.8817
      256      4   198,913   0.9020
      256      6   330,497   0.9176
      512      4   791,041   0.9099

16x the parameters buys 0.03 r2 and depth 6 beats width 512. The residual is
not sitting in the function class, it is sitting in the optimizer — which is
what makes the number worth comparing optimizers on. At ``d=8`` the same sweep
is flat to within noise around r2 0.6.

Difficulty is governed by ``--dim`` far more than anything else (20k AdamW
steps, 128 modes, alpha=0.3): d=4 -> r2 0.90, d=8 -> 0.62, d=16 -> 0.32. d=4 is
the default: unmistakably unsolved, still improving at the budget, and
responsive to effort. ``--alpha`` is the second knob — it tilts variance toward
low frequencies, and the flat spectrum is the hard one:

    alpha   [0.5,1)  [1,2)   [2,4)   [4,8)     <- share of target variance
      0.0     30.5%  25.0%   16.4%   28.1%
      0.3     48.8%  26.3%   11.8%   13.1%
      1.0     80.0%  15.9%    2.9%    1.2%

alpha=1.0 is a low-frequency problem with high-frequency decoration; the
default 0.3 keeps real energy in the top two octaves, which is where an MLP's
spectral bias bites and where the Hessian's dynamic range comes from.

Because the target's mode decomposition is known exactly, the run reports a
**per-octave recovery** breakdown at the end: the validation residual is
regressed back onto the mode basis, so you can see *which spatial scales* the
optimizer actually reached rather than only how much total error is left. That
is the metric the loss curve hides, in the same spirit as ``beta_dist`` in
``ols_regression``.

    uv run python -m experiments.fourier_regression --optimizer gnome --seed 0
    uv run python -m experiments.fourier_regression --optimizer soap  --seed 0
    uv run python -m experiments.fourier_regression --optimizer adamw --seed 0
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from gnome import Gnome
from experiments.baselines import SOAP
from experiments.common import (
    DIVERGED_EXIT,
    diverged,
    RunLogger,
    baseline_cosine_scheduler,
    current_lr,
    pick_device,
)


EXPERIMENT = "fourier_regression"

# Frequency band in cycles per unit length along a mode's own direction. The
# box is 2 units wide, so FREQ_MIN = 0.5 is exactly one period across it (any
# lower and the mode is a near-linear ramp, indistinguishable from a bias) and
# FREQ_MAX = 8 is 16 periods, well inside what a 256-wide MLP can represent but
# far enough up that it is only reached late in training.
FREQ_MIN = 0.5
FREQ_MAX = 8.0


# ----------------------------------------------------------------------
# Target
# ----------------------------------------------------------------------

def make_target(dim, n_modes, alpha, seed, freq_min=FREQ_MIN, freq_max=FREQ_MAX):
    """Draw the random Fourier field: returns ``(W, phase, amp)``.

    ``W`` is ``[n_modes, dim]`` with ``W[k] = f_k * u_k``, so ``x @ W.T`` gives
    each mode's phase argument directly. Frequencies are drawn *log*-uniformly
    so the modes spread evenly across octaves rather than piling up near
    ``freq_max`` — with uniform sampling the top octave alone would hold half
    the modes and the field would have no meaningful low-frequency structure to
    fit first.

    The field is drawn from its own generator (``seed + 101``) so that changing
    the training seed re-rolls the initialization and the data stream but keeps
    the *target function* fixed. Comparing optimizers across seeds otherwise
    confounds "this optimizer is better" with "this optimizer drew an easier
    function".
    """
    g = torch.Generator().manual_seed(seed + 101)
    u = torch.randn(n_modes, dim, generator=g)
    u = u / u.norm(dim=1, keepdim=True)
    log_f = math.log(freq_min) + (math.log(freq_max) - math.log(freq_min)) * \
        torch.rand(n_modes, generator=g)
    freq = torch.exp(log_f)
    phase = 2.0 * math.pi * torch.rand(n_modes, generator=g)
    amp = freq.pow(-alpha)
    return freq.unsqueeze(1) * u, phase, amp


def evaluate(x, W, phase, amp):
    """Evaluate the field at ``x`` ``[N, dim]``; returns ``[N, 1]`` (unscaled)."""
    return (torch.sin(2.0 * math.pi * (x @ W.T) + phase) * amp).sum(dim=1, keepdim=True)


def sample_inputs(n, dim, generator=None, device=None):
    """Draw ``n`` points uniformly from the box ``[-1, 1]^dim``."""
    return torch.rand(n, dim, generator=generator, device=device) * 2.0 - 1.0


class Field:
    """The standardized target function, plus the pieces the diagnostic needs.

    Standardization constants come from a large fixed reference sample rather
    than closed form: the modes are only *near*-orthogonal at finite ``dim``,
    so the analytic variance ``sum(a_k^2)/2`` is off by the cross terms, and
    using it would leave the target's variance a few percent from 1 and make
    ``r2 = 1 - val_loss`` quietly untrue.
    """

    def __init__(self, dim, n_modes, alpha, seed, ref_size=1 << 16):
        self.dim = dim
        self.W, self.phase, self.amp = make_target(dim, n_modes, alpha, seed)
        g = torch.Generator().manual_seed(seed + 202)
        ref = evaluate(sample_inputs(ref_size, dim, generator=g),
                       self.W, self.phase, self.amp)
        self.mean = ref.mean()
        self.std = ref.std()

    def to(self, device):
        self.W = self.W.to(device)
        self.phase = self.phase.to(device)
        self.amp = self.amp.to(device)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def __call__(self, x):
        return (evaluate(x, self.W, self.phase, self.amp) - self.mean) / self.std

    @property
    def freqs(self):
        return self.W.norm(dim=1)

    @property
    def coeffs(self):
        """Mode amplitudes on the *standardized* target scale."""
        return self.amp / self.std


def octave_bands(freq_min=FREQ_MIN, freq_max=FREQ_MAX):
    """Split the band into octaves: ``[(lo, hi), ...]``, last one inclusive."""
    bands, lo = [], freq_min
    while lo < freq_max - 1e-9:
        bands.append((lo, min(lo * 2.0, freq_max)))
        lo = min(lo * 2.0, freq_max)
    return bands


# ----------------------------------------------------------------------
# Spectral diagnostic
# ----------------------------------------------------------------------

def spectral_recovery(field, x, pred, bands):
    """Per-octave share of the target that the model actually captured.

    The residual ``y - pred`` is regressed onto the true mode basis (plus a
    constant, to absorb any DC offset the model carries), giving a residual
    coefficient ``e_k`` per mode. Octave ``b``'s recovery is then

        1 - ||e_b||^2 / ||c_b||^2

    against the target's own coefficients ``c``: 1.0 means the octave is fully
    fit, 0.0 means it is untouched, and negative means the model put *more*
    energy at that scale than the target has.

    Done in float64 on CPU — the design matrix is only ``[n_val, n_modes + 1]``,
    and ``lstsq``'s rank-revealing driver is CPU-only.
    """
    resid = (field(x) - pred.reshape(-1, 1)).cpu().double()
    x = x.cpu().double()
    W = field.W.cpu().double()
    phase = field.phase.cpu().double()

    phi = torch.sin(2.0 * math.pi * (x @ W.T) + phase)
    design = torch.cat([phi, torch.ones(phi.shape[0], 1, dtype=phi.dtype)], dim=1)
    e = torch.linalg.lstsq(design, resid, driver="gelsd").solution[:-1, 0]

    c = field.coeffs.cpu().double()
    f = field.freqs.cpu().double()
    total = (c ** 2).sum()
    rows = []
    for lo, hi in bands:
        sel = (f >= lo) & (f < hi) if hi < FREQ_MAX else (f >= lo)
        c_energy = (c[sel] ** 2).sum()
        recovered = 1.0 - (e[sel] ** 2).sum() / c_energy.clamp_min(1e-30)
        rows.append((lo, hi, int(sel.sum()), float(c_energy / total),
                     float(recovered)))
    return rows


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

def build_mlp(dim, width, depth, activation):
    """Plain MLP, ``depth`` hidden layers of ``width``, scalar output.

    Deliberately no Fourier feature embedding: the whole difficulty of this
    task is the spectral bias of a coordinate-input MLP, and an embedding would
    hand the high-frequency octaves over for free.
    """
    act = {"gelu": nn.GELU, "tanh": nn.Tanh, "silu": nn.SiLU}[activation]
    layers, prev = [], dim
    for _ in range(depth):
        layers += [nn.Linear(prev, width), act()]
        prev = width
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


# ----------------------------------------------------------------------
# Optimizer + schedule
# ----------------------------------------------------------------------

def build_optimizer(name, params, lr, weight_decay, warmup, total_steps,
                    cosine_decay, eps=1e-6, beta1=0.9, beta2=0.99,
                    trust_region=1.0):
    """Return ``(optimizer, config, scheduler)``.

    MSE regression, so the repo protocol applies: Gnome runs at a fixed
    learning rate — its Gauss-Newton step self-anneals as the residual shrinks
    — while SOAP and AdamW get linear warmup plus cosine decay to a
    ``cosine_decay`` final-LR fraction. ``precondition_1d`` is on: the only 1D
    tensors here are the Linear biases, which are full-width and do carry
    cross-coordinate structure worth a Kronecker factor.
    """
    if name == "gnome":
        cfg = dict(
            lr=lr, weight_decay=weight_decay,
            betas=(beta1, beta2), shampoo_beta=beta2, eps=eps,
            precondition_frequency=10,
            warmup=warmup, loss="mse", precondition_1d=True,
            trust_radius=(trust_region if trust_region > 0 else None),
            norm_free=False,
        )
        opt = Gnome(params, **cfg)
        # aux_batch_size sizes the auxiliary batch the caller builds for
        # opt.step(...); it is not a Gnome constructor arg. Recorded in the
        # returned config for logging and to set K below.
        cfg["aux_batch_size"] = 16
        return opt, cfg, None
    if name == "soap":
        cfg = dict(
            lr=lr, weight_decay=weight_decay,
            betas=(beta1, beta2), shampoo_beta=beta2, eps=1e-8,
            precondition_frequency=10, precondition_1d=True,
        )
        opt = SOAP(params, **cfg)
    elif name == "adamw":
        cfg = dict(lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)
        opt = torch.optim.AdamW(params, **cfg)
    else:
        raise ValueError(f"unknown optimizer: {name}")

    cfg["warmup"] = warmup      # unified meta key across optimizers
    cfg["cosine_decay_floor"] = cosine_decay
    scheduler = baseline_cosine_scheduler(opt, warmup, total_steps, cosine_decay)
    return opt, cfg, scheduler


# ----------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device()

    field = Field(args.dim, args.n_modes, args.alpha, args.seed).to(device)
    bands = octave_bands()

    # Frozen validation points, deterministic per seed, so val metrics are
    # comparable across optimizers and across steps within a run.
    val_gen = torch.Generator().manual_seed(args.seed + 303)
    x_val = sample_inputs(args.n_val, args.dim, generator=val_gen).to(device)
    y_val = field(x_val)

    # Best-linear-fit r2 as the floor to read the metric against. The field is
    # a sum of sines with random phases, so a linear model should explain
    # essentially nothing; a non-trivial number here means the draw is degenerate.
    xb = torch.cat([x_val, torch.ones_like(x_val[:, :1])], dim=1).cpu().double()
    yb = y_val.cpu().double()
    lin = torch.linalg.lstsq(xb, yb).solution
    lin_r2 = float(1.0 - ((xb @ lin - yb) ** 2).mean() / yb.var())

    model = build_mlp(args.dim, args.width, args.depth, args.activation).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    opt, opt_cfg, scheduler = build_optimizer(
        args.optimizer, model.parameters(), args.lr, args.weight_decay,
        args.warmup_steps, args.steps, args.cosine_decay, eps=args.eps,
        beta1=args.beta1, beta2=args.beta2, trust_region=args.trust_region,
    )
    K = opt_cfg.get("aux_batch_size", 16) if args.optimizer == "gnome" else 0

    print(f"[{EXPERIMENT}] {args.optimizer} | dim={args.dim} "
          f"modes={args.n_modes} alpha={args.alpha} "
          f"f in [{FREQ_MIN}, {FREQ_MAX}] | mlp {args.width}x{args.depth} "
          f"({n_params:,} params) | device={device}", flush=True)
    print(f"[{EXPERIMENT}] baselines: predict-the-mean r2=0.0000, "
          f"best linear fit r2={lin_r2:.4f}", flush=True)

    hyperparameters = {
        "optimizer": args.optimizer,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "dim": args.dim,
        "n_modes": args.n_modes,
        "alpha": args.alpha,
        "freq_min": FREQ_MIN,
        "freq_max": FREQ_MAX,
        "width": args.width,
        "depth": args.depth,
        "activation": args.activation,
        "n_val": args.n_val,
        "n_params": n_params,
        "linear_r2": lin_r2,
        "warmup_steps": args.warmup_steps,
        "cosine_decay": args.cosine_decay,
        "device": str(device),
        **{f"opt.{k}": v for k, v in opt_cfg.items()},
    }

    def validate():
        model.eval()
        with torch.no_grad():
            preds = [model(x_val[i:i + 4096])
                     for i in range(0, x_val.shape[0], 4096)]
            pred = torch.cat(preds)
            loss = F.mse_loss(pred, y_val).item()
            r2 = 1.0 - loss / max(float(y_val.var()), 1e-12)
            rel = float((pred - y_val).norm() / y_val.norm())
        model.train()
        return pred, loss, r2, rel

    train_gen = torch.Generator(device="cpu").manual_seed(args.seed + 404)
    best_r2 = -float("inf")
    val_loss = r2 = rel = float("nan")

    with RunLogger(EXPERIMENT, args.optimizer, args.seed, hyperparameters,
                   runs_dir=args.runs_dir) as run:
        window_sum, window_n = 0.0, 0
        for step in range(args.steps):
            # Fresh inputs every step -- see the module docstring on why this
            # is not a fixed training set.
            x = sample_inputs(args.batch_size, args.dim,
                              generator=train_gen).to(device)
            y = field(x)

            if args.optimizer == "gnome":
                k = min(K, max(1, args.batch_size - 1))
                a_idx = torch.randperm(args.batch_size, device=device)[:k]
                x_aux, y_aux = x[a_idx], y[a_idx]

                def main_closure():
                    return model(x), y

                def aux_closure():
                    return model(x_aux), y_aux

                loss = opt.step(main_closure, aux_closure)
            else:
                opt.zero_grad()
                loss = F.mse_loss(model(x), y)
                loss.backward()
                opt.step()
            if scheduler is not None:
                scheduler.step()

            loss_val = float(loss.detach().item())
            if diverged(loss_val):
                run.finish(completed=False, diverged=True, diverged_step=step)
                print(f"[{EXPERIMENT}] diverged at step {step} — stopping.",
                      flush=True)
                raise SystemExit(DIVERGED_EXIT)
            run.log_train(step, loss=loss_val)
            window_sum += loss_val
            window_n += 1

            if (step + 1) % args.eval_every == 0:
                _, val_loss, r2, rel = validate()
                best_r2 = max(best_r2, r2)
                run.log_val(step + 1, loss=val_loss, r2=r2, rel_l2=rel,
                            lr=current_lr(opt))
                if not args.quiet:
                    avg = window_sum / max(window_n, 1)
                    print(f"  step {step + 1:7d}/{args.steps}  "
                          f"train_loss[{window_n}]={avg:.4e}  "
                          f"val_loss={val_loss:.4e}  r2={r2:.4f}  "
                          f"rel_l2={rel:.4f}", flush=True)
                window_sum, window_n = 0.0, 0

        pred, val_loss, r2, rel = validate()
        best_r2 = max(best_r2, r2)
        rows = spectral_recovery(field, x_val, pred, bands)
        run.finish(completed=True, final_val_loss=val_loss, final_r2=r2,
                   final_rel_l2=rel, best_r2=best_r2,
                   **{f"octave_{lo:g}_{hi:g}": rec for lo, hi, _, _, rec in rows})

    print(f"[{EXPERIMENT}] done → val_loss={val_loss:.4e}  r2={r2:.4f}  "
          f"rel_l2={rel:.4f}")
    print("  per-octave recovery (1.0 = scale fully fit, 0.0 = untouched):")
    for lo, hi, n, share, rec in rows:
        print(f"    f in [{lo:4.1f}, {hi:4.1f})  {n:3d} modes  "
              f"{share:5.1%} of variance   recovered {rec:6.3f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--optimizer", required=True, choices=["gnome", "soap", "adamw"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--dim", type=int, default=4,
                   help="Input dimension, and the primary difficulty knob. At "
                        "20k AdamW steps: d=4 -> r2 0.90, d=8 -> 0.62, "
                        "d=16 -> 0.32. Above ~8 the task stops responding to "
                        "extra capacity or steps and just measures how much of "
                        "the low band you got.")
    p.add_argument("--n-modes", type=int, default=128,
                   help="Number of Fourier modes summed into the target.")
    p.add_argument("--alpha", type=float, default=0.3,
                   help="Amplitude spectrum exponent, a_k = f_k^-alpha. 0.0 is "
                        "flat (energy spread evenly over octaves, hardest); "
                        "1.0 puts 80%% of the variance in the lowest octave and "
                        "makes the high frequencies decoration.")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--depth", type=int, default=4,
                   help="Hidden layers. Capacity is not the binding constraint "
                        "here — see the docstring sweep — so raising these "
                        "mostly buys wall time.")
    p.add_argument("--activation", choices=["gelu", "tanh", "silu"], default="gelu")
    p.add_argument("--n-val", type=int, default=8192,
                   help="Frozen validation points, drawn once per seed.")
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
                   help="Second-moment / preconditioner EMA (also shampoo_beta) "
                        "for Gnome and SOAP.")
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="Default 0: inputs are resampled every step, so there "
                        "is nothing to regularize and decay only biases the fit.")
    p.add_argument("--warmup-steps", type=int, default=200,
                   help="LR warmup steps. Baselines warm up then cosine-decay; "
                        "Gnome uses this as its internal warmup only.")
    p.add_argument("--cosine-decay", type=float, default=1.0,
                   help="Final-LR fraction for the SOAP/AdamW cosine decay: "
                        "0.0 decays to zero (default), 1.0 disables decay. "
                        "Gnome (MSE) never decays regardless.")
    p.add_argument("--eval-every", type=int, default=1000,
                   help="Validation cadence in steps. Per-step train loss is "
                        "logged to the artifact regardless.")
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
