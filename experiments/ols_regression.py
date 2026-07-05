"""OLS regression: a clean optimizer-comparison testbed.

A single ``nn.Linear(d, 1, bias=False)`` is trained on synthetic data
``y = X @ beta_true + noise``. Because the model is exactly the linear map,
``model.weight.view(-1)`` *is* the current beta estimate, so we track
``||beta_hat - beta_true||_2`` directly — the headline metric here, because it
exposes a failure the loss curve hides.

The point: **AdamW and SOAP divide the update by the gradient's RMS**
(``sqrt(v)``), so near the optimum the step collapses to ``lr * sign(g)`` and
never vanishes — at a fixed lr the iterate hovers in a ball of radius ~``lr``
around the minimizer and never settles onto it. **Gnome divides by the
curvature** (``v``, un-square-rooted), i.e. the diagonal Gauss-Newton step
``(J^T J)^{-1} J^T r``, which vanishes as the residual shrinks — so it drives
``beta_hat`` onto the exact normal-equations solution *with no schedule at
all*. In the loss this gap is invisible (the floor is negligible in a quadratic
loss); in ``||beta_hat - beta_true||`` it is plain. The one-line version:
*AdamW can't fit a linear regression — so why trust it on any regression
problem?*

To give the baselines their standard strong treatment they get a linear-warmup
+ cosine-decay schedule (``--cosine-decay`` sets the final-lr fraction; ``0.0``
decays to zero, ``1.0`` disables decay to show the raw hovering failure). Gnome
runs at a fixed lr regardless. Decaying the baselines to zero lets them *settle*
— but at whatever point the hovering ball froze, which still sits above the
exact solution Gnome reaches on its own. That is the honest, harder-to-rebut
framing: not "the baselines can't converge," but "the baselines need a
hand-tuned schedule and still land short of where Gnome self-anneals to."

The design matrix is an anisotropic Gaussian with eigenvalues log-spaced over
``[1, condition_number]``. By default the spectrum is axis-aligned (diagonal
Hessian), which isolates the denominator effect above: all three optimizers
share the same per-coordinate basis, so the only difference is ``v`` vs
``sqrt(v)``. Pass ``--rotate`` to additionally apply a random orthogonal
rotation, making the Hessian non-axis-aligned — a *second*, independent
handicap for AdamW, whose per-coordinate scaling can rescale axes but cannot
rotate (SOAP and Gnome track the off-diagonal curvature and can). Crank
``--condition-number`` to widen the gap.

Usage:

    uv run -m experiments.ols_regression --optimizer gnome --seed 0
    uv run -m experiments.ols_regression --optimizer adamw --seed 0
    uv run -m experiments.ols_regression --optimizer soap  --seed 0
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from gnome import Gnome
from experiments.baselines.soap import SOAP
from experiments.common import (
    RunLogger,
    baseline_cosine_scheduler,
    current_lr,
    pick_device,
)


EXPERIMENT = "ols_regression"


def build_optimizer(
    name: str, params, lr: float, weight_decay: float,
    warmup: int, total_steps: int, cosine_decay: float,
):
    """Construct the optimizer and its LR schedule.

    Returns ``(optimizer, config, scheduler_or_None)``. Gnome (MSE) runs at a
    fixed lr — its Gauss-Newton step self-anneals as the residual shrinks — so
    it gets no scheduler (just its own internal warmup). SOAP and AdamW get the
    standard linear-warmup + cosine-decay treatment; ``cosine_decay`` is the
    final-lr fraction (``0.0`` → decay to zero, ``1.0`` → decay disabled).
    """
    if name == "gnome":
        cfg = dict(
            lr=lr, weight_decay=weight_decay,
            betas=(0.95, 0.95), shampoo_beta=0.95, eps=1e-4,
            precondition_frequency=10, aux_batch_size=10,
            clip=1.0, warmup=warmup,
            loss="mse", precondition_1d=True,
        )
        return Gnome(params, **cfg), cfg, None
    if name == "soap":
        cfg = dict(
            lr=lr, weight_decay=weight_decay,
            betas=(0.95, 0.95), shampoo_beta=0.95, eps=1e-8,
            precondition_frequency=10, precondition_1d=True,
        )
        opt = SOAP(params, **cfg)
    elif name == "adamw":
        cfg = dict(
            lr=lr, weight_decay=weight_decay,
            betas=(0.9, 0.99), eps=1e-8,
        )
        opt = torch.optim.AdamW(params, **cfg)
    else:
        raise ValueError(f"unknown optimizer: {name}")

    scheduler = baseline_cosine_scheduler(opt, warmup, total_steps, cosine_decay)
    cfg["warmup"] = warmup
    cfg["cosine_decay_floor"] = cosine_decay
    return opt, cfg, scheduler


def make_dataset(
    n_samples: int,
    d: int,
    seed: int,
    beta_true: torch.Tensor,
    eigenvalues: torch.Tensor,
    rotation: torch.Tensor | None,
    noise_std: float,
):
    """Draw X with anisotropic Gaussian columns and y = X @ beta_true + eps.

    ``eigenvalues`` has shape (d,) and is the eigenvalue spectrum of cov(X).
    ``rotation`` is an optional (d, d) orthogonal matrix; if provided,
    ``X = Z @ diag(sqrt(lambda)) @ rotation`` so cov(X) = R^T diag(lambda) R
    has the requested spectrum but is not axis-aligned. With ``rotation=None``
    the Hessian is diagonal (the default; the denominator effect still shows).
    """
    rng = torch.Generator().manual_seed(seed)
    Z = torch.randn(n_samples, d, generator=rng)
    X = Z * eigenvalues.sqrt().unsqueeze(0)
    if rotation is not None:
        X = X @ rotation
    y = X @ beta_true + noise_std * torch.randn(n_samples, 1, generator=rng)
    return X.to(torch.float32), y.to(torch.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--optimizer", required=True,
                   choices=["gnome", "soap", "adamw"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-2)
    # Weight decay biases beta_hat away from beta_true (ridge), which
    # contaminates the distance metric. Default to 0 here.
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--n-train", type=int, default=32_000)
    p.add_argument("--n-val", type=int, default=2048)
    p.add_argument("--dim", type=int, default=64,
                   help="Number of features d.")
    p.add_argument("--condition-number", type=float, default=1e3,
                   help="Ratio of largest to smallest eigenvalue of cov(X). "
                        "Larger -> more ill-conditioned -> bigger gap.")
    p.add_argument("--noise-std", type=float, default=0.1,
                   help="Standard deviation of additive Gaussian label noise.")
    p.add_argument("--warmup-steps", type=int, default=200,
                   help="Linear LR warmup steps for the SOAP/AdamW baselines "
                        "(Gnome uses its own internal warmup).")
    p.add_argument("--cosine-decay", type=float, default=0.0,
                   help="Final-LR fraction for the baseline cosine decay: 0.0 "
                        "decays to zero (standard treatment), 1.0 disables "
                        "decay entirely (raw SOAP/AdamW). Gnome (MSE) never "
                        "decays regardless.")
    p.add_argument("--rotate", action="store_true",
                   help="Apply a random orthogonal rotation to cov(X) so the "
                        "Hessian is non-axis-aligned. Off by default: the "
                        "exact-recovery gap is a denominator effect that shows "
                        "even with a diagonal Hessian. Turning it on adds a "
                        "second handicap for AdamW (per-coordinate scaling "
                        "cannot rotate).")
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device()

    # Ground truth beta and the eigenvalue spectrum that defines cov(X).
    # beta_true is drawn once per seed and stays fixed across the run.
    gen = torch.Generator().manual_seed(args.seed + 101)
    beta_true = torch.randn(args.dim, 1, generator=gen)
    eigenvalues = torch.logspace(
        0.0, torch.log10(torch.tensor(args.condition_number)).item(),
        args.dim,
    )
    if args.rotate:
        # Random orthogonal d x d matrix via QR of a Gaussian.
        A = torch.randn(args.dim, args.dim, generator=gen)
        Q, R = torch.linalg.qr(A)
        # Sign-canonicalize so the rotation is deterministic across torch versions.
        rotation = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)
    else:
        rotation = None

    print(f"[{EXPERIMENT}] generating data: d={args.dim}, "
          f"cond(cov)={args.condition_number:.1e}, "
          f"rotation={'on' if args.rotate else 'off'}, "
          f"noise_std={args.noise_std}")
    X_train, y_train = make_dataset(
        args.n_train, args.dim, args.seed, beta_true,
        eigenvalues, rotation, args.noise_std,
    )
    X_val, y_val = make_dataset(
        args.n_val, args.dim, args.seed + 7, beta_true,
        eigenvalues, rotation, args.noise_std,
    )

    # Closed-form OLS solution via the normal equations, beta_hat =
    # (X^T X)^{-1} X^T y. This is the best any unbiased optimizer can do on
    # this finite dataset; any iterative method should be compared against
    # it. ``torch.linalg.lstsq`` solves it in float64 for numerical safety
    # since cond(X^T X) = condition_number^2 here.
    beta_ols = torch.linalg.lstsq(
        X_train.double(), y_train.double()
    ).solution.to(torch.float32)
    with torch.no_grad():
        ols_val_pred = X_val @ beta_ols
        ols_val_loss = F.mse_loss(ols_val_pred, y_val).item()
        ss_res = ((ols_val_pred - y_val) ** 2).sum().item()
        ss_tot = ((y_val - y_val.mean()) ** 2).sum().item()
        ols_val_r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        ols_beta_dist = (beta_ols.view(-1) - beta_true.view(-1)).norm().item()
    print(f"[{EXPERIMENT}] OLS closed-form (normal equations):  "
          f"beta_dist={ols_beta_dist:.4e}  val_loss={ols_val_loss:.4e}  "
          f"r2={ols_val_r2:.4f}")

    # No standardization: we want model.weight to be directly comparable
    # to beta_true. The optimizer sees the raw, ill-conditioned problem,
    # which is the whole point of the experiment.
    model = nn.Linear(args.dim, 1, bias=False).to(device)
    beta_true_dev = beta_true.to(device).view(-1)

    loader = DataLoader(
        TensorDataset(X_train.to(device), y_train.to(device)),
        batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    X_val_t = X_val.to(device)
    y_val_t = y_val.to(device)

    total_steps = args.epochs * len(loader)
    opt, opt_cfg, scheduler = build_optimizer(
        args.optimizer, model.parameters(), args.lr, args.weight_decay,
        args.warmup_steps, total_steps, args.cosine_decay,
    )

    K = opt_cfg.get("aux_batch_size", 10) if args.optimizer == "gnome" else 0

    def beta_dist() -> float:
        with torch.no_grad():
            return (model.weight.view(-1) - beta_true_dev).norm().item()

    init_beta_dist = beta_dist()
    hyperparameters = {
        "optimizer": args.optimizer,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "n_features": args.dim,
        "n_train": args.n_train,
        "n_val": args.n_val,
        "condition_number": args.condition_number,
        "rotation": args.rotate,
        "noise_std": args.noise_std,
        "warmup_steps": args.warmup_steps,
        "cosine_decay": args.cosine_decay,
        "beta_true_norm": float(beta_true.norm().item()),
        "init_beta_dist": init_beta_dist,
        "ols_val_loss": ols_val_loss,
        "ols_val_r2": ols_val_r2,
        "ols_beta_dist": ols_beta_dist,
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

    step = 0
    val_beta_dist = init_beta_dist
    val_loss = float("nan")
    r2 = float("nan")
    for epoch in range(args.epochs):
        model.train()
        for x, y in loader:
            if args.optimizer == "gnome":
                k = min(K, max(1, x.shape[0] - 1))
                perm = torch.randperm(x.shape[0], device=x.device)
                aux_idx = perm[:k]

                def main_closure():
                    return model(x), y

                def aux_closure():
                    return model(x[aux_idx]), y[aux_idx]

                loss = opt.step(main_closure, aux_closure)
            else:
                opt.zero_grad()
                loss = F.mse_loss(model(x), y)
                loss.backward()
                opt.step()

            if scheduler is not None:
                scheduler.step()

            run.log_train(step, loss=float(loss.detach().item()),
                          beta_dist=beta_dist())
            step += 1

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = F.mse_loss(val_pred, y_val_t).item()
            ss_res = ((val_pred - y_val_t) ** 2).sum().item()
            ss_tot = ((y_val_t - y_val_t.mean()) ** 2).sum().item()
            r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        val_beta_dist = beta_dist()
        run.log_val(step, epoch=epoch, lr=current_lr(opt),
                    beta_dist=val_beta_dist, loss=val_loss, r2=r2)
        if not args.quiet:
            print(
                f"  epoch {epoch:3d}/{args.epochs}  "
                f"beta_dist={val_beta_dist:.4e}  "
                f"val_loss={val_loss:.4e}  r2={r2:.4f}",
                flush=True,
            )

    path = run.finish(
        completed=True,
        final_beta_dist=val_beta_dist,
        final_val_loss=val_loss,
        final_val_r2=r2,
        ols_beta_dist=ols_beta_dist,
    )
    print(f"[{EXPERIMENT}] saved → {path}")
    print(f"  init  beta_dist={init_beta_dist:.4e}")
    print(f"  final beta_dist={val_beta_dist:.4e}  "
          f"(OLS optimum {ols_beta_dist:.4e})")
    print(f"  final val_loss={val_loss:.4e}  final val_r2={r2:.4f}")


if __name__ == "__main__":
    main()
