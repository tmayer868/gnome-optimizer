"""Deep MNIST autoencoder: Gnome vs SOAP vs AdamW vs L-BFGS.

This is the classic difficult-optimization benchmark from Martens (2010),
trained from random initialization without layerwise pretraining.  The encoder
is ``784 -> 1000 -> 500 -> 250 -> 30`` and the decoder mirrors it.  Every
non-code layer uses the selected nonlinearity; the 30-dimensional code is
linear.  The default sparse initialization gives every unit exactly 15
nonzero incoming weights, as in the Hessian-free experiments.

The default sigmoid run minimizes binary cross-entropy with logits, matching
the original MNIST objective.  Gnome uses its Rademacher-only
``loss="bce_hutchinson"`` path.  ``--loss mse`` remains available for the
tanh comparison.  Irrespective of the training objective, evaluation logs:

* ``sse``: mean sum of squared errors per image (the historical reporting unit)
* ``pixel_mse``: mean squared error per pixel, equal to ``sse / 784``

For BCE, logits are passed directly to the numerically stable BCE-with-logits
loss and sigmoid is applied only for evaluation.  For tanh MSE, inputs and
reconstructions are mapped back to ``[0, 1]`` before metrics are reported, so
the reconstruction numbers remain comparable.

Examples:

    # Edit experiments/configs/mnist_autoencoder.py: optimizer='gnome', activation='sigmoid'
    uv run -m experiments.mnist_autoencoder
    # Edit experiments/configs/mnist_autoencoder.py: optimizer='soap', activation='sigmoid'
    uv run -m experiments.mnist_autoencoder
    # Edit experiments/configs/mnist_autoencoder.py: optimizer='adamw', activation='sigmoid'
    uv run -m experiments.mnist_autoencoder
    # Edit experiments/configs/mnist_autoencoder.py: optimizer='lbfgs', activation='sigmoid'
    uv run -m experiments.mnist_autoencoder

The default 2,000 steps are intended as a diagnostic run, not a claim of full
convergence.  Use ``--save-checkpoint`` to write a model checkpoint beside the
JSONL run, suitable for later spectrum analysis.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Iterable, Sequence

import torch
from torch import nn

from gnome import DEFAULT_METRICS, Gnome, JsonlDiagnostics
from experiments.baselines import SOAP
from experiments.common import (
    DIVERGED_EXIT,
    RunLogger,
    cosine_scheduler,
    current_lr,
    diverged,
    pick_device,
)


EXPERIMENT = "mnist_autoencoder"
DEFAULT_DATA_DIR = "experiments/data"
INPUT_DIM = 28 * 28
ENCODER_DIMS = (INPUT_DIM, 1000, 500, 250, 30)


class DeepAutoencoder(nn.Module):
    """Symmetric deep autoencoder with a linear bottleneck."""

    def __init__(
        self,
        encoder_dims: Sequence[int] = ENCODER_DIMS,
        activation: str = "sigmoid",
        init: str = "sparse",
        sparse_connections: int = 15,
        sparse_std: float | None = None,
        output_logits: bool = False,
    ) -> None:
        super().__init__()
        if len(encoder_dims) < 2:
            raise ValueError("encoder_dims must include input and code dimensions")
        if activation not in ("sigmoid", "tanh"):
            raise ValueError("activation must be 'sigmoid' or 'tanh'")
        if init not in ("sparse", "xavier"):
            raise ValueError("init must be 'sparse' or 'xavier'")
        if sparse_connections < 1:
            raise ValueError("sparse_connections must be positive")

        encoder_dims = tuple(int(d) for d in encoder_dims)
        full_dims = encoder_dims + encoder_dims[-2::-1]
        self.layers = nn.ModuleList(
            nn.Linear(n_in, n_out)
            for n_in, n_out in zip(full_dims[:-1], full_dims[1:])
        )
        self.activation_name = activation
        self.output_logits = output_logits
        self.code_layer = len(encoder_dims) - 2
        self.input_dim = encoder_dims[0]
        self.reset_parameters(init, sparse_connections, sparse_std)

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) if self.activation_name == "sigmoid" else torch.tanh(x)

    @torch.no_grad()
    def reset_parameters(
        self,
        init: str,
        sparse_connections: int,
        sparse_std: float | None,
    ) -> None:
        """Initialize every layer without tying encoder/decoder weights."""
        if init == "xavier":
            gain = nn.init.calculate_gain("tanh") if self.activation_name == "tanh" else 1.0
            for layer in self.layers:
                nn.init.xavier_uniform_(layer.weight, gain=gain)
                nn.init.zeros_(layer.bias)
            return

        # Martens sparse initialization: exactly k incoming connections per
        # unit, drawn from N(0, 1).  For tanh, Martens & Sutskever's equivalent
        # sigmoid-to-tanh transform scales weights by 0.25 and uses bias 0.5.
        std = sparse_std
        if std is None:
            std = 0.25 if self.activation_name == "tanh" else 1.0
        for index, layer in enumerate(self.layers):
            n_out, n_in = layer.weight.shape
            k = min(sparse_connections, n_in)
            scores = torch.rand(n_out, n_in, device=layer.weight.device)
            connections = scores.topk(k, dim=1, largest=False).indices
            values = torch.randn(
                n_out, k, device=layer.weight.device, dtype=layer.weight.dtype
            ).mul_(std)
            layer.weight.zero_()
            layer.weight.scatter_(1, connections, values)
            bias = (
                0.5
                if self.activation_name == "tanh" and index != self.code_layer
                else 0.0
            )
            layer.bias.fill_(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1)
        for index, layer in enumerate(self.layers):
            x = layer(x)
            is_output = index == len(self.layers) - 1
            if index != self.code_layer and not (is_output and self.output_logits):
                x = self._activate(x)
        return x


class NormTrackingGnome(Gnome):
    """Gnome variant exposing current main/surrogate gradient norms.

    Gnome obtains both gradients internally via ``autograd.grad``, so ordinary
    ``parameter.grad`` inspection cannot observe them.  Accumulating their
    squared norms in ``_param_step`` adds no extra forward or backward pass.
    """

    last_main_grad_norm: torch.Tensor | None = None
    last_surrogate_grad_norm: torch.Tensor | None = None

    def step(self, main_closure, aux_closure):
        self._tracked_main_sq = None
        self._tracked_surrogate_sq = None
        loss = super().step(main_closure, aux_closure)
        self.last_main_grad_norm = (
            None if self._tracked_main_sq is None else self._tracked_main_sq.sqrt()
        )
        self.last_surrogate_grad_norm = (
            None
            if self._tracked_surrogate_sq is None
            else self._tracked_surrogate_sq.sqrt()
        )
        return loss

    def _param_step(self, p, g, G_s, group, idx=0):
        main_sq = g.detach().square().sum()
        surrogate_sq = G_s.detach().square().sum()
        if self._tracked_main_sq is None:
            self._tracked_main_sq = main_sq
            self._tracked_surrogate_sq = surrogate_sq
        else:
            self._tracked_main_sq.add_(main_sq)
            self._tracked_surrogate_sq.add_(surrogate_sq)
        return super()._param_step(p, g, G_s, group, idx)


def load_mnist_tensors(data_dir: str = DEFAULT_DATA_DIR):
    """Return MNIST train/test images as flattened uint8 tensors.

    Keeping pixels as uint8 uses about 55 MB instead of roughly 220 MB for a
    float32 copy.  Batches are converted and scaled only when used.
    """
    import ssl
    import warnings

    from torchvision import datasets

    ssl._create_default_https_context = ssl._create_unverified_context
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train = datasets.MNIST(data_dir, train=True, download=True)
        test = datasets.MNIST(data_dir, train=False, download=True)
    return train.data.reshape(-1, INPUT_DIM), test.data.reshape(-1, INPUT_DIM)


def prepare_images(
    images_u8: torch.Tensor,
    activation: str,
    device: torch.device,
) -> torch.Tensor:
    """Convert uint8 pixels to the model's input/output range."""
    x = images_u8.to(device=device, dtype=torch.float32).div_(255.0)
    if activation == "tanh":
        x.mul_(2.0).sub_(1.0)
    return x


def reconstruction_objective(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss: str = "mse",
) -> torch.Tensor:
    """Training objective with sum-over-pixels, mean-over-batch reduction."""
    if loss == "mse":
        return prediction.sub(target).square().sum() / target.shape[0]
    if loss == "bce":
        return nn.functional.binary_cross_entropy_with_logits(
            prediction, target, reduction="sum"
        ) / target.shape[0]
    raise ValueError(f"unknown reconstruction loss: {loss}")


def objective_to_raw_metrics(
    loss: float,
    activation: str,
    input_dim: int = INPUT_DIM,
) -> tuple[float, float]:
    """Convert model-scale per-image SSE to raw-[0,1] SSE and pixel MSE."""
    raw_sse = loss / (4.0 if activation == "tanh" else 1.0)
    return raw_sse, raw_sse / input_dim


@torch.no_grad()
def evaluate(
    model: nn.Module,
    images_u8: torch.Tensor,
    activation: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    """Evaluate comparable raw-[0,1] reconstruction metrics."""
    was_training = model.training
    model.eval()
    squared_error = 0.0
    count = 0
    for start in range(0, images_u8.shape[0], batch_size):
        target = prepare_images(images_u8[start:start + batch_size], activation, device)
        prediction = model(target)
        if getattr(model, "output_logits", False):
            prediction = torch.sigmoid(prediction)
        elif activation == "tanh":
            target = target.add(1.0).mul(0.5)
            prediction = prediction.add(1.0).mul(0.5)
        squared_error += prediction.sub(target).square().sum().item()
        count += target.shape[0]
    if was_training:
        model.train()
    sse = squared_error / count
    return {"sse": sse, "pixel_mse": sse / images_u8.shape[1]}


def infinite_shuffled_batches(
    n: int,
    batch_size: int,
    generator: torch.Generator,
) -> Iterable[torch.Tensor]:
    """Yield shuffled CPU indices forever, including each epoch's tail."""
    while True:
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            yield permutation[start:start + batch_size]


def gradient_norm(parameters: Iterable[nn.Parameter]) -> torch.Tensor | None:
    """L2 norm of populated ``.grad`` tensors, or ``None`` if unavailable."""
    total = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        contribution = parameter.grad.detach().square().sum()
        total = contribution if total is None else total.add(contribution)
    return None if total is None else total.sqrt()


def synchronize(device: torch.device) -> None:
    """Make accelerator timings measure completed work rather than dispatch."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def build_optimizer(
    name: str,
    params,
    lr: float,
    weight_decay: float = 0.0,
    beta1: float = 0.9,
    beta2: float = 0.99,
    eps: float = 1e-12,
    trust_region: float = 1.0,
    precondition_frequency: int = 10,
    max_precond_dim: int = 2048,
    lbfgs_max_iter: int = 10,
    lbfgs_history: int = 10,
    loss: str = "mse",
):
    """Build one optimizer and return it with its logged configuration."""
    if name == "gnome":
        config = dict(
            lr=lr,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
            shampoo_beta=beta2,
            eps=eps,
            trust_radius=trust_region if trust_region > 0 else None,
            precondition_frequency=precondition_frequency,
            max_precond_dim=max_precond_dim,
            precondition_1d=False,
            norm_free=False,
            loss="bce_hutchinson" if loss == "bce" else "mse",
        )
        optimizer = NormTrackingGnome(params, **config)
    elif name == "soap":
        config = dict(
            lr=lr,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
            shampoo_beta=beta2,
            eps=1e-8,
            precondition_frequency=precondition_frequency,
            max_precond_dim=max_precond_dim,
            precondition_1d=False,
        )
        optimizer = SOAP(params, **config)
    elif name == "adamw":
        config = dict(
            lr=lr,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
            eps=1e-8,
        )
        optimizer = torch.optim.AdamW(params, **config)
    elif name == "lbfgs":
        if weight_decay != 0.0:
            raise ValueError("L-BFGS does not support decoupled weight decay")
        config = dict(
            lr=lr,
            max_iter=lbfgs_max_iter,
            history_size=lbfgs_history,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-12,
            tolerance_change=1e-12,
        )
        optimizer = torch.optim.LBFGS(params, **config)
    else:
        raise ValueError(f"unknown optimizer: {name}")
    return optimizer, config


def diagnostic_parameter_indices(model: nn.Module, selection: str) -> list[int] | None:
    """Resolve ``weights``, ``all``, or comma-separated parameter indices."""
    if selection == "all":
        return None
    if selection == "weights":
        return [
            index
            for index, (name, _) in enumerate(model.named_parameters())
            if name.endswith("weight")
        ]
    try:
        return [int(value) for value in selection.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError(
            "diagnostics-params must be 'weights', 'all', or comma-separated integers"
        ) from exc


def _checkpoint_path(args: argparse.Namespace, run: RunLogger) -> str | None:
    if args.checkpoint_path:
        return args.checkpoint_path
    if args.save_checkpoint:
        return os.path.splitext(run.path)[0] + ".pt"
    return None


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    if args.batch_size < 2 and args.optimizer == "gnome":
        raise ValueError("Gnome needs batch-size >= 2 for its auxiliary batch")
    if args.loss == "bce" and args.activation != "sigmoid":
        raise ValueError("BCE requires --activation sigmoid; use MSE with tanh")

    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"[{EXPERIMENT}] {args.optimizer} | loading MNIST...", flush=True)
    train_u8, test_u8 = load_mnist_tensors(args.data_dir)

    model = DeepAutoencoder(
        activation=args.activation,
        init=args.init,
        sparse_connections=args.sparse_connections,
        sparse_std=args.sparse_std,
        output_logits=args.loss == "bce",
    ).to(device)
    optimizer, optimizer_config = build_optimizer(
        args.optimizer,
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        trust_region=args.trust_region,
        precondition_frequency=args.precondition_frequency,
        max_precond_dim=args.max_precond_dim,
        lbfgs_max_iter=args.lbfgs_max_iter,
        lbfgs_history=args.lbfgs_history,
        loss=args.loss,
    )
    scheduler = cosine_scheduler(
        optimizer, args.warmup_steps, args.steps, args.cosine_decay
    )

    subset_size = min(args.eval_train_size, train_u8.shape[0])
    subset_generator = torch.Generator().manual_seed(args.seed + 10_000)
    subset_indices = torch.randperm(
        train_u8.shape[0], generator=subset_generator
    )[:subset_size]
    train_eval_u8 = train_u8[subset_indices]

    batch_generator = torch.Generator().manual_seed(args.seed + 1)
    batches = infinite_shuffled_batches(
        train_u8.shape[0], args.batch_size, batch_generator
    )
    fixed_lbfgs_batch = None
    if args.optimizer == "lbfgs":
        fixed_size = min(args.lbfgs_batch_size, train_u8.shape[0])
        fixed_indices = torch.randperm(
            train_u8.shape[0], generator=batch_generator
        )[:fixed_size]
        fixed_lbfgs_batch = prepare_images(
            train_u8[fixed_indices], args.activation, device
        )

    parameter_names = [name for name, _ in model.named_parameters()]
    hyperparameters = {
        "optimizer": args.optimizer,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "aux_batch_size": args.aux_batch_size if args.optimizer == "gnome" else 0,
        "activation": args.activation,
        "loss": args.loss,
        "init": args.init,
        "sparse_connections": args.sparse_connections,
        "sparse_std": args.sparse_std,
        "encoder_dims": list(ENCODER_DIMS),
        "linear_code": True,
        "objective": (
            "mean_per_image_bce_with_logits"
            if args.loss == "bce"
            else "mean_per_image_sse"
        ),
        "reporting_scale": "raw_[0,1]",
        "matches_original_mnist_loss_family": args.loss == "bce",
        "n_train": int(train_u8.shape[0]),
        "n_test": int(test_u8.shape[0]),
        "eval_train_size": subset_size,
        "warmup_steps": args.warmup_steps,
        "cosine_decay": args.cosine_decay,
        "target_sse": args.target_sse,
        "n_params": sum(p.numel() for p in model.parameters()),
        "parameter_names": parameter_names,
        "device": str(device),
        "diagnostics_every": args.diagnostics_every,
        "diagnostics_params": args.diagnostics_params,
        **{f"opt.{key}": value for key, value in optimizer_config.items()},
    }

    if not args.quiet:
        print(
            f"[{EXPERIMENT}] {args.optimizer} | "
            f"{args.activation}/{args.loss}/{args.init} | "
            f"params={hyperparameters['n_params']:,} | device={device}\n"
            f"  architecture={'→'.join(str(d) for d in ENCODER_DIMS)}"
            f"→{'→'.join(str(d) for d in ENCODER_DIMS[-2::-1])} | "
            f"steps={args.steps} batch={args.batch_size} "
            f"aux_K={hyperparameters['aux_batch_size']}",
            flush=True,
        )

    diagnostics = None
    with RunLogger(
        EXPERIMENT,
        args.optimizer,
        args.seed,
        hyperparameters,
        runs_dir=args.runs_dir,
    ) as run:
        if args.optimizer == "gnome" and args.diagnostics_every > 0:
            diagnostics_path = os.path.splitext(run.path)[0] + ".diagnostics.jsonl"
            diagnostics = JsonlDiagnostics(
                diagnostics_path,
                params=diagnostic_parameter_indices(model, args.diagnostics_params),
            )
            optimizer.diagnostics = diagnostics
            optimizer.diagnostics_every = args.diagnostics_every
            optimizer.metrics = {
                **DEFAULT_METRICS,
                "v_median": lambda context: context["v"].median(),
                "lam_over_v_median": lambda context: (
                    context["lam"]
                    / context["v"].median().clamp_min(1e-30)
                ),
            }

        elapsed_train_seconds = 0.0
        total_step_time_ms = 0.0
        window_time_ms = 0.0
        window_loss = 0.0
        window_count = 0
        time_to_target_seconds = None
        step_to_target = None
        best_test_sse = float("inf")

        initial_train = evaluate(
            model, train_eval_u8, args.activation, device, args.eval_batch_size
        )
        initial_test = evaluate(
            model, test_u8, args.activation, device, args.eval_batch_size
        )
        run.log_val(
            0,
            train_sse=initial_train["sse"],
            train_pixel_mse=initial_train["pixel_mse"],
            test_sse=initial_test["sse"],
            test_pixel_mse=initial_test["pixel_mse"],
            elapsed_train_seconds=0.0,
        )
        best_test_sse = initial_test["sse"]
        if initial_train["sse"] <= args.target_sse:
            time_to_target_seconds = 0.0
            step_to_target = 0

        try:
            for step_index in range(args.steps):
                model.train()
                if fixed_lbfgs_batch is None:
                    indices = next(batches)
                    x_batch = prepare_images(
                        train_u8[indices], args.activation, device
                    )
                else:
                    x_batch = fixed_lbfgs_batch

                lr_now = current_lr(optimizer)
                synchronize(device)
                step_started = time.perf_counter()
                surrogate_grad_norm = None

                if args.optimizer == "gnome":
                    k = min(args.aux_batch_size, x_batch.shape[0] - 1)
                    aux_indices = torch.randperm(
                        x_batch.shape[0], device=device
                    )[:k]
                    x_aux = x_batch[aux_indices]

                    loss = optimizer.step(
                        lambda: (model(x_batch), x_batch),
                        lambda: (model(x_aux), x_aux),
                    )
                    main_grad_norm = optimizer.last_main_grad_norm
                    surrogate_grad_norm = optimizer.last_surrogate_grad_norm
                elif args.optimizer == "lbfgs":
                    def closure():
                        optimizer.zero_grad()
                        closure_loss = reconstruction_objective(
                            model(x_batch), x_batch, args.loss
                        )
                        closure_loss.backward()
                        return closure_loss

                    loss = optimizer.step(closure)
                    main_grad_norm = gradient_norm(model.parameters())
                else:
                    optimizer.zero_grad()
                    loss = reconstruction_objective(
                        model(x_batch), x_batch, args.loss
                    )
                    loss.backward()
                    main_grad_norm = gradient_norm(model.parameters())
                    optimizer.step()

                scheduler.step()
                synchronize(device)
                step_time_ms = (time.perf_counter() - step_started) * 1000.0
                elapsed_train_seconds += step_time_ms / 1000.0
                total_step_time_ms += step_time_ms

                loss_value = float(loss.detach().item())
                if args.loss == "mse":
                    train_sse, train_pixel_mse = objective_to_raw_metrics(
                        loss_value, args.activation
                    )
                else:
                    train_sse, train_pixel_mse = None, None
                completed_step = step_index + 1
                if diverged(loss_value):
                    run.finish(
                        completed=False,
                        diverged=True,
                        diverged_step=completed_step,
                    )
                    raise SystemExit(DIVERGED_EXIT)

                run.log_train(
                    completed_step,
                    loss=loss_value,
                    sse=train_sse,
                    pixel_mse=train_pixel_mse,
                    lr=lr_now,
                    step_time_ms=step_time_ms,
                    elapsed_train_seconds=elapsed_train_seconds,
                    main_grad_norm=main_grad_norm,
                    surrogate_grad_norm=surrogate_grad_norm,
                )
                window_loss += loss_value
                window_time_ms += step_time_ms
                window_count += 1

                if (
                    not args.quiet
                    and args.log_every > 0
                    and completed_step % args.log_every == 0
                ):
                    print(
                        f"    step {completed_step:6d}  "
                        f"train_loss[last {window_count}]="
                        f"{window_loss / window_count:.4f}  "
                        f"lr={lr_now:.2e}  "
                        f"{window_time_ms / window_count:.1f} ms/step",
                        flush=True,
                    )
                    window_loss = 0.0
                    window_time_ms = 0.0
                    window_count = 0

                should_evaluate = (
                    completed_step == args.steps
                    or (
                        args.eval_every > 0
                        and completed_step % args.eval_every == 0
                    )
                )
                if should_evaluate:
                    train_metrics = evaluate(
                        model,
                        train_eval_u8,
                        args.activation,
                        device,
                        args.eval_batch_size,
                    )
                    test_metrics = evaluate(
                        model,
                        test_u8,
                        args.activation,
                        device,
                        args.eval_batch_size,
                    )
                    best_test_sse = min(best_test_sse, test_metrics["sse"])
                    if (
                        time_to_target_seconds is None
                        and train_metrics["sse"] <= args.target_sse
                    ):
                        time_to_target_seconds = elapsed_train_seconds
                        step_to_target = completed_step
                    run.log_val(
                        completed_step,
                        train_sse=train_metrics["sse"],
                        train_pixel_mse=train_metrics["pixel_mse"],
                        test_sse=test_metrics["sse"],
                        test_pixel_mse=test_metrics["pixel_mse"],
                        lr=current_lr(optimizer),
                        elapsed_train_seconds=elapsed_train_seconds,
                    )
                    if not args.quiet:
                        print(
                            f"  eval step {completed_step:6d}  "
                            f"train_sse={train_metrics['sse']:.4f}  "
                            f"test_sse={test_metrics['sse']:.4f}  "
                            f"test_pixel_mse={test_metrics['pixel_mse']:.3e}",
                            flush=True,
                        )

            final_train = evaluate(
                model, train_u8, args.activation, device, args.eval_batch_size
            )
            final_test = evaluate(
                model, test_u8, args.activation, device, args.eval_batch_size
            )
            checkpoint_path = _checkpoint_path(args, run)
            if checkpoint_path is not None:
                os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "activation": args.activation,
                        "loss": args.loss,
                        "output_logits": model.output_logits,
                        "encoder_dims": ENCODER_DIMS,
                        "init": args.init,
                        "step": args.steps,
                        "seed": args.seed,
                    },
                    checkpoint_path,
                )
            mean_step_time_ms = total_step_time_ms / args.steps
            run.finish(
                completed=True,
                final_train_sse=final_train["sse"],
                final_train_pixel_mse=final_train["pixel_mse"],
                final_test_sse=final_test["sse"],
                final_test_pixel_mse=final_test["pixel_mse"],
                best_test_sse=best_test_sse,
                mean_step_time_ms=mean_step_time_ms,
                elapsed_train_seconds=elapsed_train_seconds,
                target_sse=args.target_sse,
                step_to_target=step_to_target,
                time_to_target_seconds=time_to_target_seconds,
                checkpoint_path=checkpoint_path,
            )
        finally:
            if diagnostics is not None:
                diagnostics.close()

    print(
        f"[{EXPERIMENT}] done → train_sse={final_train['sse']:.4f}  "
        f"test_sse={final_test['sse']:.4f}  "
        f"test_pixel_mse={final_test['pixel_mse']:.3e}  "
        f"mean={mean_step_time_ms:.1f} ms/step",
        flush=True,
    )
    if checkpoint_path is not None:
        print(f"[{EXPERIMENT}] checkpoint → {checkpoint_path}", flush=True)
    if diagnostics is not None:
        print(f"[{EXPERIMENT}] diagnostics → {diagnostics.path}", flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    from experiments.common.config import ConfigParser
    parser = ConfigParser(__file__)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
