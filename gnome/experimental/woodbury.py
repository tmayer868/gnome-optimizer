"""Finite-history, tensor-block GGN with exact damped Woodbury solves."""

from collections import deque
import math

import torch
from torch.optim import Optimizer

from ..optimizer import Gnome


class WoodburyGGN(Optimizer):
    """Precondition each 2D parameter with its finite EMA of GGN samples.

    ``step(main_closure, aux_closure)`` uses Gnome's unchanged loss and random
    surrogate. Each closure returns ``(prediction, target)`` on an independent
    graph, as for Gnome. Alternatively, populate ordinary ``param.grad`` and
    call ``step(surrogate_grads=[...])`` (or ``step([...])``), with one already
    scaled surrogate gradient per trainable parameter in parameter-group order.
    A ``None`` entry adds no sample; ``step()`` uses the existing history.

    ``curvature_warmup_steps`` (default 100) holds parameters fixed for the
    first N calls while accumulating curvature and ordinary-gradient momentum.
    Updates start on call N+1. The scheduled ``group['lr']`` is left intact;
    ``group['applied_lr']`` records the effective LR used by the last call.

    Histories contain detached matrix copies, oldest first. ``max_history``
    overrides ceil(log(history_tol) / log(beta2)); beta2=0 retains one sample.
    With ``trust_radius`` (default 1), damping is increased from its configured
    floor until ||P|| <= trust_radius * sqrt(||W||**2 + 1). Set it to None for
    the original fixed-damping update. No clipping, weight decay, or additional
    surrogate scaling is applied. Unsupported shapes are rejected, including 1D biases;
    use bias-free layers or the repository's fused [W|b] matrices.

    ``gram_dtype=torch.float64`` promotes inner products, the small solve, and
    temporary reconstruction, while leaving stored samples in parameter
    precision (at least float32). On MPS these promoted operations run on CPU.
    MPS also uses CPU for the small linear solve in float32 mode.
    Diagnostics follow the existing callable-record interface and are only
    computed at ``diagnostics_every`` steps when a sink is attached.
    """

    # Reuse the existing implementation verbatim, including probe normalization.
    _compute_loss = Gnome._compute_loss
    _build_surrogate = Gnome._build_surrogate
    _SQRT_TWO = Gnome._SQRT_TWO

    def __init__(
        self, params, lr=1e-3, beta1=0.0, beta2=0.99, damping=1e-3,
        history_tol=0.01, max_history=None, gram_dtype=None, loss="mse",
        diagnostics=None, diagnostics_every=1000, curvature_warmup_steps=100,
        trust_radius=1.0,
    ):
        if loss not in ("mse", "bce_hutchinson", "cce", "cce_hutchinson"):
            raise ValueError(f"Unsupported loss: {loss}")
        if not isinstance(diagnostics_every, int) or diagnostics_every < 0:
            raise ValueError("diagnostics_every must be a nonnegative integer")
        self._loss_mode = loss
        self._step_count = 0
        self.diagnostics = diagnostics
        self.diagnostics_every = diagnostics_every
        super().__init__(params, dict(
            lr=lr, beta1=beta1, beta2=beta2, damping=damping,
            history_tol=history_tol, max_history=max_history, gram_dtype=gram_dtype,
            curvature_warmup_steps=curvature_warmup_steps,
            trust_radius=trust_radius,
        ))

    def add_param_group(self, param_group):
        group = dict(param_group)
        params = group["params"]
        group["params"] = [params] if isinstance(params, torch.Tensor) else list(params)
        config = {**self.defaults, **group}
        radius = config["trust_radius"]
        if radius is not None and (not math.isfinite(radius) or radius <= 0):
            raise ValueError("trust_radius must be finite and positive, or None")
        warmup = config["curvature_warmup_steps"]
        if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
            raise ValueError("curvature_warmup_steps must be a nonnegative integer")
        for key in ("lr", "damping"):
            v = config[key]
            if not math.isfinite(v) or v < 0 or (key == "damping" and v == 0):
                raise ValueError(f"{key} must be finite and {'positive' if key == 'damping' else 'nonnegative'}")
        for key in ("beta1", "beta2"):
            if not 0 <= config[key] < 1:
                raise ValueError(f"{key} must be in [0, 1)")
        if not 0 < config["history_tol"] < 1:
            raise ValueError("history_tol must be in (0, 1)")
        cap = config["max_history"]
        if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 1):
            raise ValueError("max_history must be a positive integer or None")
        if config["gram_dtype"] not in (None, torch.float32, torch.float64):
            raise ValueError("gram_dtype must be None, torch.float32 or torch.float64")
        for p in group["params"]:
            if p.ndim != 2 or not p.is_floating_point():
                raise ValueError("WoodburyGGN supports only real 2D matrix parameters; "
                                 "omit 1D biases or use fused [W|b] matrices")
        group["history_size"] = cap or (
            math.ceil(math.log(config["history_tol"]) / math.log(config["beta2"]))
            if config["beta2"] else 1
        )
        group["applied_lr"] = 0.0 if warmup else config["lr"]
        super().add_param_group(group)

    @staticmethod
    def _sample_dtype(p):
        return torch.float64 if p.dtype == torch.float64 else torch.float32

    @classmethod
    def _work_options(cls, p, group):
        dtype = group["gram_dtype"] or cls._sample_dtype(p)
        device = torch.device("cpu") if p.device.type == "mps" and dtype == torch.float64 else p.device
        return dict(device=device, dtype=dtype)

    def _init_state(self, p, group):
        state = self.state[p]
        if not state:
            state.update(history=deque(), gram=torch.empty((0, 0), **self._work_options(p, group)),
                         step=0, solve_failures=0, jitter_events=0)
        return state

    def _append_sample(self, p, sample, group, state):
        history, gram = state["history"], state["gram"]
        # Drop before computing products: the discarded row is never used again.
        if len(history) == group["history_size"]:
            history.popleft()
            gram = gram[1:, 1:]
        sample = sample.detach().to(device=p.device, dtype=self._sample_dtype(p)).clone()
        work = sample.to(device=gram.device, dtype=gram.dtype)
        n = len(history)
        updated = gram.new_empty((n + 1, n + 1))
        updated[:n, :n] = gram
        for i, old in enumerate(history):
            product = (old.to(work) * work).sum()
            updated[i, n] = product
            updated[n, i] = product
        updated[n, n] = (work * work).sum()
        history.append(sample)
        state["gram"] = updated

    @staticmethod
    def _weights(n, beta2, like):
        if n == 0:
            return like.new_empty(0)
        ages = torch.arange(n - 1, -1, -1, device=like.device, dtype=like.dtype)
        if beta2 == 0:
            return (ages == 0).to(like.dtype)
        # expm1 avoids cancellation in the bias correction near beta2=1.
        return ages.mul(math.log(beta2)).exp() * ((1 - beta2) / -math.expm1(n * math.log(beta2)))

    @staticmethod
    def _solve(system, b, state, return_factor=False):
        output_like = b
        # MPS does not implement every dense linear-algebra operation. Transfer
        # only the small system in ordinary float32 mode, not the history.
        if system.device.type == "mps":
            system, b = system.cpu(), b.cpu()
        factor, info = torch.linalg.cholesky_ex(system)
        if info.item() == 0:
            result = torch.cholesky_solve(b[:, None], factor)[:, 0].to(output_like)
            return (result, factor) if return_factor else result
        state["solve_failures"] += 1
        # A generic solve is the fallback requested for Cholesky roundoff.
        try:
            result = torch.linalg.solve(system, b)
            if torch.isfinite(result).all():
                result = result.to(output_like)
                return (result, None) if return_factor else result
        except torch.linalg.LinAlgError:
            pass
        # Last resort: tiny numerical jitter, distinct from optimizer damping.
        jitter = torch.finfo(system.dtype).eps * system.diagonal().abs().max().clamp_min(1)
        state["jitter_events"] += 1
        result = torch.linalg.solve(
            system + jitter * torch.eye(len(b), device=b.device, dtype=b.dtype), b
        ).to(output_like)
        return (result, None) if return_factor else result

    @staticmethod
    def _history_direction(x, history, a, c, damping):
        correction = torch.zeros_like(x)
        for s, coefficient in zip(history, a * c):
            correction.add_(s.to(x) * coefficient)
        return (x - correction) / damping

    def _trust_damping(self, x, history, a, k, b, floor, budget, state):
        """Safeguarded reciprocal-norm Newton solve in history space.

        No eigenvectors or parameter-space matrices. Norm cancellation triggers
        a direct Frobenius evaluation; also check the final reconstructed step.
        Always return the feasible side of the scalar bracket.
        """
        x2 = x.square().sum().item()
        eps = torch.finfo(k.dtype).eps
        tol = max(1e-6, 32 * eps)
        iterations = 0

        def evaluate(lam, direct):
            nonlocal iterations
            iterations += 1
            system = k.clone()
            system.diagonal().add_(lam)
            c, factor = self._solve(system, b, state, return_factor=True)
            bc, ckc = (b * c).sum().item(), (c * (k @ c)).sum().item()
            numerator = x2 - 2 * bc + ckc
            # Near the history span these terms cancel. Do not interpret a
            # negative/roundoff-sized difference as a zero Newton direction.
            uncertain = numerator <= 64 * eps * (x2 + 2 * abs(bc) + abs(ckc))
            if direct or uncertain:
                phi = self._history_direction(x, history, a, c, lam).norm().item()
            else:
                phi = math.sqrt(numerator) / lam
            if not math.isfinite(phi):
                raise RuntimeError("Woodbury trust-region norm is non-finite; check gradients/precision")
            return phi, c, system, factor

        def search(direct=False):
            lo = floor
            phi, c, system, factor = evaluate(lo, direct)
            if phi <= budget:
                return lo, c
            # PSD curvature implies ||P(lam)|| <= ||X||/lam.
            hi = max(lo, math.sqrt(x2) / budget) * (1 + 8 * eps)
            hi_phi, hi_c, _, _ = evaluate(hi, direct)
            for _ in range(8):
                if hi_phi <= budget:
                    break
                hi *= 2
                hi_phi, hi_c, _, _ = evaluate(hi, direct)
            else:
                raise RuntimeError("Could not bracket Woodbury trust-region damping")
            lam = lo
            previous = state.get("lm_lambda", lo)
            if lo < previous < hi:
                lam = previous
                phi, c, system, factor = evaluate(lam, direct)
            for _ in range(40):
                if phi <= budget:
                    hi, hi_c = lam, c
                    if phi >= budget * (1 - tol):
                        return hi, hi_c
                else:
                    lo = lam
                if hi - lo <= tol * hi:
                    return hi, hi_c
                # phi' = -(phi**2 - c.T (K+lam I)^-1 c)/(lam*phi).
                # Newton on 1/budget - 1/phi converges rapidly in the nullspace.
                t = (torch.cholesky_solve(c.to(factor)[:, None], factor)[:, 0].to(c)
                     if factor is not None else self._solve(system, c, state))
                denominator = phi * phi - (c * t).sum().item()
                candidate = (lam + lam * phi * phi * (phi / budget - 1) / denominator
                             if denominator > 0 else float("nan"))
                if not math.isfinite(candidate) or not lo < candidate < hi or candidate == lam:
                    candidate = (lo + hi) * 0.5
                lam = candidate
                phi, c, system, factor = evaluate(lam, direct)
            return hi, hi_c

        lam, c = search()
        direction = self._history_direction(x, history, a, c, lam)
        if direction.norm().item() > budget * (1 + tol):
            # The history-space norm can underestimate after floating-point
            # cancellation. Re-solve with actual matrix norms, never clip P.
            state["trust_fallbacks"] = state.get("trust_fallbacks", 0) + 1
            lam, c = search(direct=True)
            direction = self._history_direction(x, history, a, c, lam)
        state["trust_iterations"] = iterations
        return direction, lam

    def _precondition(self, x, group, state, trust_budget=None):
        gram, history = state["gram"], state["history"]
        x = x.to(device=gram.device, dtype=gram.dtype)
        a = self._weights(len(history), group["beta2"], gram).sqrt()
        k = a[:, None] * gram * a[None, :]
        k = (k + k.T) * 0.5
        damping = group["damping"]
        state["trust_iterations"] = 0
        if history:
            b = a * torch.stack([(s.to(x) * x).sum() for s in history])
            if trust_budget is not None:
                direction, damping = self._trust_damping(
                    x, history, a, k, b, damping, trust_budget, state)
            else:
                system = k.clone()
                system.diagonal().add_(damping)
                c = self._solve(system, b, state)
                direction = self._history_direction(x, history, a, c, damping)
        else:
            if trust_budget is not None:
                damping = max(damping, x.norm().item() / trust_budget)
            direction = x / damping
        state["lm_lambda"] = damping
        state["trust_budget"] = trust_budget
        return direction, k

    def _emit_diagnostics(self, p, g, x, direction, k, group, state, index):
        if self.diagnostics is None or not self.diagnostics_every or self._step_count % self.diagnostics_every:
            return
        x = x.to(direction)
        x_norm, p_norm = x.norm().item(), direction.norm().item()
        eigs = torch.linalg.eigvalsh(k.cpu() if k.device.type == "mps" else k)
        lo, hi = (eigs[0].item(), eigs[-1].item()) if len(eigs) else (0.0, 0.0)
        damping = state["lm_lambda"]
        budget = state["trust_budget"]
        self.diagnostics(dict(
            step=self._step_count, param=index, shape=list(p.shape), numel=p.numel(),
            lr=group["applied_lr"], scheduled_lr=group["lr"],
            curvature_warmup=self._step_count <= group["curvature_warmup_steps"],
            history_length=len(state["history"]),
            history_max=group["history_size"], damping=damping,
            damping_floor=group["damping"], trust_budget=budget,
            trust_ratio=p_norm / budget if budget is not None else None,
            trust_binding=damping > group["damping"],
            trust_iterations=state["trust_iterations"],
            trust_fallbacks=state.get("trust_fallbacks", 0),
            grad_norm=g.to(direction).norm().item(), x_norm=x_norm,
            direction_norm=p_norm, direction_ratio=p_norm / x_norm if x_norm else 0.0,
            cosine=(x * direction).sum().item() / (x_norm * p_norm) if x_norm * p_norm else 0.0,
            k_eig_min=lo, k_eig_max=hi,
            condition_number=(hi + damping) / (lo + damping) if lo + damping > 0 else float("inf"),
            solve_failures=state["solve_failures"], jitter_events=state["jitter_events"],
        ))

    @torch.no_grad()
    def step(self, main_closure=None, aux_closure=None, *, surrogate_grads=None):
        """Apply one damped step; return the loss when using closures."""
        params = [p for group in self.param_groups for p in group["params"] if p.requires_grad]
        loss = None
        if callable(main_closure):
            if not callable(aux_closure) or surrogate_grads is not None:
                raise ValueError("Supply both main/aux closures, or only surrogate_grads")
            with torch.enable_grad():
                prediction, target = main_closure()
                loss = self._compute_loss(prediction, target)
                ordinary = torch.autograd.grad(loss, params, allow_unused=True)
                prediction, _ = aux_closure()
                indices = torch.arange(prediction.shape[0], device=prediction.device)
                surrogate = self._build_surrogate(prediction, indices)
                samples = torch.autograd.grad(surrogate, params, allow_unused=True)
            loss = loss.detach()
            # An unused aux parameter has a zero surrogate derivative, just as
            # in Gnome; this is a real zero sample, not missing curvature data.
            samples = [torch.zeros_like(p) if s is None else s for p, s in zip(params, samples)]
        else:
            if aux_closure is not None or (main_closure is not None and surrogate_grads is not None):
                raise ValueError("Supply both main/aux closures, or only surrogate_grads")
            samples = surrogate_grads if main_closure is None else main_closure
            samples = [None] * len(params) if samples is None else list(samples)
            ordinary = [p.grad for p in params]
        if len(samples) != len(params):
            raise ValueError("Expected one surrogate gradient per trainable parameter")
        # Validate the entire step before modifying any state or parameters.
        for p, g, s in zip(params, ordinary, samples):
            for value in (g, s):
                if value is not None and (value.shape != p.shape or value.is_sparse or value.is_complex()):
                    raise ValueError("Gradients must be dense real matrices matching their parameter")
        self._step_count += 1
        index = 0
        for group in self.param_groups:
            group["applied_lr"] = (0.0 if self._step_count <= group["curvature_warmup_steps"]
                                   else group["lr"])
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                g, sample = ordinary[index], samples[index]
                state = self._init_state(p, group)
                if sample is not None:
                    self._append_sample(p, sample, group, state)
                if g is not None:
                    state["step"] += 1
                    x = g.detach().to(dtype=self._sample_dtype(p), device=p.device)
                    if group["beta1"]:
                        if "momentum" not in state:
                            state["momentum"] = torch.zeros_like(x)
                        state["momentum"].lerp_(x, 1 - group["beta1"])
                        x = state["momentum"] / (1 - group["beta1"] ** state["step"])
                    diagnostics_due = (self.diagnostics is not None and self.diagnostics_every
                                       and self._step_count % self.diagnostics_every == 0)
                    # Zero LR still updates history/momentum, but needs no solve
                    # unless diagnostics request the hypothetical direction.
                    if group["applied_lr"] != 0 or diagnostics_due:
                        budget = (group["trust_radius"] * (p.to(state["gram"]).square().sum() + 1).sqrt().item()
                                  if group["trust_radius"] is not None else None)
                        direction, k = self._precondition(x, group, state, trust_budget=budget)
                        self._emit_diagnostics(p, g, x, direction, k, group, state, index)
                        if group["applied_lr"] != 0:
                            p.add_(direction.to(p), alpha=-group["applied_lr"])
                index += 1
        return loss

    def state_dict(self):
        result = super().state_dict()
        # Lists are supported by torch.load(weights_only=True); deques are not.
        result["state"] = {key: {**state, "history": list(state["history"])}
                           for key, state in result["state"].items()}
        result["woodbury"] = dict(step_count=self._step_count, loss=self._loss_mode)
        return result

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        # Optimizer's default loader casts floating state to parameter dtype.
        # Restore directly from the saved tensors to preserve double Grams and
        # >=float32 samples/momentum for low-precision parameters.
        for saved_group, group in zip(state_dict["param_groups"], self.param_groups):
            for key, p in zip(saved_group["params"], group["params"]):
                if key not in state_dict["state"]:
                    continue
                saved, state = state_dict["state"][key], self.state[p]
                state["history"] = deque(s.to(device=p.device, dtype=self._sample_dtype(p)).clone()
                                         for s in saved["history"])
                state["gram"] = saved["gram"].to(**self._work_options(p, group)).clone()
                if "momentum" in saved:
                    state["momentum"] = saved["momentum"].to(device=p.device, dtype=self._sample_dtype(p)).clone()
        metadata = state_dict.get("woodbury", {})
        self._step_count = metadata.get("step_count", 0)
        self._loss_mode = metadata.get("loss", self._loss_mode)
        for group in self.param_groups:
            # Checkpoints predating curvature warmup retain their old behavior.
            group.setdefault("curvature_warmup_steps", 0)
            group.setdefault("trust_radius", None)
            group.setdefault("applied_lr", group["lr"])
