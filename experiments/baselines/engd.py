"""Energy Natural Gradient Descent (ENGD), Müller & Zeinhofer (ICML 2023).

ENGD is the *natural gradient* for a PINN under the metric induced by the PDE
operator itself — the "energy" (Sobolev) inner product ``⟨v,w⟩_E = ∫ (Dv)(Dw)``
rather than the plain L² metric on the network output. Concretely: form the
Gram matrix of the parameter tangents of the **residual** (operator included),

    G(θ) = J_rᵀ J_r,   J_r = ∂r/∂θ,   r = the stacked PINN residual,

and take the Gauss-Newton / least-squares step ``Δ = -(G + λI)⁻¹ J_rᵀ r``. For a
linear PDE this is one Newton step on a loss that is genuinely quadratic in the
energy norm, which is why ENGD reaches ~1e-7 relative error where first-order
methods floor around 1e-3 — it exactly cancels the operator-induced
ill-conditioning (the derivative operator's spectrum spans orders of magnitude)
that cripples Adam. It is the exact, dense counterpart of what Gnome
approximates with a Kronecker-factored, EMA'd Hutchinson surrogate.

This is a *reference* optimizer: it forms and solves a dense ``p×p`` system each
step (``O(Mp² + p³)``), so it only scales to small networks (a few thousand
parameters). Run it in float64 — the whole point is accuracy the float32 floor
can't express.

Design notes:

* The step is computed as an **augmented least-squares** solve of
  ``min_Δ ‖J_r Δ + r‖² + λ‖Δ‖²`` rather than by forming and inverting
  ``J_rᵀJ_r``. Forming the normal equations squares the condition number of
  ``J_r`` — fatal in the regime where ENGD earns its keep. The augmented system
  keeps the conditioning at ``cond(J_r)``.
* A backtracking line search along ``Δ`` handles the nonlinearity of the
  parameterization (the one-step GN solve overshoots on nonlinear PDEs).
* ``λ`` (Levenberg-Marquardt damping) adapts: it shrinks when a full step is
  accepted and grows when the line search has to backtrack hard or fails,
  giving a lightweight trust region.

This class is PDE-agnostic. The caller supplies:

* ``residual_jacobian_fn() -> (r, J)`` — the stacked residual ``r`` (shape
  ``(M,)``) and its parameter-Jacobian ``J`` (shape ``(M, p)``, columns in
  ``torch.nn.utils.parameters_to_vector`` order) at the current parameters.
* ``loss_fn() -> float`` — the scalar training loss at the current parameters,
  recomputed on the same batch (used for the line search and logging).
"""

from __future__ import annotations

import math

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class ENGD:
    def __init__(
        self,
        params,
        lr: float = 1.0,
        damping: float = 1e-8,
        min_damping: float = 1e-12,
        max_damping: float = 1e2,
        max_line_search: int = 20,
        ls_decay: float = 0.5,
    ):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.damping = damping
        self.min_damping = min_damping
        self.max_damping = max_damping
        self.max_line_search = max_line_search
        self.ls_decay = ls_decay
        # Diagnostics for logging.
        self.last_step_size = 0.0
        self.last_damping = damping

    @torch.no_grad()
    def step(self, residual_jacobian_fn, loss_fn) -> float:
        """One ENGD step. Returns the loss at the accepted parameters."""
        r, J = residual_jacobian_fn()          # r: (M,), J: (M, p)
        r = r.reshape(-1)
        M, p = J.shape
        dtype, device = J.dtype, J.device

        base = float((r @ r).item()) / M       # mean-square loss (matches loss_fn)

        theta0 = parameters_to_vector(self.params).detach().clone()

        # Solve min_Δ ‖J Δ + r‖² + λ‖Δ‖² via the augmented least-squares system
        #   [ J          ] Δ ≈ [ -r ]
        #   [ sqrt(λ)·I  ]     [  0  ]
        # (Tikhonov by augmentation — avoids squaring cond(J) that forming JᵀJ
        # would incur). Δ is the descent step; θ ← θ + step·Δ.
        sqrt_lam = math.sqrt(self.damping)
        eye = torch.eye(p, dtype=dtype, device=device)
        A = torch.cat([J, sqrt_lam * eye], dim=0)          # (M+p, p)
        b = torch.cat([-r, torch.zeros(p, dtype=dtype, device=device)], dim=0)
        delta = torch.linalg.lstsq(A, b.unsqueeze(1)).solution.reshape(-1)

        # Backtracking line search along Δ.
        step = self.lr
        accepted = False
        new = base
        for _ in range(self.max_line_search):
            vector_to_parameters(theta0 + step * delta, self.params)
            new = float(loss_fn())
            if math.isfinite(new) and new < base:
                accepted = True
                break
            step *= self.ls_decay

        if not accepted:
            vector_to_parameters(theta0, self.params)   # revert; keep θ0
            new = base

        # Levenberg-style adaptive damping: a full-size accepted step means the
        # quadratic model is trustworthy (loosen); a failed/heavily-backtracked
        # step means it isn't (tighten).
        if accepted and step == self.lr:
            self.damping = max(self.min_damping, self.damping * 0.5)
        elif not accepted:
            self.damping = min(self.max_damping, self.damping * 2.0)

        self.last_step_size = step if accepted else 0.0
        self.last_damping = self.damping
        return new
