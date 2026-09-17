"""Energy Natural Gradient Descent with the Woodbury identity.

Guzmán-Cordero et al. (2025), Section 3, equation (5):
https://arxiv.org/html/2505.12149v2#S3

Extracted from the Poisson-5D benchmark so PDEs share the same solver.
Residual blocks must be divided by sqrt(their sample count), giving
energy 0.5 * sum(block MSEs). step() reports twice this energy to match
experiment training losses. Damping is fixed; there is no weight decay,
momentum, sketching, or learning-rate schedule inside this baseline.
"""

from __future__ import annotations

import torch
from torch.func import jacrev


class ENGDW:
    """Energy Natural Gradient Descent via the Woodbury / push-through identity.

        δ = (JᵀJ + λI)⁻¹ Jᵀr  ==  Jᵀ(JJᵀ + λI)⁻¹ r,     θ ← θ − η·δ

    The right form solves an ``N×N`` system instead of ``D×D`` — that identity
    *is* ENGD-W. Cost ``O(N²D)`` to form the Gram, ``O(N³)`` to factor; the
    binding constraint is batch ``N``, not model size ``D``. Use float64 on
    CPU or CUDA for small damping values and ill-conditioned PINN Jacobians.

    ``residual_fn(params, *args) → (N, 1)`` with energy ``L = 0.5‖r‖²``.
    """

    def __init__(self, model, residual_fn, lr=5.2289e-2, damping=6.804474e-8,
                 line_search=False, ls_grid=None, chunk_size=None, jacobian_fn=None):
        self.model = model
        self.residual_fn = residual_fn
        self.lr = lr
        self.lam = damping
        self.line_search = line_search
        self.ls_grid = ls_grid or torch.logspace(-3, 0, 13).tolist()
        self.chunk_size = chunk_size
        # Optional per-sample implementation: same dict (N, *parameter.shape)
        # as jacrev(residual_fn), avoiding cross-sample differentiation work.
        self.jacobian_fn = jacobian_fn
        self.params = dict(model.named_parameters())
        self.shapes = [(k, v.shape, v.numel()) for k, v in self.params.items()]
        # Duck-type shim so common.current_lr(opt) works; updated per step to the
        # actual η applied (meaningful when line search is on).
        self.param_groups = [{"lr": lr}]

    def _unflat(self, flat):
        out, i = {}, 0
        for k, shape, n in self.shapes:
            out[k] = flat[i:i + n].view(shape)
            i += n
        return out

    def _loss_at(self, eta, delta, args):
        d = self._unflat(delta)
        cand = {k: (v - eta * d[k]).detach() for k, v in self.params.items()}
        r = self.residual_fn(cand, *args).squeeze(-1)   # x-derivs still needed
        return (r ** 2).sum().item()

    def step(self, *args):
        def r_vec(p):
            return self.residual_fn(p, *args).squeeze(-1)            # (N,)

        params = {k: v.detach() for k, v in self.params.items()}
        r = r_vec(params).detach()
        N = r.shape[0]

        if self.jacobian_fn is None:
            Jd = jacrev(r_vec, chunk_size=self.chunk_size)(params)
        else:
            Jd = self.jacobian_fn(params, *args)
        J = torch.cat([Jd[k].reshape(N, -1) for k in self.params], dim=1)
        del Jd

        G = J @ J.T
        G = 0.5 * (G + G.T)                    # symmetrize before factoring
        G.diagonal().add_(self.lam)

        try:                                    # G is SPD; Cholesky is the right call
            L = torch.linalg.cholesky(G)
            z = torch.cholesky_solve(r.unsqueeze(1), L).squeeze(1)
        except torch.linalg.LinAlgError:        # damping too small for this dtype
            z = torch.linalg.lstsq(G, r.unsqueeze(1)).solution.squeeze(1)

        delta = J.T @ z                                              # (D,)

        if self.line_search:
            eta = min((self._loss_at(e, delta, args), e) for e in self.ls_grid)[1]
        else:
            eta = self.lr
        self.param_groups[0]["lr"] = eta

        upd = self._unflat(eta * delta)
        with torch.no_grad():
            for k, v in self.params.items():
                v -= upd[k]

        return float((r ** 2).sum())          # PINN loss = sum of block MSEs
