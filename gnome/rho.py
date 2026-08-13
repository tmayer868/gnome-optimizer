"""Exact rho(Q) — how much curvature a separable basis fails to diagonalize.

For an MSE/PINN objective ``L = (1/N) sum_k r_k^2`` the Gauss-Newton matrix of
one weight tensor ``W`` (m x n) is exact, not an approximation::

    H = (2/N) sum_k vec(g_k) vec(g_k)^T ,     g_k = d r_k / d W

and for a separable rotation ``Q = Q_L (x) Q_R``::

    rho(Q) = ||offdiag(Q^T H Q)||_F / ||H||_F

i.e. the fraction of curvature mass the basis leaves off-diagonal. Gnome's
Newton step is diagonal in its own basis, so rho(Q_G) *is* the modelling error
that step accepts.

This module knows nothing about any model, PDE or experiment: give
:func:`measure_rho` a residual vector and some parameters and it measures.
Call it from a training loop to get rho at real points on the trajectory,
without checkpoints::

    from gnome import measure_rho

    if step % 5000 == 0:
        records = measure_rho(residual_vector, params, opt=opt)

Relation to :mod:`gnome.diagnostics`
------------------------------------
``diagnostics`` is continuous and cheap: it reads state the optimizer already
maintains, so it runs every N steps for free, but its ``rho_proxy`` is a
Kronecker-model estimate that clamps to a useless 0.0 whenever the model is
cold. This is occasional and expensive — one backward per residual entry — and
exact. Use both: ``rho_proxy`` for the shape of the curve, ``measure_rho`` for
the number.

Cost and scaling
----------------
``rho`` never forms the (P x P) matrix. ``||H||_F`` is rotation-invariant and
comes from an N x N Gram of the per-sample gradients; the rotated diagonal is a
per-coordinate sum. Cost is O(N^2 + N*P), not O(P^2), so 256x256 layers
(P = 65536, where H alone would be 4.3e9 entries) are fine. The binding cost is
the N backward passes, hence ``max_samples``.

``kron_floor=True`` is the exception: the Van Loan-Pitsianis bound needs the
rearrangement of H, so it does build (P x P) and is gated by ``max_kron_dim``.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch


# ----------------------------------------------------------------------
# Core: rho without ever forming H
# ----------------------------------------------------------------------

def gram_hnorm_sq(G: torch.Tensor) -> torch.Tensor:
    """``||H||_F^2`` from per-sample gradients, via the N x N Gram.

    ``H = c sum_k h_k h_k^T`` with ``h_k = vec(g_k)`` and ``c = 2/N``, so
    ``||H||_F^2 = tr(H^2) = c^2 sum_{k,l} (h_k . h_l)^2``. Never O(P^2) in
    memory, and rotation-invariant so it is computed once for every basis.
    """
    N = G.shape[0]
    U = G.reshape(N, -1)
    M = U @ U.T                       # (N, N) pairwise <g_k, g_l>
    return (2.0 / N) ** 2 * (M * M).sum()


def rho(G: torch.Tensor, Q_L: torch.Tensor, Q_R: torch.Tensor,
        hnorm_sq: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``rho(Q_L (x) Q_R)`` — exact, no (P x P) matrix.

    The rotated curvature diagonal is ``diag(H~)_p = c sum_k g~_k[p]^2`` with
    ``g~_k = Q_L^T g_k Q_R``, and since ``||H~||_F = ||H||_F``::

        rho^2 = 1 - sum_p diag(H~)_p^2 / ||H||_F^2

    The rotation matches ``Gnome._project``, which computes ``Q_0^T g Q_1``.
    For row-major ``vec`` the corresponding full basis is ``kron(Q_L, Q_R)``;
    the column-major ``kron(Q_R, Q_L)`` is a different, wrong number that is
    equally finite and plausible-looking. :func:`self_test` pins it down.
    """
    N = G.shape[0]
    if hnorm_sq is None:
        hnorm_sq = gram_hnorm_sq(G)
    Gt = torch.einsum("mi,nmk,kj->nij", Q_L, G, Q_R)   # g~_k = Q_L^T g_k Q_R
    diag = (2.0 / N) * Gt.square().sum(dim=0)
    return (1.0 - diag.square().sum() / hnorm_sq).clamp_min(0.0).sqrt()


def gnome_factors(G: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Gnome's Kronecker factors from per-sample gradients.

    Matches ``Gnome._update_preconditioner``, which contracts the surrogate
    gradient over every mode *except* one::

        GG[0][i,k] = sum_j g[i,j] g[k,j]   (m x m, output side -> Q_L)
        GG[1][j,l] = sum_i g[i,j] g[i,l]   (n x n, input side  -> Q_R)

    The optimizer EMAs these across steps; here it is the plain batch average,
    giving the *ideal* basis Gnome's estimator is converging to. The gap to the
    live basis is estimator lag rather than model error.
    """
    return (torch.einsum("nij,nkj->ik", G, G),
            torch.einsum("nij,nil->jl", G, G))


def eigvecs_descending(M: torch.Tensor) -> torch.Tensor:
    """Eigenvectors of a symmetric matrix, sorted by descending eigenvalue."""
    _, Q = torch.linalg.eigh(M)
    return Q.flip(1)


def kron_floor(G: torch.Tensor, m: int, n: int) -> torch.Tensor:
    """Van Loan-Pitsianis: relative error of the *best* Kronecker product.

    Rearranging ``R[(i,k),(j,l)] = H[(i,j),(k,l)]`` turns Kronecker
    approximation into rank-1 approximation, so the best achievable
    ``||H - A (x) B||_F / ||H||_F`` is ``sqrt(sum_{t>0} sigma_t^2) / ||H||_F``.

    **Not a lower bound on rho, and the distinction is the point.** ``rho``
    minimises over separable *bases*; this minimises over Kronecker
    *products*. ``Q diag(v) Q^T`` with a non-factorising ``v`` is diagonal in a
    separable basis without being any ``A (x) B``, so the Kronecker set is
    strictly smaller and its optimum strictly larger. Hence::

        rho(Q_G) < kron_floor  <=>  a rotated + re-estimated diagonal beats
                                    *every* Kronecker-product model of H

    The method this bounds is **KFAC**, which approximates the GGN/Fisher
    directly as ``A (x) S`` and so cannot do better than ``kron_floor``.

    It does *not* bound Shampoo or SOAP, for two separate reasons. Shampoo's
    factors come from loss-gradient outer products — a gradient second-moment
    object, not the GGN — so it is not approximating H at all, and a floor on
    approximations to H says nothing about it. SOAP escapes for the same reason
    Gnome does: it keeps only the *eigenvectors* and re-estimates the diagonal
    in that basis, so it is bounded by ``rho(its basis)`` rather than by any
    product model. Gnome and SOAP differ in where the basis comes from (GGN
    surrogate vs loss gradients), which is a ``rho`` question, not this one.

    This is the only function here that forms H (P x P).
    """
    N = G.shape[0]
    U = G.reshape(N, -1)
    H = (2.0 / N) * (U.T @ U)
    R = H.reshape(m, n, m, n).permute(0, 2, 1, 3).reshape(m * m, n * n)
    sigma = torch.linalg.svdvals(R)
    return (sigma[1:] ** 2).sum().sqrt() / H.norm("fro")


# ----------------------------------------------------------------------
# Per-sample gradients
# ----------------------------------------------------------------------

def per_sample_grads(residual: torch.Tensor, params) -> list:
    """``d r_k / d p`` for every residual entry k and every parameter p.

    One batched vector-Jacobian product where the graph supports it, falling
    back to a per-sample loop. PINN residuals are built with
    ``create_graph=True`` for the PDE derivative terms and vmap does not cover
    every double-backward kernel, so the fallback is load-bearing.
    """
    r = residual.reshape(-1)
    N = r.numel()
    eye = torch.eye(N, device=r.device, dtype=r.dtype)
    try:
        return list(torch.autograd.grad(
            r, params, grad_outputs=eye, is_grads_batched=True,
            retain_graph=True, allow_unused=True,
        ))
    except Exception:
        out = [torch.zeros((N,) + p.shape, device=p.device, dtype=p.dtype)
               for p in params]
        for k in range(N):
            gs = torch.autograd.grad(r[k], params, retain_graph=True,
                                     allow_unused=True)
            for j, g in enumerate(gs):
                if g is not None:
                    out[j][k] = g
        return out


# ----------------------------------------------------------------------
# The entry point
# ----------------------------------------------------------------------

def measure_rho(
    residual: torch.Tensor,
    params: Iterable[torch.Tensor],
    *,
    opt=None,
    max_samples: int = 256,
    with_kron_floor: bool = False,
    max_kron_dim: int = 4096,
    float64: bool = True,
) -> list[dict]:
    """Measure rho for every 2D parameter, at the current point.

    Args:
        residual: 1-D residual vector with a live autograd graph — the same
            ``r`` whose mean square is the loss. Its length is the sample count
            N before subsampling.
        params: Parameters to measure. 1-D entries are skipped: a separable
            basis question needs two modes.
        opt: Optional :class:`~gnome.optimizer.Gnome`. When given, its *live*
            eigenbasis ``state[p]["Q"]`` is measured too, so the gap to the
            ideal basis reports estimator lag separately from model error.
        max_samples: Cap on residual entries used. Cost is one backward pass
            per entry, so this is the knob that decides whether a measurement
            costs a second or a minute. Entries are chosen uniformly at random.
        with_kron_floor: Also compute the best-Kronecker-product error. Builds
            a (P x P) matrix, so it is skipped for layers above
            ``max_kron_dim``.
        float64: Promote the gradients before reducing. rho is a ratio of
            near-cancelling sums and float32 costs real digits; this does not
            require the model itself to be float64.

    Returns:
        One dict per measured parameter, with ``param`` (index into ``params``),
        ``shape``, ``n_samples``, ``rho_I``, ``rho_gnome``, and — when
        available — ``rho_live`` and ``kron_floor``. ``float('nan')`` marks
        a value that was not computed.
    """
    params = [p for p in params]
    r = residual.reshape(-1)
    N = r.numel()
    if max_samples and N > max_samples:
        idx = torch.randperm(N, device=r.device)[:max_samples]
        r = r[idx]

    grads = per_sample_grads(r, params)

    out: list[dict] = []
    for i, (p, G) in enumerate(zip(params, grads)):
        if G is None or p.dim() != 2:
            continue
        G = G.detach()
        if float64:
            # MPS has no float64 at all, so accuracy means a CPU round-trip.
            # These are occasional measurements, and rho is a ratio of
            # near-cancelling sums where float32 costs real digits.
            if G.device.type == "mps":
                G = G.cpu()
            G = G.to(torch.float64)
        m, n = p.shape
        hn = gram_hnorm_sq(G)
        if not torch.isfinite(hn) or hn <= 0:
            continue

        eye_m = torch.eye(m, device=G.device, dtype=G.dtype)
        eye_n = torch.eye(n, device=G.device, dtype=G.dtype)
        A0, A1 = gnome_factors(G)
        QL, QR = eigvecs_descending(A0), eigvecs_descending(A1)

        rec = {
            "param": i,
            "shape": (m, n),
            "n_samples": int(G.shape[0]),
            "rho_I": rho(G, eye_m, eye_n, hn).item(),
            "rho_gnome": rho(G, QL, QR, hn).item(),
            "rho_live": float("nan"),
            "kron_floor": float("nan"),
            # N/P. H_N is a Monte-Carlo estimate of an integral, with hard
            # rank <= N, so this says how well-determined it is. Near or below
            # 1 the sample spectrum is badly spread (Marchenko-Pastur) even if
            # the true curvature is well conditioned — rho is a Frobenius
            # ratio and degrades gracefully, but read low values with care.
            "rank_ratio": G.shape[0] / float(m * n),
        }

        if opt is not None:
            Q = opt.state.get(p, {}).get("Q")
            if (Q is not None and len(Q) == 2
                    and all(torch.is_tensor(q) for q in Q)
                    and Q[0].shape == (m, m) and Q[1].shape == (n, n)):
                # Move device first, then cast: a single .to(device=, dtype=)
                # attempts the cast on the source device, and MPS refuses
                # float64 outright.
                rec["rho_live"] = rho(
                    G,
                    Q[0].to(G.device).to(G.dtype),
                    Q[1].to(G.device).to(G.dtype),
                    hn,
                ).item()

        if with_kron_floor and m * n <= max_kron_dim:
            rec["kron_floor"] = kron_floor(G, m, n).item()

        out.append(rec)
    return out


def format_records(records: list[dict], prefix: str = "  ") -> str:
    """Render :func:`measure_rho` output as a table."""
    hdr = (f"{'param':<7}{'shape':<12}{'N':>6}{'rho(I)':>9}"
           f"{'rho(Q_G)':>10}{'rho(live)':>11}{'kron_fl':>9}{'N/P':>8}")
    f = lambda v: "   -  " if v is None or v != v else f"{v:.4f}"
    lines = [prefix + hdr, prefix + "-" * len(hdr)]
    for r in records:
        m, n = r["shape"]
        row = (f"p{r['param']:<6}{f'{m}x{n}':<12}{r['n_samples']:>6}"
               f"{r['rho_I']:>9.4f}{r['rho_gnome']:>10.4f}"
               f"{f(r['rho_live']):>11}{f(r['kron_floor']):>9}"
               f"{r['rank_ratio']:>8.2f}")
        lines.append(prefix + row)
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------

def self_test(device=None) -> int:
    """Validate the machinery on cases with known answers. Returns #failures."""
    device = device or torch.device("cpu")
    torch.manual_seed(0)
    fails = 0
    f64 = dict(dtype=torch.float64, device=device)

    def check(name, got, want, tol):
        nonlocal fails
        ok = abs(got - want) <= tol
        fails += (not ok)
        print(f"   [{'PASS' if ok else 'FAIL'}] {name}: got {got:.6f}, "
              f"want {want:.6f} (tol {tol})")

    m, n, N = 6, 5, 400
    L = torch.randn(m, m, **f64)
    R = torch.randn(n, n, **f64)
    Z = torch.randn(N, m, n, **f64)
    G = torch.einsum("ab,nbc,dc->nad", L, Z, R)
    A0, A1 = gnome_factors(G)
    r_g = rho(G, eigvecs_descending(A0), eigvecs_descending(A1)).item()
    r_i = rho(G, torch.eye(m, **f64), torch.eye(n, **f64)).item()
    print(f"   structured curvature: rho(I)={r_i:.4f} rho(Q_G)={r_g:.4f}")
    check("rho(Q_G) <= rho(I)", float(r_g <= r_i + 1e-9), 1.0, 0.0)

    S_L = torch.linalg.qr(torch.randn(m, m, **f64))[0]
    S_R = torch.linalg.qr(torch.randn(n, n, **f64))[0]
    G2 = torch.einsum("mi,nmk,kj->nij", S_L, G, S_R)
    A0b, A1b = gnome_factors(G2)
    check("rho invariant under a consistent rotation",
          rho(G2, eigvecs_descending(A0b), eigvecs_descending(A1b)).item(),
          r_g, 1e-8)

    # Cross-check against the explicit (P x P) definition: this is what pins
    # down the kron operand order.
    N2, m2, n2 = 60, 4, 3
    Gs = torch.randn(N2, m2, n2, **f64)
    A0c, A1c = gnome_factors(Gs)
    QL, QR = eigvecs_descending(A0c), eigvecs_descending(A1c)
    U = Gs.reshape(N2, -1)
    H = (2.0 / N2) * (U.T @ U)
    for label, Q in (("kron(Q_L, Q_R)", torch.kron(QL, QR)),
                     ("kron(Q_R, Q_L)", torch.kron(QR, QL))):
        Ht = Q.T @ H @ Q
        off = Ht - torch.diag(torch.diag(Ht))
        explicit = (off.norm("fro") / H.norm("fro")).item()
        print(f"   explicit via {label}: {explicit:.6f}")
        if label == "kron(Q_L, Q_R)":
            check("fast rho == explicit rho", rho(Gs, QL, QR).item(),
                  explicit, 1e-9)

    Rr = H.reshape(m2, n2, m2, n2).permute(0, 2, 1, 3).reshape(m2 * m2, n2 * n2)
    check("rearrangement preserves ||.||_F", Rr.norm("fro").item(),
          H.norm("fro").item(), 1e-10)
    Us, S, Vh = torch.linalg.svd(Rr)
    A = (Us[:, 0] * S[0].sqrt()).reshape(m2, m2)
    B = (Vh[0, :] * S[0].sqrt()).reshape(n2, n2)
    check("kron_floor == explicit best-Kronecker error",
          kron_floor(Gs, m2, n2).item(),
          ((H - torch.kron(A, B)).norm("fro") / H.norm("fro")).item(), 1e-8)

    # End-to-end through measure_rho on a real autograd graph.
    W = torch.nn.Linear(5, 4, bias=False).double()
    x = torch.randn(40, 5, **f64)
    res = (W(x) ** 2).sum(dim=1)
    recs = measure_rho(res, list(W.parameters()), max_samples=40)
    ok = len(recs) == 1 and recs[0]["shape"] == (4, 5) and recs[0]["n_samples"] == 40
    fails += (not ok)
    print(f"   [{'PASS' if ok else 'FAIL'}] measure_rho end-to-end: {recs}")

    print("\n   self-test: " + ("ALL PASS" if not fails else f"{fails} FAILED"))
    return fails


if __name__ == "__main__":
    raise SystemExit(1 if self_test() else 0)
