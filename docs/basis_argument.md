# The Basis Argument

*Companion to `kfac_gnome_convergence_targets.md`. That document compares estimands
under Frobenius norm. This one asks the narrower question that Gnome's implementation
actually poses: **which rotation?** Same setup — fixed weights, infinite averaging.*

---

## 1. Gnome's estimand is not $M_{\mathrm G}$

Let $Q = Q_R \otimes Q_L$ with $Q_L \in O(m)$, $Q_R \in O(n)$; the Kronecker product of
orthogonal matrices is orthogonal, so $Q \in O(mn)$. Gnome's $\hat v$ estimates
$\operatorname{diag}(Q^\top H Q)$, so the implied curvature model is

$$
D_Q \;=\; Q \operatorname{diag}\!\left(Q^\top H Q\right) Q^\top .
\tag{1}
$$

For **fixed** $Q$, this is the exact Frobenius projection of $H$ onto matrices diagonal in
that basis: $\lVert H - QDQ^\top\rVert_F = \lVert Q^\top H Q - D\rVert_F$ is minimised over
diagonal $D$ by taking the diagonal. Hence the error is precisely the off-diagonal mass,

$$
\boxed{\;\lVert H - D_Q \rVert_F \;=\; \rho(Q)\,\lVert H \rVert_F,
\qquad
\rho(Q) \;\triangleq\; \frac{\lVert \operatorname{offdiag}(Q^\top H Q) \rVert_F}{\lVert H \rVert_F}\;}
\tag{2}
$$

**There is no eigenvalue approximation error in Gnome.** The eigenvalues are re-estimated
from data in whatever basis is supplied. The entire approximation is $\rho(Q)$, and
$\rho$ is invariant to the eigenvalues of $\hat A, \hat B$ — it is a function of the
*basis alone*.

---

## 2. Three nested assumptions

$$
\underbrace{H = \mathbb{E}[aa^\top] \otimes \mathbb{E}[\delta\delta^\top]}_{\text{C1 — K-FAC exact}}
\;\Longrightarrow\;
\underbrace{\varepsilon^\star = 0}_{M_{\mathrm G} \text{ exact}}
\;\Longrightarrow\;
\underbrace{\rho^\star = 0}_{\text{Gnome exact}}
$$

with $\rho^\star = \min_{Q_L, Q_R} \rho(Q_R \otimes Q_L)$. Both implications are **strict**:

* **C1 $\not\Leftarrow \varepsilon^\star = 0$.** Two equiprobable samples,
  $a^{(1)} = \delta^{(1)} = e_1$, $a^{(2)} = e_2$, $\delta^{(2)} = 3e_1$. Then
  $H = \operatorname{diag}(0.5, 4.5) \otimes \operatorname{diag}(1,0)$ is *exactly*
  Kronecker, $M_{\mathrm G} = H$, and yet
  $\lVert H - M_{\mathrm K}\rVert_F / \lVert H\rVert_F = 0.625$.
  K-FAC's error is **estimator error, not model error** — it survives even when the
  Kronecker model is perfectly correct.
* **$\varepsilon^\star = 0 \not\Leftarrow \rho^\star = 0$.**
  $H = \operatorname{diag}(1,5,5,1)$ is diagonal, hence trivially diagonal in
  $I \otimes I$, so $\rho^\star = 0$. It is not a Kronecker product: $a_1b_1 = 1$,
  $a_1b_2 = a_2b_1 = 5$, $a_2b_2 = 1$ forces $1 = a_1a_2b_1b_2 = 25$.
  This is the Shampoo/SOAP gap — eigenvalues that do not factor as $\lambda_i \mu_k$.

**Separable eigenbasis, stated concretely.** The columns of $Q_R \otimes Q_L$ are
$q_R^{(k)} \otimes q_L^{(i)}$, which un-vec to $q_L^{(i)} (q_R^{(k)})^\top$. So the
assumption is: *every eigenvector of the GGN, viewed as an $m \times n$ matrix, is rank
one, and all $mn$ of them are built from a shared pool of $m + n$ vectors.* The $mn$
eigenvalues remain free. Dimension of the separable subset of $O(mn)$ is
$\binom{m}{2} + \binom{n}{2}$ against $\binom{mn}{2}$.

---

## 3. The Frobenius error splits orthogonally

$H - D_Q$ is purely off-diagonal in $Q$; $D_Q - M$ is purely diagonal. They are
Frobenius-orthogonal, so for any $M$ diagonal in its own eigenbasis $Q$,

$$
\underbrace{\lVert H - M \rVert_F^2}_{\text{estimand error}}
\;=\;
\underbrace{\rho(Q)^2 \lVert H \rVert_F^2}_{\text{basis error}}
\;+\;
\underbrace{\lVert \operatorname{diag}(Q^\top H Q) - \operatorname{diag}(Q^\top M Q) \rVert_F^2}_{\text{eigenvalue error}} .
\tag{3}
$$

Two consequences.

1. **The Frobenius numbers in the companion document are upper bounds** on what the
   optimizers actually incur: $\lVert H - D_{Q}\rVert_F \le \lVert H - M \rVert_F$ for
   both methods, since $D_Q$ is the best matrix diagonal in $Q$ and $M$ is one such.
2. **Only the first term survives into Gnome.** The second is discarded and re-estimated
   by $\hat v$. Reporting the split is therefore more informative than reporting the total.

---

## 4. Basis recovery: the partial traces are marginals of the eigenvalues

Suppose $H = (Q_R \otimes Q_L)\operatorname{diag}(\Lambda)(Q_R \otimes Q_L)^\top$ exactly,
with $\Lambda_{ik}$ **arbitrary** (not factorising). Then

$$
\hat A_{cc'} = \sum_r H_{(r,c),(r,c')}
= \sum_{i,k} \Lambda_{ik} (Q_R)_{ci}(Q_R)_{c'i} \underbrace{\sum_r (Q_L)_{rk}^2}_{=1},
$$

so

$$
\boxed{\;\hat A = Q_R \operatorname{diag}(r) Q_R^\top,
\qquad
\hat B = Q_L \operatorname{diag}(c) Q_L^\top,
\qquad
r_i = \sum_k \Lambda_{ik},\;\; c_k = \sum_i \Lambda_{ik}\;}
\tag{4}
$$

> **Proposition.** $\rho^\star = 0 \Longrightarrow Q_{\mathrm G} = Q_R \otimes Q_L$
> exactly (up to column permutation and sign), provided the marginals $r_i$ and $c_k$ are
> non-degenerate.

Gnome is consistent on the entire separable-eigenbasis class — strictly larger than the
Kronecker class on which $M_{\mathrm G}$ is consistent, which is in turn strictly larger
than the class C1 on which $M_{\mathrm K}$ is consistent. Degeneracy is the only failure
mode: repeated marginals leave $Q_R$ underdetermined by $\hat A$, and nothing then forces
the within-eigenspace rotation to diagonalise $H$.

---

## 5. Why the partial traces, when the assumption only holds approximately

Write $\alpha = Q_R^\top a$, $d = Q_L^\top \delta$. The diagonal of the rotated curvature
is the nonnegative **energy matrix**

$$
C_{ik} = \mathbb{E}\!\left[\alpha_i^2 d_k^2\right] \ge 0,
\qquad
\sum_{ik} C_{ik} = \mathbb{E}\!\left[\lVert a\rVert^2 \lVert\delta\rVert^2\right] = \operatorname{tr}(H),
$$

the total being basis-independent. So minimising $\rho$ is exactly **maximising
$\lVert C \rVert_F^2$** — concentrating curvature energy into as few coordinate pairs as
possible. The marginals of $C$ are

$$
\sum_k C_{ik} = \big(Q_R^\top \hat A\, Q_R\big)_{ii},
\qquad
\sum_i C_{ik} = \big(Q_L^\top \hat B\, Q_L\big)_{kk}.
\tag{5}
$$

Since $0 \le C_{ik} \le r_i$,

$$
\lVert C \rVert_F^2 = \sum_{ik} C_{ik}^2 \;\le\; \sum_{ik} C_{ik} r_i \;=\; \lVert r \rVert_2^2
= \big\lVert \operatorname{diag}(Q_R^\top \hat A Q_R) \big\rVert_2^2 ,
\tag{6}
$$

tight iff each row of $C$ has a single nonzero — i.e. tight in the concentrated regime we
are aiming for. By Schur–Horn (the diagonal of $Q^\top M Q$ is majorised by the
eigenvalues of $M$; $\sum x_i^2$ is Schur-convex), the right side of $(6)$ is maximised
over $O(n)$ at the **eigenbasis of $\hat A$**.

> Gnome's basis is the exact maximiser of the tightest marginal upper bound on the true
> objective. K-FAC's basis maximises concentration of $\mathbb{E}[\alpha_i^2]$, the
> diagonal of $A_{\mathrm K}$, which is not a marginal of $C$ and bounds nothing.

This reaches the same object as the one-step-power-iteration argument by a route that
never invokes Van Loan–Pitsianis, and it is the argument matching what the code does.
Note that the $\operatorname{tr}(H)$ normalisation in eq. (3) of the companion document is
irrelevant here: eigenvectors do not see scale.

---

## 6. Scalar coupling is not enough — the coupling must be directional

The two arguments require different premises, and conflating them is the main trap.

| coupling | breaks | separates in Frobenius | separates the basis |
|---|---|---|---|
| $\lVert\delta\rVert^2$ depends on $\lVert a \rVert$ | C2 | yes | **no** |
| $\lVert\delta\rVert^2$ depends on $\langle a, u\rangle$ | C2 | yes | yes |

For elliptically symmetric $a$, scalar coupling leaves the bases identical: in $\Sigma$'s
eigenbasis, flipping the sign of $a_i$ alone flips $a_i a_j$ while leaving $\lVert a\rVert$
fixed, so $\mathbb{E}[f(\lVert a\rVert) a_i a_j] = 0$ and $\hat A$ stays diagonal there.
The §5 example of the companion document is of this type — every matrix in it is diagonal,
so $\rho = 0$ for **both** methods and the example is silent on the question at hand.

The load-bearing empirical premise is therefore narrower than "coupling exists":
**curvature magnitude must be anisotropic in activation space.** Hard examples, tail
classes, or collocation points near a shock occupying a particular subspace of the hidden
representation. Note also that $Q_{\mathrm G} = Q_{\mathrm K}$ requires only
$[\hat A, A_{\mathrm K}] = 0$ — commuting, strictly weaker than C2's
$\hat A \propto A_{\mathrm K}$ — so the methods agree *more often* at the basis level than
at the estimand level, and the separation is correspondingly harder to demonstrate.

---

## 7. Numbers

**(a) Directional coupling, $n=6$, $m=5$, $N = 2 \times 10^5$.**

| | $\varepsilon^\star/\lVert H\rVert$ | $\frac{\lVert H - M_{\mathrm G}\rVert}{\lVert H\rVert}$ | $\frac{\lVert H - M_{\mathrm K}\rVert}{\lVert H\rVert}$ | $\rho(I)$ | $\rho(Q_{\mathrm G})$ | $\rho(Q_{\mathrm K})$ |
|---|---|---|---|---|---|---|
| coupled | 0.0143 | 0.0148 | 0.3193 | 0.2492 | **0.0144** | **0.2497** |
| decoupled (control) | 0.0087 | 0.0091 | 0.0106 | 0.0101 | 0.0085 | 0.0089 |

* $\rho(Q_{\mathrm K}) = 0.2497 \approx \rho(I) = 0.2492$: **K-FAC's rotation buys nothing
  over not rotating at all.** Against a diagonal-GGN baseline it is a no-op; Gnome's cuts
  off-diagonal mass $17\times$.
* $\rho(Q_{\mathrm G}) = 0.0144 \approx \varepsilon^\star/\lVert H\rVert = 0.0143$: the
  one-step estimator is at the achievable floor, so $\rho^\star \le 0.0144$.
* Per-eigenvector angles $Q_{\mathrm G}$ vs $Q_{\mathrm K}$: $[24°, 73°, 72°, 7°, 13°, 4°]$
  coupled; all $< 0.5°$ decoupled. (Compare columns pairwise — principal angles between
  the full subspaces are identically zero, both being complete bases.)

Applying the split $(3)$: K-FAC's $0.319$ is $62\%$ basis and $38\%$ eigenvalue by squared
mass ($\rho = 0.2497$, eigenvalue $= 0.1989$). Gnome's $0.0148$ is $0.0144$ basis against
$0.0034$ eigenvalue. Only the basis column survives into either optimizer.

**(b) The middle tier is non-empty.** Constructing $H$ with random orthogonal
$Q_L, Q_R$ and deliberately non-factorising $\Lambda$:

| quantity | value |
|---|---|
| $\varepsilon^\star / \lVert H\rVert_F$ | $0.387$ — badly non-Kronecker |
| $\rho(Q_R \otimes Q_L)$, true basis | $5.6 \times 10^{-16}$ |
| $\rho(Q_{\mathrm G})$, from partial traces | $1.7 \times 10^{-15}$ |
| eigenvalues of $\hat A$ | exactly the marginals $\{\sum_k \Lambda_{ik}\}$ |
| $\lvert\cos(q_{\mathrm G}^{(i)}, q_R^{(\pi(i))})\rvert$ | all $1.000$ |

Shampoo is $39\%$ wrong here; Gnome is exact. Recovery is up to a **permutation** $\pi$
— `eigh` orders by marginal, which is unrelated to any construction order. This is the
bookkeeping at method.md line 320. A unit test asserting $\rho$ is unchanged across an
eigenbasis refresh catches the whole class of ordering bugs with one scalar.

---

## 8. Graceful degradation

Because $\hat v$ is re-estimated in whatever basis is supplied, a poor $Q$ costs accuracy
but not correctness: $D_Q$ remains PSD, remains the optimal diagonal model in that basis,
and at $Q = I$ degrades exactly to a diagonal-GGN optimizer. There is no basis for which
the preconditioner becomes singular or the eigenvalues become wrong.

Contrast Shampoo/K-FAC, where a poor Kronecker fit corrupts the eigenvalues too. In §5 of
the companion document the Van Loan–Pitsianis optimum is
$\operatorname{diag}(0,0,0,4.5)$ — Frobenius-optimal, rank-deficient, and uninvertible,
while the one-step $M_{\mathrm G} = \operatorname{diag}(0.05,0.45,0.45,4.05)$ is worse by
$0.9$ against $0.5$ and strictly better to invert. **Frobenius optimality on $H$ is not
optimality for $H^{-1}$-like operators**, and the shrinkage-toward-identity implicit in one
power step is a feature, not a shortfall to apologise for.

---

## 9. What this does not establish

1. **Nothing about finite samples.** $Q_{\mathrm K}$ comes from second moments already
   computed in the forward/backward pass; $Q_{\mathrm G}$ from a $K$-probe Hutchinson
   estimate with genuine Monte Carlo variance. Eigenvector perturbation scales as noise
   divided by the *eigenvalue gap*, so near-degenerate spectra — common in wide layers —
   are where a variance disadvantage bites hardest, independently of any bias analysis.
   The ordering can reverse at fixed budget.
2. **$\rho$ is still Frobenius-flavoured.** It weights by absolute magnitude and
   under-reports damage in the small-eigenvalue directions that govern step size. Report
   $\kappa\big(D_Q^{-1/2} H D_Q^{-1/2}\big)$ alongside it; a basis can lower off-diagonal
   mass while worsening conditioning.
3. **Nothing about which layers.** The assumption "every GGN eigenvector is rank one in
   parameter shape" is most plausible in small dense MLP layers without normalisation
   (the PINN benchmarks) and least plausible where residual connections or normalisation
   make the effective map from $W$ depend on $W$ in ways that do not factor through a
   single activation direction. Attention projections are the natural stress test.

**Measurement ladder** on a layer small enough to form $H$ explicitly
($m, n \lesssim 64$): $\rho(I) \ge \rho(Q_{\mathrm K}) \ge \rho(Q_{\mathrm G}) \ge \rho^\star \ge 0$
across training. The gap $\rho(I) - \rho(Q_{\mathrm G})$ is what the SOAP machinery buys;
$\rho(Q_{\mathrm K}) - \rho(Q_{\mathrm G})$ is what the partial-trace construction buys over
K-FAC. Add basis staleness, $\rho(Q_{t-100})$ vs $\rho(Q_t)$, to justify the refresh
interval independently.

If $\rho(Q_{\mathrm G}) \approx \rho(Q_{\mathrm K})$ in real networks, the honest reading is
that Gnome's advantage comes from the GGN surrogate replacing empirical Fisher rather than
from the partial-trace construction. That would not undercut the method — the EF$\to$GGN
change is plausibly the larger effect on MSE and PINN objectives — but it determines which
section is load-bearing.

---

## 10. Reproduction

```python
import numpy as np
rng = np.random.default_rng(0); n, m, N = 6, 5, 200_000

def diagnostics(coupled):
    rng = np.random.default_rng(0)
    u = rng.standard_normal(n); u /= np.linalg.norm(u)
    a = rng.standard_normal((N, n)) @ np.diag([1.5, 1., .9, .8, .6, .4])
    d = rng.standard_normal((N, m)) @ np.diag([1.3, 1., .9, .7, .5])
    if coupled:
        d *= (1.0 + 3.0 * np.abs(a @ u))[:, None]        # DIRECTIONAL coupling
    g = (a[:, :, None] * d[:, None, :]).reshape(N, -1)   # vec(G), G = d a^T
    H = g.T @ g / N; nH = np.linalg.norm(H, 'fro'); trH = np.trace(H)
    G = g.reshape(N, n, m)

    A_K, S_K = a.T @ a / N, d.T @ d / N
    A_h = np.einsum('nij,nkj->ik', G, G) / N             # E[||delta||^2 a a^T]
    B_h = np.einsum('nji,njk->ik', G, G) / N             # E[||a||^2 delta delta^T]
    M_K = np.kron(A_K, S_K); M_K *= trH / np.trace(M_K)
    M_G = np.kron(A_h, B_h) / trH

    R = H.reshape(n, m, n, m).transpose(0, 2, 1, 3).reshape(n*n, m*m)
    eps = np.sqrt((np.linalg.svd(R, compute_uv=False)[1:] ** 2).sum())
    eb = lambda M: np.linalg.eigh(M)[1][:, ::-1]

    def split(QR, QL, M):
        W = np.kron(QR, QL); Ht, Mt = W.T @ H @ W, W.T @ M @ W
        off = np.linalg.norm(Ht - np.diag(np.diag(Ht)), 'fro')
        dia = np.linalg.norm(np.diag(Ht) - np.diag(Mt))
        return off / nH, dia / nH, np.linalg.norm(H - M, 'fro') / nH

    print(f"{'coupled' if coupled else 'control'}  eps*={eps/nH:.4f}  "
          f"rho(I)={split(np.eye(n), np.eye(m), H)[0]:.4f}")
    for nm_, QR, QL, M in [('KFAC ', eb(A_K), eb(S_K), M_K),
                           ('Gnome', eb(A_h), eb(B_h), M_G)]:
        r, e, t = split(QR, QL, M)
        print(f"   {nm_}  rho={r:.4f}  eigval={e:.4f}  total={t:.4f}"
              f"  (check {np.hypot(r, e):.4f})")

diagnostics(True); diagnostics(False)

# --- middle tier: separable eigenbasis, non-factorising eigenvalues ---
rng = np.random.default_rng(7)
QR = np.linalg.qr(rng.standard_normal((n, n)))[0]
QL = np.linalg.qr(rng.standard_normal((m, m)))[0]
Lam = rng.uniform(0.5, 8.0, size=(n, m))
H = np.kron(QR, QL) @ np.diag(Lam.reshape(-1)) @ np.kron(QR, QL).T
T = H.reshape(n, m, n, m); nH = np.linalg.norm(H, 'fro')
A_h, B_h = np.einsum('irjr->ij', T), np.einsum('crcs->rs', T)
eb = lambda M: np.linalg.eigh(M)[1][:, ::-1]
rho = lambda qr, ql: (lambda Ht: np.linalg.norm(
    Ht - np.diag(np.diag(Ht)), 'fro') / nH)(np.kron(qr, ql).T @ H @ np.kron(qr, ql))
eps = np.sqrt((np.linalg.svd(
    T.transpose(0, 2, 1, 3).reshape(n*n, m*m), compute_uv=False)[1:] ** 2).sum())
print(f"\nmiddle tier  eps*={eps/nH:.4f}  rho(true)={rho(QR, QL):.2e}  "
      f"rho(Q_G)={rho(eb(A_h), eb(B_h)):.2e}")
print("A_h eigvals vs marginals:", np.allclose(
    np.sort(np.linalg.eigh(A_h)[0])[::-1], np.sort(Lam.sum(1))[::-1]))
```
