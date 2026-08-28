# Why Gnome's Kronecker Basis Approximates the GGN

*A comparison with K-FAC at fixed weights and under infinite averaging. This is a
statement about population targets and bases; finite-sample estimation is treated
separately in §9.*

Gnome and K-FAC both attach two small symmetric matrices to a weight tensor and use
their eigenvectors as a Kronecker-structured rotation. It is tempting to compare the
full Kronecker products implied by those factors. That comparison is useful, but it is
not the question Gnome's implementation ultimately asks: Gnome discards the factor
eigenvalues and re-estimates the curvature diagonal after rotating.

The operative question is therefore:

> **How much of the GGN remains off-diagonal in the basis selected by the factors?**

This note derives Gnome's factors as partial traces of the GGN, compares their basis
with K-FAC's, gives an exact recovery result for the separable-eigenbasis class, and
states precisely what is and is not known in the approximate and finite-sample cases.

---

## 1. The object Gnome actually approximates

For a layer with weight $W \in \mathbb{R}^{m \times n}$, let
$P = mn$ and let $H \in \mathbb{R}^{P \times P}$ be its GGN block. Gradients
reshaped like the weight are written $G \in \mathbb{R}^{m \times n}$, with
$g = \operatorname{vec}(G)$ using column-major vectorization.

Gnome constructs two orthogonal matrices $Q_L \in O(m)$ and $Q_R \in O(n)$ and
uses the joint basis

$$
Q = Q_R \otimes Q_L \in O(P).
$$

Its curvature EMA in that basis estimates $\operatorname{diag}(Q^\top H Q)$.
The implied curvature model is consequently

$$
D_Q
= Q\,\operatorname{diag}\!\left(Q^\top H Q\right)Q^\top .
\tag{1}
$$

For fixed $Q$, this is the exact Frobenius projection of $H$ onto the matrices
diagonal in that basis. Indeed,

$$
\min_{D\ \mathrm{diagonal}}
\left\|H-QDQ^\top\right\|_F
= \left\|\operatorname{offdiag}(Q^\top H Q)\right\|_F.
$$

Define the relative basis error

$$
\boxed{
\rho(Q)
\;\triangleq\;
\frac{\left\|\operatorname{offdiag}(Q^\top H Q)\right\|_F}
     {\left\|H\right\|_F}
}
\tag{2}
$$

so that $\|H-D_Q\|_F = \rho(Q)\|H\|_F$.

This framing has two important consequences:

1. **Gnome imposes a Kronecker structure on the eigenbasis, not on the
   eigenvalues.** The $mn$ diagonal entries estimated after rotation are free; they
   need not factor as $\lambda_i\mu_j$.
2. **There is no inherited factor-eigenvalue error.** The eigenvalues of the two
   factor matrices determine ordering during the basis refresh, but they are not used
   as the curvature denominator. The approximation that survives into the optimizer
   is the residual off-diagonal mass $\rho(Q)$.

This is the spine of the comparison below. Full Kronecker matrix approximations enter
later as supporting motivation, not as Gnome's final curvature model.

## 2. Gnome's factors are partial traces of the GGN

Gnome's surrogate gradient is constructed so that

$$
\mathbb{E}[g_s g_s^\top] = H.
\tag{3}
$$

Reshape $g_s$ into $G_s \in \mathbb{R}^{m \times n}$. Under infinite averaging,
the two factor EMAs converge to

$$
\hat A = \mathbb{E}[G_s^\top G_s] \in \mathbb{R}^{n \times n},
\qquad
\hat B = \mathbb{E}[G_s G_s^\top] \in \mathbb{R}^{m \times m}.
\tag{4}
$$

These are exactly the two partial traces of $H$. In indexed form,

$$
\hat A_{cc'} = \sum_{r=1}^{m} H_{(r,c),(r,c')},
\qquad
\hat B_{rr'} = \sum_{c=1}^{n} H_{(r,c),(r',c)}.
\tag{5}
$$

They share the trace

$$
\operatorname{tr}(\hat A)
= \operatorname{tr}(\hat B)
= \operatorname{tr}(H).
\tag{6}
$$

Gnome takes $Q_R$ to be an eigenbasis of $\hat A$ and $Q_L$ an eigenbasis of
$\hat B$. Equations (3)--(5) are general: they do not require a rank-one
per-sample gradient or a layer-specific curvature derivation. This matters for weight
sharing and for PINN residuals, where a parameter-gradient matrix can be a sum of
several outer products.

## 3. Comparison with K-FAC on an ordinary dense layer

For a dense layer with input activation $a \in \mathbb{R}^n$ and a target-matched
backpropagated curvature probe $\delta \in \mathbb{R}^m$, the per-sample gradient is

$$
G = \delta a^\top,
\qquad
g = \operatorname{vec}(G) = a \otimes \delta.
$$

The exact layerwise curvature is therefore

$$
\boxed{
H
= \mathbb{E}[g g^\top]
= \mathbb{E}\!\left[(aa^\top)\otimes(\delta\delta^\top)\right].
}
\tag{7}
$$

The expectation in (7) is over data and whatever output-space randomness is needed
to make the covariance equal the Fisher/GGN. This idealized population comparison
should not be confused with the empirical Fisher obtained from real-label loss
gradients; that distinction is revisited in §8.

K-FAC moves the expectation through the Kronecker product:

$$
A_K = \mathbb{E}[aa^\top],
\qquad
B_K = \mathbb{E}[\delta\delta^\top],
\qquad
M_K = A_K \otimes B_K.
\tag{8}
$$

By contrast, substituting $G=\delta a^\top$ into Gnome's partial traces gives

$$
\boxed{
\hat A = \mathbb{E}[\|\delta\|^2 aa^\top],
\qquad
\hat B = \mathbb{E}[\|a\|^2\delta\delta^\top].
}
\tag{9}
$$

Thus:

> **Gnome's factors are K-FAC's second moments, each reweighted by the squared
> norm of the signal on the other side of the layer.**

The reweighting is not an added heuristic. It follows mechanically from taking the
partial traces of the target matrix $H$. A direction of activation space receives more
weight when it occurs on examples carrying more output-side curvature, and vice versa.

## 4. When K-FAC and Gnome agree

Several conditions are useful here, and they should not be conflated.

### 4.1 Second-moment independence: K-FAC is exact

If

$$
\mathbb{E}\!\left[(aa^\top)\otimes(\delta\delta^\top)\right]
= \mathbb{E}[aa^\top]\otimes\mathbb{E}[\delta\delta^\top],
\tag{C1}
$$

then $H=M_K$ exactly. The norm weights in (9) also factor, so

$$
\hat A = \mathbb{E}[\|\delta\|^2]A_K,
\qquad
\hat B = \mathbb{E}[\|a\|^2]B_K.
$$

K-FAC and Gnome therefore select the same basis, and both diagonalize $H$ exactly.

### 4.2 Norm decoupling: the factors are proportional

The methods can agree without (C1). It is enough that

$$
\mathbb{E}[\|\delta\|^2aa^\top]
= \mathbb{E}[\|\delta\|^2]\,\mathbb{E}[aa^\top],
$$

$$
\mathbb{E}[\|a\|^2\delta\delta^\top]
= \mathbb{E}[\|a\|^2]\,\mathbb{E}[\delta\delta^\top].
\tag{C2}
$$

Under (C2), $\hat A \propto A_K$ and $\hat B \propto B_K$, so the bases are
identical even though $H$ need not be a Kronecker product and neither method need be
exact.

### 4.3 Commuting factors: only the bases agree

Gnome requires still less. Two real symmetric matrices share an eigenbasis when they
commute, so

$$
[\hat A,A_K]=0,
\qquad
[\hat B,B_K]=0
\tag{C3}
$$

is enough for the two pairs of factors to **admit** a common eigenbasis. With
nondegenerate spectra this gives $Q_G=Q_K$ up to permutation and sign. Under
degeneracy, independently run eigensolvers can choose incompatible rotations inside a
shared eigenspace, so equality of the implemented bases requires compatible resolution
of those degeneracies. The factor eigenvalues may otherwise differ arbitrarily.

The hierarchy relevant to equality of the two bases is therefore

$$
\text{(C1)} \Longrightarrow \text{(C2)} \Longrightarrow \text{(C3)},
$$

with neither converse holding in general. This explains why showing a difference
between two full Kronecker approximants does not, by itself, show a difference between
the rotations used by the optimizers.

## 5. Exact basis recovery beyond the Kronecker class

Suppose the GGN has an exactly separable eigenbasis,

$$
H
= (Q_R\otimes Q_L)\operatorname{diag}(\Lambda)
  (Q_R\otimes Q_L)^\top,
\tag{10}
$$

where $\Lambda \in \mathbb{R}_+^{n\times m}$ contains **arbitrary** eigenvalues.
They need not factor as $\Lambda_{ik}=r_i c_k$, so $H$ need not itself be a
Kronecker product.

Taking the partial traces of (10) gives

$$
\boxed{
\hat A = Q_R\operatorname{diag}(r)Q_R^\top,
\qquad
\hat B = Q_L\operatorname{diag}(c)Q_L^\top,
}
\tag{11}
$$

with the marginal eigenvalues

$$
r_i = \sum_k \Lambda_{ik},
\qquad
c_k = \sum_i \Lambda_{ik}.
\tag{12}
$$

Hence:

> **Proposition.** If $H$ has a separable eigenbasis and the row and column
> marginals in (12) are nondegenerate, Gnome's partial traces recover that basis
> exactly, up to column permutation and sign.

The nondegeneracy condition is necessary for the partial traces alone to identify the
basis uniquely. Repeated marginals leave the corresponding factor eigenspace
underdetermined; recovery can still occur, but an arbitrary rotation inside that
eigenspace need not diagonalize $H$.

This yields three strictly nested model classes:

$$
\underbrace{H=A_K\otimes B_K}_{\text{K-FAC exact under (C1)}}
\quad\subsetneq\quad
\underbrace{H=A\otimes B}_{\text{some Kronecker product}}
\quad\subsetneq\quad
\underbrace{H\text{ has a separable eigenbasis}}_{\rho^\star=0},
\tag{13}
$$

where

$$
\rho^\star
= \min_{Q_R\in O(n),\,Q_L\in O(m)}
\rho(Q_R\otimes Q_L).
$$

Gnome is basis-consistent on the largest class in (13). This is the
Shampoo/SOAP gap in its cleanest form: a Kronecker-structured basis does not require
Kronecker-factorized eigenvalues.

Concretely, the eigenvectors $q_R^{(i)}\otimes q_L^{(k)}$ un-vectorize into
rank-one matrices $q_L^{(k)}(q_R^{(i)})^\top$. The assumption is that all $mn$
eigenvectors of the layerwise GGN can be built from one shared pool of $m+n$ vectors.
This is still restrictive, but it is strictly weaker than assuming the whole GGN is one
Kronecker product.

## 6. Why use the partial-trace basis when separability is approximate?

For candidate bases $Q_R,Q_L$, write

$$
\alpha = Q_R^\top a,
\qquad
d = Q_L^\top\delta.
$$

The diagonal of the rotated curvature is the nonnegative energy matrix

$$
C_{ik}=\mathbb{E}[\alpha_i^2d_k^2]\ge 0,
\qquad
\sum_{ik}C_{ik}=\operatorname{tr}(H).
\tag{14}
$$

Because the Frobenius norm is rotation-invariant,

$$
\rho(Q_R\otimes Q_L)^2
= 1-\frac{\|C\|_F^2}{\|H\|_F^2}.
\tag{15}
$$

Minimizing basis error is therefore equivalent to concentrating as much curvature
energy as possible into the diagonal coordinate pairs.

The row and column marginals of $C$ are

$$
\sum_k C_{ik} = (Q_R^\top\hat A Q_R)_{ii},
\qquad
\sum_i C_{ik} = (Q_L^\top\hat B Q_L)_{kk}.
\tag{16}
$$

Since $0\le C_{ik}\le r_i$ for $r_i=\sum_kC_{ik}$,

$$
\|C\|_F^2
\le \sum_i r_i^2
= \left\|\operatorname{diag}(Q_R^\top\hat A Q_R)\right\|_2^2,
\tag{17}
$$

and analogously for $\hat B$. By Schur--Horn, the right side is maximized when
$Q_R$ is an eigenbasis of $\hat A$; likewise for $Q_L$ and $\hat B$.

Thus Gnome's basis exactly maximizes the natural marginal upper bounds on the true
diagonal-energy objective. These bounds become tight when the energy matrix is
concentrated, which is precisely the regime the rotation is meant to discover.

This is a principled surrogate objective, not a proof that the two independently chosen
factor eigenbases globally minimize $\rho$. The latter is a coupled non-convex problem,
and the gap should be measured rather than hidden.

### 6.1 Supporting view: one Kronecker power-iteration step

Van Loan--Pitsianis rearrangement converts

$$
\min_{A,B}\|H-A\otimes B\|_F
$$

into a best rank-one approximation of a rearrangement of $H$. The corresponding power
iteration, initialized at the identity and run for one step, yields the partial traces
in (4), up to the common trace normalization. This supplies a second reason that the
partial traces are natural: they are the first iterate toward the best full Kronecker
approximation.

That result should not be overstated. One step is not guaranteed to be close to the
optimum for every architecture, and the best full Kronecker approximation is not the
object Gnome ultimately inverts. The direct basis argument above is the one matching
the code.

## 7. Full Kronecker approximants: useful but secondary

For comparison at a common trace, define

$$
M_G = \frac{\hat A\otimes\hat B}{\operatorname{tr}(H)},
\qquad
M_K = \frac{\operatorname{tr}(H)}
{\operatorname{tr}(A_K)\operatorname{tr}(B_K)}
(A_K\otimes B_K).
\tag{18}
$$

Equation (9) shows that $M_G$ reweights each side of K-FAC by the norm of the
other. Under (C1), $M_G=M_K=H$. Under (C2), $M_G=M_K$ even when neither is exact.
When directional curvature and activation geometry are coupled, the two generally
differ.

These full-matrix errors split orthogonally. If $M$ is diagonal in basis $Q$, then

$$
\|H-M\|_F^2
= \underbrace{\|\operatorname{offdiag}(Q^\top H Q)\|_F^2}_{\text{basis error}}
+ \underbrace{\left\|\operatorname{diag}(Q^\top H Q)
 -\operatorname{diag}(Q^\top M Q)\right\|_2^2}_{\text{eigenvalue error}}.
\tag{19}
$$

Gnome keeps the first term and re-estimates away the second. Consequently,

$$
\|H-D_Q\|_F \le \|H-M\|_F,
$$

and a comparison of $\|H-M_G\|_F$ with $\|H-M_K\|_F$ is only an upper-bound
comparison for the actual diagonal-in-basis optimizers.

This also explains why a Frobenius-optimal Kronecker approximation is not necessarily a
better preconditioner. It can achieve lower matrix error by setting weak directions to
zero, producing a singular inverse. Gnome's freely re-estimated diagonal, damping, and
trust region make the relevant question the quality and conditioning of $D_Q$, not only
the fit of $M_G$.

## 8. Directional coupling is what separates the bases

The norm reweighting in (9) changes the basis only when curvature magnitude is coupled
to **direction** in activation or error space. Scalar coupling is insufficient.

For elliptically symmetric $a$, a multiplier depending only on $\|a\|$ changes the
eigenvalues of $\mathbb{E}[aa^\top]$ but not its eigenvectors: sign symmetry keeps the
off-diagonal entries zero in the same basis. Thus a construction in which
$\|\delta\|^2$ merely grows with $\|a\|$ can make $M_G$ and $M_K$ differ while
leaving $Q_G=Q_K$ and $\rho(Q_G)=\rho(Q_K)$.

A basis separation requires anisotropic coupling, for example
$\|\delta\|^2$ growing with $|\langle a,u\rangle|$ for a particular direction $u$.
This is plausible when hard examples, rare classes, boundaries, or shocks occupy a
particular subspace of a hidden representation. It is an empirical premise, not a
universal property of neural networks.

In a synthetic directional-coupling example with $n=6$, $m=5$, and
$N=2\times10^5$, the population diagnostics are:

| setting | $\rho(I)$ | $\rho(Q_G)$ | $\rho(Q_K)$ |
|---|---:|---:|---:|
| directionally coupled | 0.2492 | **0.0144** | 0.2497 |
| decoupled control | 0.0101 | 0.0085 | 0.0089 |

In the coupled case K-FAC's rotation buys essentially nothing over the identity, while
the partial-trace basis reduces off-diagonal mass by about $17\times$. In the control,
both methods agree to Monte Carlo noise. The example establishes that the separation
can occur; it does not establish its prevalence in real networks.

## 9. Population targets are not finite-sample performance

Everything above fixes the weights and takes infinite averaging. It establishes what
the factors target, not which target is estimated more accurately at a fixed compute
budget.

Gnome's factors are built from randomized GGN probes. K-FAC's activation and error
moments can often be collected directly during a forward/backward pass. Therefore:

- Gnome can have the better population basis and the noisier finite-sample estimate.
- Eigenvector perturbation scales like estimation noise divided by the eigengap, so
  nearly degenerate factor spectra are particularly sensitive.
- EMA length, auxiliary batch size, basis-refresh frequency, and basis staleness can
  reverse the population ordering at finite compute.
- The variance analysis in `variance.md` applies to Gnome's rank-one GGN estimates;
  it should not be read as a completed finite-sample comparison with K-FAC.

There are also two scope limits:

1. The dense-layer formulas (7)--(9) assume $G=\delta a^\top$. Gnome's
   partial-trace construction (3)--(5) remains valid under weight sharing and PINN
   differential operators, but the simple K-FAC comparison then needs the corresponding
   weight-sharing treatment.
2. A target-matched K-FAC Fisher uses output-space sampling or an analytic expectation.
   Practical K-FAC and SOAP variants sometimes use real-label loss gradients instead,
   producing an empirical-Fisher target. That is a separate target mismatch, not a
   Kronecker-factorization error.

## 10. Relationship to SOAP

The partial-trace argument applies to the covariance supplied to the factor updates.
Gnome supplies $H$ because its zero-mean surrogate satisfies (3). SOAP instead forms
outer products from observed minibatch loss gradients. If $g$ is a per-example
real-label gradient, $\mu=\mathbb{E}[g]$, and $\bar g$ averages a batch of size $B$,
then

$$
\mathbb{E}[\bar g\bar g^\top]
= \frac{1}{B}\mathbb{E}[gg^\top]
+ \frac{B-1}{B}\mu\mu^\top.
\tag{20}
$$

The first term is a scaled empirical-Fisher moment and the second is a rank-one
mean-gradient term that dominates as $B$ grows. Neither is generally the GGN. Even if
SOAP's partial traces were estimated without noise, they would therefore be the partial
traces of a different target matrix.

This cleanly separates two possible advantages:

1. **Target advantage:** GGN rather than empirical Fisher, likely the dominant effect on
   MSE and PINN objectives.
2. **Basis advantage over conventional K-FAC:** partial traces retain directional
   coupling through the opposite-side norm weights in (9).

If real-network measurements find $Q_G\approx Q_K$, the second advantage is small; that
does not weaken the first.

## 11. What to measure

On a layer small enough to form $H$ explicitly, report

$$
\rho(I),
\qquad
\rho(Q_K),
\qquad
\rho(Q_G),
\qquad
\rho(Q_{\mathrm{SOAP}}),
\qquad
\rho^\star\ \text{when numerically feasible}.
$$

These answer different questions:

- $\rho(I)-\rho(Q_G)$: what the rotated-basis machinery buys over a diagonal GGN.
- $\rho(Q_K)-\rho(Q_G)$: what the partial-trace weighting buys over K-FAC's basis.
- $\rho(Q_{\mathrm{SOAP}})-\rho(Q_G)$: what changing the covariance target buys over
  the empirical-Fisher basis.
- $\rho(Q_G)-\rho^\star$: the remaining gap to the best separable basis.

Frobenius off-diagonal mass is not the whole optimizer story: it underweights errors in
small-eigenvalue directions. Also report

$$
\kappa\!\left(D_Q^{-1/2}HD_Q^{-1/2}\right)
$$

with the same damping convention used by the optimizer. Finally, compare fresh and stale
bases, such as $\rho(Q_t)$ against $\rho(Q_{t-100})$, to justify the refresh interval
independently of the population construction.

## 12. Claims, in order of strength

| claim | status |
|---|---|
| Gnome's factor targets are the partial traces of the GGN | exact from (3)--(5) |
| K-FAC and Gnome coincide under second-moment independence | exact |
| Gnome recovers a nondegenerate separable GGN eigenbasis | exact |
| The partial-trace basis maximizes marginal energy bounds | exact |
| It globally minimizes off-diagonal mass among separable bases | **not established** |
| One Kronecker power step is near the best full Kronecker fit | empirical, architecture-dependent |
| Gnome's finite-sample basis is better than K-FAC's | empirical question |
| Directional activation--curvature coupling is substantial in PINNs or LLMs | empirical question |

The defensible summary is:

> K-FAC selects a basis from unweighted activation and error moments. Gnome selects
> one from the GGN's partial traces, which retain directional coupling between the two
> sides of a layer. The resulting basis is exact for any nondegenerate separable GGN
> eigenbasis, even when the eigenvalues are not Kronecker-factorizable, and otherwise
> gives a principled rotation whose quality is measured directly by residual
> off-diagonal GGN mass. Whether its additional estimation variance is worth that
> population advantage is a finite-sample empirical question.
