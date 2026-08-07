# What KFAC and Gnome Converge To

*Fixed weights, infinite averaging. A statement about estimands, not estimators —
finite-sample behaviour is a separate question and is addressed in §6.*

---

## 1. Setup

A linear layer with weight $W \in \mathbb{R}^{m \times n}$ maps input activations
$a \in \mathbb{R}^{n}$ to pre-activations $s = Wa \in \mathbb{R}^{m}$. Write
$\delta = \partial \ell / \partial s \in \mathbb{R}^{m}$ for the backpropagated
pre-activation gradient. The per-sample gradient with respect to $W$ is then

$$
G \;=\; \delta a^{\top} \;\in\; \mathbb{R}^{m \times n},
\qquad
g \;=\; \operatorname{vec}(G) \;=\; a \otimes \delta \;\in\; \mathbb{R}^{P},
\quad P = mn,
$$

using column-major $\operatorname{vec}$ and the identity
$\operatorname{vec}(uv^{\top}) = v \otimes u$.

**The layerwise curvature is exactly a Kronecker-structured expectation.** Applying
$(u \otimes v)(x \otimes y)^{\top} = (ux^{\top}) \otimes (vy^{\top})$,

$$
\boxed{\;H \;=\; \mathbb{E}\!\left[g g^{\top}\right] \;=\; \mathbb{E}\!\left[(a a^{\top}) \otimes (\delta \delta^{\top})\right]\;}
\tag{1}
$$

with no approximation. For exponential-family losses with the model emitting natural
parameters — softmax cross-entropy on logits, Gaussian/MSE on the mean — the Fisher
equals the GGN, so $(1)$ is a statement about the curvature we actually want.

Note that $(1)$ is an *expectation of Kronecker products*, not a Kronecker product of
expectations. Everything below is about how the two methods close that gap.

---

## 2. KFAC's estimand

K-FAC pushes the expectation through the tensor product:

$$
H \;\approx\; M_{\mathrm{K}} \;=\; A_{\mathrm{K}} \otimes S_{\mathrm{K}},
\qquad
A_{\mathrm{K}} = \mathbb{E}[a a^{\top}],
\quad
S_{\mathrm{K}} = \mathbb{E}[\delta \delta^{\top}].
\tag{2}
$$

Cheap: both factors are ordinary second moments of quantities already computed in the
forward and backward pass.

---

## 3. Gnome's estimand

Gnome's surrogate is built so that $\mathbb{E}[g_s g_s^{\top}] = H$, and its Kronecker
factors are EMAs of $G_s G_s^{\top}$ and $G_s^{\top} G_s$. In the infinite-averaging
limit these converge to the **partial traces** of $H$. Substituting $G = \delta a^{\top}$:

$$
\mathbb{E}\!\left[G^{\top} G\right] \;=\; \mathbb{E}\!\left[a \delta^{\top} \delta a^{\top}\right] \;=\; \mathbb{E}\!\left[\lVert \delta \rVert^{2} \, a a^{\top}\right] \;=:\; \hat A \;\in\; \mathbb{R}^{n \times n},
$$
$$
\mathbb{E}\!\left[G G^{\top}\right] \;=\; \mathbb{E}\!\left[\delta a^{\top} a \delta^{\top}\right] \;=\; \mathbb{E}\!\left[\lVert a \rVert^{2} \, \delta \delta^{\top}\right] \;=:\; \hat B \;\in\; \mathbb{R}^{m \times m}.
$$

Since $\operatorname{tr}(\hat A) = \operatorname{tr}(\hat B) = \mathbb{E}[\lVert a \rVert^{2}\lVert \delta \rVert^{2}] = \operatorname{tr}(H)$, the product
$\hat A \otimes \hat B$ has trace $\operatorname{tr}(H)^{2}$ and requires one division:

$$
\boxed{\;M_{\mathrm{G}} \;=\; \frac{\mathbb{E}\!\left[\lVert \delta \rVert^{2} a a^{\top}\right] \;\otimes\; \mathbb{E}\!\left[\lVert a \rVert^{2} \delta \delta^{\top}\right]}{\mathbb{E}\!\left[\lVert a \rVert^{2}\lVert \delta \rVert^{2}\right]}\;}
\tag{3}
$$

**Gnome's factors are KFAC's factors, each reweighted by the squared norm of the other
side.** This is not a design choice — it falls out of the partial trace.

By Van Loan–Pitsianis, the optimal Kronecker approximation of $H$ is a rank-one
approximation of a rearrangement $\mathcal{R}(H)$, obtainable by power iteration; by
Morwani et al., initialising at the identity and taking **one step** yields exactly the
partial traces $(3)$. So $M_{\mathrm{G}}$ is a principled rank-one approximant of $H$.
$M_{\mathrm{K}}$ carries no such property.

---

## 4. When the two coincide

Two conditions, of different strength.

**(C1) Second-moment independence.** If $a \perp \delta$ in the sense that
$\mathbb{E}[(aa^{\top}) \otimes (\delta\delta^{\top})] = \mathbb{E}[aa^{\top}] \otimes \mathbb{E}[\delta\delta^{\top}]$,
then $H = A_{\mathrm{K}} \otimes S_{\mathrm{K}}$ **exactly**: $H$ is genuinely Kronecker,
$\varepsilon^{\star} = 0$, and $M_{\mathrm{K}} = H$.

Under C1 the norm weights in $(3)$ also factor,
$\mathbb{E}[\lVert\delta\rVert^{2} a a^{\top}] = \mathbb{E}[\lVert\delta\rVert^{2}]\,A_{\mathrm{K}}$
and likewise for $\hat B$, while
$\operatorname{tr}(H) = \mathbb{E}[\lVert a\rVert^{2}]\,\mathbb{E}[\lVert\delta\rVert^{2}]$.
The scalars cancel:

$$
M_{\mathrm{G}} \;=\; \frac{\mathbb{E}[\lVert\delta\rVert^{2}]\,\mathbb{E}[\lVert a\rVert^{2}]}{\mathbb{E}[\lVert a\rVert^{2}]\,\mathbb{E}[\lVert\delta\rVert^{2}]}\;\bigl(A_{\mathrm{K}} \otimes S_{\mathrm{K}}\bigr) \;=\; M_{\mathrm{K}} \;=\; H.
$$

**All three objects coincide.**

**(C2) Norm decoupling.** The equality $M_{\mathrm{G}} = M_{\mathrm{K}}$ needs only the
three scalar-level conditions

$$
\mathbb{E}[\lVert\delta\rVert^{2} a a^{\top}] = \mathbb{E}[\lVert\delta\rVert^{2}]\,\mathbb{E}[a a^{\top}],
\quad
\mathbb{E}[\lVert a\rVert^{2} \delta \delta^{\top}] = \mathbb{E}[\lVert a\rVert^{2}]\,\mathbb{E}[\delta \delta^{\top}],
$$
$$
\mathbb{E}[\lVert a\rVert^{2}\lVert\delta\rVert^{2}] = \mathbb{E}[\lVert a\rVert^{2}]\,\mathbb{E}[\lVert\delta\rVert^{2}],
$$

which are strictly weaker than C1. Under C2 the two methods agree, but neither need be
exact — $H$ can still be far from any Kronecker product.

$$
\text{C1} \;\Longrightarrow\; \text{C2}, \qquad \text{C2} \;\not\Longrightarrow\; \text{C1}.
$$

**Summary.**

| condition | $M_{\mathrm{K}}$ | $M_{\mathrm{G}}$ | relation |
|---|---|---|---|
| C1 holds | $= H$ | $= H$ | identical, both exact |
| C2 only | approximate | approximate | identical, neither exact |
| C2 fails | approximate | approximate | **differ** |

---

## 5. A minimal case where they differ

Two equiprobable samples, $m = n = 2$, $c > 0$:

$$
a^{(1)} = \begin{pmatrix}1\\0\end{pmatrix},\;
\delta^{(1)} = \begin{pmatrix}1\\0\end{pmatrix};
\qquad
a^{(2)} = \begin{pmatrix}0\\1\end{pmatrix},\;
\delta^{(2)} = \begin{pmatrix}0\\c\end{pmatrix}.
$$

Here $\lVert a \rVert^{2} = 1$ always but $\lVert \delta \rVert^{2} \in \{1, c^{2}\}$
varies *with which* $a$ occurred — C2 fails for $c \neq 1$. At $c = 3$:

$$
H = \operatorname{diag}(0.5,\; 0,\; 0,\; 4.5), \qquad
M_{\mathrm{K}} = \operatorname{diag}(0.25,\; 2.25,\; 0.25,\; 2.25), \qquad
M_{\mathrm{G}} = \operatorname{diag}(0.05,\; 0.45,\; 0.45,\; 4.05).
$$

$$
\lVert H - M_{\mathrm{K}} \rVert_{F} = 3.202,
\qquad
\lVert H - M_{\mathrm{G}} \rVert_{F} = 0.900,
\qquad
\varepsilon^{\star} = 0.500 .
$$

KFAC spreads mass onto the two structurally-zero blocks and misses the dominant one by a
factor of two. Gnome recovers the dominant block to within 10% and lands within a factor
of $1.8$ of the Van Loan–Pitsianis optimum. At $c = 1$, C2 holds and the two agree exactly
(both $\operatorname{diag}(\tfrac14,\tfrac14,\tfrac14,\tfrac14)$), illustrating that
agreement is a property of the *distribution*, not of the construction.

A continuous version with $\lVert \delta \rVert$ made to grow with $\lVert a \rVert$
gives, at $N = 4\times10^{5}$ samples, relative Frobenius errors of $0.134$ for KFAC
against $0.006$ for Gnome, while the decoupled version gives $0.0075$ and $0.0052$ —
agreement to Monte Carlo noise.

---

## 6. What this does and does not establish

**Does.** The comparison of estimands is exact and requires no empirical premise. In
particular, §1 of the Kronecker-basis argument — "a good Kronecker factorisation exists,
imported from K-FAC" — is not load-bearing for the *relative* claim. Wherever K-FAC's
assumption holds, Gnome reproduces K-FAC; wherever it fails, the two differ and Gnome's
estimand is the one with a rank-one optimality property. The chain

$$
\lVert H - M_{\mathrm{G}} \rVert_{F} \;\approx\; \varepsilon^{\star} \;\le\; \lVert H - M_{\mathrm{K}} \rVert_{F}
$$

can therefore be asserted directly rather than inherited.

**Does not.**

1. *Nothing about finite samples.* $M_{\mathrm{K}}$ is estimated from per-layer second
   moments; $M_{\mathrm{G}}$ from probe covariances, which carry additional Monte Carlo
   variance. At fixed budget the ordering can reverse. This is the subject of the
   variance analysis and is orthogonal to everything above.

2. *Nothing about exactness.* $M_{\mathrm{G}}$ is one power-iteration step, empirically
   close to the Van Loan–Pitsianis optimum but not equal to it — $0.900$ against
   $0.500$ in §5.

3. *Nothing about eigenbases.* Frobenius closeness of the matrix is not the operative
   quantity for a diagonal preconditioner; off-diagonal mass in the induced basis is.
   That is a separate (and short) argument.

4. *Nothing about PINNs specifically.* Eq. $(1)$ assumes the per-sample gradient is
   rank one, $G = \delta a^{\top}$. Residuals containing input derivatives give
   $G = \sum_{r} \delta_{r} a_{r}^{\top}$ of rank up to $k+1$ for a $k$-th order
   operator, which is the weight-sharing regime. The partial-trace construction is
   unchanged there; K-FAC's is not, and requires the separate KFAC-expand treatment.

---

## 7. Reproduction

```python
import numpy as np
rng = np.random.default_rng(0); m, n, N = 6, 5, 400_000

a = rng.standard_normal((N, n)) * np.array([1.5, 1., .8, .6, .4])
d = rng.standard_normal((N, m)) * np.array([1.3, 1., .9, .7, .5, .4])
d *= 1.0 + 2.0 * (np.linalg.norm(a, axis=1, keepdims=True) / np.sqrt(n))   # break C2

g = (a[:, :, None] * d[:, None, :]).reshape(N, -1)     # vec(G), G = d a^T
H = g.T @ g / N
trH = np.trace(H)

M_K = np.kron(a.T @ a / N, d.T @ d / N); M_K *= trH / np.trace(M_K)

G = g.reshape(N, n, m)
A_hat = np.einsum('nij,nkj->ik', G, G) / N             # E[||delta||^2 a a^T]
B_hat = np.einsum('nji,njk->ik', G, G) / N             # E[||a||^2 delta delta^T]
M_G = np.kron(A_hat, B_hat) / trH

for nm, M in [('KFAC', M_K), ('Gnome', M_G)]:
    print(nm, np.linalg.norm(H - M, 'fro') / np.linalg.norm(H, 'fro'))
```

Removing the third line (the coupling) drives the two errors together.
