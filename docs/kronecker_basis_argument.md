# Why Gnome's Eigenbasis Approximates the GGN's

A composition of two existing results plus one inference. The composition is available to
Gnome specifically, and not to SOAP, for reasons made explicit in §5.

---

## 0. Notation

For a layer with weight $W \in \mathbb{R}^{m \times n}$, write $P = mn$ and let
$H \in \mathbb{R}^{P \times P}$ denote the layerwise GGN block. Gradients reshaped to
matrix form are written $G \in \mathbb{R}^{m \times n}$, so $g = \operatorname{vec}(G)$.

A *Kronecker approximation* of $H$ is any $A \otimes B$ with $A \in \mathbb{R}^{n \times n}$,
$B \in \mathbb{R}^{m \times m}$. Write

$$
\varepsilon^\star \;=\; \min_{A, B} \; \bigl\| H - A \otimes B \bigr\|_F
$$

for the best achievable Frobenius error over that family.

---

## 1. Premise (K-FAC): a good Kronecker factorization exists

For a layer with input activations $a$ and pre-activation gradients $\delta$, the per-sample
gradient is $G = \delta a^\top$, so the exact Fisher block is

$$
F \;=\; \mathbb{E}\bigl[\, a a^\top \otimes \delta \delta^\top \,\bigr].
$$

K-FAC's approximation is to push the expectation inside,

$$
F \;\approx\; \mathbb{E}[a a^\top] \otimes \mathbb{E}[\delta \delta^\top] \;=\; A_{\text{KFAC}} \otimes S_{\text{KFAC}},
$$

which is exact iff $a$ and $\delta$ are uncorrelated in second moments. Martens & Grosse
justify this by approximate statistical independence of activations and backpropagated
errors, with the residual controlled by higher cumulants of their joint distribution; the
assumption is argued to improve with large batches and near-Gaussian layer statistics.

For exponential-family losses with the model emitting natural parameters (softmax CCE on
logits, Gaussian/MSE on the mean), $F = H$, so this is a statement about the GGN.

**What we take from this:** $\varepsilon^\star \le \|H - A_{\text{KFAC}} \otimes S_{\text{KFAC}}\|_F$,
and the latter is empirically small. I.e. *some* good Kronecker approximation exists.

> Note this is an empirical/structural premise about neural networks, not a theorem. See §6.

---

## 2. Optimality (Morwani et al.): the factors we compute are near-best

Van Loan & Pitsianis (1993): under a rearrangement $\mathcal{R}(\cdot)$ of the entries of $H$,

$$
\bigl\| H - A \otimes B \bigr\|_F \;=\; \bigl\| \mathcal{R}(H) - \operatorname{vec}(A)\operatorname{vec}(B)^\top \bigr\|_F,
$$

so the optimal Kronecker approximation is a **rank-one** approximation of $\mathcal{R}(H)$,
obtainable by power iteration.

Morwani et al. extend this to networks with weight sharing and show: for any covariance
$H = \mathbb{E}[g g^\top]$, initializing the power iteration at the identity and taking **one
step** yields exactly

$$
\mathbb{E}[G G^\top] \;\otimes\; \mathbb{E}[G^\top G],
$$

the partial traces — i.e. Shampoo's two factors, reached from an entirely different
direction. (Shampoo maintains the $1/2$ powers of each, so it is *Shampoo squared* that is
the approximation to $H$; this is the misconception their paper corrects.) Empirically the
one-step result tracks the optimal Kronecker approximation closely.

**What we take from this:** the factors are near-optimal *for whatever covariance they are
built from*. The result is agnostic to whether that covariance is curvature.

---

## 3. Composition

Gnome's surrogate satisfies $\mathbb{E}[g_s g_s^\top] = \Sigma = H$ by construction. So the
covariance fed to the factor updates **is** the GGN, and §2 applies with $H$ as its input:

$$
\hat A \otimes \hat B \;=\; \mathbb{E}[G_s G_s^\top] \otimes \mathbb{E}[G_s^\top G_s]
\qquad\text{with}\qquad
\bigl\| H - \hat A \otimes \hat B \bigr\|_F \;\approx\; \varepsilon^\star .
$$

Chaining with §1:

$$
\boxed{\;\bigl\| H - \hat A \otimes \hat B \bigr\|_F \;\approx\; \varepsilon^\star \;\le\; \bigl\| H - A_{\text{KFAC}} \otimes S_{\text{KFAC}} \bigr\|_F \;\ll\; \|H\|_F \;}
$$

Gnome's Kronecker approximation of the GGN is at least as good as K-FAC's, up to the
one-power-iteration gap — **without** requiring the independence assumption to hold, and
without layer-specific derivations.

---

## 4. From Frobenius error to eigenspaces

§3 bounds the *matrix* error. The eigenbasis claim needs a further step, and it is ours,
not Morwani's.

Frobenius closeness does **not** imply eigenvector closeness. By Davis–Kahan, the rotation
of an eigenvector is bounded by $\|\Delta\| / \text{gap}$; where the spectrum is degenerate
or nearly so, an arbitrarily small perturbation can rotate individual eigenvectors
arbitrarily. For PINNs — many collocation directions of comparable curvature — near-degeneracy
is the common case.

The resolution is that the per-coordinate diagonal we fit does not require individual
eigenvectors to be correct. Partition the spectrum into clusters separated by gap $\delta$
from the remainder. Then:

- **Between clusters:** the spectral projector $\hat\Pi$ onto a cluster satisfies
  $\|\hat\Pi - \Pi\| \le c\,\|\Delta\|/\delta$ — stable, and this is what we need.
- **Within a cluster:** the true curvature restricted to the cluster is $\approx \lambda I$,
  which is rotation-invariant and therefore stays diagonal under *any* orthonormal basis of
  that subspace. Eigenvector instability inside a cluster costs nothing.

So the correct statement is:

> A near-optimal Kronecker approximation of $H$ induces near-correct **spectral
> subspaces** wherever the spectrum separates, and the free diagonal absorbs the rest.

---

## 5. Why this composition is unavailable to SOAP

§2 delivers the optimal Kronecker approximation *of the matrix supplied to it*. SOAP supplies
the empirical Fisher $\mathbb{E}[gg^\top]$ with $g$ the real-label loss gradient. §1 is a
statement about $F = H$; it says nothing about the empirical Fisher, and Morwani et al. §4.2
separately measures the degradation from substituting one for the other.

The chain therefore closes for Gnome and breaks at step 3 for SOAP. **The surrogate is what
makes the input to Morwani's theorem the right matrix.** That is the load-bearing role of §4
of the method, stated in one line.

The same holds for Morwani §4.1 (batch gradients): since
$\mathbb{E}[\bar g \bar g^\top] = \tfrac{1}{B}\mathbb{E}[gg^\top] + \tfrac{B-1}{B}\mu\mu^\top$,
Shampoo's factor is dominated by a rank-one mean term at large $B$. Gnome's probes
$\epsilon_k$ are independent and zero-mean, so all $k \neq l$ cross terms vanish and
$\mathbb{E}[g_s g_s^\top]$ is exactly the average per-sample GGN over the aux batch — from a
single aggregated backward pass.

---

## 6. Caveats to state, not to hide

**(a) §1 is imported from classification.** K-FAC's independence assumption is validated on
classification networks, and it is argued to hold better with large batches and near-Gaussian
layer statistics. PINN collocation geometry is the adversarial case: points near a boundary
layer or shock couple large $\|a\|$ with large $\|\delta\|$ — exactly the correlation the
approximation assumes away.

This weakens the *bound*, not the *method*. §3 gives the best available Kronecker
approximation regardless of how good the best one is; a weaker §1 means we know less about
$\varepsilon^\star$, not that we do worse than K-FAC. §1 functions as a lower bound on
quality, not a load-bearing assumption.

**(b) "Optimal eigenbasis" means optimal *among Kronecker-structured bases*.** The GGN's true
eigenbasis is not generally of the form $Q_A \otimes Q_B$ — such matrices are a measure-zero
subset of $O(P)$. §3 says our Kronecker basis is a good approximation to the true one; it does
not say we recover it.

**(c) Morwani's guarantee is one power-iteration step,** empirically close to optimal on
MNIST-2 / CIFAR-5M / ImageNet, with a larger gap reported on ViT. Architecture-dependent, and
unmeasured for PINN MLPs.

---

## 7. The measurement that closes it

All three caveats reduce to one unmeasured quantity: the off-diagonal mass of the rotated
GGN. On a Poisson problem small enough to form $H$ exactly:

1. Build $Q = Q_{\hat A} \otimes Q_{\hat B}$ from Gnome's factors.
2. Report $\;\bigl\|\operatorname{offdiag}(Q^\top H Q)\bigr\|_F \big/ \bigl\|Q^\top H Q\bigr\|_F\;$ over training.
3. Report the same for SOAP's factors as a control.

If small: the chain closes empirically for our setting, and it is a data point on the question
Morwani et al. list as open — characterizing when $H$ is close to a Kronecker product beyond
the conditions K-FAC established.

If large *and Gnome still works*: more interesting. It means the free diagonal is doing the
heavy lifting and the Kronecker basis is a weaker constraint than the literature assumes.

Either outcome is worth reporting.
