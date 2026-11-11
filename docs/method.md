# Gnome: Gauss-Newton Optimizer via Matrix Eigen-decomposition

Gnome is a second-order PyTorch optimizer that extends [SOAP](https://arxiv.org/abs/2409.11321) into a true Gauss-Newton method. It keeps SOAP's Kronecker-factored eigenbasis machinery but changes the signal those factors are built from, the update rule inside the rotated basis, how step size is controlled, and how diagonal curvature statistics follow basis changes.

This document describes the algorithm in detail: how it builds off SOAP, the role of the auxiliary surrogate function, and the concrete construction of that surrogate for each supported loss.

---

## 1. Background: SOAP in one paragraph

SOAP runs Adam inside a rotated parameter basis derived from the Kronecker factors of the loss gradient. For a 2D parameter tensor $G \in \mathbb{R}^{m \times n}$ with per-step gradient $g_t \in \mathbb{R}^{m \times n}$, SOAP maintains an EMA of two outer products

$$
L_t = \beta \, L_{t-1} + (1-\beta)\, g_t g_t^\top, \qquad
R_t = \beta \, R_{t-1} + (1-\beta)\, g_t^\top g_t,
$$

periodically diagonalizes them ($L = Q_L \Lambda_L Q_L^\top$, $R = Q_R \Lambda_R Q_R^\top$), rotates the gradient into the joint eigenbasis ($\tilde g = Q_L^\top g \, Q_R$), runs Adam there ($\tilde m$, $\tilde v$ EMAs, update $\tilde m / (\sqrt{\tilde v} + \varepsilon)$), and rotates the result back. The Kronecker structure is what makes the eigendecompositions tractable for large weight matrices; the rotation is what gives Adam a basis in which the curvature is approximately axis-aligned.

The piece of SOAP we want to inherit is the Kronecker-eigenbasis machinery itself. The piece we want to replace is the **signal** it is fed. SOAP's $L_t$ and $R_t$ are built from past loss gradients, i.e. an empirical-Fisher proxy.

## 2. From SOAP to Gnome: three changes

Gnome makes three changes to the SOAP recipe.

**(a) Build the Kronecker factors from a GGN estimate, not from the loss gradient.**

Empirical Fisher (gradient outer products) is a known-poor curvature proxy for regression losses, and it conflates noise with curvature in classification. We replace the loss gradient $g_t$ in the factor update with a *surrogate gradient* $g_s$ obtained from a small auxiliary batch and a randomized sketch of the loss's output Hessian. The surrogate is constructed so that

$$
\mathbb{E}\!\left[g_s\, g_s^\top\right] \;=\; J_\theta^\top H_y J_\theta \;=\; \text{GGN},
$$

where $J_\theta = \partial \hat y / \partial \theta$ is the model Jacobian and $H_y = \partial^2 \ell / \partial \hat y^2$ is the intrinsic output Hessian of the loss. The Kronecker factors then become EMAs of $g_s g_s^\top$ and $g_s^\top g_s$, which in expectation are the Kronecker factors of the GGN itself.

The factor refresh schedule, initial eigendecomposition, subsequent subspace iteration with QR, and dimension-merging machinery follow SOAP. The diagonal EMA is transported across basis changes using squared overlaps, as described in (c) and §6.3.

**(b) Newton step inside the rotated basis, with an l2 trust region.**

SOAP's rotated-basis update is $\tilde m / (\sqrt{\tilde v} + \varepsilon)$. Gnome's is

$$
\Delta \tilde\theta \;=\; \frac{\hat m}{\hat v + \lambda}
$$

with the un-square-rooted denominator. This is the diagonal Gauss-Newton update inside the eigenbasis: $\hat v$ is the second-moment EMA of the rotated surrogate gradient, which in expectation is the diagonal of the GGN in the eigenbasis, so dividing the rotated loss gradient by $\hat v$ is the per-coordinate Newton step.

A pure Newton step has no built-in step-size control. Gnome imposes an **l2 trust region** on the update in the rotated basis:

$$
\| \Delta\tilde\theta \|_2 \;\le\; T, \qquad
T =
\begin{cases}
c\sqrt{N}, & \text{CCE}, \\
c\sqrt{\|W\|_2^2 + 1}, & \text{MSE}.
\end{cases}
$$

Here $c$ is `trust_radius`, $W$ is the current parameter tensor, and $N = \operatorname{numel}(W)$ is its number of scalar parameters. The norm $\|W\|_2$ is the Euclidean norm of the flattened tensor (the Frobenius norm for a matrix). Both CCE surrogate variants, `cce` and `cce_hutchinson`, use the CCE budget. The smallest $\lambda \ge \varepsilon$ that satisfies this constraint is found by solving the secular equation

$$
\phi(\lambda) \;=\; \left\| \frac{\hat m}{\hat v + \lambda} \right\|_2 \;=\; T,
$$

via Hebden's reciprocal Newton method. Convergence takes 2 to 3 iterations, warm-started from the previous step's $\lambda$, with an analytic bracket derived from $\min(\hat v)$ and $\max(\hat v)$. All computation stays on-device with no Python scalar syncs.

**Why l2 rather than per-coordinate clipping.** The rotation matrices $Q$ are orthonormal (eigenvectors of real symmetric Kronecker factors), so the L2 norm is preserved: $\|\Delta\tilde\theta\|_2 = \|\Delta\theta\|_2$. The trust region is therefore meaningful in both spaces. An l2 bound has two advantages over per-coordinate clipping:

1. **It preserves the update direction.** Uniform damping shrinks the step along the Newton-to-gradient-descent continuum without distorting which coordinates move relative to each other. Per-coordinate clipping projects onto an l-infinity ball, which can scramble the relative ordering of updates.
2. **No single coordinate dominates.** The l2 norm aggregates all $P$ coordinates, so an outlier with spuriously small $\hat v_i$ cannot single-handedly set the damping for the entire tensor.

**Loss-dependent scale.** For CCE, $T=c\sqrt{N}$ gives a constant RMS update budget $T/\sqrt{N}=c$, independent of the parameter magnitudes. For MSE, $T=c\sqrt{\|W\|_2^2+1}$ grows with the current parameter norm and has a nonzero floor of $c$ at $W=0$. Its RMS form is $T/\sqrt{N}=c\sqrt{\operatorname{mean}(W^2)+1/N}$.

These budgets depend on the tensor's size or current norm; neither requires an initialization-norm reference.

The existing coordinate clamp to `[-1, 1]` after projection remains in place; it can only reduce the update norm. The trust budget applies before learning-rate scaling and excludes decoupled weight decay.

**Behavior at the extremes.** When curvature is well-estimated and the unconstrained Newton step already satisfies $\|\hat m / \hat v\|_2 \le T$, the constraint is slack: $\lambda = \varepsilon$ and the update is a pure diagonally-preconditioned Newton step. When curvature is unreliable (small $\hat v_i$, e.g. early in training while the eigenbasis is still warming up), $\lambda$ grows and the update degrades smoothly toward $\hat m / \lambda$, which is bounded gradient descent in the rotated basis. The trust region is a safeguard, not the primary step-size control; at sensible learning rates and $c \sim 1$ it rarely binds on well-conditioned steps.

The dimensional argument for the un-square-rooted denominator: if we model the rotated gradient as $\tilde g \sim \mathcal{N}(0, \tilde H)$ with $\tilde H$ the rotated GGN, then $\mathbb{E}[\tilde g^2] = \tilde H$. The Newton-rescaled coordinate has units $[\text{grad}] / [\text{curvature}]$, i.e. $\tilde g / \tilde H$. Adam's $\sqrt{\tilde v}$ has the wrong units ($\sqrt{[\text{curvature}]}$); it is a heuristic for the gradient *norm*, not the curvature. Gnome's $\hat v$ has the right units.

**(c) Squared-overlap transport across basis updates.**

The curvature EMA is stored in the current rotated basis. When that basis
changes, its entries must follow the new coordinates. Gnome uses
**squared-overlap transport**: if
$A=Q_{L,\mathrm{new}}^\top Q_{L,\mathrm{old}}$ and
$B=Q_{R,\mathrm{old}}^\top Q_{R,\mathrm{new}}$, it transports the diagonal
curvature EMA $V$ as

$$
V_{\mathrm{new}} = A^{\odot 2} V_{\mathrm{old}} B^{\odot 2}.
$$

This replaces permutation-only handling with a diagonal approximation to the
second-moment change of basis. It preserves nonnegativity and total diagonal
mass, and handles eigenvector reordering and sign flips automatically. The
gradient EMA is transported as a vector using the unsquared overlaps.
Section 6.3 gives the derivation, assumptions, and implementation details.

## 3. The GGN matrix

For a model $\hat y = f(\theta; x)$ and a per-sample loss $\ell(\hat y, y)$, the Generalized Gauss-Newton matrix is

$$
G(\theta) \;=\; \mathbb{E}_{x,y}\!\left[ J_\theta(x)^\top \, H_y(\hat y, y) \, J_\theta(x) \right],
$$

where $J_\theta$ is the model Jacobian and $H_y$ is the intrinsic Hessian of the loss with respect to its own input (the model output). The GGN equals the true Hessian when $f$ is linear in $\theta$ or when the residuals are zero, and otherwise drops the second-derivative-of-$f$ terms. It is positive semi-definite whenever $H_y$ is (i.e. for any convex output loss), which makes it a safe preconditioner.

The classical way to use the GGN is to apply it directly to the gradient (Gauss-Newton or natural gradient). Gnome instead uses it to *rotate* into a basis where it's approximately diagonal, then runs a diagonal Newton step in that basis.

## 4. The surrogate function: principle

We want a scalar $S(\theta; \text{aux batch}, R)$, where $R$ is auxiliary randomness, such that

$$
g_s \;\triangleq\; \frac{\partial S}{\partial \theta}, \qquad
\mathbb{E}_R\!\left[g_s\, g_s^\top\right] \;=\; J_\theta^\top H_y J_\theta \;=\; \text{GGN}.
$$

The construction is a five-step composition.

**Step 1: Factor the output Hessian.** Because $H_y$ is positive semi-definite for any convex output loss, it admits a square-root factorization

$$
H_y \;=\; A A^\top.
$$

Substituting into the GGN definition,

$$
\text{GGN} \;=\; J^\top H_y J \;=\; J^\top A \, A^\top J \;=\; (J^\top A)(J^\top A)^\top.
$$

The crucial observation: for every loss in §5, $H_y$ depends only on the model output $\hat y$ (and the target, in some cases), not on $\theta$. So the square root $A$ inherits the same property and can be written in closed form for each loss family. **No second-order autograd is required at any point**; we never differentiate $H_y$ through $\theta$.

**Step 2: Express $J^\top A$ as a gradient.** Treating $A$ as detached from autograd, define

$$
z \;=\; \hat y^\top A \quad \text{(a row vector of length $\dim A$)}.
$$

Then $J^\top A = \partial z / \partial \theta$, which is precisely what `autograd.grad` returns when differentiating $z$'s components w.r.t.\ $\theta$.

**Step 3: Reduce to a scalar via a random probe.** A single scalar suffices to backward through. Inner-product $z$ with a random vector $R$:

$$
S \;=\; \hat y^\top A \, R \quad \text{(with $A$ and $R$ both detached)}.
$$

Then

$$
g_s \;=\; \frac{\partial S}{\partial \theta} \;=\; J^\top A R.
$$

**Step 4: Take the expectation.** With $R$ random,

$$
\mathbb{E}\!\left[g_s\, g_s^\top\right] \;=\; J^\top A \, \mathbb{E}[R R^\top] \, A^\top J.
$$

This equals the GGN $J^\top A A^\top J$ **if and only if** $\mathbb{E}[R_i R_j] = \delta_{ij}$, i.e. $R$ has uncorrelated unit-variance components. Two cheap distributions satisfy this:

- **Rademacher**: $R_i \in \{\pm 1\}$ independently with equal probability.
- **Unit Gaussian**: $R_i \sim \mathcal{N}(0, 1)$ iid.

Both yield unbiased GGN estimators. The Rademacher choice has strictly lower variance (the standard Hutchinson estimator argument: same first two moments, lowest fourth moment) and is the default in Gnome.

**Step 5: Average over the aux batch.** The aux batch contains $K$ samples; each has its own $\hat y_k$, its own output Hessian factor $A_k$ (which generally depends on $\hat y_k$ and so varies per sample; see §5.3 for the CCE case), and its own independent random vector $R_k$. The surrogate sums them with a $1/\sqrt K$ normalization:

$$
S \;=\; \frac{1}{\sqrt K} \sum_{k=1}^{K} \hat y_k^\top A_k R_k,
\qquad
g_s \;=\; \frac{1}{\sqrt K} \sum_{k=1}^{K} J_k^\top A_k R_k.
$$

Taking expectations, with each $R_k$ independent across $k$ so $\mathbb{E}[R_k R_j^\top] = I \cdot \delta_{kj}$, gives the empirical GGN over the aux batch:

$$
\mathbb{E}_R\!\left[g_s g_s^\top\right] \;=\; \frac{1}{K} \sum_{k=1}^{K} J_k^\top H_{y,k} J_k.
$$

Taking the further expectation over the aux batch sampling $(x_k, y_k) \sim \text{data}$ recovers the population GGN:

$$
\mathbb{E}_{R,\,\text{batch}}\!\left[g_s g_s^\top\right] \;=\; \mathbb{E}_{x,y}\!\left[J^\top H_y J\right] \;=\; \text{GGN}, \qquad \text{for any } K \ge 1.
$$

The estimator is unbiased at every batch size. $K$ controls **variance**, not bias: with the $1/\sqrt K$ scaling, $\mathbb{E}\|g_s\|^2$ is independent of $K$ but $\mathrm{Var}(g_s)$ decreases as $K$ grows. The $1/\sqrt K$ normalization thus decouples the surrogate's scale from the aux batch size $K$ entirely, so `eps` and `trust_radius` calibrate against a fixed-scale surrogate regardless of $K$. (`tests/test_surrogate_scaling.py` pins this empirically across $K \in \{1,4,16,64\}$.)

The explicit per-sample sum is the uniform notation that handles all four loss modes in §5. For MSE and BCE, $A_k$ is diagonal (either constant or a per-element function of detached logits), and the sum collapses into a single inner product over `(batch × output_dim)` axes. Both losses can be written in batched-tensor form. For CCE, $A_k$ varies per sample through the softmax probabilities $p_k$, so the sum is genuinely a per-sample operation.

So the entire construction reduces to: **given a loss, write down a closed-form square root $A$ of its output Hessian.** Section 5 collects $A$ for each loss family. The optimizer exposes four modes via the `loss=` argument: `'mse'`, `'bce_hutchinson'`, `'cce'`, and `'cce_hutchinson'`. BCE intentionally has only the Hutchinson construction; there is no separate sampling variant.

- **MSE**: $H_y = 2I$, so $A = \sqrt 2 \, I$.
- **BCE**: $H_y = \mathrm{diag}(p(1-p))$, so $A = \mathrm{diag}\!\big(\sqrt{p(1-p)}\big)$.
- **CCE**: $H_y = \mathrm{diag}(p) - p p^\top$, with an analytic square root $A = \mathrm{diag}(\sqrt p)(I - \sqrt p \, \sqrt p^\top)$ computable in $O(K_c)$ work per sample.

## 5. Per-loss surrogate constructions

### 5.1 MSE (regression)

Per-element loss $\ell(\hat y, y) = (\hat y - y)^2$. Output Hessian:

$$
H_y \;=\; \frac{\partial^2 \ell}{\partial \hat y^2} \;=\; 2 I.
$$

A trivial square root is $A = \sqrt 2 \cdot I$. Drawing per-element Rademacher signs $\varepsilon \in \{\pm 1\}^{B \times D}$ and forming $v = \sqrt 2 \, \varepsilon$,

$$
S_{\text{MSE}} \;=\; \frac{1}{\sqrt K} \sum_{k=1}^{K} \sqrt 2 \, \varepsilon_k \cdot \hat y_k \;=\; \frac{\sqrt 2}{\sqrt K} \langle \varepsilon, \hat y \rangle.
$$

The gradient $g_s = (\sqrt 2 / \sqrt K) \sum_k J_k^\top \varepsilon_k$, and

$$
\mathbb{E}[g_s g_s^\top] \;=\; \frac{2}{K} \sum_k J_k^\top J_k \;\to\; \mathbb{E}_x[2 J^\top J] \;=\; \text{GGN}_{\text{MSE}}.
$$

Two notes:
- The main loss is built internally as `((y_hat - y)**2).sum() / B` (sum over output dim, mean over batch). This matches the per-element $H_y = 2I$ contract. Using `mean` over both dims would divide $g_{\text{main}}$ by the output dim while the surrogate scale is per-element, breaking the relative scale that `eps` and `trust_radius` are calibrated against.
- No double backward is needed. Knowing $H_y = 2I$ in closed form means we don't have to call any second-order autograd primitive. Just multiply logits by random signs and `autograd.grad` through.

### 5.2 BCE (binary cross-entropy with logits)

Shipped as `loss='bce_hutchinson'`. It is the diagonal special case of the CCE surrogate in §5.3 and intentionally has no separate sampling variant.

Per-element: predict logit $z$, target $y \in \{0, 1\}$, with $p = \sigma(z)$ and

$$
\ell \;=\; -y \log p - (1-y) \log(1-p).
$$

Derivatives of $\ell$ in the logit:

$$
\frac{\partial \ell}{\partial z} \;=\; p - y, \qquad
\frac{\partial^2 \ell}{\partial z^2} \;=\; p(1-p).
$$

So $H_y$ is diagonal with entries $p_i(1 - p_i)$, the per-element Bernoulli variance. The natural square root is $A = \mathrm{diag}(\sqrt{p(1-p)})$. Drawing per-element Rademacher signs $\varepsilon$:

$$
v \;=\; \sqrt{p(1-p)} \odot \varepsilon, \qquad
S_{\text{BCE}} \;=\; \frac{1}{\sqrt K} \sum_{k=1}^{K} \langle z_k, \mathrm{detach}(v_k) \rangle.
$$

The gradient is $g_s = (1/\sqrt K) \sum_k J_k^\top v_k$, and

$$
\mathbb{E}[g_s g_s^\top] \;=\; \frac{1}{K} \sum_k J_k^\top \, \mathrm{diag}(p_k(1-p_k)) \, J_k \;\to\; \text{GGN}_{\text{BCE}}.
$$

Construction notes:
- $p$ must be `detach`ed before being used to build $v$. The gradient should flow through $z$ (the logits), not through $p$. The probabilities enter only as a scaling factor for the Rademacher probe. Otherwise we'd be sampling an estimator of $J^\top (\partial_\theta H_y) J + J^\top H_y J$, not the GGN.
- Like MSE, no double backward: $H_y$ is known in closed form once we have $p$.
- The construction is the diagonal special case of the CCE surrogate below (with $K_{\text{classes}} = 2$ and the simplex-projection term vanishing).

### 5.3 CCE (categorical cross-entropy)

Per-sample: $K_c$-way softmax cross-entropy. Output Hessian:

$$
H_y \;=\; \mathrm{diag}(p) - p p^\top,
$$

which is non-diagonal but admits a closed-form square root computable in $O(K_c)$:

$$
A \;=\; \mathrm{diag}(\sqrt p)\, \big(I - \sqrt p \, \sqrt p^{\,\top}\big).
$$

Verification, using $\|\sqrt p\|^2 = \sum_i p_i = 1$:

$$
\begin{aligned}
A A^\top
&= \mathrm{diag}(\sqrt p)\, (I - \sqrt p \sqrt p^\top)^2 \, \mathrm{diag}(\sqrt p) \\
&= \mathrm{diag}(\sqrt p)\, (I - 2\sqrt p \sqrt p^\top + \sqrt p \sqrt p^\top \sqrt p \sqrt p^\top)\, \mathrm{diag}(\sqrt p) \\
&= \mathrm{diag}(\sqrt p)\, (I - \sqrt p \sqrt p^\top)\, \mathrm{diag}(\sqrt p) \\
&= \mathrm{diag}(p) - p p^\top \;=\; H_y. \quad\checkmark
\end{aligned}
$$

For per-sample Rademacher $R \in \{\pm 1\}^{K_c}$, the per-sample probe is

$$
A R \;=\; \mathrm{diag}(\sqrt p) R - \mathrm{diag}(\sqrt p)\, \sqrt p \big(\sqrt p^\top R\big)
\;=\; \sqrt p \odot R - (\sqrt p \cdot R) \, p,
$$

where the second equality uses $\mathrm{diag}(\sqrt p)\, \sqrt p = \sqrt p \odot \sqrt p = p$. This is $O(K_c)$ work per sample (no explicit $K_c \times K_c$ matrix is ever materialized) and plugs straight into the §4 recipe with $\mathbb{E}[(AR)(AR)^\top] = H_y$ *exactly* per sample, not just in expectation over labels.

Gnome ships two CCE surrogate strategies, selectable via the `loss=` argument:

| `loss=` | Surrogate | Mechanism |
|---|---|---|
| `'cce'` | Fisher sampling | Draw $\tilde y \sim \text{Categorical}(p)$, use `cross_entropy(z, \tilde y)` |
| `'cce_hutchinson'` | Rademacher Hutchinson | Use the $A R$ probe above directly |

**Fisher sampling (`'cce'`).** The classical K-FAC / Fisher approach: draw a fake label $\tilde y_k \sim \mathrm{Categorical}(p_k)$ from the model's own predictive distribution and use $\mathtt{F.cross\_entropy}(z_k, \tilde y_k)$ as the surrogate. Its logit gradient is $v = p - e_{\tilde y}$, and since $\mathbb{E}[e_{\tilde y}] = p$,

$$
\mathbb{E}_{\tilde y \sim p}\big[v v^\top\big] \;=\; \mathrm{Cov}(e_{\tilde y}) \;=\; \mathrm{diag}(p) - p p^\top \;=\; H_y ,
$$

so the outer products have the correct GGN expectation with no explicit factorization of $H_y$. Simple to implement and understand; the label draw plays the role of the random probe.

**Hutchinson (`'cce_hutchinson'`).** Uses the Rademacher $A R$ probe from the §4 recipe directly. Compared to Fisher sampling:

- **Lower variance.** The $A R$ probe uses every class simultaneously (a dense vector in $\mathbb{R}^{K_c}$), while Fisher sampling's $p - e_{\tilde y}$ is a rank-1 contribution aligned with a single class direction. For a given aux batch size $K$, Hutchinson extracts more information per sample.
- **Variance decays with confidence.** As the model becomes confident ($p$ approaches a one-hot vector), $\sqrt p \odot R$ collapses to a near-one-hot vector and the simplex-projection term $(\sqrt p \cdot R)\, p$ cancels the surviving coordinate. The estimator variance decays toward zero with the Gini impurity of $p$. Late in training, when classification gradients are sharp and Fisher sampling's single-class probe becomes most wasteful, Hutchinson approaches a noise-free preconditioner.
- **Cost.** Both are $O(K_c)$ per aux sample. Hutchinson has a small constant-factor overhead from the additional `sqrt`, `sum`, and elementwise ops.

**Common pitfalls (both modes).** Two ways to silently break the estimator:

1. **Broadcasting.** Random probes must be independent per (sample, class). Sharing one $K_c$-vector across the batch makes $\mathbb{E}[R_n R_m^\top] = I \ne 0$ for $n \ne m$, injecting a covariance bias between unrelated samples that destroys the batched-GGN identity.
2. **BatchNorm.** The per-sample GGN sum assumes the Jacobian of sample $n$ does not depend on sample $m$. BatchNorm couples the forward pass across the batch dim, breaking the block-diagonal structure of the true batched Hessian. Use GroupNorm or LayerNorm in any architecture trained under `'cce'` or `'cce_hutchinson'`.

## 6. The full step

For each minibatch, with `main_closure` returning $(\hat y, y)$ on the $B-K$ main slice and `aux_closure` returning $(\hat y_{\text{aux}}, y_{\text{aux}})$ on the disjoint $K$ aux slice:

1. **Main forward + backward.** Compute the main loss internally (`mse` → sum-over-D-mean-over-B, `cce` / `cce_hutchinson` → `reduction='mean'`). Gradient $g_{\text{main}} = \nabla_\theta \ell_{\text{main}}$ via `autograd.grad`. Free the main graph.

   *Gradient accumulation.* When the full main batch does not fit in memory, `main_closure` can be a **list of closures**, one per micro-batch. Each runs an independent forward+backward, and their gradients are combined as a size-weighted average (weighted by each micro-batch's sample count). The result matches a single forward over the concatenation of all micro-batches exactly, regardless of how they were chunked. The eigenbasis $Q$ and all EMAs update once per logical step; the aux surrogate still runs once.

   *Global gradient clipping.* If `max_grad_norm` is set, the accumulated main gradient is rescaled so its total L2 norm (across all parameters) does not exceed the given value, matching `torch.nn.utils.clip_grad_norm_`. The curvature surrogate gradient $g_s$ is intentionally left unclipped. Its scale is calibrated by the $1/\sqrt{K}$ normalization and should not be distorted.

2. **Aux forward + backward.** Build $S$ per §5 from the aux logits and sampled randomness. Gradient $g_s = \nabla_\theta S$. Free the aux graph. With multiple `surrogate_partition_sizes`, compute the partition gradients in one batched backward and use the sums of curvature contributions described in §6.2.

3. **Per-parameter update.** For each parameter tensor:
   - On the first call for this tensor, initialize the Kronecker factors from the basis-gradient outer products and build $Q_L,Q_R$ with full `eigh`. This call builds the basis and skips the parameter update.
   - On subsequent calls, project $g_{\text{main}}$ and $g_s$ into the current rotated basis: $\tilde g = Q_L^\top g\, Q_R$.
   - Update the first-moment EMA from $\tilde g_{\text{main}}$ (`beta1`) and the second-moment EMA from $\tilde g_s^{\,2}$ (`beta2`).
   - Apply bias correction to obtain $\hat m$ and $\hat v$.
   - Solve for the trust-region damping $\lambda$: the smallest value $\ge \varepsilon$ such that $\|\hat m / (\hat v + \lambda)\|_2 \le T$, using $T=c\sqrt{N}$ for CCE and $T=c\sqrt{\|W\|_2^2+1}$ for MSE. Here $c$ is `trust_radius`, $N$ is the parameter tensor's element count, and $W$ is its current value, as in §2. Solved via Hebden's reciprocal Newton method (2 to 3 iterations), warm-started from the previous step's $\lambda$, with an analytic bracket from $\min(\hat v)$ and $\max(\hat v)$.
   - Compute the damped Newton step: $\Delta \tilde\theta = \hat m / (\hat v + \lambda)$.
   - Project back to the parameter basis: $\Delta\theta = Q_L \, \Delta\tilde\theta \, Q_R^\top$. Because the $Q$ matrices are orthonormal, $\|\Delta\theta\|_2 = \|\Delta\tilde\theta\|_2$. The trust region bound is preserved exactly.
   - Apply with $-\text{lr}$, then decoupled weight decay. Gnome owns no learning-rate schedule: `lr` is read from the parameter group on every step and multiplies the update *after* the trust-region solve, so it scales step length linearly and any `torch.optim.lr_scheduler` drives Gnome exactly as it drives AdamW.
   - Update the Kronecker factor EMAs from $g_s g_s^\top$ and $g_s^\top g_s$ (`shampoo_beta`), or from the main gradient when `use_soap_basis=True`.
   - At a scheduled refresh, compute the new eigenbases using one subspace iteration plus QR. Apply squared-overlap transport to the curvature EMA and unsquared-overlap transport to the gradient EMA (§6.3). These transported EMAs are ready before the next step incorporates its rotated gradients. Between refreshes, both remain in the current basis.

Two independent forward passes (one for main, one for aux) avoid holding the main batch's activations alive while running the surrogate backward. Neither the default single-surrogate path nor the batched partition path needs `retain_graph=True`; each releases the auxiliary graph in its gradient call.

Basis refresh and EMA transport occur after the current parameter update. Transport changes the coordinates of the stored statistics; it does not increment the EMA counters or apply another decay factor.

### 6.1 The `norm_free` option

By default, the squared surrogate $\tilde g_s^2$ enters the curvature EMA directly. When the surrogate magnitude decays faster than the EMA window (common early in classification training), the oldest entries still in the window carry the most mass and the curvature estimate tracks an obsolete scale. Setting `norm_free=True` normalizes each contribution to unit norm before it enters the EMA, so `beta2` controls recency rather than being overridden by a decaying $\|g_s\|$. Off by default: on MSE objectives the shrinking $\hat v$ is the self-annealing mechanism that lets Gnome hold a fixed learning rate, and normalizing would remove it. Turn it on for fast-learning objectives (classification) where the loss moves a long way inside one EMA window. Note that it rescales $\hat v$, so `lr` generally needs retuning when toggled.

### 6.2 Partitioned surrogates: compute versus variance

`opt.step(main_closure, aux_closure, surrogate_partition_sizes=[n_1, n_2, ...])`
partitions the auxiliary output into contiguous slices along its first axis.
The sizes must be positive integers with $\sum_j n_j=K$, the total auxiliary
batch size. Omitting the argument is equivalent to `[K]`.

For partition $I_j$, form

$$
S_j = \frac{1}{\sqrt K}\sum_{k\in I_j}\hat y_k^\top A_kR_k,
\qquad g_j=\nabla_\theta S_j.
$$

The normalization stays global. Equivalently, a locally normalized surrogate
would need an additional factor $\sqrt{n_j/K}$. The original surrogate gradient
is $\sum_j g_j$, so its outer product includes cross terms $g_jg_l^\top$.
Independent zero-mean probes make those cross terms vanish in expectation.
Partitioning removes them on every step by using

$$
\widehat H=\sum_jg_jg_j^\top,
\qquad \mathbb E_R[\widehat H]=\frac1K\sum_kJ_k^\top H_{y,k}J_k.
$$

For matrix-shaped parameter gradients $G_j$, update the Kronecker factor EMAs
once from $\sum_jG_jG_j^\top$ and $\sum_jG_j^\top G_j$. Project each gradient
in the same current basis, $\widetilde G_j=Q_L^\top G_jQ_R$, and update the
curvature EMA once from $\sum_j\widetilde G_j^{\,2}$. Never sum the gradients
before taking outer products or squares. The main-gradient momentum and the
basis-refresh schedule still advance once per logical step. With
`use_soap_basis=True`, only the curvature EMA uses the partitioned contributions;
the basis factors still use the main gradient. With `norm_free=True`, normalize
the summed squared contribution once, preserving relative partition weights.

For the already-scaled PINN vector from `stack_residuals`, partition sizes are
the auxiliary blocks' element counts, such as `[n_pde, n_ic, n_bc]`. Combining
the global normalization with the block scaling in §8 gives
$S_j=\sqrt{2\lambda_j/n_j}\,r_j^\top R_j$ for raw residual block $r_j$.
Thus unequal block sizes keep their intended loss weights. A block with multiple
output components contributes all of its elements to its partition size.

There is one auxiliary forward. For multiple partitions, stack the scalar
surrogates into a vector and call `torch.autograd.grad` with identity seeds and
`is_grads_batched=True`. PyTorch's vmap backend batches the vector-Jacobian
products, returning gradients shaped `[num_partitions, *parameter.shape]`.
The partition axis is kept separate through curvature accumulation. A single
partition uses ordinary `autograd.grad` without vmap overhead. The auxiliary
graph is released by the gradient call, and the separate parameter gradients
are stored until the update. Finer partitions remove more cross-partition noise
at the cost of more backward work and gradient storage. Sizes are supplied per
step, can vary with the auxiliary batch, and are not optimizer checkpoint state.

### 6.3 Squared-overlap transport

Let $M_{\mathrm{old}}$ and $V_{\mathrm{old}}$ denote the stored, uncorrected
gradient and diagonal curvature EMAs (`grad_m` and `gnd_m`)
immediately before a basis refresh. For a matrix-shaped parameter, define

$$
A=Q_{L,\mathrm{new}}^\top Q_{L,\mathrm{old}},
\qquad
B=Q_{R,\mathrm{old}}^\top Q_{R,\mathrm{new}}.
$$

A historical gradient with old coordinates $X$ has new coordinates $AXB$.
The first moment therefore has an exact linear change of basis:

$$
M_{\mathrm{new}} = A M_{\mathrm{old}} B.
$$

For the second moment, retain only its diagonal in the old basis. The resulting
**squared-overlap transport** is

$$
\boxed{V_{\mathrm{new}} = A^{\odot 2} V_{\mathrm{old}} B^{\odot 2}},
\qquad
(V_{\mathrm{new}})_{ij}
=\sum_{a,b} A_{ia}^{\,2}(V_{\mathrm{old}})_{ab}B_{bj}^{\,2}.
$$

Here $\odot 2$ is elementwise squaring. Applying the unsquared overlaps to $V$
would treat second moments as vector coordinates and could produce negative
curvature estimates. The squared overlaps give the diagonal of the transformed
diagonal second-moment model instead.

**Approximation and exact cases.** Using column-major vectorization, let
$T=B^\top\otimes A$ and
$D=\operatorname{Diag}(\operatorname{vec}V_{\mathrm{old}})$.
The formula above computes $\operatorname{diag}(TDT^\top)$ without forming
either full matrix. If the stored diagonal equals the diagonal of the historical
second-moment matrix $C$, write $C=D+E$, with $E$ its omitted off-diagonal
part. The exact new diagonal also
contains $\operatorname{diag}(TET^\top)$. The transport is exact for signed
permutations, or whenever that historical second-moment matrix is diagonal in
the old basis.

For a single transport with $D=\operatorname{Diag}(\operatorname{diag}C)$,
orthogonality gives the absolute-error bound

$$
\left\|\operatorname{diag}(TCT^\top)
      -\operatorname{diag}(TDT^\top)\right\|_2
\leq \|E\|_F.
$$

Thus small historical off-diagonal mass makes the local transport error small.
The relevant matrix is the historical second moment represented by the EMA;
the instantaneous GGN being nearly diagonal alone does not establish this
condition. Repeated transports discard newly introduced off-diagonal terms,
so approximation error can accumulate and diagonal statistics can become
smoother. The bound above does not include error already present in the stored
diagonal.

**Preserved properties.** Squared orthogonal overlap matrices are doubly
stochastic. Consequently, for nonnegative $V_{\mathrm{old}}$,

$$
V_{\mathrm{new}}\geq 0,
\qquad
\sum_{i,j}(V_{\mathrm{new}})_{ij}
=\sum_{i,j}(V_{\mathrm{old}})_{ij},
$$

up to numerical roundoff. Eigenvalue reordering is already included in the
overlaps, so there is no additional permutation of $V$. Eigenvector sign flips
cancel under squaring. Preservation of total diagonal mass does not imply
preservation of individual diagonal entries or of the full covariance.

**Tensor implementation and timing.** Along each preconditioned tensor axis
$k$, form $C_k=Q_{k,\mathrm{new}}^\top Q_{k,\mathrm{old}}$. Contract that axis
of `grad_m` with $C_k$ and of `gnd_m` with $C_k^{\odot 2}$.
For the right axis of a matrix, $C_R=B^\top$, so the axis contraction is exactly
right multiplication by $B^{\odot 2}$ in the boxed formula. Merged dimensions
are handled in their grouped layout and restored afterward; unpreconditioned
axes are left unchanged. The matrix case adds two overlap products and two
matrix multiplications for the curvature transport, plus the first-moment
transport, without storing historical gradients or a full covariance.

The implementation applies this transformation at the end of each scheduled
basis refresh, after incorporating the current gradient and surrogate in the
old basis and applying the current parameter update. The next step projects
its gradients into the new basis and incorporates them into the transported
EMAs. Transport preserves the EMA step count and does not apply extra decay;
the usual bias-correction factors remain in use. With residual partitioning,
transport the single aggregate curvature EMA once per refresh.

**Configuration.** Squared-overlap transport is the default methodology,
selected by `Gnome(..., basis_transport="overlap")`. The retained
`basis_transport="soap"` option provides the previous permutation-only
curvature handling and momentum projection for ablations. It is independent of
`use_soap_basis`, which selects the source of the Kronecker factors.
The selected transport mode is stored per parameter group in checkpoints.
Reaction, convection, and Allen–Cahn expose it as
`CONFIG["optimizers"]["gnome"]["basis_transport"]`.

## 7. Why the un-square-rooted denominator (again, briefly)

Suppose the rotated GGN is approximately diagonal with diagonal $d$. For a fixed basis and current parameters, the squared surrogate satisfies

$$
\mathbb{E}\!\left[\tilde g_s^{\,2}\right]_i \;=\; \mathbb{E}\!\left[(Q^\top g_s)_i^2\right] \;=\; (Q^\top \text{GGN}\, Q)_{ii} \;\approx\; d_i.
$$

Averaging these contributions gives $\hat v \approx d$ when curvature varies slowly over the EMA window and the squared-overlap transport error (§6.3) is small. Then $\hat m / \hat v$ is the per-coordinate Newton step in the rotated basis. The Adam denominator $\sqrt{\hat v}$ has the wrong units for a Newton step (it scales like $\sqrt{\text{curvature}}$, not curvature), which is why Adam needs $\varepsilon$ as a numerical-safety constant calibrated against the gradient norm. Moving curvature from "noisy outer product of gradients" to "true GGN estimate" only pays off when we also move the denominator from $\sqrt{\hat v}$ to $\hat v$.

## 8. Multi-block losses (PINNs)

A PINN loss is a weighted sum of per-block mean-squared residuals,
$L = \sum_j \lambda_j \, \mathrm{mean}_i(r_{j,i}^2)$. One block for the PDE residual on collocation points, one for the initial condition, one per boundary condition, and (for inverse problems) one per data-fit term. Because every block is an MSE, the whole loss rides the §5.1 MSE surrogate unchanged: `gnome.stack_residuals` folds the blocks into a single flat residual vector whose plain `mean(·²)` reproduces $L$ exactly, and the automatic $\sqrt 2\,\varepsilon$ Rademacher probe on that vector decomposes into the per-block independent GGN estimator.

Concretely, scaling block $j$ by $\alpha_j = \sqrt{\lambda_j N / N_j}$ (with $N_j$ the block's element count and $N = \sum_j N_j$) gives $\mathrm{mean}(\text{stacked}^2) = \sum_j \lambda_j\,\mathrm{mean}_i(r_{j,i}^2)$, and the surrogate gradient $g_s = \sum_j \sqrt{2\lambda_j / N_j}\,J_j^\top \varepsilon_{(j)}$ has

$$
\mathbb{E}[g_s g_s^\top] \;=\; \sum_j 2\lambda_j\,\tfrac{1}{N_j}\sum_k J_{j,k}^\top J_{j,k},
$$

the true multi-block GGN with the right $\lambda_j$ weights. Cross-block terms vanish exactly because $\varepsilon$ is drawn per coordinate, so no probe is shared across blocks. Higher-order input derivatives ($u_t$, $u_{xx}$, ...) built with `create_graph=True` differentiate back to $\theta$ through the stacked tensor unchanged.

The four PINN benchmarks in this repo (Poisson, Burgers, Kuramoto–Sivashinsky, and the Navier–Stokes inverse problem) all use this with **equal block weights** ($\lambda_j = 1$): no causal training, no grad-norm balancing, no hand-tuned loss weights.

## 9. What's not covered here

* **Adaptive block weighting.** Causal/temporal weighting and grad-norm balancing schemes are orthogonal to Gnome and are deliberately omitted so the optimizer comparison is clean. `stack_residuals` accepts static per-block $\lambda_j$ but does not adapt them during training.
* **Losses without a closed-form output-Hessian square root.** The whole construction hinges on writing $A$ with $A A^\top = H_y$ in closed form (§5). A loss whose intrinsic output Hessian is not analytically factorizable would need a different surrogate.
