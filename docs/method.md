# Gnome: Gauss-Newton Optimizer via Matrix Eigen-decomposition

Gnome is a second-order PyTorch optimizer that extends [SOAP](https://arxiv.org/abs/2409.11321) into a true Gauss-Newton method. It keeps SOAP's Kronecker-factored eigenbasis machinery but changes the signal those factors are built from, the update rule inside the rotated basis, and how step size is controlled.

This document describes the algorithm in detail: how it builds off SOAP, the role of the auxiliary surrogate function, and the concrete construction of that surrogate for each supported loss.

---

## 1. Background: SOAP in one paragraph

SOAP runs Adam inside a rotated parameter basis derived from the Kronecker factors of the loss gradient. For a 2D parameter tensor $G \in \mathbb{R}^{m \times n}$ with per-step gradient $g_t \in \mathbb{R}^{m \times n}$, SOAP maintains an EMA of two outer products

$$
L_t = \beta \, L_{t-1} + (1-\beta)\, g_t g_t^\top, \qquad
R_t = \beta \, R_{t-1} + (1-\beta)\, g_t^\top g_t,
$$

periodically diagonalizes them ($L = Q_L \Lambda_L Q_L^\top$, $R = Q_R \Lambda_R Q_R^\top$), rotates the gradient into the joint eigenbasis ($\tilde g = Q_L^\top g \, Q_R$), runs Adam there ($\tilde m$, $\tilde v$ EMAs, update $\tilde m / (\sqrt{\tilde v} + \varepsilon)$), and rotates the result back. The Kronecker structure is what makes the eigendecompositions tractable for large weight matrices; the rotation is what gives Adam a basis in which the curvature is approximately axis-aligned.

The piece of SOAP we want to inherit is the Kronecker-eigenbasis machinery itself. The piece we want to replace is the **signal** it is fed — SOAP's $L_t$ and $R_t$ are built from past loss gradients, i.e. an empirical-Fisher proxy.

## 2. From SOAP to Gnome: two changes

Gnome makes two changes to the SOAP recipe.

**(a) Build the Kronecker factors from a GGN estimate, not from the loss gradient.**

Empirical Fisher (gradient outer products) is a known-poor curvature proxy for regression losses, and it conflates noise with curvature in classification. We replace the loss gradient $g_t$ in the factor update with a *surrogate gradient* $g_s$ obtained from a small auxiliary batch and a randomized sketch of the loss's output Hessian. The surrogate is constructed so that

$$
\mathbb{E}\!\left[g_s\, g_s^\top\right] \;=\; J_\theta^\top H_y J_\theta \;=\; \text{GGN},
$$

where $J_\theta = \partial \hat y / \partial \theta$ is the model Jacobian and $H_y = \partial^2 \ell / \partial \hat y^2$ is the intrinsic output Hessian of the loss. The Kronecker factors then become EMAs of $g_s g_s^\top$ and $g_s^\top g_s$, which in expectation are the Kronecker factors of the GGN itself.

The factor refresh schedule, eigendecomposition, dim-merging, and projection logic are all carried over from SOAP unchanged.

**(b) Newton step inside the rotated basis; clip as trust region.**

SOAP's rotated-basis update is $\tilde m / (\sqrt{\tilde v} + \varepsilon)$. Gnome's is

$$
\Delta \tilde\theta \;=\; \frac{\hat m}{\hat v + \varepsilon}
$$

— the un-square-rooted denominator. This is the diagonal Gauss-Newton update inside the eigenbasis: $\hat v$ is the second-moment EMA of the rotated surrogate gradient, which in expectation is the diagonal of the GGN in the eigenbasis, so dividing the rotated loss gradient by $\hat v$ is the per-coordinate Newton step.

A pure Newton step has no built-in step-size control. We bound the per-coordinate update with `clip` (default 1.0) in *both* the rotated basis (before un-rotating) and the parameter basis (after un-rotating). At sensible learning rates the clip almost never binds; its role is to prevent blow-ups from small denominators while the eigenbasis is warming up, not to be the primary step-size knob. `eps` retains its physical meaning as a curvature *damping* term (larger pulls toward gradient descent, smaller toward pure Newton), not as Adam's numerical-safety constant.

The dimensional argument for the un-square-rooted denominator: if we model the rotated gradient as $\tilde g \sim \mathcal{N}(0, \tilde H)$ with $\tilde H$ the rotated GGN, then $\mathbb{E}[\tilde g^2] = \tilde H$. The Newton-rescaled coordinate has units $[\text{grad}] / [\text{curvature}]$ — i.e. $\tilde g / \tilde H$. Adam's $\sqrt{\tilde v}$ has the wrong units ($\sqrt{[\text{curvature}]}$); it is a heuristic for the gradient *norm*, not the curvature. Gnome's $\hat v$ has the right units.

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

The crucial observation: for every loss in §5, $H_y$ depends only on the model output $\hat y$ (and the target, in some cases) — *not* on $\theta$. So the square root $A$ inherits the same property and can be written in closed form for each loss family. **No second-order autograd is required at any point**; we never differentiate $H_y$ through $\theta$.

**Step 2: Express $J^\top A$ as a gradient.** Treating $A$ as detached from autograd, define

$$
z \;=\; \hat y^\top A \quad \text{(a row vector of length $\dim A$)}.
$$

Then $J^\top A = \partial z / \partial \theta$ — precisely what `autograd.grad` returns when differentiating $z$'s components w.r.t.\ $\theta$.

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

This equals the GGN $J^\top A A^\top J$ **if and only if** $\mathbb{E}[R_i R_j] = \delta_{ij}$ — i.e.\ $R$ has uncorrelated unit-variance components. Two cheap distributions satisfy this:

- **Rademacher**: $R_i \in \{\pm 1\}$ independently with equal probability.
- **Unit Gaussian**: $R_i \sim \mathcal{N}(0, 1)$ iid.

Both yield unbiased GGN estimators. The Rademacher choice has strictly lower variance (the standard Hutchinson estimator argument: same first two moments, lowest fourth moment) and is the default in Gnome.

**Step 5: Average over the aux batch.** The aux batch contains $K$ samples; each has its own $\hat y_k$, its own output Hessian factor $A_k$ (which generally depends on $\hat y_k$ and so varies per sample — see §5.3 for the CCE case), and its own independent random vector $R_k$. The surrogate sums them with a $1/\sqrt K$ normalization:

$$
S \;=\; \frac{1}{\sqrt K} \sum_{k=1}^{K} \hat y_k^\top A_k R_k,
\qquad
g_s \;=\; \frac{1}{\sqrt K} \sum_{k=1}^{K} J_k^\top A_k R_k.
$$

Taking expectations — with each $R_k$ independent across $k$, so $\mathbb{E}[R_k R_j^\top] = I \cdot \delta_{kj}$ — gives the empirical GGN over the aux batch:

$$
\mathbb{E}_R\!\left[g_s g_s^\top\right] \;=\; \frac{1}{K} \sum_{k=1}^{K} J_k^\top H_{y,k} J_k.
$$

Taking the further expectation over the aux batch sampling $(x_k, y_k) \sim \text{data}$ recovers the population GGN:

$$
\mathbb{E}_{R,\,\text{batch}}\!\left[g_s g_s^\top\right] \;=\; \mathbb{E}_{x,y}\!\left[J^\top H_y J\right] \;=\; \text{GGN}, \qquad \text{for any } K \ge 1.
$$

The estimator is unbiased at every batch size. $K$ controls **variance**, not bias: with the $1/\sqrt K$ scaling, $\mathbb{E}\|g_s\|^2$ is independent of $K$ but $\mathrm{Var}(g_s)$ decreases as $K$ grows. The $1/\sqrt K$ normalization thus decouples the surrogate's scale from `aux_batch_size` entirely, so `eps` and `clip` calibrate against a fixed-scale surrogate regardless of $K$. (`tests/test_surrogate_scaling.py` pins this empirically across $K \in \{1,4,16,64\}$.)

The explicit per-sample sum is the uniform notation that handles all three losses in §5. For MSE and BCE, $A_k$ doesn't depend on $k$ (it's either constant or a per-element function of detached logits), and the sum collapses into a single inner product over `(batch × output_dim)` axes — both losses can be written in batched-tensor form. For CCE, $A_k$ varies per sample through the softmax probabilities $p_k$, so the sum is genuinely a per-sample operation.

So the entire construction reduces to: **given a loss, write down a closed-form square root $A$ of its output Hessian.** Section 5 collects $A$ for each loss family. The optimizer currently exposes three via the `loss=` argument — `'mse'`, `'cce'`, and `'cce_hutchinson'`; BCE is included below to show the recipe generalizes (it is the diagonal special case of CCE), not as a shipped option.

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
- The main loss is built internally as `((y_hat - y)**2).sum() / B` (sum over output dim, mean over batch). This matches the per-element $H_y = 2I$ contract — using `mean` over both dims would divide $g_{\text{main}}$ by the output dim while the surrogate scale is per-element, breaking the relative scale that `eps` and `clip` are calibrated against.
- No double backward is needed. Knowing $H_y = 2I$ in closed form means we don't have to call any second-order autograd primitive — just multiply logits by random signs and `autograd.grad` through.

### 5.2 BCE (binary cross-entropy with logits)

*Not a shipped `loss=` option — described here for completeness, since it is the diagonal special case of the CCE surrogate in §5.3.*

Per-element: predict logit $z$, target $y \in \{0, 1\}$, with $p = \sigma(z)$ and

$$
\ell \;=\; -y \log p - (1-y) \log(1-p).
$$

Derivatives of $\ell$ in the logit:

$$
\frac{\partial \ell}{\partial z} \;=\; p - y, \qquad
\frac{\partial^2 \ell}{\partial z^2} \;=\; p(1-p).
$$

So $H_y$ is diagonal with entries $p_i(1 - p_i)$ — the per-element Bernoulli variance. The natural square root is $A = \mathrm{diag}(\sqrt{p(1-p)})$. Drawing per-element Rademacher signs $\varepsilon$:

$$
v \;=\; \sqrt{p(1-p)} \odot \varepsilon, \qquad
S_{\text{BCE}} \;=\; \frac{1}{\sqrt K} \sum_{k=1}^{K} \langle z_k, \mathrm{detach}(v_k) \rangle.
$$

The gradient is $g_s = (1/\sqrt K) \sum_k J_k^\top v_k$, and

$$
\mathbb{E}[g_s g_s^\top] \;=\; \frac{1}{K} \sum_k J_k^\top \, \mathrm{diag}(p_k(1-p_k)) \, J_k \;\to\; \text{GGN}_{\text{BCE}}.
$$

Construction notes:
- $p$ must be `detach`ed before being used to build $v$. The gradient should flow through $z$ (the logits), not through $p$ — $p$ enters only as a scaling factor for the Rademacher probe. Otherwise we'd be sampling an estimator of $J^\top (\partial_\theta H_y) J + J^\top H_y J$, not the GGN.
- Like MSE, no double backward: $H_y$ is known in closed form once we have $p$.
- The construction is the diagonal special case of the CCE surrogate below (with $K_{\text{classes}} = 2$ and the simplex-projection term vanishing).

### 5.3 CCE (categorical cross-entropy)

Per-sample: $K_c$-way softmax cross-entropy. Output Hessian:

$$
H_y \;=\; \mathrm{diag}(p) - p p^\top,
$$

which is non-diagonal — but admits a closed-form square root computable in $O(K_c)$:

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

where the second equality uses $\mathrm{diag}(\sqrt p)\, \sqrt p = \sqrt p \odot \sqrt p = p$. This is $O(K_c)$ work per sample (no explicit $K_c \times K_c$ matrix is ever materialized) and plugs straight into the §4 recipe with $\mathbb{E}[(AR)(AR)^\top] = H_y$ *exactly* per sample — not just in expectation over labels.

**Aside (MC alternative).** The Fisher / K-FAC tradition replaces the square-root factorization with Monte Carlo label sampling: draw a fake label $\tilde y_k \sim \mathrm{Categorical}(p_k)$ from the model's own predictive distribution and use $\mathtt{F.cross\_entropy}(z_k, \tilde y_k)$ as the surrogate. Its logit gradient is $v = p - e_{\tilde y}$, and since $\mathbb{E}[e_{\tilde y}] = p$,

$$
\mathbb{E}_{\tilde y \sim p}\big[v v^\top\big] \;=\; \mathrm{Cov}(e_{\tilde y}) \;=\; \mathrm{diag}(p) - p p^\top \;=\; H_y ,
$$

so the outer products have the correct GGN expectation with no factorization of $H_y$ involved — the label randomness plays the role of the Rademacher probe, with the softmax's own covariance standing in for $A A^\top$. We use Rademacher Hutchinson instead for consistency with the §4 recipe and for lower variance — Hutchinson's $AR$ uses every class simultaneously, while MC sampling's $p - e_{\tilde y}$ is a rank-1 contribution aligned with a single class direction per sample.

**Variance behaviour.** As the model becomes confident ($p$ approaches a one-hot vector), $\sqrt p \odot R$ collapses to a near-one-hot vector and the simplex-projection term $(\sqrt p \cdot R)\, p$ cancels the surviving coordinate. The variance of the estimator decays toward zero with the Gini impurity of $p$, so late in training — exactly when classification gradients are sharp and the MC alternative's single-class probe becomes most wasteful — Hutchinson approaches a noise-free preconditioner.

**Implementation traps.** Two ways to silently break the estimator:

1. **Broadcasting.** $R$ must be an independent $\{\pm 1\}^{K_c}$ sample *per (sample, class)*. Sharing one $K_c$-vector across the batch makes $\mathbb{E}[R_n R_m^\top] = I \ne 0$ for $n \ne m$, injecting a covariance bias between unrelated samples that destroys the batched-GGN identity.
2. **BatchNorm.** The per-sample GGN sum assumes the Jacobian of sample $n$ does not depend on sample $m$. BatchNorm couples the forward pass across the batch dim, breaking the block-diagonal structure of the true batched Hessian. Use GroupNorm or LayerNorm in any architecture trained under `cce_hutchinson` (or `cce`).

## 6. The full step

For each minibatch, with `main_closure` returning $(\hat y, y)$ on the $B-K$ main slice and `aux_closure` returning $(\hat y_{\text{aux}}, y_{\text{aux}})$ on the disjoint $K$ aux slice:

1. **Main forward + backward.** Compute the main loss internally (`mse` → sum-over-D-mean-over-B, `cce` / `cce_hutchinson` → `reduction='mean'`). Gradient $g_{\text{main}} = \nabla_\theta \ell_{\text{main}}$ via `autograd.grad`. Free the main graph.
2. **Aux forward + backward.** Build $S$ per §5 from the aux logits and sampled randomness. Gradient $g_s = \nabla_\theta S$. Free the aux graph.
3. **Per-parameter update.** For each parameter tensor:
   - Update the Kronecker factors $L, R$ with EMA from $g_s g_s^\top$ and $g_s^\top g_s$ (`shampoo_beta`).
   - If on the refresh schedule, recompute the eigenbases $Q_L, Q_R$ (full `eigh` on first build, one step of subspace iteration + QR thereafter).
   - Project $g_{\text{main}}$ and $g_s$ into the rotated basis: $\tilde g = Q_L^\top g\, Q_R$.
   - Update the first-moment EMA from $\tilde g_{\text{main}}$ (`beta1`) and the second-moment EMA from $\tilde g_s^{\,2}$ (`beta2`).
   - Compute the rotated Newton step: $\Delta \tilde\theta = \hat m / (\hat v + \varepsilon)$, clip per-coordinate to $\pm$`clip`.
   - Project back to the parameter basis; clip again.
   - Apply with $-\text{lr}_{\text{eff}}$ (linear warmup over `warmup` steps), then decoupled weight decay.

Two independent forward passes (one for main, one for aux) cost more than a single shared forward, but mean we don't need `retain_graph=True` and don't have to hold the main batch's activations alive while running the surrogate backward. The aux batch is small (default $K=10$), so the second pass is cheap.

`grad_m` (a vector quantity in the rotated basis) is translated through eigenbasis refreshes: it is rotated back to the parameter basis before the new $Q$ is computed and projected forward into the new basis afterwards. `gnd_m` (a diagonal variance) is permuted along the corresponding tensor axis to match the new eigenvalue ordering but is not re-rotated, matching SOAP's treatment of `exp_avg_sq`.

## 7. Why the un-square-rooted denominator (again, briefly)

Suppose the rotated GGN is approximately diagonal with diagonal $d$. The second-moment EMA of the rotated surrogate gradient satisfies

$$
\mathbb{E}\!\left[\tilde g_s^{\,2}\right]_i \;=\; \mathbb{E}\!\left[(Q^\top g_s)_i^2\right] \;=\; (Q^\top \text{GGN}\, Q)_{ii} \;\approx\; d_i.
$$

So $\hat v \approx d$, and $\hat m / \hat v$ is the per-coordinate Newton step in the rotated basis. The Adam denominator $\sqrt{\hat v}$ has the wrong units for a Newton step (it scales like $\sqrt{\text{curvature}}$, not curvature), which is why Adam needs $\varepsilon$ as a numerical-safety constant calibrated against the gradient norm — and why moving curvature from "noisy outer product of gradients" to "true GGN estimate" only pays off when we also move the denominator from $\sqrt{\hat v}$ to $\hat v$.

## 8. Multi-block losses (PINNs)

A PINN loss is a weighted sum of per-block mean-squared residuals,
$L = \sum_j \lambda_j \, \mathrm{mean}_i(r_{j,i}^2)$ — one block for the PDE residual on collocation points, one for the initial condition, one per boundary condition, and (for inverse problems) one per data-fit term. Because every block is an MSE, the whole loss rides the §5.1 MSE surrogate unchanged: `gnome.stack_residuals` folds the blocks into a single flat residual vector whose plain `mean(·²)` reproduces $L$ exactly, and the automatic $\sqrt 2\,\varepsilon$ Rademacher probe on that vector decomposes into the per-block independent GGN estimator.

Concretely, scaling block $j$ by $\alpha_j = \sqrt{\lambda_j N / N_j}$ (with $N_j$ the block's element count and $N = \sum_j N_j$) gives $\mathrm{mean}(\text{stacked}^2) = \sum_j \lambda_j\,\mathrm{mean}_i(r_{j,i}^2)$, and the surrogate gradient $g_s = \sum_j \sqrt{2\lambda_j / N_j}\,J_j^\top \varepsilon_{(j)}$ has

$$
\mathbb{E}[g_s g_s^\top] \;=\; \sum_j 2\lambda_j\,\tfrac{1}{N_j}\sum_k J_{j,k}^\top J_{j,k},
$$

the true multi-block GGN with the right $\lambda_j$ weights. Cross-block terms vanish exactly because $\varepsilon$ is drawn per coordinate, so no probe is shared across blocks. Higher-order input derivatives ($u_t$, $u_{xx}$, ...) built with `create_graph=True` differentiate back to $\theta$ through the stacked tensor unchanged.

The four PINN benchmarks in this repo — Poisson, Burgers, Kuramoto–Sivashinsky, and the Navier–Stokes inverse problem — all use this with **equal block weights** ($\lambda_j = 1$): no causal training, no grad-norm balancing, no hand-tuned loss weights.

## 9. What's not covered here

* **Adaptive block weighting.** Causal/temporal weighting and grad-norm balancing schemes are orthogonal to Gnome and are deliberately omitted so the optimizer comparison is clean. `stack_residuals` accepts static per-block $\lambda_j$ but does not adapt them during training.
* **Losses without a closed-form output-Hessian square root.** The whole construction hinges on writing $A$ with $A A^\top = H_y$ in closed form (§5). A loss whose intrinsic output Hessian is not analytically factorizable would need a different surrogate.

