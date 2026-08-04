# When Monte Carlo Curvature Estimation Loses to the Factorized Probe

**Status:** working note. The decomposition in §2 is classical; the threshold computations in §3–4 were derived in a working session and should be independently verified before publication.

## 1. Setup

Two unbiased single-draw estimators of the generalized Gauss–Newton (GGN) matrix
$G = J^\top H_y J$, both costing one backward pass:

- **MC (Fisher lineage):** sample a fake label from the model's own predictive
  distribution, $\tilde y \sim p(\cdot \mid \hat y)$, and use the score
  $u_{\mathrm{mc}} = -\nabla_{\hat y} \log p(\tilde y \mid \hat y)$
  (for softmax CCE: $u_{\mathrm{mc}} = p - e_{\tilde y}$).
- **Probe (GN lineage):** factor the output Hessian analytically, $H_y = AA^\top$,
  and use $u_{\mathrm{pr}} = AR$ with $R$ Rademacher
  (for softmax CCE: $A = \mathrm{diag}(\sqrt p)(I - \sqrt p\,\sqrt p^{\,\top})$).

Both satisfy $\mathbb{E}[uu^\top] = H_y$ per sample. The question is which
outer-product estimator has lower variance, and when.

## 2. The governing identity

For any mean-zero $u$ with covariance $H_y$ and fourth-cumulant tensor $\kappa$:

$$
\mathrm{Cov}(u_i u_j,\; u_k u_l) \;=\; \kappa_{ijkl} \;+\; \underbrace{H_{ik}H_{jl} + H_{il}H_{jk}}_{\text{Gaussian (Wick) part}} .
$$

Every estimator of the same $H_y$ shares the Wick part; **estimators differ only
through their fourth cumulant.** Consequences:

- **Rademacher probe:** sits at the kurtosis floor (per-coordinate excess
  cumulant $-2$, the minimum for unit-variance symmetric variables). Its
  variance is $\le$ the Gaussian probe's for every $H_y$, every factorization,
  every contraction.
- **Gaussian probe:** $\kappa \equiv 0$ by definition.
- **MC:** inherits the fourth cumulant of the *score under the model's own
  predictive distribution* — the sampler cannot discard the likelihood's higher
  cumulants. MC's variance relative to the Gaussian probe is exactly the sign
  and size of that cumulant.

So "MC is higher variance than the probe" $\iff$ the score's fourth cumulant
(suitably contracted) exceeds the probe's floor.

## 3. The exact scalar case (Bernoulli / BCE)

Score $v = y - p \in \{1-p, -p\}$ with variance $pq$ ($q = 1-p$). Excess kurtosis:

$$
\gamma_4(v) \;=\; \frac{1 - 6pq}{pq}.
$$

- $\gamma_4 > 0 \iff pq < 1/6 \iff p_{\max} \gtrsim 0.789$. Beyond this
  confidence, MC is worse than even the **Gaussian** probe.
- As $p_{\max} \to 1$: $\gamma_4 \sim 1/(pq) \to \infty$. MC's *relative*
  variance (per unit of shrinking signal) diverges like $1/(1 - p_{\max})$:
  the score is almost always $\approx 0$ and rarely $O(1)$ — naive Monte Carlo
  of a rare event.
- The Rademacher probe for BCE gives $u = \sqrt{pq}\,\varepsilon$, so
  $u^2 = pq$ **deterministically** — zero diagonal variance at every $p$.
  Against the Rademacher probe, scalar MC never wins.

## 4. Regimes where MC variance is greater (categorical / general)

MC's estimator tail is the score's tail; it is heavy exactly when the
predictive distribution **concentrates while retaining atomic support**.
Concretely, MC is the higher-variance estimator when:

1. **Late in training.** Mean confidence rises past the crossover
   ($p_{\max} \sim 0.7$–$0.8$ as the Bernoulli-calibrated landmark; the
   Rademacher crossover is somewhat earlier). This is the primary condition.
2. **On easy or memorized examples**, at any epoch — the condition is
   per-sample confidence, not training time.
3. **Large vocabularies with peaked predictions** (the LLM regime,
   $K_c \sim 10^4$–$10^5$). Most class directions are rare events for the label
   sampler; the probe excites all of them analytically at the same $O(K_c)$
   cost. Note this is also the regime where MC and the probe are the *only*
   tractable estimators — materializing, decomposing, or column-propagating
   $H_y$ is impossible at vocabulary scale.
4. **Severe class imbalance** — small minority-class probabilities give small
   $pq$ regardless of aggregate accuracy.
5. **Under EMA accumulation.** Curvature EMAs weight recent (confident)
   history, so the integrated comparison tips toward the probe earlier than the
   instantaneous one; MC's rare $O(1)$ spikes dominate the decaying average.

## 5. The converse (stated for honesty)

At **high entropy** the score's fourth cumulant is *negative*: near uniform $p$,
every MC realization has nearly constant norm ($|u|^2 = 1 - 1/K_c$ exactly at
uniform), making MC sub-Gaussian and better than the Gaussian probe
off-diagonal (by $\sim K/2$ at uniform $p$), while losing on the diagonal
(by $\sim K/3$). Dominance fails in both directions at the same $p$ — hence no
universal ordering exists; only the confidence-resolved statement is true.
This window occurs early in training, when curvature estimates matter least.
For the Gaussian likelihood (regression/MSE) all higher cumulants vanish: MC
reduces *exactly* to the Gaussian probe, and the Rademacher probe's advantage
is the classical diagonal term $2\sum_i M_{ii}^2$ — strict but modest.

## 6. One-line upshot

> MC label sampling and the factorized Rademacher probe estimate the same GGN
> at the same cost; their variances differ by the score's fourth cumulant,
> which is negative at high entropy, crosses zero near
> $p_{\max} \approx 0.8$ (exact for Bernoulli at $pq = 1/6$), and diverges as
> $1/(1-p_{\max})$ — so the probe dominates precisely where models spend most
> of training and all of convergence.

## 7. Caveats and verification plan

- Multivariate $\kappa$ is a tensor; "MC worse" holds where the relevant
  contractions through $J^\top A$-type maps are positive. The confident regime
  makes the dominant contractions positive, but claims should be phrased as
  decomposition + regime analysis, not a universal inequality (§5 is the
  counterexample).
- To verify before use: (i) the Rademacher fourth-moment correction term and
  its sign; (ii) the $pq = 1/6$ Bernoulli crossover; (iii) the uniform-$p$
  categorical moments in §5.
- Empirical check (self-designing figure): log the single-draw
  factor-estimate variance for MC vs. probe against mean $p_{\max}$ over a real
  CCE training run; theory predicts the ratio crosses 1 near the §3 threshold
  and grows without bound thereafter.
