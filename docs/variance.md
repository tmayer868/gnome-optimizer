# Variance of the surrogate GGN estimator and the role of the aux batch size $K$

This note answers one question: **when does increasing the auxiliary batch size $K$ reduce
the variance of the rank-1 GGN estimator $G_s = g_s g_s^\top$?** The answer is a direct
matrix generalization of the scalar Hutchinson result "kurtosis $> 3$": the relevant
quantity is a *multivariate* excess kurtosis of the per-sample probe vector, measured
against a Gaussian baseline. The scalar toy is recovered exactly as the $1$-dimensional
special case.

---

## 1. Setup and notation

Per aux sample $k$, write the (matrix-valued) signal and the probe vector

$$
B_k \;\triangleq\; J_k^\top A_k \in \mathbb{R}^{P \times c}, \qquad
w_k \;\triangleq\; B_k R_k = J_k^\top A_k R_k \in \mathbb{R}^{P},
$$

where $A_k A_k^\top = H_{y,k}$ is the output-Hessian square root, $R_k \in \mathbb{R}^{c}$ has
i.i.d. mean-zero, unit-variance entries (Rademacher by default), $P$ is the parameter count
and $c$ the output/probe dimension. The surrogate gradient and its rank-1 outer product are

$$
g_s \;=\; \frac{1}{\sqrt K}\sum_{k=1}^{K} w_k, \qquad
G_s \;=\; g_s g_s^\top .
$$

The two random sources are the probes $R_k$ and the aux data $(x_k,y_k)$ (which fixes $B_k$).
Across $k$ the $w_k$ are i.i.d. and **mean zero** — mean zero because $\mathbb{E}[R_k]=0$,
regardless of whether $B_k$ is centered. Their common covariance is the population GGN:

$$
\Sigma \;\triangleq\; \mathbb{E}\!\left[w_k w_k^\top\right]
\;=\; \mathbb{E}_x\!\left[J^\top A\,\mathbb{E}[RR^\top]\,A^\top J\right]
\;=\; \mathbb{E}_x\!\left[J^\top H_y J\right] \;=\; \text{GGN} \;=\; \bar G .
$$

Two scalar summaries of $\Sigma$ will appear throughout:

$$
\tau \;\triangleq\; \operatorname{tr}\Sigma, \qquad
\|\Sigma\|_F^2 \;=\; \operatorname{tr}(\Sigma^2).
$$

As established in the method note, $\mathbb{E}[G_s] = \Sigma$ for every $K \ge 1$: the
estimator is unbiased at all batch sizes, so **$K$ moves variance, not bias.** Everything
below is about that variance.

---

## 2. What "variance" means for a matrix estimator

Use the total (Frobenius) variance about the mean,

$$
V(K) \;\triangleq\; \mathbb{E}\,\big\|\,G_s - \Sigma\,\big\|_F^2 .
$$

Because $G_s = g_s g_s^\top$ is symmetric rank-1, $\|G_s\|_F^2 = \operatorname{tr}(G_s^2)= (g_s^\top g_s)^2 = \|g_s\|^4$, so the whole problem collapses onto a single fourth moment:

$$
\boxed{\,V(K) \;=\; \mathbb{E}\|g_s\|^4 \;-\; \|\Sigma\|_F^2\,}
$$

The subtracted term is $K$-independent, so **all** $K$-dependence lives in
$\mathbb{E}\|g_s\|^4$. This is the matrix counterpart of $\mathbb{E}[M_2^2] = \mathbb{E}\|g_s\|^4$
in the scalar toy.

---

## 3. The fourth moment

Expand using independence and mean-zero-ness of the $w_k$:

$$
\|g_s\|^4 \;=\; \frac{1}{K^2}\sum_{k,l,m,n}(w_k^\top w_l)(w_m^\top w_n).
$$

Any index that appears exactly once kills the term (factor out its mean, which is $0$). The
surviving index patterns are "all four equal" and "two disjoint pairs":

$$
\mathbb{E}\|g_s\|^4
= \frac{1}{K^2}\Big[\, K\,\mathbb{E}\|w\|^4
\;+\; K(K-1)\big(\,\mathbb{E}\|w\|^2\,\mathbb{E}\|w'\|^2 + 2\,\mathbb{E}[(w^\top w')^2]\,\big)\Big],
$$

with $w,w'$ independent copies. Two sub-identities, both using $\mathbb{E}[ww^\top]=\Sigma$:

$$
\mathbb{E}\|w\|^2 = \operatorname{tr}\Sigma = \tau, \qquad
\mathbb{E}[(w^\top w')^2] = \mathbb{E}\big[w^\top \Sigma\, w\big]
= \operatorname{tr}(\Sigma^2) = \|\Sigma\|_F^2 .
$$

Substituting and simplifying $\tfrac{K-1}{K} = 1 - \tfrac1K$:

$$
\mathbb{E}\|g_s\|^4
= \frac{\mathbb{E}\|w\|^4}{K}
\;+\; \frac{K-1}{K}\big(\tau^2 + 2\|\Sigma\|_F^2\big).
$$

Subtracting $\|\Sigma\|_F^2$ gives the headline result.

---

## 4. Main result

$$
\boxed{\;
V(K) \;=\; \underbrace{\big(\tau^2 + \|\Sigma\|_F^2\big)}_{\textstyle \Phi:\ K\to\infty\ \text{floor}}
\;+\; \frac{1}{K}\,\underbrace{\Big(\mathbb{E}\|w\|^4 - \tau^2 - 2\|\Sigma\|_F^2\Big)}_{\textstyle c:\ \text{excess-kurtosis term}}
\;}
$$

$V(K)$ is **monotone in $K$**: it is the floor $\Phi$ plus a $1/K$ term whose sign is the
sign of $c$. Hence

$$
V(K) < V(1)\ \ \text{for all } K>1
\quad\Longleftrightarrow\quad
c>0
\quad\Longleftrightarrow\quad
\mathbb{E}\|w\|^4 \;>\; \tau^2 + 2\|\Sigma\|_F^2 .
$$

Endpoints, for orientation:

$$
V(1) = \mathbb{E}\|w\|^4 - \|\Sigma\|_F^2,
\qquad
V(\infty) = \Phi = \tau^2 + \|\Sigma\|_F^2,
\qquad
V(1)-V(K) = c\,\frac{K-1}{K}.
$$

At $K=1$ the probe never fluctuates in norm (a single $R$ contributes only its sign), so the
entire variance is the spread of one probe vector, $\mathbb{E}\|w\|^4-\|\Sigma\|_F^2$. As $K$
grows you trade that single-sample spread (the $1/K$ piece) against an irreducible
cross-probe floor $\Phi$. Note $V(\infty)\neq 0$: $G_s$ stays rank-1 forever, so it never
converges to the full-rank GGN — it is the **EMA** of $g_s g_s^\top$ that does the averaging,
and a smaller $V(K)$ is exactly what feeds that EMA a cleaner per-step factor.

---

## 5. The scalar toy is the 1-D special case

Set $P=c=1$, so $w = XR$ with $X=B$ scalar, $R$ scalar Rademacher, and
$M_2 = \tfrac1K(\sum_i X_iR_i)^2 = g_s^2$. Then $\Sigma = \mathbb{E}[X^2]=\mu_2$,
$\tau=\mu_2$, $\|\Sigma\|_F^2=\mu_2^2$, and $\mathbb{E}\|w\|^4 = \mathbb{E}[X^4]=\mu_4$. The boxed
formula becomes

$$
\operatorname{Var}[M_2]
= \big(\mu_2^2 + \mu_2^2\big) + \frac{1}{K}\big(\mu_4 - 3\mu_2^2\big)
= 2\mu_2^2 + \frac{\mu_4 - 3\mu_2^2}{K}
= \mu_2^2\!\left(2 + \frac{\kappa-3}{K}\right),
$$

with $\kappa = \mu_4/\mu_2^2$ the kurtosis of $X$. So $\operatorname{Var}[M_2]$ decreases in $K$
iff $\kappa>3$ — your stated result — and the "$3$" is the Gaussian kurtosis. The matrix case
keeps this structure; the only thing that changes is *what plays the role of "$3$".*

---

## 6. The generalized kurtosis: "$3$" becomes the Gaussian baseline

The threshold $\tau^2 + 2\|\Sigma\|_F^2$ is exactly the fourth moment of a **Gaussian** vector
with the same covariance. For $\eta \sim \mathcal N(0,\Sigma)$,

$$
\mathbb{E}\|\eta\|^4 = \mathbb{E}\big[(\eta^\top\eta)^2\big]
= (\operatorname{tr}\Sigma)^2 + 2\operatorname{tr}(\Sigma^2)
= \tau^2 + 2\|\Sigma\|_F^2 .
$$

Define the **generalized kurtosis** of the probe vector

$$
\bar\kappa \;\triangleq\; \frac{\mathbb{E}\|w\|^4}{\tau^2 + 2\|\Sigma\|_F^2}
\;=\; \frac{\mathbb{E}\|w\|^4}{\mathbb{E}\|\eta\|^4}\Big|_{\eta\sim\mathcal N(0,\Sigma)} .
$$

Then $c>0 \iff \bar\kappa>1$, and the rule reads identically to the scalar one with the number
$3$ replaced by "the Gaussian with matching covariance":

$$
\boxed{\;K>1 \text{ reduces } V \;\Longleftrightarrow\; \bar\kappa>1
\;\Longleftrightarrow\; w \text{ is heavier-}\|\cdot\|^4\text{ than its Gaussian.}\;}
$$

This is Mardia-style multivariate excess kurtosis: $\bar\kappa>1$ is positive excess,
$\bar\kappa=1$ is the Gaussian/flat case ($V$ independent of $K$), $\bar\kappa<1$ means
$K=1$ is variance-optimal and larger $K$ *hurts* the rank-1 Frobenius variance.

---

## 7. What actually drives $\bar\kappa$: data heterogeneity vs. probe noise

To see *why* $\bar\kappa>1$ holds in practice, split the probe and data randomness. Condition
on the batch; the Rademacher quadratic-form fourth moment is, with $M = B^\top B$,

$$
\mathbb{E}_R\big[\|w\|^4 \,\big|\, B\big]
= (\operatorname{tr}M)^2 + 2\operatorname{tr}(M^2) - 2\!\sum_i M_{ii}^2 ,
$$

where the $-2\sum_i M_{ii}^2$ is precisely the Rademacher advantage (a Gaussian probe would
have $+0$ here; this is the "lowest fourth moment" property the method note cites). Taking
$\mathbb{E}_B$ and using $BB^\top = J^\top H_y J = \text{GGN}_x$ (the **per-sample** GGN, mean
$\Sigma$) and $\operatorname{tr}(BB^\top)=\operatorname{tr}(\text{GGN}_x)$, the excess term
$c = \mathbb{E}\|w\|^4 - (\tau^2+2\|\Sigma\|_F^2)$ decomposes cleanly:

$$
\boxed{\;
c \;=\;
\underbrace{\operatorname{Var}_x\!\big[\operatorname{tr}(\text{GGN}_x)\big]}_{\ge 0}
\;+\;
\underbrace{2\,\mathbb{E}_x\big\|\text{GGN}_x - \Sigma\big\|_F^2}_{\ge 0}
\;-\;
\underbrace{2\,\mathbb{E}_x\!\Big[\textstyle\sum_i (B^\top B)_{ii}^2\Big]}_{\ge 0\ \text{(Rademacher gain)}}
\;}
$$

So $K>1$ helps exactly when

$$
\underbrace{\operatorname{Var}_x\!\big[\operatorname{tr}(\text{GGN}_x)\big]
+ 2\,\mathbb{E}_x\|\text{GGN}_x - \Sigma\|_F^2}_{\text{how much curvature varies across data}}
\;>\;
\underbrace{2\,\mathbb{E}_x\!\Big[\textstyle\sum_i (A^\top JJ^\top A)_{ii}^2\Big]}_{\text{residual within-sample probe variance}} .
$$

Reading: the first two terms are **curvature heterogeneity across the data** — how much each
example's GGN, in trace and in full Frobenius shape, deviates from the population GGN. The
last term is the **leftover probe noise** that Rademacher could not cancel inside a single
sample. Increasing $K$ averages over the data, so it pays off precisely when the data are
curvature-heterogeneous relative to the residual probe noise — which is the generic situation
in deep learning, where different examples carry very different curvature. The Rademacher
choice shrinks the right-hand side (that is its whole point), which both lowers $V$ overall
and *lowers the bar* at which $K>1$ becomes worthwhile.

---

## 8. Reading it back to Gnome

* **Each aux sample carries one probe.** Increasing $K$ buys more data coverage *and* more
  probes simultaneously; you cannot tune them separately in the current construction. The
  decomposition in §7 says which one you are really paying for: if the data term dominates,
  more samples (larger $K$) is the right lever; if the probe term dominates, Rademacher has
  already done most of the work and the marginal value of $K$ is small.

* **The variance is monotone, so the largest marginal gain is $K:1\to2$,** with total
  reducible variance $c = V(1)-V(\infty)$. Past that, returns diminish as $1/K$, which is the
  quantitative justification for a small default aux batch rather than a large one.

* **$V(\infty)=\Phi>0$ is not a failure.** $G_s$ is rank-1 by construction; the EMA over steps
  is what accumulates rank and averages the floor away. Lower $V(K)$ means each $g_sg_s^\top$
  handed to the EMA is closer to the GGN, so the Kronecker factors — and the eigenbasis built
  from them — settle faster and cleaner.

* **MSE check.** With $A=\sqrt2\,I$ the per-sample GGN is $2J^\top J$, the residual probe term
  is $4\,\mathbb{E}_x\sum_i\|J_{i,:}\|^4$ (rows of the Jacobian), and $\bar\kappa>1$ reduces to
  Jacobian-curvature heterogeneity across the batch outrunning that row-norm term — the same
  story, specialized.

---

### One-line summary

$V(K) = (\tau^2+\|\Sigma\|_F^2) + \tfrac1K\big(\mathbb{E}\|w\|^4 - \tau^2 - 2\|\Sigma\|_F^2\big)$,
so $K>1$ lowers the GGN-estimator variance **iff** the probe vector $w=J^\top A R$ has
positive multivariate excess kurtosis ($\bar\kappa>1$) — the exact generalization of
"kurtosis $>3$," with the Gaussian-with-matching-covariance fourth moment $\tau^2+2\|\Sigma\|_F^2$
playing the role of the scalar's $3\mu_2^2$.
