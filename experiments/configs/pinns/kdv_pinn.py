"""Settings for experiments.pinns.kdv_pinn. Edit CONFIG, then run the module.

None keeps an existing automatic/default-dependent setting where supported.
"""

CONFIG = {
    # Choices: 'gnome', 'soap', 'adamw', 'adamw+lbfgs', 'engd'.
    'optimizer': 'gnome',
    # Network: plain tanh MLP or the gated modified MLP (Wang et al. 2021). hidden / depth
    # control both. Choices: 'mlp', 'modified'.
    'arch': 'modified',
    # Input embedding: 'periodic' (default) feeds [t, cos(pi x), sin(pi x)]; 'none' feeds raw [t,
    # x]. Both retain the periodic BC loss for u, u_x, u_xx. Choices: 'periodic', 'none'.
    'embed': 'periodic',
    'seed': 0,
    # KdV is dispersive/stiff; may want more than default.
    'steps': 100000,
    'n_pde': 4000,
    'n_ic': 200,
    'n_bc': 200,
    # Dispersive u_xxx coefficient. Defaults to 0.022^2, the JAX-PI benchmark value, which loads
    # kdv.mat. A matching reference is generated for other values.
    'mu_sq': 0.022 ** 2,
    # Odd number of unique periodic Fourier points used for the generated reference.
    'reference_modes': 511,
    # ETDRK4 substeps per 0.005 reference-output interval.
    'reference_substeps': 20,
    # Aux batch sizes for Gnome are max(K_min, int(N * aux_frac)) per block. Each aux pass is a
    # full higher-order residual eval, so keep small.
    'aux_frac': 0.03,
    'lr': 0.001,
    # Gnome per-coordinate update bound: lambda is set to the smallest value with
    # max|m̂/(v̂+lambda)| <= this, so no coordinate moves more than lr*trust_region in a step.
    # Larger -> weaker bound -> longer steps. 0 disables it, falling back to plain m̂/(v̂+eps)
    # damping.
    'trust_region': 1.0,
    # Gnome curvature-damping epsilon in m̂/(v̂+eps): larger -> more gradient-descent-like,
    # smaller -> fuller Newton step. Gnome only; SOAP/AdamW keep eps=1e-8.
    'eps': 1e-6,
    # First-moment (momentum) EMA for Gnome and SOAP.
    'beta1': 0.99,
    # Second-moment / preconditioner EMA (also shampoo_beta) for Gnome and SOAP.
    'beta2': 0.999,
    'weight_decay': 1e-8,
    # Network width.
    'hidden': 256,
    # Network depth: Linear-layer count for arch mlp, number of gated hidden layers for arch
    # modified.
    'depth': 4,
    # Linear LR warmup steps, applied to every optimizer.
    'warmup_steps': 200,
    # Final-LR fraction for the baseline cosine decay: 0.0 decays to zero (standard treatment),
    # 1.0 disables decay. Gnome (MSE) never decays regardless.
    'cosine_decay': 0.0,
    # L-BFGS OUTER steps after the Adam phase (optimizer adamw+lbfgs only). Each outer step runs
    # up to lbfgs_max_iter inner iterations, so the total L-BFGS budget is lbfgs_steps *
    # lbfgs_max_iter. Runs full-batch on a fixed collocation set.
    'lbfgs_steps': 500,
    # Inner L-BFGS iterations per outer step. Must be >1: the strong-Wolfe line search cannot
    # recover from a cold (identity) Hessian in a single iteration and stalls. adamw+lbfgs only.
    'lbfgs_max_iter': 20,
    # L-BFGS history size (number of stored curvature pairs). adamw+lbfgs only.
    'lbfgs_history': 50,
    # L-BFGS learning rate; with the strong-Wolfe line search 1.0 is standard. adamw+lbfgs only.
    'lbfgs_lr': 1.0,
    # ENGD step size (line-searched from here). engd only.
    'engd_lr': 1.0,
    # ENGD Levenberg-Marquardt damping λ (adapts). Smaller -> fuller Gauss-Newton step / higher
    # attainable accuracy but worse conditioning. engd only.
    'engd_damping': 1e-8,
    # Run in double precision on CPU. Strongly recommended for optimizer engd: the energy-metric
    # Gauss-Newton solve targets accuracy the float32 floor (~1e-6) can't express. Forces
    # device=cpu (MPS has no float64).
    'float64': False,
    # Log Gnome's internal state — curvature spectrum, LM damping, trust-region usage — every N
    # steps to a sibling runs/.../{run_id}.diag.jsonl. 0 (default) disables it entirely. Gnome
    # only: SOAP and AdamW expose no such hook.
    'diagnostics_every': 0,
    # Comma-separated parameter indices to log, e.g. '0,4'. Default logs every parameter, which
    # is one record per tensor per logged step — narrow it to keep the file readable.
    'diagnostics_params': None,
    'log_every': 200,
    'runs_dir': 'runs',
    'quiet': False,
}
