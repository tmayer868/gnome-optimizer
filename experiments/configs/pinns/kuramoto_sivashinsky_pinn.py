"""Settings for experiments.pinns.kuramoto_sivashinsky_pinn. Edit CONFIG, then run the module.

None keeps an existing automatic/default-dependent setting where supported.
"""

CONFIG = {
    # Choices: 'gnome', 'soap', 'adamw', 'adamw+lbfgs'.
    'optimizer': 'gnome',
    # Network: plain tanh MLP or the gated modified MLP (Wang et al. 2021). hidden / depth
    # control both. Defaults to mlp — the original setup for this experiment, so existing runs
    # stay comparable. Choices: 'mlp', 'modified'.
    'arch': 'mlp',
    # Input embedding. 'none' feeds raw [t, x] and keeps the soft periodic BC block. 'periodic'
    # feeds [t, cos(2pi x/L), sin(2pi x/L)], making the network exactly period-L in x —
    # periodicity then holds for every derivative and the BC block is DROPPED (two blocks instead
    # of three). Choices: 'none', 'periodic'.
    'embed': 'none',
    'seed': 0,
    'steps': 20000,
    'n_pde': 4000,
    'n_ic': 200,
    'n_bc': 200,
    # Aux batch sizes for Gnome are max(K_min, int(N * aux_frac)) per block. Each aux pass is a
    # full higher-order residual eval, so this is not free — keep small. KS's 4th-order PDE makes
    # the aux pass ~2x more expensive than Burgers per point.
    'aux_frac': 0.1,
    'lr': 0.001,
    # Gnome per-coordinate update bound: lambda is set to the smallest value with
    # max|m̂/(v̂+lambda)| <= this, so no coordinate moves more than lr*trust_region in a step.
    # Larger -> weaker bound -> longer steps. 0 disables it, falling back to plain m̂/(v̂+eps)
    # damping.
    'trust_region': 1.0,
    # Gnome curvature-damping epsilon in m̂/(v̂+eps): larger -> more gradient-descent-like,
    # smaller -> fuller Newton step. Gnome only; SOAP/AdamW keep their fixed eps=1e-8.
    'eps': 1e-6,
    # First-moment (momentum) EMA for Gnome and SOAP.
    'beta1': 0.9,
    # Second-moment / preconditioner EMA (also shampoo_beta) for Gnome and SOAP.
    'beta2': 0.99,
    'weight_decay': 1e-8,
    # Network width. The stiff KS operator needs more capacity than Burgers; 128 is a reasonable
    # default.
    'hidden': 128,
    # Network depth: Linear-layer count for arch mlp, number of gated hidden layers for arch
    # modified.
    'depth': 6,
    # Linear LR warmup steps, applied to every optimizer.
    'warmup_steps': 200,
    # Final-LR fraction for the baseline cosine decay: 0.0 decays to zero (standard treatment),
    # 1.0 disables decay. Gnome (MSE) never decays regardless.
    'cosine_decay': 0.0,
    # L-BFGS OUTER steps after the AdamW phase (optimizer adamw+lbfgs only). Each outer step runs
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
