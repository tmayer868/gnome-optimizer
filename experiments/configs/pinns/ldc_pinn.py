"""Settings for experiments.pinns.ldc_pinn. Edit CONFIG, then run the module.

None keeps an existing automatic/default-dependent setting where supported.
"""

CONFIG = {
    # Choices: 'gnome', 'soap', 'adamw+lbfgs'.
    'optimizer': 'gnome',
    # Network: plain tanh MLP or the gated modified MLP (Wang et al. 2021). hidden / depth
    # control both. Choices: 'mlp', 'modified'.
    'arch': 'modified',
    'seed': 0,
    # Reynolds-number curriculum, as a Python list. Each stage warm-starts from the previous one's
    # weights AND optimizer state. Every value needs a reference solution; available: (100, 400,
    # 1000, 1600, 3200, 5000).
    're_stages': [100, 400, 1000, 3200, 5000],
    # Steps per curriculum stage, as a Python list. A single value applies to every stage except
    # the last, which gets final_stage_steps.
    'stage_steps': [20000],
    # Steps for the last curriculum stage when stage_steps is a single value. The final Re is
    # where the accuracy is actually earned.
    'final_stage_steps': 140000,
    # Lid velocity profile. 'regularized' is jaxpi's 1-cosh(50(x-1/2))/cosh(25), which removes
    # the corner singularity but does NOT match the reference data (generated with a uniform lid)
    # in two thin corner strips. 'uniform' matches the reference exactly and reinstates the
    # singularity. Choices: 'regularized', 'uniform'.
    'lid': 'regularized',
    # Static per-block loss weights λ_j, as a Python list in ru,rv,rc,u_bc,v_bc order, giving the
    # loss Σ_j λ_j·mean(r_j²). Default is equal weighting. jaxpi's LDC init_weights are
    # '1,1,10,100,100' — they then adapt them with grad-norm balancing, which is not ported, so
    # passing that is the static approximation of their starting point.
    'block_weights': None,
    # Interior collocation points per step.
    'n_pde': 4000,
    # Boundary points per edge per step (4x this total).
    'n_bc': 256,
    # Aux batch sizes for Gnome are max(K_min, int(N * aux_frac)) per block. Each aux pass is a
    # full second-order residual eval, so keep small.
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
    'beta1': 0.9,
    # Second-moment / preconditioner EMA (also shampoo_beta) for Gnome and SOAP.
    'beta2': 0.99,
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
