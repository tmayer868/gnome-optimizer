"""Settings for experiments.pinns.helmholtz_pinn. Edit CONFIG, then run the module.

None keeps an existing automatic/default-dependent setting where supported.
"""

CONFIG = {
    # Choices: 'gnome', 'soap', 'adamw', 'adamw+lbfgs'.
    'optimizer': 'gnome',
    'seed': 0,
    'steps': 20000,
    # x-frequency of the exact solution sin(a1*pi*x).
    'a1': 1.0,
    # y-frequency of the exact solution sin(a2*pi*y).
    'a2': 4.0,
    # Helmholtz wavenumber. Paper rungs: 1, 10, 100. k=100 wants the domain-expansion curriculum,
    # which is not implemented here.
    'k': 1.0,
    # Interior collocation points per step. Paper uses 15,000-25,000 with RAD refresh; Table 4
    # shows 10,000 already reaches rel_Linf 1.5e-8, so that is the default here.
    'n_pde': 10000,
    # Aux batch size for Gnome is max(8, int(n_pde * aux_frac)). Each aux pass is a full second-
    # order residual eval, so this is not free — keep small.
    'aux_frac': 0.03,
    'lr': 0.001,
    # Gnome l2 trust radius: lambda is set to the smallest value >= eps with
    # ||m_hat/(v_hat+lambda)||_2 <= this * sqrt(P), i.e. a bound on the RMS per-coordinate step.
    # Larger -> weaker bound -> longer steps. 0 disables it, falling back to plain
    # m_hat/(v_hat+eps) damping.
    'trust_region': 1.0,
    # Gnome curvature-damping epsilon in m_hat/(v_hat+eps): larger -> more gradient-descent-like,
    # smaller -> fuller Newton step. Gnome only; SOAP/AdamW keep their fixed eps=1e-8.
    'eps': 1e-6,
    # First-moment (momentum) EMA for Gnome and SOAP.
    'beta1': 0.9,
    # Second-moment / preconditioner EMA (also shampoo_beta) for Gnome and SOAP.
    'beta2': 0.99,
    'weight_decay': 1e-8,
    # MLP width.
    'hidden': 30,
    # Total linear layers (paper: 4 hidden + output = 5). Ladder: 5 / 7 / 9 -> 2971 / 4831 / 6691
    # params at modes 1.
    'depth': 5,
    # Fourier embedding modes M; input width is 4M. Paper uses M=1 for low frequency and M=10 for
    # the k=10/100 rungs (depth 7 modes 10 -> 5911 params). Ignored when embedding raw.
    'modes': 1,
    # fourier: periodic input features, BCs hold exactly, single-block PDE-only loss (the paper's
    # setup). raw: feed (x,y) directly — periodicity is then NOT enforced, so a BC block is
    # required (see bc) or the problem is ill-posed. Choices: 'fourier', 'raw'.
    'embedding': 'fourier',
    # Boundary treatment. auto = none with embedding fourier, dirichlet with raw. Set explicitly
    # to get the off-diagonal combinations: `embedding fourier bc dirichlet` adds a BC block
    # while keeping the embedding, which is the clean single-variable ablation (embedding fixed,
    # block count changed). Choices: 'auto', 'none', 'dirichlet'.
    'bc': 'auto',
    # Boundary points per step when the BC block is on.
    'n_bc': 1000,
    # Weight on the BC block: mean(out^2) = mean(r_pde^2) + bc_weight*mean(r_bc^2). At k=1 a DC
    # offset c gives |r_pde| = k^2*c = c and |r_bc| = c, so bc_weight=1 only doubles DC's
    # visibility — raise it to actually pin the constant mode.
    'bc_weight': 1.0,
    # float64 by default: the target here is 1e-9, which float32 cannot represent in the
    # residual. MPS has no float64 and falls back to CPU. Choices: 'float32', 'float64'.
    'dtype': 'float64',
    # Linear LR warmup steps, applied to every optimizer.
    'warmup_steps': 200,
    # Final-LR fraction for the baseline cosine decay: 0.0 decays to zero (standard treatment),
    # 1.0 disables decay. Gnome (MSE) never decays regardless.
    'cosine_decay': 0.0,
    # Points per axis in the fixed eval grid (401 -> 160k points). Raise it for high a1/a2.
    'n_eval': 401,
    # Outer L-BFGS steps after the AdamW phase (optimizer adamw+lbfgs only). Each outer step runs
    # up to lbfgs_max_iter inner iterations, so the total budget is lbfgs_steps * lbfgs_max_iter.
    # Runs on a single fixed collocation batch. 0 skips the phase.
    'lbfgs_steps': 500,
    # Inner iterations per L-BFGS outer step. Must be >1: the strong-Wolfe line search cannot
    # recover from a cold identity Hessian in one iteration and stalls. adamw+lbfgs only.
    'lbfgs_max_iter': 20,
    # L-BFGS history size (stored curvature pairs). adamw+lbfgs only.
    'lbfgs_history': 50,
    # L-BFGS step scale; with a strong-Wolfe line search 1.0 is standard. adamw+lbfgs only.
    'lbfgs_lr': 1.0,
    # Log Gnome's internal state — curvature spectrum, LM damping, trust-region usage — every N
    # steps to a sibling runs/.../{run_id}.diag.jsonl. 0 (default) disables it entirely. Gnome
    # only: SOAP and AdamW expose no such hook.
    'diagnostics_every': 0,
    # Comma-separated parameter indices to log, e.g. '0,4'. Default logs every parameter, which
    # is one record per tensor per logged step — narrow it to keep the file readable.
    'diagnostics_params': None,
    'log_every': 100,
    'runs_dir': 'runs',
    'quiet': False,
}
