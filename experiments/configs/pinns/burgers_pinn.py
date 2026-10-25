"""Settings for experiments.pinns.burgers_pinn. Edit CONFIG, then run the module.

None keeps an existing automatic/default-dependent setting where supported.
"""

CONFIG = {
    # Choices: 'gnome', 'soap', 'adamw', 'engdw'.
    'optimizer': 'gnome',
    'seed': 0,
    # Use double precision for training and evaluation; falls back to CPU on MPS, which has no
    # float64 support.
    'float64': False,
    'steps': 5000,
    'n_pde': 2000,
    'n_ic': 100,
    'n_bc': 100,
    # Reuse PDE/IC/BC training points; validation still draws fresh points.
    'fixed_dataset': False,
    # ENGD-W fixed damping in J J^T + lambda I; no weight decay or schedule.
    'engdw_damping': 1e-6,
    # ENGD-W regularization-center momentum in [0,1).
    'engdw_momentum': 0.0,
    # Same-batch grid line search over [1e-3,1], overriding lr.
    'engdw_line_search': False,
    # Per-point Jacobian vmap chunk size; the exact solve still uses all rows.
    'engdw_chunk': 32,
    # ENGD-W-only compatibility alias for fixed_dataset.
    'engdw_fixed': False,
    # Aux batch sizes for Gnome are max(K_min, int(N * aux_frac)) per block. Each aux pass is a
    # full higher-order residual eval, so this is not free — keep small.
    'aux_frac': 0.03,
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
    # MLP width.
    'hidden': 20,
    # MLP depth.
    'depth': 9,
    # Linear LR warmup for Gnome/SOAP/AdamW; ignored by ENGD-W.
    'warmup_steps': 200,
    # Final-LR fraction for the baseline cosine decay: 0.0 decays to zero (standard treatment),
    # 1.0 disables decay. Ignored by ENGD-W.
    'cosine_decay': 0.0,
    # Log Gnome's internal state — curvature spectrum, LM damping, trust-region usage — every N
    # steps to a sibling runs/.../{run_id}.diag.jsonl. 0 (default) disables it entirely. Gnome
    # only: SOAP and AdamW expose no such hook.
    'diagnostics_every': 0,
    # Comma-separated parameter indices to log, e.g. '0,4'. Default logs every parameter, which
    # is one record per tensor per logged step — narrow it to keep the file readable.
    'diagnostics_params': None,
    # Log a val entry (running train mean + per-term diagnostics on a fresh probe batch) every N
    # steps.
    'log_every': 100,
    'runs_dir': 'runs',
    # Skip the additional rel_l2 eval against jaxpi's burgers.mat (Wang et al. 2025). On by
    # default so a head-to-head number against their reported SOAP result is logged as
    # `rel_l2_jaxpi` alongside our Cole–Hopf-reference `rel_l2`.
    'no_jaxpi_ref': False,
    'quiet': False,
}
