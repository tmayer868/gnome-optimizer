"""Settings for experiments.pinns.burgers2d_pinn. Edit CONFIG, then run the module.

None keeps an existing automatic/default-dependent setting where supported.
"""

CONFIG = {
    # Choices: 'gnome', 'soap', 'adamw'.
    'optimizer': 'gnome',
    'seed': 0,
    # Paper protocol is 20000 (NG / quasi-Newton) or 21001 (SOAP); see Table 9.
    'steps': 20000,
    # Interior collocation points per step. The paper does not state this for the 2D viscous
    # case, so the default is inherited from burgers_pinn. The front is only 2nu=0.008 thick, so
    # this is the knob most likely to set the accuracy floor — sweep it first.
    'n_pde': 2000,
    # Initial-condition points per step (t=0).
    'n_ic': 100,
    # Boundary points per step, spread over all four spatial faces of the unit square.
    'n_bc': 100,
    # Aux batch sizes for Gnome are max(K_min, int(N * aux_frac)) per block. Each aux pass is a
    # full higher-order residual eval, so this is not free — keep small.
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
    # MLP width. Default 20 reproduces the paper's 3881 parameters with depth 11.
    'hidden': 20,
    # Total linear layers (paper: 10 hidden + output).
    'depth': 11,
    # float64 is likely needed to approach the paper's 5.8e-7: second-order autodiff on a
    # 0.008-thick front hits a ~1e-5 floor in float32. MPS has no float64 and falls back to CPU.
    # Choices: 'float32', 'float64'.
    'dtype': 'float32',
    # Linear LR warmup steps, applied to every optimizer.
    'warmup_steps': 200,
    # Final-LR fraction for the baseline cosine decay: 0.0 decays to zero (standard treatment),
    # 1.0 disables decay. Gnome (MSE) never decays regardless.
    'cosine_decay': 0.0,
    # Time slices in the fixed eval grid.
    'n_eval_t': 26,
    # Points per spatial axis in the fixed eval grid (201 -> spacing 0.005, just under the 0.008
    # front).
    'n_eval_x': 201,
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
    'quiet': False,
}
