"""Settings for experiments.pinns.inviscid_burgers_pinn. Edit CONFIG, then run the module.

None keeps an existing automatic/default-dependent setting where supported.
"""

CONFIG = {
    # Choices: 'gnome', 'soap', 'adamw'.
    'optimizer': 'gnome',
    'seed': 0,
    # Paper protocol: 5000 iterations for every case.
    'steps': 5000,
    # Interior collocation points. Paper sweeps 625, 1250, 2500, 5000, 10000 (Table 10).
    'n_pde': 2500,
    # Initial-condition points at t=0. Not stated in the paper for this rung.
    'n_ic': 256,
    # Weight on the IC block: mean(out^2) = mean(r_pde^2) + ic_weight*mean(r_ic^2). Eq. 50 uses
    # equal weights.
    'ic_weight': 1.0,
    # Shock-detection threshold: Omega_s = {u_x <= -delta}. Undocumented in the paper. Before
    # shock formation the slope at the origin is pi/(1-pi*t), so delta should sit well above pi.
    'roe_delta': 10.0,
    # Offset for the left/right states u(x -+ offset, t) in the Roe average. Undocumented in the
    # paper; should be below a typical collocation spacing.
    'roe_offset': 0.01,
    # Freeze the Roe average as a coefficient (the literal 'linearize' reading) instead of
    # differentiating through it. Changes the parameter gradients.
    'roe_detach': False,
    # periodic: (t, cos pi x, sin pi x), BCs exact, the paper's setup. raw: (t, x) — periodicity
    # is then not enforced at all and there is no BC block, so expect drift at the domain edges.
    # Choices: 'periodic', 'raw'.
    'embedding': 'periodic',
    # Aux batch sizes for Gnome are max(K_min, int(N * aux_frac)) per block. Each aux pass is a
    # full residual eval, so this is not free — keep small.
    'aux_frac': 0.03,
    'lr': 0.001,
    # Gnome l2 trust radius: lambda is the smallest value >= eps with ||m_hat/(v_hat+lambda)||_2
    # <= this * sqrt(P), i.e. a bound on the RMS per-coordinate step. 0 disables it.
    'trust_region': 1.0,
    # Gnome curvature-damping epsilon in m_hat/(v_hat+eps). Gnome only; SOAP/AdamW keep their
    # fixed eps=1e-8.
    'eps': 1e-6,
    'beta1': 0.9,
    'beta2': 0.99,
    'weight_decay': 1e-8,
    # MLP width.
    'hidden': 30,
    # Total linear layers (paper: 6 hidden + output = 7, giving 4801 params at hidden 30).
    'depth': 7,
    # float32 suffices here — the paper's accuracies are ~1e-2/1e-3, set by the shock, not by
    # precision. Choices: 'float32', 'float64'.
    'dtype': 'float32',
    'warmup_steps': 200,
    # Final-LR fraction for the baseline cosine decay: 0.0 decays to zero, 1.0 disables decay.
    # Gnome never decays regardless.
    'cosine_decay': 0.0,
    # Spatial points at t=1 (paper's WENO reference: 1000).
    'n_eval_x': 1000,
    # Time slices in the space-time eval grid.
    'n_eval_t': 51,
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
