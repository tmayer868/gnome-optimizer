"""Settings for experiments.pinns.navier_stokes_pinn. Edit CONFIG, then run the module.

None keeps an existing automatic/default-dependent setting where supported.
"""

CONFIG = {
    # Choices: 'adamw', 'soap', 'gnome', 'soap+ssbroyden', 'engdw'.
    'optimizer': 'gnome',
    'seed': 0,
    'steps': 200000,
    # Number of sparse (u, v) observation points (fixed at training start).
    'n_data': 5000,
    # Fresh collocation points per step.
    'n_pde': 10000,
    'hidden': 128,
    'depth': 8,
    # Log Gnome's internal state — curvature spectrum, LM damping, trust-region usage — every N
    # steps to a sibling runs/.../{run_id}.diag.jsonl. 0 (default) disables it entirely. Gnome
    # only: SOAP and AdamW expose no such hook.
    'diagnostics_every': 0,
    # Comma-separated parameter indices to log, e.g. '0,4'. Default logs every parameter, which
    # is one record per tensor per logged step — narrow it to keep the file readable.
    'diagnostics_params': None,
    'log_every': 200,
    'w_data_u': 1.0,
    'w_data_v': 1.0,
    'w_continuity': 1.0,
    'w_momentum_x': 1.0,
    'w_momentum_y': 1.0,
    'lambda1_init': 0.0,
    'lambda2_init': 0.0,
    'runs_dir': 'runs',
    'quiet': False,
    # Each optimizer keeps its own hyperparameters and learning-rate schedule.
    'optimizers': {
        'adamw': {
            'lr': 0.001,
            'weight_decay': 1e-08,
            'warmup_steps': 200,
            'cosine_decay': 0.0,
        },
        'soap': {
            'lr': 0.001,
            'beta1': 0.9,
            'beta2': 0.99,
            'weight_decay': 1e-08,
            'warmup_steps': 200,
            'cosine_decay': 0.0,
        },
        'gnome': {
            'aux_frac': 0.03,
            'lr': 0.001,
            'trust_region': 1.0,
            'eps': 1e-06,
            'beta1': 0.9,
            'beta2': 0.99,
            'weight_decay': 1e-08,
            'warmup_steps': 200,
            'cosine_decay': 0.0,
        },
        'soap+ssbroyden': {
            'lr': 0.001,
            'beta1': 0.9,
            'beta2': 0.99,
            'weight_decay': 1e-08,
            'warmup_steps': 200,
            'cosine_decay': 0.0,
            'soap_steps': 30000,
            'ssbroyden_lr': 1.0,
            'ssbroyden_tolerance_grad': 1e-10,
        },
        'engdw': {
            'lr': 0.05,
            'engdw_damping': 1e-06,
            'engdw_momentum': 0.0,
            'engdw_line_search': True,
            'engdw_chunk': 32,
            'engdw_fixed': False,
        },
    },
    # Input features: 'none', 'periodic', or 'fourier'.
    'embedding': 'none',
    # Even number of random Fourier features.
    'embed_dim': 32,
    # Initial standard deviation of the frozen Fourier projection.
    'embed_scale': 1.0,
    'fixed_dataset': False,
    'fix_after': None,
    'float64': False,
}
