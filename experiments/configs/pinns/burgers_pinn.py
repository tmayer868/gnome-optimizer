"""Settings for experiments.pinns.burgers_pinn. Edit CONFIG, then run the module.

None keeps an existing automatic/default-dependent setting where supported.
"""

CONFIG = {
    # Choices: 'adamw', 'soap', 'gnome', 'soap+ssbroyden', 'engdw'.
    'optimizer': 'gnome',
    'seed': 0,
    # Use double precision for training and evaluation; falls back to CPU on MPS, which has no
    # float64 support.
    'float64': True,
    'steps': 100000,
    'n_pde': 4000,
    'n_ic': 200,
    'n_bc': 200,
    # Reuse PDE/IC/BC training points; validation still draws fresh points.
    'fixed_dataset': False,
    # MLP width.
    'hidden': 64,
    # MLP depth.
    'depth': 6,
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
            'aux_frac': 0.1,
            'lr': 0.01,
            'trust_region': 0.03,
            'eps': 1e-12,
            'beta1': 0.99,
            'beta2': 0.99,
            'weight_decay': 1e-08,
            'warmup_steps': 200,
            'cosine_decay': 0.1,
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
            'engdw_damping': 1e-06,
            'engdw_momentum': 0.0,
            'engdw_line_search': False,
            'engdw_chunk': 32,
            'engdw_fixed': False,
            'lr': 0.001,
        },
    },
    # Input features: 'none', 'periodic', or 'fourier'.
    'embedding': 'none',
    # Even number of random Fourier features.
    'embed_dim': 32,
    # Initial standard deviation of the frozen Fourier projection.
    'embed_scale': 1.0,
    'fix_after': None,
}
