"""Settings for experiments.pinns.heat2d_pinn; edit CONFIG, then run the module.

Paper's Heat 2D means (t, x): u_t = 0.25 u_xx on [0, 1]^2.
Training budgets and sampling counts below are local defaults.
"""

CONFIG = {
    'optimizer': 'gnome',  # 'gnome', 'soap', 'adamw', or 'engdw'
    'seed': 0,
    'steps': 100000,
    'n_pde': 2000,
    'n_ic': 100,
    'n_bc': 100,  # Total across x=0 and x=1; resampled each step.
    # Table 2's [2, 256, 256, 256, 1]. Depth includes the output layer.
    # hidden=32, depth=2 gives its smallest [2, 32, 1] architecture.
    'hidden': 96,
    'depth': 6,
    'dtype': 'float64',  # 'float64' falls back to CPU on MPS.
    'n_eval_t': 101 * 2,
    'n_eval_x': 201 * 2,
    'diagnostics_every': 0,  # Gnome only; 0 disables optimizer diagnostics.
    'diagnostics_params': None,
    'log_every': 200,  # Final metrics are always evaluated, including when 0.
    'runs_dir': 'runs',
    'quiet': False,
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
            'lr': 0.02,
            'trust_region': 0.01,
            'eps': 1e-12,
            'beta1': 0.95,
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
            'lr': 0.05,
            'engdw_damping': 1e-07,
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
}
