"""Settings for experiments.pinns.poisson_pinn. Edit CONFIG, then run the module.

None keeps an existing automatic/default-dependent setting where supported.
"""

CONFIG = {
    # Choices: 'gnome', 'soap', 'adamw', 'engdw', 'woodbury'.
    'optimizer': 'gnome',
    'seed': 0,
    'steps': 100_000,
    'n_pde': 2000,
    'n_bc_per_edge': 50,
    'fixed_dataset': False,
    'hidden': 128,
    'depth': 4,
    'arch': 'mlp',
    'fuse_every': 2,
    'float64': True,
    'diagnostics_every': 0,
    'diagnostics_params': None,
    'measure_rho_every': 0,
    'measure_rho_samples': 256,
    'measure_rho_kron_floor': False,
    'log_every': 200,
    'runs_dir': 'runs',
    'quiet': False,
    # Each optimizer keeps its own hyperparameters and learning-rate schedule.
    'optimizers': {
        'gnome': {
            'aux_frac': 0.10,
            'lr': 2.5e-2,
            'trust_region': .01,
            'eps': 1e-12,
            'beta1': .99,
            'beta2': 0.99,
            'weight_decay': 1e-8,
            'warmup_steps': 200,
            'cosine_decay': 0.10,
        },
        'soap': {
            'lr': 1e-2,
            'beta1': .95,
            'beta2': 0.99,
            'weight_decay': 1e-8,
            'warmup_steps': 200,
            'cosine_decay': 0.10,
        },
        'adamw': {
            'lr': 1e-2,
            'weight_decay': 1e-8,
            'warmup_steps': 200,
            'cosine_decay': 0.10,
        },
        'engdw': {
            'lr': 1e-2,
            'engdw_damping': 1e-6,
            'engdw_momentum': 0.0,
            'engdw_line_search': True,
            'engdw_chunk': 32,
            'engdw_fixed': False,
        },
        'woodbury': {
            'aux_frac': 0.10,
            'lr': 1e-2,
            'trust_region': .05,
            'beta1': .95,
            'beta2': 0.99,
            'woodbury_damping': 0.001,
            'woodbury_warmup_steps': 100,
            'history_tol': 0.01,
            'max_history': None,
            'gram_float64': False,
            'warmup_steps': 200,
            'cosine_decay': 0.10,
        },
    },
}
