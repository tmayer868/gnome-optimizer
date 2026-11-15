"""Settings for experiments.pinns.convection_pinn. Edit CONFIG, then run the module.

None keeps an existing automatic/default-dependent setting where supported.
"""

CONFIG = {
    # Choices: 'adamw', 'soap', 'gnome', 'soap+ssbroyden', 'engdw'.
    'optimizer': 'soap+ssbroyden',
    # Convection speed. THE knob for this benchmark: the PDE is trivial at every value, but
    # first-order optimizers start failing around 20 and are comprehensively broken by 30-40. The
    # published comparison numbers in the module docstring are for 30-40. Sweep it to draw the
    # failure curve — that curve is the result, more than any single number.
    'beta': 50.0,
    # Network. Defaults to the plain MLP, which is what the published failure results use —
    # switching to 'modified' changes the architecture as well as the optimizer and muddies the
    # comparison. Choices: 'mlp', 'modified', 'fused', 'fused-modified'.
    'arch': 'mlp',
    # arch fused / fused-modified only: consecutive hidden layers per parameter tensor. 1 = one
    # tensor per layer, the control every larger value is compared against. 0 (default) fuses the
    # whole stack.
    'fuse_every': 0,
    'seed': 0,
    'steps': 250000,
    'n_pde': 8_000,
    'n_ic': 256,
    'n_bc': 256,
    'weights': [1.0, 1.0, 1.0],  # [pde, IC, periodic BC]
    # The BC weight is ignored for the periodic embedding (exact BC).
    # Reuse PDE/IC/BC points for any optimizer; Gnome draws auxiliary subsets.
    # Validation still samples fresh points.
    'fixed_dataset': False,
    # Resample N steps, then freeze PDE/IC/BC points. None disables delayed freezing.
    # fixed_dataset / engdw_fixed freeze from step 0; SS-Broyden freezes by its handoff.
    'fix_after': None,
    # Network width. Default 50 follows Krishnapriyan.
    'hidden': 55,
    'depth': 8,
    'float64': True,
    'diagnostics_every': 0,
    # Comma-separated parameter indices to log.
    'diagnostics_params': None,
    'log_every': 200,
    'runs_dir': 'runs',
    'quiet': False,
    # Each optimizer keeps its own hyperparameters and learning-rate schedule.
    'optimizers': {
        'adamw': {
            'lr': 0.009,
            'weight_decay': 1e-08,
            'warmup_steps': 600,
            'cosine_decay': 0.1,
        },
        'soap': {
            'lr': 0.003,
            'beta1': 0.99,
            'beta2': 0.999,
            'weight_decay': 1e-08,
            'warmup_steps': 600,
            'cosine_decay': 0.1,
        },
        'gnome': {
            'basis_transport': 'overlap',
            'partition_residuals': False,
            'aux_frac': 0.1,
            'lr': 0.007,
            'trust_region': 0.05,
            'eps': 1e-12,
            'beta1': 0.95,
            'beta2': 0.99,
            'weight_decay': 1e-08,
            'warmup_steps': 600,
            'cosine_decay': 0.1,
        },
        'soap+ssbroyden': {
            'beta1': 0.99,
            'beta2': 0.999,
            'weight_decay': 0.0,
            'lr': 0.003,
            'warmup_steps': 0,
            'cosine_decay': 0.0,
            'soap_steps': 1000,
            'ssbroyden_lr': 1.0,
            'ssbroyden_tolerance_grad': 1e-10,
        },
        'engdw': {
            'lr': 0.009,
            'engdw_damping': 1e-06,
            'engdw_momentum': 0.0,
            'engdw_line_search': True,
            'engdw_fixed': False,
            'engdw_chunk': 32,
        },
    },
    # Input features: 'none', 'periodic', or 'fourier'.
    'embedding': 'none',
    # Even number of random Fourier features.
    'embed_dim': 32,
    # Initial standard deviation of the frozen Fourier projection.
    'embed_scale': 1.0,
}
