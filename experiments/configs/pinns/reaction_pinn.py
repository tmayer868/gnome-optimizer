"""Settings for experiments.pinns.reaction_pinn. Edit CONFIG, then run the module.

None keeps an existing automatic/default-dependent setting where supported.
"""

CONFIG = {
    # Choices: 'adamw', 'soap', 'gnome', 'soap+ssbroyden', 'engdw'.
    'optimizer': 'gnome',
    # Reaction coefficient. Vanilla whole-domain PINNs are already near 100% relative error at
    # rho=5; rho=10 is the harder published rung.
    'rho': 5.0,
    # The plain MLP matches the published benchmark; other choices are architecture or tensor-
    # grouping ablations. Choices: 'mlp', 'modified', 'fused', 'fused-modified'.
    'arch': 'mlp',
    # Fused architectures only: hidden layers per parameter tensor. One is the per-layer control;
    # zero fuses the whole stack.
    'fuse_every': 2,
    # Raw inputs plus a soft BC (published setup), or an exact periodic embedding that drops the
    # BC block. Choices: 'none', 'periodic'.
    'seed': 105,
    'steps': 100000,
    'n_pde': 4000,
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
    # Network width; 50 matches the published model.
    'hidden': 76,
    # Linear-layer count for MLP/fused (four hidden layers at the default); gated hidden-layer
    # count for modified variants.
    'depth': 9,
    # Use float64 (always enabled for ENGD-W), falling back to CPU when the selected device is
    # MPS.
    'float64': True,
    # Write Gnome internals every N steps; zero disables it.
    'diagnostics_every': 0,
    # Comma-separated parameter indices for Gnome diagnostics.
    'diagnostics_params': None,
    'log_every': 200,
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
            'lr': 0.003,
            'beta1': 0.99,
            'beta2': 0.999,
            'weight_decay': 1e-08,
            'warmup_steps': 200,
            'cosine_decay': 0.0,
        },
        'gnome': {
            'basis_transport': 'overlap',
            'partition_residuals': False,
            'aux_frac': 0.1,
            'lr': 0.01,
            'trust_region': 0.07,
            'eps': 1e-12,
            'beta1': 0.95,
            'beta2': 0.99,
            'weight_decay': 1e-08,
            'warmup_steps': 200,
            'cosine_decay': 0.1,
        },
        'soap+ssbroyden': {
            'beta1': 0.99,
            'beta2': 0.999,
            'weight_decay': 0.0,
            'lr': 0.001,
            'warmup_steps': 0,
            'cosine_decay': 1.0,
            'soap_steps': 30000,
            'ssbroyden_lr': 1.0,
            'ssbroyden_tolerance_grad': 1e-10,
        },
        'engdw': {
            'lr': 0.001,
            'engdw_damping': 1e-06,
            'engdw_momentum': 0.0,
            'engdw_line_search': True,
            'engdw_fixed': True,
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
