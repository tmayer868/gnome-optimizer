"""Settings for experiments.pinns.convection_pinn. Edit CONFIG, then run the module.

None keeps an existing automatic/default-dependent setting where supported.
"""

CONFIG = {
    # Choices: 'gnome', 'soap', 'adamw', 'engdw', 'soap+ssbroyden', 'gnome+ssbroyden'.
    'optimizer': 'gnome',
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
    'embed': 'none',
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
    'hidden': 128,
    'depth': 6,
    'float64': True,
    'diagnostics_every': 0,
    # Comma-separated parameter indices to log.
    'diagnostics_params': None,
    'log_every': 200,
    'runs_dir': 'runs',
    'quiet': False,
    # Each optimizer keeps its own hyperparameters and learning-rate schedule.
    'optimizers': {
        'gnome': {
            # EMA basis handling: 'overlap' (squared overlaps) or 'soap' (permutation only).
            'basis_transport': 'overlap',
            # Estimate curvature separately for each auxiliary residual block.
            'partition_residuals': False,
            # Aux batch fraction for Gnome's curvature surrogate.
            'aux_frac': 0.10,
            'lr': 5e-3,
            'trust_region': .25,
            # Gnome curvature damping in m/(v+eps). Gnome only.
            'eps': 1e-12,
            # First-moment EMA for Gnome and SOAP.
            'beta1': 0.95,
            # Second-moment / preconditioner EMA for Gnome, SOAP.
            'beta2': 0.99,
            'weight_decay': 1e-8,
            'warmup_steps': 600,
            # Final-LR fraction for the cosine decay: 0.0 decays to zero, 1.0 disables decay.
            'cosine_decay': 0.10,
        },
        'gnome+ssbroyden': {
            # EMA basis handling: 'overlap' (squared overlaps) or 'soap' (permutation only).
            'basis_transport': 'overlap',
            # Estimate curvature separately for each auxiliary residual block.
            'partition_residuals': False,
            # Aux batch fraction for Gnome's curvature surrogate.
            'aux_frac': 0.10,
            'lr': 8e-3,
            'trust_region': .10,
            # Gnome curvature damping in m/(v+eps). Gnome only.
            'eps': 1e-12,
            # First-moment EMA for Gnome and SOAP.
            'beta1': 0.95,
            # Second-moment / preconditioner EMA for Gnome, SOAP.
            'beta2': 0.99,
            'weight_decay': 1e-8,
            'warmup_steps': 600,
            # Final-LR fraction for the cosine decay: 0.0 decays to zero, 1.0 disables decay.
            'cosine_decay': 0.10,
            'gnome_steps': 15_000,
            'ssbroyden_lr': 1.0,
            'ssbroyden_tolerance_grad': 1e-10,
        },
        'soap': {
            'lr': 3e-3,
            # First-moment EMA for Gnome and SOAP.
            'beta1': 0.99,
            # Second-moment / preconditioner EMA for Gnome, SOAP.
            'beta2': 0.999,
            'weight_decay': 1e-8,
            'warmup_steps': 600,
            # Final-LR fraction for the cosine decay: 0.0 decays to zero, 1.0 disables decay.
            'cosine_decay': 0.10,
        },
        'adamw': {
            'lr': 9e-3,
            'weight_decay': 1e-8,
            'warmup_steps': 600,
            # Final-LR fraction for the cosine decay: 0.0 decays to zero, 1.0 disables decay.
            'cosine_decay': 0.10,
        },
        # Total steps includes both phases; SOAP settings apply to the warm start.
        'soap+ssbroyden': {
            'beta1': 0.99,
            'beta2': 0.999,
            'weight_decay': 0.0,
            'lr': 3e-3,
            'warmup_steps': 0,
            'cosine_decay': 0.0,
            'soap_steps': 10_000,
            'ssbroyden_lr': 1.0,
            'ssbroyden_tolerance_grad': 1e-10,
        },
        'engdw': {
            'lr': 9e-3,
            # ENGD-W fixed damping in JJ^T + lambda I; convection starting value, not a tuned result. No
            # weight decay or LR schedule is used.
            'engdw_damping': 1e-6,
            # ENGD-W SPRING-style regularization-center momentum in [0, 1). Zero preserves the original
            # solver; try 0.9, 0.95, or 0.99.
            'engdw_momentum': 0.0,
            # Choose the lowest same-batch loss on 13 log-spaced step sizes from 1e-3 to 1, overriding lr
            # (same grid as Poisson-5D).
            'engdw_line_search': True,
            # ENGD-W only: sample PDE/IC/BC training points once and reuse them every step. Validation
            # points remain independently sampled.
            'engdw_fixed': False,
            # Samples per vectorized Jacobian chunk for ENGD-W; limits differentiation workspace, while
            # the full Jacobian and sample Gram remain dense.
            'engdw_chunk': 32,
        },
    },
}
