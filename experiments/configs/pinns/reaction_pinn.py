"""Settings for experiments.pinns.reaction_pinn. Edit CONFIG, then run the module.

None keeps an existing automatic/default-dependent setting where supported.
"""

CONFIG = {
    # Choices: 'gnome', 'soap', 'adamw', 'engdw'.
    'optimizer': 'gnome',
    # Reaction coefficient. Vanilla whole-domain PINNs are already near 100% relative error at
    # rho=5; rho=10 is the harder published rung.
    'rho': 5.0,
    # The plain MLP matches the published benchmark; other choices are architecture or tensor-
    # grouping ablations. Choices: 'mlp', 'modified', 'fused', 'fused-modified'.
    'arch': 'mlp',
    # Fused architectures only: hidden layers per parameter tensor. One is the per-layer control;
    # zero fuses the whole stack.
    'fuse_every': 0,
    # Raw inputs plus a soft BC (published setup), or an exact periodic embedding that drops the
    # BC block. Choices: 'none', 'periodic'.
    'embed': 'none',
    'seed': 0,
    'steps': 100000,
    'n_pde': 4000,
    'n_ic': 256,
    'n_bc': 256,
    # Fraction of each residual block used by Gnome's auxiliary curvature pass.
    'aux_frac': 0.05,
    'lr': 0.001,
    # Gnome trust radius; zero disables the trust-region solve.
    'trust_region': 1.0,
    # Gnome curvature damping. Baselines retain their own epsilon.
    'eps': 1e-6,
    'beta1': 0.9,
    'beta2': 0.99,
    'weight_decay': 1e-8,
    # ENGD-W fixed damping in JJ^T + lambda I; reaction starting value, not a tuned result. No
    # weight decay or LR schedule is used.
    'engdw_damping': 1e-6,
    # ENGD-W SPRING-style regularization-center momentum in [0, 1). Zero preserves the original
    # solver; try 0.9, 0.95, or 0.99.
    'engdw_momentum': 0.0,
    # Choose the lowest same-batch loss on 13 log-spaced step sizes from 1e-3 to 1, overriding lr
    # (same grid as Poisson-5D).
    'engdw_line_search': False,
    # ENGD-W only: sample PDE/IC/BC training points once and reuse them every step. Validation
    # points remain independently sampled.
    'engdw_fixed': False,
    # Samples per vectorized Jacobian chunk for ENGD-W; limits differentiation workspace, while
    # the full Jacobian and sample Gram remain dense.
    'engdw_chunk': 32,
    # Network width; 50 matches the published model.
    'hidden': 50,
    # Linear-layer count for MLP/fused (four hidden layers at the default); gated hidden-layer
    # count for modified variants.
    'depth': 5,
    'warmup_steps': 200,
    # Final learning-rate fraction after cosine decay.
    'cosine_decay': 0.0,
    # Use float64 (always enabled for ENGD-W), falling back to CPU when the selected device is
    # MPS.
    'float64': False,
    # Write Gnome internals every N steps; zero disables it.
    'diagnostics_every': 0,
    # Comma-separated parameter indices for Gnome diagnostics.
    'diagnostics_params': None,
    'log_every': 200,
    'runs_dir': 'runs',
    'quiet': False,
}
