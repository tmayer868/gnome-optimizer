"""Settings for experiments.pinns.kdv_pinn. Edit CONFIG, then run the module.

None keeps an existing automatic/default-dependent setting where supported.
"""

CONFIG = {
    # Choices: 'adamw', 'soap', 'gnome', 'soap+ssbroyden', 'engdw'.
    'optimizer': 'gnome',
    # Network: plain tanh MLP or the gated modified MLP (Wang et al. 2021). hidden / depth
    # control both. Choices: 'mlp', 'modified'.
    'arch': 'mlp',
    # Input embedding: 'periodic' (default) feeds [t, cos(pi x), sin(pi x)]; 'none' feeds raw [t,
    # x]. Both retain the periodic BC loss for u, u_x, u_xx. Choices: 'periodic', 'none'.
    'seed': 0,
    # KdV is dispersive/stiff; may want more than default.
    'steps': 100000,
    'n_pde': 4000,
    'n_ic': 200,
    'n_bc': 200,
    # Loss weights [PDE, IC, BC]; BC averages the u, u_x, and u_xx residuals together.
    'weights': [.01, 1.0, 1.0],
    # Dispersive u_xxx coefficient. Defaults to 0.022^2, the JAX-PI benchmark value, which loads
    # kdv.mat. A matching reference is generated for other values.
    'mu_sq':  0.022 ** 2,
    # Odd number of unique periodic Fourier points used for the generated reference.
    'reference_modes': 511,
    # ETDRK4 substeps per 0.005 reference-output interval.
    'reference_substeps': 20,
    # Network width.
    'hidden': 128,
    # Network depth: Linear-layer count for arch mlp, number of gated hidden layers for arch
    # modified.
    'depth': 8,
    # Run in double precision on CPU. ENGD-W forces float64: the energy-metric
    # Gauss-Newton solve targets accuracy the float32 floor (~1e-6) can't express. Forces
    # device=cpu (MPS has no float64).
    'float64': False,
    # Switch after N float32 steps, keeping optimizer state and the LR schedule.
    # None disables; 0 starts in float64. Requires N < steps. MPS switches to CPU.
    'float64_after': None,
    # Log Gnome's internal state — curvature spectrum, LM damping, trust-region usage — every N
    # steps to a sibling runs/.../{run_id}.diag.jsonl. 0 (default) disables it entirely. Gnome
    # only: SOAP and AdamW expose no such hook.
    'diagnostics_every': 0,
    # Comma-separated parameter indices to log, e.g. '0,4'. Default logs every parameter, which
    # is one record per tensor per logged step — narrow it to keep the file readable.
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
            'lr': 0.001,
            'beta1': 0.99,
            'beta2': 0.999,
            'weight_decay': 1e-08,
            'warmup_steps': 200,
            'cosine_decay': 0.0,
        },
        'gnome': {
            'partition_residuals': False,
            'aux_frac': 0.1,
            'lr': 0.03,
            'trust_region': 0.05,
            'eps': 1e-12,
            'beta1': 0.99,
            'beta2': 0.99,
            'weight_decay': 1e-08,
            'warmup_steps': 200,
            'cosine_decay': 0.0,
        },
        'soap+ssbroyden': {
            'lr': 0.001,
            'beta1': 0.99,
            'beta2': 0.999,
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
    'embedding': 'periodic',
    # Even number of random Fourier features.
    'embed_dim': 32,
    # Initial standard deviation of the frozen Fourier projection.
    'embed_scale': 1.0,
    'fixed_dataset': False,
    'fix_after': None,
}
