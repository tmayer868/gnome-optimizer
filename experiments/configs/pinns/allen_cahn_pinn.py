"""Settings for experiments.pinns.allen_cahn_pinn. Edit CONFIG, then run the module.

None keeps an existing automatic/default-dependent setting where supported.
"""

CONFIG = {
    # Choices: 'adamw', 'soap', 'gnome', 'soap+ssbroyden', 'engdw'.
    'optimizer': 'gnome',
    # Network: plain tanh MLP, the gated modified MLP (Wang et al. 2021), or 'fused' — the plain
    # MLP with [W|b] merged per layer and hidden layers grouped per fuse_every. hidden / depth
    # control all three. Choices: 'mlp', 'modified', 'fused', 'fused-modified'.
    'arch': 'mlp',
    # arch fused only: consecutive hidden layers per parameter tensor. 1 = one tensor per layer,
    # the control every larger value is compared against (identical function and initialization,
    # only the grouping differs). 0 (default) fuses the whole stack. Note depth 4 leaves only k=2
    # hidden layers to group, so raise depth to give this room.
    'fuse_every': 2,
    # Even number of random Fourier features.
    'embed_dim': 96,
    # Initial standard deviation of the frozen Fourier projection.
    'embed_scale': 1.0,
    'seed': 0,
    # Allen-Cahn is stiff; may want more than the default.
    'steps': 200_000,
    'n_pde': 4000,
    'n_ic': 400,
    'n_bc': 400,
    'weights': [.05, 1.0, 1.0],  # [pde, IC, periodic BC]
    # The BC weight is ignored for periodic and Fourier embeddings (exact BC).
    # Reuse the same PDE/IC/BC training points every step for any optimizer. Gnome draws fresh
    # auxiliary subsets; validation still resamples.
    'fixed_dataset': False,
    # Resample for this many steps, then freeze a newly sampled PDE/IC/BC dataset.
    # None keeps resampling; 0 fixes from the start. fixed_dataset=True takes precedence.
    # soap+ssbroyden freezes no later than the optimizer handoff.
    'fix_after': None,
    # Network width.
    'hidden': 50,
    # Network depth: Linear-layer count for arch mlp, number of gated hidden layers for arch
    # modified.
    'depth': 5,
    # Run in double precision. Allen-Cahn's stiff canyon geometry can push the residual below the
    # float32 floor (~1e-6 relative), where the Gauss-Newton curvature estimate is reading round-
    # off; float64 moves that floor out of the way. MPS has no double support, so this falls back
    # to CPU there.
    'float64': True,
    # Switch to float64 after N float32 steps, preserving optimizer state and LR schedule.
    # Requires 0 <= N < steps; 0 starts in float64. MPS moves to CPU at the switch.
    # soap+ssbroyden promotes no later than its handoff, even if this is None.
    'float64_after': None,
    # Reference used for rel_l2. 'jaxpi' is the published 201x512 Chebfun table and remains the
    # compatibility default. 'highres' uses the validated 8191-point tight reference
    # experiments/reference_solutions/allen_cahn_highres_8191_tight.mat and generates it
    # automatically when missing (rtol=2e-13, atol=2e-15). Choices: 'jaxpi', 'highres'.
    'reference': 'highres',
    # Log Gnome's internal state — curvature spectrum, LM damping, trust-region usage — every N
    # steps to a sibling r5s/.../{run_id}.diag.jsonl. 0 (default) disables it entirely. Gnome
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
            'basis_transport': 'overlap',
            'partition_residuals': False,
            'aux_frac': 0.1,
            'lr': 0.008,
            'trust_region': 0.05,
            'eps': 1e-12,
            'beta1': 0.99,
            'beta2': 0.99,
            'weight_decay': 1e-08,
            'warmup_steps': 200,
            'cosine_decay': 0.1,
        },
        'soap+ssbroyden': {
            'lr': 0.002,
            'beta1': 0.99,
            'beta2': 0.999,
            'weight_decay': 1e-08,
            'warmup_steps': 200,
            'cosine_decay': 0.0,
            'soap_steps': 10000,
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
}
