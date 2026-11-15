"""Settings for experiments.pinns.poisson5d_pinn. Edit CONFIG, then run the module.

None keeps an existing automatic/default-dependent setting where supported.
"""

CONFIG = {
    # Choices: 'adamw', 'soap', 'gnome', 'soap+ssbroyden', 'engdw'.
    'optimizer': 'gnome',
    'seed': 0,
    # Compute precision. float64 pushes below the ~1e-5 single-precision floor of second-order
    # PDE autodiff, but MPS has no float64 so it falls back to CPU (slow) — use CUDA for real
    # float64 runs. Note: Gnome computes its preconditioner eigenbasis internally in float32
    # regardless; only the Laplacian/gradient/param path is true float64. Choices: 'float32',
    # 'float64'.
    'dtype': 'float32',
    'steps': 100_000,
    # Problem dimension.
    'd': 5,
    # Choices: 'mlp' (one parameter tensor per layer), 'fused' (group hidden layers).
    'arch': 'fused',
    # Fused only: consecutive hidden layers per tensor. 0 fuses the full hidden stack;
    # 1 keeps separate layers; 2 groups pairs, with a shorter final chunk if needed.
    'fuse_every': 2,
    # MLP width.
    'hidden': 128,
    # Total number of linear layers (input + hidden + output).
    'depth': 7,
    # Interior collocation points per step (paper: 3000).
    'n_int': 3000,
    # Boundary collocation points per step (paper: 500).
    'n_bc': 500,
    # Use |∂Ω|=2d weighting on the BC block (paper's Background loss) instead of equal block
    # weights (paper's Methods residual). See header note.
    'bc_volume_weight': False,
    # Size of the fixed held-out set for rel_l2 (paper: 30k).
    'n_eval': 30000,
    # Seed for the fixed held-out eval set.
    'eval_seed': 1234,
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
            'beta1': 0.9,
            'beta2': 0.99,
            'weight_decay': 1e-08,
            'warmup_steps': 200,
            'cosine_decay': 0.0,
        },
        'gnome': {
            'aux_n_int': 180,
            'aux_n_bc': 30,
            'lr': 0.01,
            'trust_region': 0.05,
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
            'lr': 0.001,
            'engdw_damping': 6.804474e-08,
            'engdw_line_search': False,
            'engdw_chunk': 32,
            'engdw_momentum': 0.0,
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
