"""Settings for experiments.pinns.poisson5d_pinn. Edit CONFIG, then run the module.

None keeps an existing automatic/default-dependent setting where supported.
"""

CONFIG = {
    # Choices: 'gnome', 'soap', 'adamw', 'engdw'.
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
        'gnome': {
            # Interior points for Gnome's GGN aux batch (disjoint, small). Gnome only.
            'aux_n_int': 180,
            # Boundary points for Gnome's GGN aux batch. Gnome only.
            'aux_n_bc': 30,
            # Base learning rate. Gnome typically wants a larger lr here (e.g. 1e-2) than the SOAP/AdamW
            # baselines.
            'lr': 1e-2,
            # Gnome per-coordinate update bound: lambda is set to the smallest value with
            # max|m̂/(v̂+lambda)| <= this, so no coordinate moves more than lr*trust_region in a step.
            # Larger -> weaker bound -> longer steps. 0 disables it, falling back to plain m̂/(v̂+eps)
            # damping.
            'trust_region': .05,
            # Gnome curvature-damping epsilon in m̂/(v̂+eps): larger -> more gradient-descent-like,
            # smaller -> fuller Newton step. Gnome only; SOAP/AdamW keep their fixed eps=1e-8.
            'eps': 1e-12,
            # First-moment (momentum) EMA for Gnome and SOAP.
            'beta1': 0.99,
            # Second-moment / preconditioner EMA (also shampoo_beta) for Gnome and SOAP.
            'beta2': 0.99,
            'weight_decay': 1e-8,
            # Linear LR warmup steps, applied to every optimizer.
            'warmup_steps': 200,
            # Final-LR fraction for the baseline cosine decay: 0.0 decays to zero (standard treatment),
            # 1.0 disables decay. Gnome (MSE) never decays regardless.
            'cosine_decay': 0.10,
        },
        'soap': {
            # Base learning rate. Gnome typically wants a larger lr here (e.g. 1e-2) than the SOAP/AdamW
            # baselines.
            'lr': 0.001,
            # First-moment (momentum) EMA for Gnome and SOAP.
            'beta1': 0.9,
            # Second-moment / preconditioner EMA (also shampoo_beta) for Gnome and SOAP.
            'beta2': 0.99,
            'weight_decay': 1e-8,
            # Linear LR warmup steps, applied to every optimizer.
            'warmup_steps': 200,
            # Final-LR fraction for the baseline cosine decay: 0.0 decays to zero (standard treatment),
            # 1.0 disables decay. Gnome (MSE) never decays regardless.
            'cosine_decay': 0.0,
        },
        'adamw': {
            # Base learning rate. Gnome typically wants a larger lr here (e.g. 1e-2) than the SOAP/AdamW
            # baselines.
            'lr': 0.001,
            'weight_decay': 1e-8,
            # Linear LR warmup steps, applied to every optimizer.
            'warmup_steps': 200,
            # Final-LR fraction for the baseline cosine decay: 0.0 decays to zero (standard treatment),
            # 1.0 disables decay. Gnome (MSE) never decays regardless.
            'cosine_decay': 0.0,
        },
        'engdw': {
            # Base learning rate. Gnome typically wants a larger lr here (e.g. 1e-2) than the SOAP/AdamW
            # baselines.
            'lr': 0.001,
            # ENGD-W Tikhonov damping λ in (JJᵀ+λI). Default is the paper's Fig 8 fixed-lr 5D value; pair
            # with lr 5.2289e-2. engdw only.
            'engdw_damping': 6.804474e-8,
            # ENGD-W: use the paper's inherited line search instead of a fixed lr (then pair with damping
            # 3.173212e-12, the Fig 7 value). engdw only.
            'engdw_line_search': False,
            # jacrev chunk_size for the ENGD-W Jacobian (memory vs speed; lower = less memory). None = no
            # chunking. engdw only.
            'engdw_chunk': None,
        },
    },
}
