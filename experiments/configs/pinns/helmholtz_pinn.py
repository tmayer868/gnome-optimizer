"""Settings for experiments.pinns.helmholtz_pinn. Edit CONFIG, then run the module.

None keeps an existing automatic/default-dependent setting where supported.
"""

CONFIG = {
    # Choices: 'adamw', 'soap', 'gnome', 'soap+ssbroyden', 'engdw'.
    'optimizer': 'gnome',
    'seed': 0,
    'steps': 20000,
    # x-frequency of the exact solution sin(a1*pi*x).
    'a1': 1.0,
    # y-frequency of the exact solution sin(a2*pi*y).
    'a2': 4.0,
    # Helmholtz wavenumber. Paper rungs: 1, 10, 100. k=100 wants the domain-expansion curriculum,
    # which is not implemented here.
    'k': 1.0,
    # Interior collocation points per step. Paper uses 15,000-25,000 with RAD refresh; Table 4
    # shows 10,000 already reaches rel_Linf 1.5e-8, so that is the default here.
    'n_pde': 10000,
    # MLP width.
    'hidden': 30,
    # Total linear layers (paper: 4 hidden + output = 5). Ladder: 5 / 7 / 9 -> 2971 / 4831 / 6691
    # params at modes 1.
    'depth': 5,
    # Fourier embedding modes M; input width is 4M. Paper uses M=1 for low frequency and M=10 for
    # the k=10/100 rungs (depth 7 modes 10 -> 5911 params). Ignored when embedding raw.
    'modes': 1,
    # periodic: deterministic spatial harmonics; fourier: random Fourier projection;
    # none: raw coordinates. The periodic options satisfy periodic BCs exactly.
    # Input features: 'none', 'periodic', or 'fourier'.
    'embedding': 'periodic',
    # auto uses no BC block with periodic features and Dirichlet BCs with none.
    # Explicit 'dirichlet' retains the BC loss with any embedding.
    'bc': 'auto',
    # Boundary points per step when the BC block is on.
    'n_bc': 1000,
    # Weight on the BC block: mean(out^2) = mean(r_pde^2) + bc_weight*mean(r_bc^2). At k=1 a DC
    # offset c gives |r_pde| = k^2*c = c and |r_bc| = c, so bc_weight=1 only doubles DC's
    # visibility — raise it to actually pin the constant mode.
    'bc_weight': 1.0,
    # float64 by default: the target here is 1e-9, which float32 cannot represent in the
    # residual. MPS has no float64 and falls back to CPU. Choices: 'float32', 'float64'.
    'dtype': 'float64',
    # Points per axis in the fixed eval grid (401 -> 160k points). Raise it for high a1/a2.
    'n_eval': 401,
    # Log Gnome's internal state — curvature spectrum, LM damping, trust-region usage — every N
    # steps to a sibling runs/.../{run_id}.diag.jsonl. 0 (default) disables it entirely. Gnome
    # only: SOAP and AdamW expose no such hook.
    'diagnostics_every': 0,
    # Comma-separated parameter indices to log, e.g. '0,4'. Default logs every parameter, which
    # is one record per tensor per logged step — narrow it to keep the file readable.
    'diagnostics_params': None,
    'log_every': 100,
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
            'aux_frac': 0.03,
            'lr': 0.001,
            'trust_region': 1.0,
            'eps': 1e-06,
            'beta1': 0.9,
            'beta2': 0.99,
            'weight_decay': 1e-08,
            'warmup_steps': 200,
            'cosine_decay': 0.0,
        },
        'soap+ssbroyden': {
            'lr': 0.001,
            'beta1': 0.9,
            'beta2': 0.99,
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
    # Even number of random Fourier features.
    'embed_dim': 32,
    # Initial standard deviation of the frozen Fourier projection.
    'embed_scale': 1.0,
    'fixed_dataset': False,
    'fix_after': None,
}
