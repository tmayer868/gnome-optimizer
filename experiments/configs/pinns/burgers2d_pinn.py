"""Settings for experiments.pinns.burgers2d_pinn. Edit CONFIG, then run the module.

None keeps an existing automatic/default-dependent setting where supported.
"""

CONFIG = {
    # Choices: 'adamw', 'soap', 'gnome', 'soap+ssbroyden', 'engdw'.
    # Select the optimizer here; keep the names of the settings blocks below.
    'optimizer': 'soap+ssbroyden',
    'seed': 0,
    # Paper protocol is 20000 (NG / quasi-Newton) or 21001 (SOAP); see Table 9.
    'steps': 100000,
    # Interior collocation points per step. The paper does not state this for the 2D viscous
    # case, so the default is inherited from burgers_pinn. The front is only 2nu=0.008 thick, so
    # this is the knob most likely to set the accuracy floor — sweep it first.
    'n_pde': 6000,
    # Initial-condition points per step (t=0).
    'n_ic': 200,
    # Boundary points per step, spread over all four spatial faces of the unit square.
    'n_bc': 200,
    # Reuse one training batch. SS-Broyden always freezes it by the handoff.
    'fixed_dataset': False,
    # Freeze after N warm-start iterations; None freezes only at the hybrid handoff.
    # fixed_dataset=True takes precedence.
    'fix_after': None,
    # MLP width. Default 20 reproduces the paper's 3881 parameters with depth 11.
    'hidden': 48,
    # Total linear layers (paper: 10 hidden + output).
    'depth': 8,
    # float64 is likely needed to approach the paper's 5.8e-7: second-order autodiff on a
    # 0.008-thick front hits a ~1e-5 floor in float32. MPS has no float64 and falls back to CPU.
    # Choices: 'float32', 'float64'. ENGD-W and SOAP + SS-Broyden always use float64.
    'dtype': 'float64',
    # Time slices in the fixed eval grid.
    'n_eval_t': 51,
    # Points per spatial axis in the fixed eval grid (201 -> spacing 0.005, just under the 0.008
    # front).
    'n_eval_x': 201,
    # Log Gnome's internal state — curvature spectrum, LM damping, trust-region usage — every N
    # steps to a sibling runs/.../{run_id}.diag.jsonl. 0 (default) disables it entirely. Gnome
    # only: SOAP and AdamW expose no such hook.
    'diagnostics_every': 0,
    # Comma-separated parameter indices to log, e.g. '0,4'. Default logs every parameter, which
    # is one record per tensor per logged step — narrow it to keep the file readable.
    'diagnostics_params': None,
    # Log a val entry (running train mean + per-term diagnostics on a fresh probe batch) every N
    # steps.
    'log_every': 5,
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
            'lr': 0.05,
            'trust_region': 0.02,
            'eps': 1e-12,
            'beta1': 0.98,
            'beta2': 0.99,
            'weight_decay': 1e-08,
            'warmup_steps': 200,
            'cosine_decay': 0.1,
        },
        'soap+ssbroyden': {
            'lr': 0.001,
            'beta1': 0.99,
            'beta2': 0.999,
            'weight_decay': 0.0,
            'warmup_steps': 0,
            'cosine_decay': 0.1,
            'soap_steps': 0,
            'ssbroyden_lr': 1.0,
            'ssbroyden_tolerance_grad': 1e-10,
        },
        'engdw': {
            'engdw_damping': 1e-09,
            'engdw_momentum': 0.0,
            'engdw_line_search': True,
            'engdw_chunk': 32,
            'engdw_fixed': False,
            'lr': 0.05,
        },
    },
    # Input features: 'none', 'periodic', or 'fourier'.
    'embedding': 'none',
    # Even number of random Fourier features.
    'embed_dim': 32,
    # Initial standard deviation of the frozen Fourier projection.
    'embed_scale': 1.0,
}
