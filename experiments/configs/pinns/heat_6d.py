"""Settings for experiments.pinns.heat_6d; edit CONFIG, then run the module.

Five spatial dimensions in the unit ball plus time in [0,1].
Inputs: (t, x1, x2, x3, x4, x5).
u_t - (1/5) Laplacian(u) = -(1/5) ||x||² exp(||x||²/2+t).
Initial value exp(||x||²/2); outward Neumann flux exp(1/2+t).
Analytic solution exp(||x||²/2+t). Defaults follow the Heat 5D template.
"""

CONFIG = {
    'optimizer': 'gnome',  # 'gnome', 'soap', 'adamw', or 'engdw'
    'seed': 0,
    'steps': 100000,
    # Freeze the main PDE/IC/BC batch after N resampled steps, for any optimizer.
    # None keeps resampling; 0 fixes it from the start. Requires 0 <= N < steps.
    # Gnome then draws fresh auxiliary subsets; engdw_fixed=True takes precedence.
    'fix_after': 50000,
    'n_pde': 2000,
    'n_ic': 100,
    'n_bc': 100,  # Total on the unit sphere; resampled until fix_after.
    # Local six-input tanh MLP; depth counts all linear layers.
    'hidden': 96,
    'depth': 6,
    'dtype': 'float64',  # 'float64' falls back to CPU on MPS.
    # Fixed uniform time/ball-volume points for relative L2; independent of training RNG.
    'n_eval': 30000,
    'eval_seed': 1234,
    'diagnostics_every': 0,  # Gnome only; 0 disables optimizer diagnostics.
    'diagnostics_params': None,
    'log_every': 200,  # Final metrics are always evaluated, including when 0.
    'runs_dir': 'runs',
    'quiet': False,
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
            'lr': 0.01,
            'trust_region': 0.04,
            'eps': 1e-12,
            'beta1': 0.99,
            'beta2': 0.99,
            'weight_decay': 1e-08,
            'warmup_steps': 200,
            'cosine_decay': 0.1,
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
            'lr': 0.03,
            'engdw_damping': 1e-07,
            'engdw_momentum': 0.0,
            'engdw_line_search': True,
            'engdw_chunk': 32,
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
}
