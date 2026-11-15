"""Settings for experiments.pinns.heat5d_pinn; edit CONFIG, then run the module.

Four spatial dimensions plus time, ordered (t, x1, x2, x3, x4), all in [0, 1].
Dangel et al. (2024), Kronecker-Factored Approximate Curvature for
Physics-Informed Neural Networks, 4+1D heat experiment: u_t = 0.25 * sum_i u_xixi,
u(0,x) = sum_i sin(2*x_i), and analytic reference exp(-t) * sum_i sin(2*x_i).
Nonhomogeneous Dirichlet targets on all eight faces come from this reference.
All settings are local defaults.
"""

CONFIG = {
    'optimizer': 'gnome',  # 'gnome', 'soap', 'adamw', or 'engdw'
    'seed': 0,
    'steps': 100000,
    'n_pde': 3000,
    'n_ic': 250,
    'n_bc': 250,  # Total across all eight spatial faces; resampled each step.
    # Local five-input tanh MLP; depth counts all linear layers.
    'hidden': 96,
    'depth': 5,
    'dtype': 'float64',  # 'float64' falls back to CPU on MPS.
    # Fixed uniform space-time points for relative L2; independent of training RNG.
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
            'lr': 0.03,
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
            'lr': 0.05,
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
    'fix_after': None,
}
