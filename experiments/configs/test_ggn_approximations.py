"""Settings for experiments.test_ggn_approximations. Edit CONFIG, then run the module.

None keeps an existing automatic/default-dependent setting where supported.
"""

CONFIG = {
    'seed': 42,
    'steps': 100,
    # Interior Poisson collocation points per step.
    'n_pde': 32,
    # Boundary points per edge (four times this many in total).
    'n_bc_per_edge': 8,
    'ema_beta': 0.95,
    'lr': 0.001,
    'report_every': 25,
    # Hidden-layer widths, for example: hidden_dims 32 64 32
    'hidden_dims': [32, 64, 32],
}
