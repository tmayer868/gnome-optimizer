"""Settings for experiments.mnist_autoencoder. Edit CONFIG, then run the module.

None keeps an existing automatic/default-dependent setting where supported.
"""

CONFIG = {
    # Choices: 'gnome', 'soap', 'adamw', 'lbfgs'.
    'optimizer': 'gnome',
    'seed': 0,
    'steps': 2000,
    'batch_size': 256,
    'aux_batch_size': 32,
    'lr': 0.001,
    'weight_decay': 0.0,
    'beta1': 0.9,
    'beta2': 0.99,
    'eps': 1e-12,
    'trust_region': 1.0,
    'precondition_frequency': 10,
    'max_precond_dim': 2048,
    # Choices: 'sigmoid', 'tanh'.
    'activation': 'sigmoid',
    # Training objective. BCE is the historical sigmoid benchmark and uses Gnome's bce_hutchinson
    # path; tanh requires MSE. Choices: 'bce', 'mse'.
    'loss': 'bce',
    # Choices: 'sparse', 'xavier'.
    'init': 'sparse',
    'sparse_connections': 15,
    # Nonzero sparse-weight std. Defaults to 1.0 for sigmoid and 0.25 for tanh, matching the
    # historical initialization.
    'sparse_std': None,
    'warmup_steps': 100,
    # Final learning-rate fraction; 1.0 means warmup then constant.
    'cosine_decay': 1.0,
    'eval_every': 100,
    'eval_batch_size': 1024,
    # Fixed train subset used for intermediate time-to-threshold checks.
    'eval_train_size': 10000,
    # Raw-[0,1] train SSE threshold whose first step/time is recorded.
    'target_sse': 2.0,
    'lbfgs_max_iter': 10,
    'lbfgs_history': 10,
    # One fixed batch reused for the entire L-BFGS run (Martens used 7500).
    'lbfgs_batch_size': 7500,
    # Write Gnome curvature/trust-region diagnostics every N steps; 0 disables and non-Gnome
    # optimizers ignore it.
    'diagnostics_every': 100,
    # Gnome diagnostics parameter selection: weights, all, or comma-separated indices.
    'diagnostics_params': 'weights',
    'save_checkpoint': False,
    'checkpoint_path': None,
    'data_dir': 'experiments/data',
    'runs_dir': 'runs',
    'log_every': 20,
    'quiet': False,
}
