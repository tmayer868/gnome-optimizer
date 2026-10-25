"""Settings for experiments.tabular.covertype. Edit CONFIG, then run the module.

None keeps an existing automatic/default-dependent setting where supported.
"""

CONFIG = {
    # Choices: 'gnome', 'gnome_hutchinson', 'gnome_fisher', 'adamw', 'soap', 'muon'.
    'optimizer': 'gnome',
    # Choices: 'auto', 'cpu', 'mps', 'cuda'.
    'device': 'auto',
    'data_dir': 'experiments/data/covertype',
    'runs_dir': 'runs',
    'seed': 0,
    'split_seed': 42,
    'epochs': 20,
    # Optional shorter training budget
    'max_steps': None,
    'width': 256,
    'blocks': 3,
    'batch_size': 512,
    'aux_batch_size': 128,
    'eval_batch_size': 4096,
    # Default: .001 Gnome/AdamW, .003 SOAP, .02 Muon
    'lr': None,
    # Muon companion AdamW LR
    'aux_lr': 0.001,
    'weight_decay': 0.01,
    'beta1': 0.9,
    # Default: .99 Gnome/SOAP, .999 AdamW/Muon companion
    'beta2': None,
    'eps': 1e-8,
    'shampoo_beta': 0.99,
    'precondition_frequency': 10,
    'max_precond_dim': 1024,
    'precondition_1d': False,
    'norm_free': False,
    # Gnome trust_radius argument; 0 passes None
    'trust_region': 0.15,
    # Main-gradient clipping; 0 disables
    'max_grad_norm': 0,
    'muon_momentum': 0.95,
    'warmup_steps': 30,
    'min_lr_frac': 0.1,
    'log_every': 50,
    'save_checkpoint': False,
    'quiet': False,
}
