"""Settings for experiments.resnets.cub200. Edit CONFIG, then run the module.

None keeps an existing automatic/default-dependent setting where supported.
"""

CONFIG = {
    # Choices: 'gnome_fisher', 'gnome_hutchinson', 'soap', 'adamw'.
    'optimizer': 'gnome_hutchinson',
    'data_dir': 'experiments/data',
    # Download CUB if not extracted locally.
    'download': False,
    # Architecture; resnet18_pretrained loads cached/downloaded ImageNet weights. Choices:
    # 'resnet8', 'resnet12', 'resnet18', 'resnet18_pretrained'.
    'model': 'resnet18',
    # Default: bn for pretrained, gn for scratch. Pretrained gn converts BN, retaining affine
    # weights. Choices: 'gn', 'bn'.
    'norm': None,
    'image_size': 224,
    'epochs': 100,
    'batch_size': 32,
    'aux_batch_size': 10,
    # DataLoader workers; increase on GPU hosts.
    'workers': 0,
    # Choices: 'auto', 'cpu', 'mps', 'cuda'.
    'device': 'auto',
    'lr': 0.001,
    'weight_decay': 0.01,
    'beta1': 0.9,
    'beta2': 0.99,
    'eps': 1e-6,
    # Gnome RMS update bound before LR scaling; 0 disables.
    'trust_region': 1.0,
    # Choices: 'none', 'greedy', 'spatial', 'patch'.
    'merge_dims': 'none',
    'warmup_steps': 200,
    'lr_min_frac': 0.1,
    # Disable random crop, flip and color jitter; use evaluation transforms.
    'no_augment': False,
    'seed': 0,
    # Cap total steps for short diagnostic runs.
    'max_steps': None,
    # Save final model and optimizer beside the JSONL.
    'save_checkpoint': False,
    'log_every': 50,
    'runs_dir': 'runs',
    'quiet': False,
}
