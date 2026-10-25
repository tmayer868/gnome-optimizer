"""Settings for experiments.resnets.cifar100. Edit CONFIG, then run the module.

None keeps an existing automatic/default-dependent setting where supported.
"""

CONFIG = {
    # Choices: 'gnome_fisher', 'gnome_hutchinson', 'soap', 'adamw'.
    'optimizer': 'gnome_hutchinson',
    'seed': 0,
    'epochs': 30,
    'batch_size': 256,
    'lr': 0.001,
    # Gnome per-coordinate update bound: lambda is set to the smallest value with
    # max|m̂/(v̂+lambda)| <= this, so no coordinate moves more than lr*trust_region in a step.
    # Larger -> weaker bound -> longer steps. 0 disables it, falling back to plain m̂/(v̂+eps)
    # damping.
    'trust_region': 1.0,
    # Gnome curvature-damping epsilon; with the trust region active this is only the lower bound
    # on lambda, so it can go much smaller than the pre-trust-region default. Gnome only;
    # SOAP/AdamW keep their fixed eps=1e-8.
    'eps': 1e-6,
    # First-moment (momentum) EMA for Gnome and SOAP.
    'beta1': 0.9,
    # Second-moment / preconditioner EMA (also shampoo_beta) for Gnome and SOAP.
    'beta2': 0.99,
    'weight_decay': 0.01,
    # Aux batch K for the Gnome surrogate, drawn as a subset of the main batch. Cheap here: the
    # Hutchinson surrogate's per-sample tensor is only (K, 100).
    'aux_batch_size': 10,
    # Gnome/SOAP convolution-factor grouping: 'none'=[O][I][H][W], 'greedy'=existing size-bounded
    # merge, 'spatial'=[O][I][HW], or 'patch'=[O][IHW]. Ignored by AdamW. Choices: 'none',
    # 'greedy', 'spatial', 'patch'.
    'merge_dims': 'none',
    # Architecture (default resnet12). Choices: 'resnet8', 'resnet12', 'resnet18'.
    'model': 'resnet12',
    # Normalization: GroupNorm (default) or BatchNorm. GroupNorm keeps the per-sample block
    # structure the surrogate assumes (see docs/method.md 5.3). Choices: 'gn', 'bn'.
    'norm': 'gn',
    # Disable flip + random-crop augmentation.
    'no_augment': False,
    # LR warmup steps, applied to every optimizer including Gnome (CCE does not self-anneal).
    'warmup_steps': 200,
    # Cosine floor: final lr as a fraction of peak.
    'lr_min_frac': 0.1,
    # Print a running-mean train loss every N steps (0 disables). Per-step train loss is logged
    # regardless.
    'log_every': 50,
    'runs_dir': 'runs',
    'quiet': False,
}
