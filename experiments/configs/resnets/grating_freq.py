"""Settings for experiments.resnets.grating_freq. Edit CONFIG, then run the module.

None keeps an existing automatic/default-dependent setting where supported.
"""

CONFIG = {
    # Choices: 'gnome', 'soap', 'adamw'.
    'optimizer': 'gnome',
    'seed': 0,
    'epochs': 20,
    'batch_size': 256,
    'lr': 0.001,
    # Lower end of the per-sample grating amplitude (drawn log-uniformly), on the same [-1,1]
    # scale as the image. The amplitude range is the difficulty knob: widening it confounds the
    # energy-magnitude shortcut but lowers SNR against image clutter. amp_min == amp_max makes
    # the task ~94% solvable by total variation alone.
    'amp_min': 0.05,
    # Upper end of the per-sample grating amplitude.
    'amp_max': 1.0,
    # Gnome per-coordinate update bound: lambda is set to the smallest value with
    # max|m̂/(v̂+lambda)| <= this, so no coordinate moves more than lr*trust_region in a step.
    # Larger -> weaker bound -> longer steps. 0 disables it, falling back to plain m̂/(v̂+eps)
    # damping.
    'trust_region': 1.0,
    # Gnome curvature-damping epsilon in m̂/(v̂+eps): larger -> more gradient-descent-like,
    # smaller -> fuller Newton step. Gnome only; SOAP/AdamW keep their fixed eps=1e-8.
    'eps': 1e-6,
    # First-moment (momentum) EMA for Gnome and SOAP.
    'beta1': 0.9,
    # Second-moment / preconditioner EMA (also shampoo_beta) for Gnome and SOAP.
    'beta2': 0.99,
    'weight_decay': 0.01,
    # Architecture (default resnet12). Choices: 'resnet8', 'resnet12', 'resnet18'.
    'model': 'resnet12',
    # Normalization: GroupNorm (default) or BatchNorm. Choices: 'gn', 'bn'.
    'norm': 'gn',
    # LR warmup steps, applied to every optimizer.
    'warmup_steps': 100,
    # Final-LR fraction for the SOAP/AdamW cosine decay: 0.0 decays to zero (default), 1.0
    # disables decay. Gnome (MSE) never decays regardless.
    'cosine_decay': 1.0,
    # Print a running-mean train loss every N steps within an epoch (0 disables). Per-step train
    # loss is logged to the artifact regardless.
    'log_every': 50,
    'runs_dir': 'runs',
    'quiet': False,
}
