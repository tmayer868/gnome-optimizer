"""Settings for experiments.fourier_regression. Edit CONFIG, then run the module.

None keeps an existing automatic/default-dependent setting where supported.
"""

CONFIG = {
    # Choices: 'gnome', 'soap', 'adamw'.
    'optimizer': 'gnome',
    'seed': 0,
    'steps': 20000,
    'batch_size': 256,
    'lr': 0.003,
    # Input dimension, and the primary difficulty knob. At 20k AdamW steps: d=4 -> r2 0.90, d=8
    # -> 0.62, d=16 -> 0.32. Above ~8 the task stops responding to extra capacity or steps and
    # just measures how much of the low band you got.
    'dim': 4,
    # Number of Fourier modes summed into the target.
    'n_modes': 128,
    # Amplitude spectrum exponent, a_k = f_k^-alpha. 0.0 is flat (energy spread evenly over
    # octaves, hardest); 1.0 puts 80% of the variance in the lowest octave and makes the high
    # frequencies decoration.
    'alpha': 0.3,
    'width': 256,
    # Hidden layers. Capacity is not the binding constraint here — see the docstring sweep — so
    # raising these mostly buys wall time.
    'depth': 4,
    # Choices: 'gelu', 'tanh', 'silu'.
    'activation': 'gelu',
    # Frozen validation points, drawn once per seed.
    'n_val': 8192,
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
    # Default 0: inputs are resampled every step, so there is nothing to regularize and decay
    # only biases the fit.
    'weight_decay': 0.0,
    # LR warmup steps, applied to every optimizer.
    'warmup_steps': 200,
    # Final-LR fraction for the SOAP/AdamW cosine decay: 0.0 decays to zero (default), 1.0
    # disables decay. Gnome (MSE) never decays regardless.
    'cosine_decay': 1.0,
    # Validation cadence in steps. Per-step train loss is logged to the artifact regardless.
    'eval_every': 1000,
    'runs_dir': 'runs',
    'quiet': False,
}
