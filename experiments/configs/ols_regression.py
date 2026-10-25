"""Settings for experiments.ols_regression. Edit CONFIG, then run the module.

None keeps an existing automatic/default-dependent setting where supported.
"""

CONFIG = {
    # Choices: 'gnome', 'soap', 'adamw'.
    'optimizer': 'gnome',
    'seed': 0,
    'epochs': 50,
    'batch_size': 256,
    'lr': 0.01,
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
    'weight_decay': 0.0,
    'n_train': 32000,
    'n_val': 2048,
    # Number of features d.
    'dim': 64,
    # Ratio of largest to smallest eigenvalue of cov(X). Larger -> more ill-conditioned -> bigger
    # gap.
    'condition_number': 1000.0,
    # Standard deviation of additive Gaussian label noise.
    'noise_std': 0.1,
    # Linear LR warmup steps, applied to every optimizer.
    'warmup_steps': 200,
    # Final-LR fraction for the baseline cosine decay: 0.0 decays to zero (standard treatment),
    # 1.0 disables decay entirely (raw SOAP/AdamW). Gnome (MSE) never decays regardless.
    'cosine_decay': 0.0,
    # Apply a random orthogonal rotation to cov(X) so the Hessian is non-axis-aligned. Off by
    # default: the exact-recovery gap is a denominator effect that shows even with a diagonal
    # Hessian. Turning it on adds a second handicap for AdamW (per-coordinate scaling cannot
    # rotate).
    'rotate': False,
    'runs_dir': 'runs',
    'quiet': False,
}
