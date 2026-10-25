"""Settings for experiments.transformers.wikitext_gpt. Edit CONFIG, then run the module.

None keeps an existing automatic/default-dependent setting where supported.
"""

CONFIG = {
    # Choices: 'gnome_fisher', 'gnome_hutchinson', 'soap', 'adamw'.
    'optimizer': 'gnome_hutchinson',
    'seed': 0,
    # Total optimizer steps to run.
    'max_steps': 5000,
    # Batch size per optimizer step. Raise this to scale the effective batch (single-batch steps
    # here).
    'batch_size': 48,
    # Gnome per-coordinate update bound: lambda is set to the smallest value with
    # max|m̂/(v̂+lambda)| <= this, so no coordinate moves more than lr*trust_region in a step.
    # Larger -> weaker bound -> longer steps. 0 disables it, falling back to plain m̂/(v̂+eps)
    # damping.
    'trust_region': 1.0,
    'lr': 0.001,
    # Cosine schedule floor as a fraction of lr (final lr).
    'lr_min_frac': 0.1,
    'warmup_steps': 100,
    'weight_decay': 1e-5,
    # Global-norm gradient clipping. Gnome applies it internally to the main gradient; other
    # optimizers use torch.nn.utils.clip_grad_norm_.
    'max_grad_norm': 1.0,
    'seq_len': 128,
    'n_layer': 10,
    'n_head': 8,
    'n_embd': 352,
    # Run validation every N optimizer steps.
    'val_every': 500,
    # Print a running-mean train loss every N steps.
    'log_every': 50,
    'runs_dir': 'runs',
    'quiet': False,
}
