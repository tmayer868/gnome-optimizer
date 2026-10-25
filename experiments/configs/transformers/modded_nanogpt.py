"""Settings for experiments.transformers.modded_nanogpt. Edit CONFIG, then run the module.

None keeps an existing automatic/default-dependent setting where supported.
"""

CONFIG = {
    # Choices: 'benchmark', 'dev'.
    'preset': 'benchmark',
    # Choices: 'muon', 'gnome_hutchinson', 'gnome_fisher', 'soap', 'adamw'.
    'optimizer': 'gnome_hutchinson',
    # Choices: 'auto', 'cuda', 'mps', 'cpu'.
    'device': 'auto',
    # Choices: 'auto', 'float32', 'bfloat16'.
    'dtype': 'auto',
    # Opt-in CUDA model compilation
    'compile': False,
    # Output weight initialization std (default: reference zero init); overrides only the output
    # head
    'head_init_std': 0.0,
    # Keep embedding weights/optimizer state in FP32 while retaining the chosen compute dtype
    'fp32_embedding': False,
    'seed': 0,
    'steps': None,
    'vocab_size': None,
    'n_layer': None,
    'n_embd': None,
    'head_dim': None,
    'seq_len': None,
    'batch_tokens': None,
    'val_tokens': None,
    # Sequences per forward; global batch is unchanged
    'microbatch_size': 4,
    # Gnome curvature sequences sampled from the main batch
    'aux_batch_size': 1,
    # Gnome/SOAP/AdamW LR, or Muon hidden-matrix LR
    'lr': None,
    'weight_decay': None,
    'beta1': None,
    'beta2': None,
    'eps': None,
    'shampoo_beta': 0.99,
    'precondition_frequency': 10,
    'max_precond_dim': 4096,
    # Gnome relative L2 trust radius; 0 disables
    'trust_region': 1.0,
    # 0 disables clipping (reference default)
    'max_grad_norm': 0,
    'muon_momentum': 0.95,
    # For Gnome runs, optionally update input embeddings with AdamW inside Gnome Choices: 'same',
    # 'adamw'.
    'embedding_optimizer': 'same',
    # Embedding AdamW LR (default: 0.7 for Muon, 0.001 for Gnome)
    'embed_lr': None,
    # Gnome embedding AdamW only
    'embed_beta1': 0.9,
    # Gnome embedding AdamW only
    'embed_beta2': 0.999,
    # Gnome embedding AdamW only
    'embed_eps': 1e-8,
    # Gnome embedding AdamW only
    'embed_weight_decay': 0.0,
    # Muon auxiliary AdamW only
    'head_lr': 0.004,
    # Muon auxiliary AdamW only
    'scalar_lr': 0.015,
    # Muon auxiliary AdamW only
    'aux_weight_decay': 0.001,
    'warmup_steps': 0,
    # Linear warmup, then reference linear cooldown or immediate cosine decay Choices: 'linear',
    # 'cosine', 'polynomial'.
    'lr_schedule': 'linear',
    # Power for polynomial decay.
    'lr_pow': 1.2,
    # Cosine schedule's final LR as a fraction of each group's initial LR
    'min_lr_frac': 0.1,
    # Final fraction with linear LR decay; ignored for lr_schedule cosine
    'cooldown_frac': 0.7,
    # Default: 125, then 25 during the final 10%
    'val_every': None,
    'log_every': 10,
    'data_dir': 'experiments/data/fineweb10B',
    # Opt-in download: N training shards, about 200 MB each
    'download_shards': 0,
    'download_only': False,
    # Offline development check, requires preset dev
    'synthetic': False,
    'runs_dir': 'runs',
    'save_checkpoint': False,
    'quiet': False,
}
