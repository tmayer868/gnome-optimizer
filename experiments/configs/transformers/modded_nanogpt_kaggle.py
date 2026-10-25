"""Kaggle run settings used by scripts/run_kaggle_nanogpt.sh."""

from copy import deepcopy
from experiments.configs.transformers.modded_nanogpt import CONFIG as BASE_CONFIG

CONFIG = deepcopy(BASE_CONFIG)
CONFIG.update({'preset': 'benchmark',
 'optimizer': 'gnome_hutchinson',
 'device': 'auto',
 'dtype': 'auto',
 'fp32_embedding': True,
 'embedding_optimizer': 'same',
 'download_shards': 24,
 'steps': 4200,
 'microbatch_size': 16,
 'aux_batch_size': 16,
 'lr': 0.001,
 'trust_region': 0.15,
 'eps': 1e-12,
 'warmup_steps': 30,
 'lr_schedule': 'linear',
 'cooldown_frac': 0.75,
 'seed': 0,
 'log_every': 1,
 'save_checkpoint': True})
