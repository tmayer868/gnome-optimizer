# Upstream reference

`train_gpt_simple.py` is the unmodified upstream Track 3 training script from
[KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/records/track_3_optimization/train_gpt_simple.py),
revision `4ea6b937337a4889b8cfe3f38a93d120048d8f71` (retrieved October 2, 2026).
It is preserved for architecture/initialization/optimizer parity checks and
attribution. Its MIT license is included alongside it.

Run the portable implementation with
`python -m experiments.transformers.modded_nanogpt`, not this reference file.
The reference itself assumes CUDA and a torchrun process group.
