#!/usr/bin/env bash
set -euo pipefail

# In Kaggle, enable Internet and a GPU accelerator before running this script.
cd /kaggle/working

if [[ ! -d gnome-optimizer/.git ]]; then
  git clone https://github.com/tmayer868/gnome-optimizer.git
fi

cd gnome-optimizer
git fetch origin experimental
git checkout experimental
git merge --ff-only origin/experimental

if ! command -v uv >/dev/null 2>&1; then
  python -m pip install --quiet uv
  export PATH="$(python -m site --user-base)/bin:$PATH"
fi

uv run --extra nanogpt python -u -m experiments.transformers.modded_nanogpt \
  --config experiments/configs/transformers/modded_nanogpt_kaggle.py
