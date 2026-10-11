#!/usr/bin/env bash
# Source this file. It does not enable or initialize CUDA.
COMP9501_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export COMP9501_ROOT
export PATH="$COMP9501_ROOT/.venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=9.0
export TCNN_CUDA_ARCHITECTURES=90
export HF_HOME="$COMP9501_ROOT/.cache/huggingface"
export TORCH_HOME="$COMP9501_ROOT/.cache/torch"
export TORCH_EXTENSIONS_DIR="$COMP9501_ROOT/.cache/torch_extensions"
export PIP_CACHE_DIR="$COMP9501_ROOT/.cache/pip"
export XDG_CACHE_HOME="$COMP9501_ROOT/.cache"
export XDG_CONFIG_HOME="$COMP9501_ROOT/.cache/config"
export MPLCONFIGDIR="$COMP9501_ROOT/.cache/matplotlib"
export IPYTHONDIR="$COMP9501_ROOT/.cache/ipython"
export PYTHONPATH="$COMP9501_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export MAX_JOBS=4
export OMP_NUM_THREADS=4
export HF_HUB_DISABLE_TELEMETRY=1
export GIT_CONFIG_GLOBAL="$COMP9501_ROOT/.gitconfig.local"
export LD_LIBRARY_PATH="$COMP9501_ROOT/vendor/runtime/usr/lib/x86_64-linux-gnu:$COMP9501_ROOT/vendor/runtime/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
