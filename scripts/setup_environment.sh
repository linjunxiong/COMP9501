#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/env.sh"
if [[ "${1:-}" == "--direct" ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
fi
unset PIP_EXTRA_INDEX_URL PIP_INDEX_URL
export PIP_CONFIG_FILE=/dev/null
# Setup/compilation may never acquire a GPU, even if one is free.
export CUDA_VISIBLE_DEVICES=-1
export TCNN_CUDA_ARCHITECTURES=90
export TORCH_CUDA_ARCH_LIST=9.0
mkdir -p "$COMP9501_ROOT"/{logs,vendor,models,.cache/torch_extensions}
if [[ ! -d "$COMP9501_ROOT/threestudio/.git" ]]; then
  git clone https://github.com/threestudio-project/threestudio.git "$COMP9501_ROOT/threestudio"
  git -C "$COMP9501_ROOT/threestudio" checkout 28d9d80d9d00f308244adfcf3be8b17ca0cb6465
fi
if [[ "$(git -C "$COMP9501_ROOT/threestudio" rev-parse HEAD)" != "28d9d80d9d00f308244adfcf3be8b17ca0cb6465" ]]; then
  echo "Unexpected upstream revision; review before replacing any existing checkout." >&2
  exit 1
fi
if [[ ! -x "$COMP9501_ROOT/.venv/bin/python" ]]; then
  /usr/bin/python -m venv --without-pip "$COMP9501_ROOT/.venv"
fi
if ! python -m pip --version >/dev/null 2>&1; then
  /usr/bin/python -m pip --python "$COMP9501_ROOT/.venv/bin/python" install --index-url https://pypi.org/simple pip==24.3.1
fi
python -m pip install --index-url https://pypi.org/simple pip==24.3.1 setuptools==69.5.1 wheel==0.44.0
python -m pip install --retries 2 --timeout 30 --index-url https://download.pytorch.org/whl/cu124 torch==2.4.1 torchvision==0.19.1 numpy==1.26.4 Pillow==10.4.0
python -m pip install --index-url https://pypi.org/simple -r "$COMP9501_ROOT/requirements.txt"
python "$COMP9501_ROOT/scripts/install_runtime_libs.py"
bash "$COMP9501_ROOT/scripts/install_native.sh"
