#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/env.sh"
export CUDA_VISIBLE_DEVICES=-1
unset PIP_EXTRA_INDEX_URL PIP_INDEX_URL
export PIP_CONFIG_FILE=/dev/null
if [[ ! -d "$COMP9501_ROOT/vendor/tiny-cuda-nn/.git" ]]; then
  git clone --no-checkout https://github.com/NVlabs/tiny-cuda-nn.git "$COMP9501_ROOT/vendor/tiny-cuda-nn"
  git -C "$COMP9501_ROOT/vendor/tiny-cuda-nn" checkout c91138bcd4c6877c8d5e60e483c0581aafc70cce
fi
if [[ "$(git -C "$COMP9501_ROOT/vendor/tiny-cuda-nn" rev-parse HEAD)" != "c91138bcd4c6877c8d5e60e483c0581aafc70cce" ]]; then
  echo "Unexpected tiny-cuda-nn revision; review the checkout before continuing." >&2
  exit 1
fi
git -C "$COMP9501_ROOT/vendor/tiny-cuda-nn" submodule update --init --recursive
python -m pip install --no-build-isolation --no-deps "$COMP9501_ROOT/vendor/tiny-cuda-nn/bindings/torch"
if [[ ! -d "$COMP9501_ROOT/vendor/nvdiffrast/.git" ]]; then
  git clone --branch v0.3.3 https://github.com/NVlabs/nvdiffrast.git "$COMP9501_ROOT/vendor/nvdiffrast"
fi
if [[ "$(git -C "$COMP9501_ROOT/vendor/nvdiffrast" rev-parse HEAD)" != "729261dc64c4241ea36efda84fbf532cc8b425b8" ]]; then
  echo "Unexpected nvdiffrast revision; review the checkout before continuing." >&2
  exit 1
fi
python -m pip install --no-build-isolation --no-deps "$COMP9501_ROOT/vendor/nvdiffrast"
if [[ ! -d "$COMP9501_ROOT/vendor/envlight/.git" ]]; then
  git clone --no-checkout https://github.com/ashawkey/envlight.git "$COMP9501_ROOT/vendor/envlight"
  git -C "$COMP9501_ROOT/vendor/envlight" checkout 05b5851e854429d72ecaf5b206ed64ce55fae677
fi
if [[ "$(git -C "$COMP9501_ROOT/vendor/envlight" rev-parse HEAD)" != "05b5851e854429d72ecaf5b206ed64ce55fae677" ]]; then
  echo "Unexpected envlight revision; review the checkout before continuing." >&2
  exit 1
fi
python -m pip install --no-build-isolation --no-deps "$COMP9501_ROOT/vendor/envlight"
python -m pip check
python "$COMP9501_ROOT/scripts/precompile_extensions.py"
python -m pip freeze > "$COMP9501_ROOT/requirements.lock.txt"
python "$COMP9501_ROOT/scripts/freeze_sources.py"
