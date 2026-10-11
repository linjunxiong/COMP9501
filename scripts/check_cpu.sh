#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/env.sh"
export CUDA_VISIBLE_DEVICES=-1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1
cd "$COMP9501_ROOT"
python -m pip check
python -m pytest -q tests
python scripts/preflight_cpu.py --registry --load-models
python scripts/run_experiments.py --stage smoke --variants baseline lcgen-paper > logs/experiment_plan.json
