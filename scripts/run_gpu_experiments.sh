#!/usr/bin/env bash
# Foreground entry point for a scheduler or a platform's task command.
# Default: 3 prompts x baseline/LCGen x 10,000 steps, then CPU evaluation.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$COMP9501_ROOT"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1

plan_only=false
arguments=()
for argument in "$@"; do
  case "$argument" in
    --dry-run|--plan) plan_only=true ;;
    --check-only) plan_only=true; arguments+=("$argument") ;;
    --execute|--allow-gpu)
      printf '%s\n' 'This entry point handles execution flags. Omit --execute/--allow-gpu; use --dry-run to plan.' >&2
      exit 2 ;;
    -h|--help)
      printf '%s\n' 'Usage: bash run_gpu_experiments.sh [--dry-run|--check-only] [suite options]' \
        'Runs in the foreground by default. --dry-run prints the plan; --check-only checks CPU imports without using a GPU.'
      exec "$COMP9501_ROOT/.venv/bin/python" "$COMP9501_ROOT/scripts/run_comparison_suite.py" --help ;;
    *) arguments+=("$argument") ;;
  esac
done

command=("$COMP9501_ROOT/.venv/bin/python" "$COMP9501_ROOT/scripts/run_comparison_suite.py")
command+=("${arguments[@]}")
if [[ "$plan_only" == false ]]; then
  command+=(--execute --allow-gpu)
fi
# Keep CUDA_VISIBLE_DEVICES as assigned by the scheduler. No nohup/backgrounding:
# cancellation reaches the supervisor, which cleans up its training process group.
exec "${command[@]}"
