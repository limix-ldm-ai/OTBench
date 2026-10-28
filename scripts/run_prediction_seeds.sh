#!/usr/bin/env bash
# Run the prediction protocol for independent checkpoint seeds sequentially.
# Usage: [SEEDS="0 1 2"] bash scripts/run_prediction_seeds.sh METHOD [DATASET ...]
# Other environment variables match run_prediction_all.sh. Requires Bash >= 4.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if (( $# < 1 )); then
  echo "usage: bash scripts/run_prediction_seeds.sh METHOD [DATASET ...]" >&2
  exit 2
fi
if [[ -n "${TRAIN_SEED:-}" ]]; then
  echo "ERROR: set SEEDS for this wrapper; TRAIN_SEED applies to run_prediction_all.sh." >&2
  exit 2
fi
read -r -a TARGET_SEEDS <<< "${SEEDS:-0 1 2}"
declare -A SEEN_SEEDS=()
for seed in "${TARGET_SEEDS[@]}"; do
  if ! [[ "${seed}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "ERROR: SEEDS entries must be non-negative integers: '${seed}'." >&2
    exit 2
  fi
  if [[ -n "${SEEN_SEEDS[${seed}]+set}" ]]; then
    echo "ERROR: duplicate seed '${seed}'." >&2
    exit 2
  fi
  SEEN_SEEDS["${seed}"]=1
done
for seed in "${TARGET_SEEDS[@]}"; do
  echo ">>> prediction checkpoint seed: ${seed}"
  TRAIN_SEED="${seed}" bash "${SCRIPT_DIR}/run_prediction_all.sh" "$@"
done
