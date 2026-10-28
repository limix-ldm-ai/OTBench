#!/usr/bin/env bash
# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
# Run ONE method/task over the whole dataset store on several local GPUs.
#
# The datasets a method/task can actually run on are split into cost-balanced
# shards (scripts/shard_plan.py), and each shard is handed to one GPU as an
# ordinary run_<task>_all.sh invocation. All per-task logic -- two-phase
# checkpoint reuse, skip-existing, per-job logs -- therefore stays in the
# existing drivers, and a dataset is only ever touched by one GPU.
#
# Two equivalent ways to say what to run -- a TOML file, command-line flags, or
# a file plus flags that override it. The last form is what a multi-machine
# launch wants: one shared config, one differing --method per machine.
#
# Usage:
#   (activate the target env first, e.g. `conda activate benchmark`)
#   ./scripts/run_sharded.sh --config scripts/sharded.example.toml
#   ./scripts/run_sharded.sh --method tabpfn --task imputation --gpus "0 1 2 3 4 5 6 7"
#   ./scripts/run_sharded.sh --config scripts/sharded.example.toml --method limix --task prediction
#   ./scripts/run_sharded.sh --help
#
# Flags mirror the config keys of scripts/sharded.example.toml with dashes
# instead of underscores. These env vars override both for operational knobs
# only:
#   OTBENCH_GPUS  space/comma GPU ids
#   MAX_WORKERS      concurrent driver processes (default: one per GPU).
#                    Set according to available host memory.
#   RUN_SHARDS       "all" or space-separated shard ids
#   SKIP_EXISTING    0 | 1
#   DRY_RUN          1 to print each shard's job list without running it
#   RUN_ID           one invocation's identity; defaults to a unique value
#   OTBENCH_WORKSPACE  writable root for outputs, checkpoints, and data
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
PYTHON="${PYTHON:-python}"

usage() {
  cat >&2 <<'EOF'
usage: [env vars] ./scripts/run_sharded.sh --config <file.toml> [--flag value ...]
       [env vars] ./scripts/run_sharded.sh --method <name> --task <task> [--flag value ...]

Flags are the config keys of scripts/sharded.example.toml with dashes instead of
underscores, and they override the file when both are given.

  --config FILE           TOML baseline (optional; a bare path also works)
  --method NAME           registry method name (required)
  --task TASK             generation | imputation | prediction (required)
  --shards N              cost-balanced dataset groups (default 8)
  --run-shards "0 3 5"    shard subset to run (default all)
  --dataset-type TYPE     real | scm | all (default all)
  --store PATH            dataset store root override
  --workspace PATH        writable workspace (default OTBENCH_WORKSPACE or cwd)
  --run-id ID             launch identity (default: UTC timestamp + random suffix)
  --gpus "0 1 2"          GPU ids (default: all visible)
  --max-workers N         concurrent driver processes (default: one per GPU)
  --extra "ARGS"          extra args forwarded to otbench.run
  --n-trials N            generation total trial target (default 3)
  --seeds "1 2"           generation / imputation explicit seed override
  --metric NAME           metric selector (all tasks)
  --scenarios "MCAR MAR"  imputation only
  --rates "0.1 0.3"       imputation only
  --train-seed N          prediction checkpoint seed
  --skip-existing         reuse validated results     (negate: --no-skip-existing)
  --dry-run               print jobs without running  (negate: --no-dry-run)
EOF
}

# Keys that are flags rather than name/value pairs. Everything else must take a
# value, because a value may legitimately start with a dash (--extra "--override
# train.steps=5000"), which rules out guessing from the next argument.
BOOL_KEYS=" skip_existing dry_run "
CONFIG=""
OVERRIDES=()
while (( $# > 0 )); do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --config)
      if (( $# < 2 )); then echo "ERROR: --config requires a value." >&2; exit 2; fi
      CONFIG="$2"; shift 2
      ;;
    --config=*)
      CONFIG="${1#--config=}"; shift
      ;;
    --no-*)
      key="${1#--no-}"; key="${key//-/_}"
      if [[ "${BOOL_KEYS}" != *" ${key} "* ]]; then
        echo "ERROR: --no-* applies to boolean flags only; got '$1'." >&2
        exit 2
      fi
      OVERRIDES+=("${key}=false"); shift
      ;;
    --*=*)
      pair="${1#--}"; key="${pair%%=*}"
      OVERRIDES+=("${key//-/_}=${pair#*=}"); shift
      ;;
    --*)
      key="${1#--}"; key="${key//-/_}"
      if [[ "${BOOL_KEYS}" == *" ${key} "* ]]; then
        OVERRIDES+=("${key}=true"); shift
      else
        if (( $# < 2 )); then echo "ERROR: $1 requires a value." >&2; exit 2; fi
        OVERRIDES+=("${key}=$2"); shift 2
      fi
      ;;
    *)
      if [[ -n "${CONFIG}" ]]; then
        echo "ERROR: unexpected argument '$1'." >&2
        exit 2
      fi
      CONFIG="$1"; shift
      ;;
  esac
done

if [[ -z "${CONFIG}" && ${#OVERRIDES[@]} -eq 0 ]]; then
  usage
  exit 2
fi
if [[ -n "${CONFIG}" && ! -f "${CONFIG}" ]]; then
  echo "ERROR: config file not found: ${CONFIG}" >&2
  exit 2
fi

config_cmd=("${PYTHON}" "${SCRIPT_DIR}/shard_plan.py" config)
if [[ -n "${CONFIG}" ]]; then config_cmd+=(--file "${CONFIG}"); fi
for override in "${OVERRIDES[@]}"; do config_cmd+=(--set "${override}"); done
# NOTE: capture rc via `|| rc=$?`; under `set -e` a bare `x="$(failing cmd)"`
# aborts the script at the assignment with no message of our own.
rc=0
config_assignments="$("${config_cmd[@]}")" || rc=$?
if (( rc != 0 )); then
  echo "ERROR: invalid run settings (see above)." >&2
  exit 2
fi
eval "${config_assignments}"
if [[ -n "${CFG_WORKSPACE}" ]]; then export OTBENCH_WORKSPACE="${CFG_WORKSPACE}"; fi
source "${SCRIPT_DIR}/_workspace.sh"

METHOD="${CFG_METHOD}"
TASK="${CFG_TASK}"
DRIVER="${SCRIPT_DIR}/run_${TASK}_all.sh"
if [[ ! -f "${DRIVER}" ]]; then
  echo "ERROR: no driver for task '${TASK}': ${DRIVER}" >&2
  exit 2
fi

# --- GPUs: env override, then config, then the ambient/visible devices ---
gpu_src="${OTBENCH_GPUS:-${CFG_GPUS}}"
if [[ -z "${gpu_src}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  case "${CUDA_VISIBLE_DEVICES}" in
    -1|NoDevFiles|"") ;;
    *) gpu_src="${CUDA_VISIBLE_DEVICES}" ;;
  esac
fi
if [[ -z "${gpu_src}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  gpu_src="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd' ' -)"
fi
if [[ -z "${gpu_src}" ]]; then
  echo "ERROR: no GPU ids resolved; set gpus in the config or OTBENCH_GPUS=\"0 1 ...\"." >&2
  exit 2
fi
read -r -a GPUS <<< "${gpu_src//,/ }"

MAX_WORKERS="${MAX_WORKERS:-${CFG_MAX_WORKERS}}"
MAX_WORKERS="${MAX_WORKERS:-${#GPUS[@]}}"
if ! [[ "${MAX_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: MAX_WORKERS must be a positive integer; got '${MAX_WORKERS}'." >&2
  exit 2
fi

# --- plan ---
RUN_ID="${RUN_ID:-${CFG_RUN_ID:-$("${PYTHON}" -c 'from datetime import datetime, timezone; import uuid; print(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "." + uuid.uuid4().hex[:12])')}}"
RUN_BASE="$(workspace_path "${SHARDED_ROOT:-${WORKSPACE_ROOT}/outputs/sharded}")/${TASK}/${METHOD}"
RUN_DIR="${RUN_BASE}/${RUN_ID}"
STATUS_DIR="${RUN_DIR}/status"
mkdir -p "${RUN_BASE}"
PLAN_FILE="$(mktemp "${RUN_BASE}/.plan_XXXXXX")"
trap 'rm -f "${PLAN_FILE}"' EXIT

plan_cmd=("${PYTHON}" "${SCRIPT_DIR}/shard_plan.py" plan
          --task "${TASK}" --method "${METHOD}" --shards "${CFG_SHARDS}"
          --dataset-type "${CFG_DATASET_TYPE}" --json "${PLAN_FILE}")
PLAN_STORE="$(workspace_path "${CFG_STORE:-${STORE:-${WORKSPACE_ROOT}/tabdata/store}}")"
plan_cmd+=(--store-root "${PLAN_STORE}")
rc=0
plan_output="$("${plan_cmd[@]}")" || rc=$?
if (( rc != 0 )); then
  echo "ERROR: could not plan shards for ${METHOD}/${TASK} (see above)." >&2
  exit 2
fi
mapfile -t PLAN_ROWS <<< "${plan_output}"

selection="${RUN_SHARDS:-${CFG_RUN_SHARDS}}"
if [[ "${selection}" == "all" ]]; then
  SELECTED=()
  for ((i = 0; i < ${#PLAN_ROWS[@]}; i++)); do SELECTED+=("${i}"); done
else
  read -r -a SELECTED <<< "${selection}"
  declare -A SEEN_SHARDS=()
  for id in "${SELECTED[@]}"; do
    if ! [[ "${id}" =~ ^(0|[1-9][0-9]*)$ ]] || (( id >= ${#PLAN_ROWS[@]} )); then
      echo "ERROR: shard id '${id}' is outside [0, $(( ${#PLAN_ROWS[@]} - 1 ))]." >&2
      exit 2
    fi
    if [[ -n "${SEEN_SHARDS[$id]+set}" ]]; then
      echo "ERROR: duplicate selected shard '${id}'." >&2
      exit 2
    fi
    SEEN_SHARDS["${id}"]=1
  done
fi

# One shard owns one GPU for the whole run, so a shard per GPU is the ceiling.
if (( ${#SELECTED[@]} > ${#GPUS[@]} )); then
  echo "ERROR: ${#SELECTED[@]} selected shard(s) exceed ${#GPUS[@]} GPU(s)." >&2
  echo "       Reduce 'shards', select fewer with run_shards, or expose more GPUs." >&2
  exit 2
fi

"${PYTHON}" "${SCRIPT_DIR}/scheduler_state.py" freeze-sharded \
  --run-root "${RUN_DIR}" --run-id "${RUN_ID}" --plan-file "${PLAN_FILE}" \
  --selected "${SELECTED[@]}" --settings "${config_assignments}" >/dev/null
mkdir -p "${STATUS_DIR}"

balance="$(printf '%s\n' "${PLAN_ROWS[@]}" \
  | awk -F'\t' '{sum += $3; if ($3 > max) max = $3} END {if (sum > 0) printf "%.3f", max / (sum / NR); else printf "n/a"}')"

settings_source="${CONFIG:-(command line only)}"
if (( ${#OVERRIDES[@]} > 0 )); then
  settings_source+=" + ${#OVERRIDES[@]} flag override(s)"
fi
echo ">>> settings   : ${settings_source}"
echo ">>> task       : ${TASK}"
echo ">>> method     : ${METHOD}"
echo ">>> run id     : ${RUN_ID}"
echo ">>> shards     : ${#PLAN_ROWS[@]} (max/ideal cost = ${balance})"
echo ">>> selected   : ${#SELECTED[@]} (${SELECTED[*]})"
echo ">>> gpus       : ${GPUS[*]}"
echo ">>> concurrency: ${MAX_WORKERS} driver process(es)"
echo ">>> plan + logs: ${RUN_DIR}"

run_shard() {
  local id="$1" gpu="$2" log="$3"; shift 3
  local -a datasets=("$@")
  local -a child_env=(OTBENCH_GPUS="${gpu}")
  child_env+=(STORE="${PLAN_STORE}" RUN_ID="${RUN_ID}")
  if [[ -n "${CFG_EXTRA}" ]]; then child_env+=(EXTRA="${CFG_EXTRA}"); fi
  if [[ -n "${CFG_SEEDS}" ]]; then child_env+=(SEEDS="${CFG_SEEDS}"); fi
  if [[ -n "${CFG_N_TRIALS}" ]]; then child_env+=(N_TRIALS="${CFG_N_TRIALS}"); fi
  if [[ -n "${CFG_METRIC}" ]]; then child_env+=(METRIC="${CFG_METRIC}"); fi
  if [[ -n "${CFG_SCENARIOS}" ]]; then child_env+=(SCENARIOS="${CFG_SCENARIOS}"); fi
  if [[ -n "${CFG_RATES}" ]]; then child_env+=(RATES="${CFG_RATES}"); fi
  if [[ -n "${CFG_TRAIN_SEED}" ]]; then child_env+=(TRAIN_SEED="${CFG_TRAIN_SEED}"); fi
  child_env+=(SKIP_EXISTING="${SKIP_EXISTING:-${CFG_SKIP_EXISTING:-0}}")
  child_env+=(DRY_RUN="${DRY_RUN:-${CFG_DRY_RUN:-0}}")

  local shard_rc=0
  "${PYTHON}" "${SCRIPT_DIR}/scheduler_state.py" write-status \
    --plan "${RUN_DIR}/plan.json" --slot "${id}" --state running
  env "${child_env[@]}" bash "${DRIVER}" "${METHOD}" "${datasets[@]}" >"${log}" 2>&1 &
  local driver_pid=$!
  "${PYTHON}" "${SCRIPT_DIR}/scheduler_state.py" heartbeat \
    --plan "${RUN_DIR}/plan.json" --slot "${id}" --pid "${driver_pid}" \
    --interval "${STATUS_HEARTBEAT_SECONDS:-60}" &
  local heartbeat_pid=$!
  wait "${driver_pid}" || shard_rc=$?
  kill "${heartbeat_pid}" 2>/dev/null || true
  wait "${heartbeat_pid}" 2>/dev/null || true
  local state=succeeded
  if (( shard_rc != 0 )); then state=failed
  elif [[ "${DRY_RUN:-${CFG_DRY_RUN:-0}}" == "1" ]]; then state=planned; fi
  "${PYTHON}" "${SCRIPT_DIR}/scheduler_state.py" write-status \
    --plan "${RUN_DIR}/plan.json" --slot "${id}" --state "${state}" --exit-code "${shard_rc}"
}

active=0
slot=0
declare -A SHARD_GPU=()
for id in "${SELECTED[@]}"; do
  IFS=$'\t' read -r _ count cost names <<< "${PLAN_ROWS[$id]}"
  if (( count == 0 )); then
    echo "[skip] shard ${id}: no compatible datasets (more shards than datasets)"
    "${PYTHON}" "${SCRIPT_DIR}/scheduler_state.py" write-status \
      --plan "${RUN_DIR}/plan.json" --slot "${id}" --state succeeded --exit-code 0 --reason "empty shard"
    continue
  fi
  gpu="${GPUS[$slot]}"
  log="${RUN_DIR}/shard_$(printf '%02d' "${id}").log"
  SHARD_GPU["${id}"]="${gpu}"
  read -r -a shard_datasets <<< "${names}"
  echo "[spawn] shard ${id} -> gpu ${gpu}: ${count} dataset(s), cost ${cost} -> ${log}"

  while (( active >= MAX_WORKERS )); do
    wait -n || true
    active=$((active - 1))
  done
  run_shard "${id}" "${gpu}" "${log}" "${shard_datasets[@]}" &
  active=$((active + 1))
  slot=$((slot + 1))
done
wait

echo
echo "=== summary: ${METHOD}/${TASK} ==="
failed=0
for id in "${SELECTED[@]}"; do
  if [[ -z "${SHARD_GPU[$id]:-}" ]]; then continue; fi
  IFS=$'\t' read -r _ count cost _ <<< "${PLAN_ROWS[$id]}"
  status="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["exit_code"])' \
    "${STATUS_DIR}/shard_$(printf '%02d' "${id}").json" 2>/dev/null || echo "?")"
  if [[ "${status}" == "0" ]]; then
    if [[ "${DRY_RUN:-${CFG_DRY_RUN:-0}}" == "1" ]]; then
      echo "  [plan] shard ${id} gpu ${SHARD_GPU[$id]}: ${count} dataset(s), cost ${cost}"
    else
      echo "  [ok]   shard ${id} gpu ${SHARD_GPU[$id]}: ${count} dataset(s), cost ${cost}"
    fi
  else
    echo "  [FAIL] shard ${id} gpu ${SHARD_GPU[$id]}: exit=${status} (see ${RUN_DIR}/shard_$(printf '%02d' "${id}").log)"
    failed=$((failed + 1))
  fi
done
if (( failed > 0 )); then
  echo ">>> ${failed} shard(s) failed; per-job logs are under outputs/${TASK}/${METHOD}/logs/."
  exit 1
fi
if [[ "${DRY_RUN:-${CFG_DRY_RUN:-0}}" == "1" ]]; then
  echo ">>> selected shards planned (dry run)."
else
  echo ">>> all selected shards completed."
fi
