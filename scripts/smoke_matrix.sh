#!/usr/bin/env bash
# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
# Smoke matrix: every registered method x every task it declares, on the
# smallest dataset of each regime it supports. Verifies the full pipeline
# (train-or-reuse -> inference -> metrics on disk), NOT model quality —
# training is truncated to single-digit steps.
#
#   ./scripts/smoke_matrix.sh                        # quick: one regime per method
#   LEVEL=full ./scripts/smoke_matrix.sh             # every regime each method supports
#   ONLY="tabdiff tabpfn" ./scripts/smoke_matrix.sh  # subset of methods
#   DRY_RUN=1 ./scripts/smoke_matrix.sh              # print the matrix, run nothing
#
# The matrix itself lives in scripts/smoke_plan.py and is derived from the
# method registry, so registering a method adds a row here automatically.
# tests/test_smoke_plan.py fails if the two ever drift apart.
#
# The target environment must already be active (plain `python`, no `conda run`);
# override with PYTHON=/path/to/python. Env: RESULTS (result file), TIMEOUT
# (per-run seconds, default 1800), ARTIFACT_ROOT (isolated smoke outputs and
# checkpoints; defaults to a fresh workspace/outputs/smoke/run_* directory).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
PYTHON="${PYTHON:-python}"
source "${SCRIPT_DIR}/_workspace.sh"
SMOKE_ROOT="${WORKSPACE_ROOT}/outputs/smoke"
mkdir -p "${SMOKE_ROOT}"

# The package is imported as `otbench` but this checkout may be in a
# directory with another name, and a sibling `otbench/` would then win
# `python -m otbench.run` — silently smoke-testing the wrong tree. Point the
# name at this checkout through a symlink that leads PYTHONPATH, so the script
# always exercises the code it ships with. (tests/conftest.py solves the same
# problem in-process.)
SHIM_DIR="$(mktemp -d "${SMOKE_ROOT}/.shim_XXXXXX")"
ln -s "${PROJECT_ROOT}" "${SHIM_DIR}/otbench"
trap 'rm -rf "${SHIM_DIR}"' EXIT
export PYTHONPATH="${SHIM_DIR}"
export SCIPY_ARRAY_API=1
PARENT_ROOT="${SHIM_DIR}"
LEVEL="${LEVEL:-quick}"
ONLY="${ONLY:-}"
TIMEOUT="${TIMEOUT:-1800}"
LOG_DIR="$(mktemp -d "${SMOKE_ROOT}/run_XXXXXX")"
RESULTS="${RESULTS:-${LOG_DIR}/results.txt}"
ARTIFACT_ROOT="$(workspace_path "${ARTIFACT_ROOT:-${LOG_DIR}/artifacts}")"
mkdir -p "${ARTIFACT_ROOT}/checkpoints" "${ARTIFACT_ROOT}/outputs"

# Threads: several backends default to every core and then fail to spawn under
# a container CPU quota (libgomp "Thread creation failed").
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

# --- preflight ------------------------------------------------------------
# A broken interpreter would turn every row into an identical failure, so
# surface the real import error once instead of 56 times.
if ! preflight="$(cd "${PARENT_ROOT}" && "${PYTHON}" -c \
      'import otbench, sklearn, scipy; print("ok")' 2>&1)"; then
  echo "ERROR: preflight failed (running '${PYTHON}' from '${PARENT_ROOT}')." >&2
  echo "       Activate the environment first and make sure 'import otbench' works." >&2
  echo "       A stale PYTHONPATH entry is the usual cause: it is searched ahead of" >&2
  echo "       the environment's own site-packages." >&2
  echo "----------------------------- underlying error -----------------------------" >&2
  echo "${preflight}" >&2
  exit 2
fi

plan_args=(--level "${LEVEL}")
if [[ -n "${ONLY}" ]]; then
  # shellcheck disable=SC2206
  plan_args+=(--only ${ONLY})
fi
if ! mapfile -t PLAN < <("${PYTHON}" "${SCRIPT_DIR}/smoke_plan.py" "${plan_args[@]}"); then
  echo "ERROR: could not build the smoke plan." >&2
  exit 2
fi
if [[ "${#PLAN[@]}" -eq 0 ]]; then
  echo "ERROR: the plan is empty (ONLY='${ONLY}' matched no registered method)." >&2
  exit 2
fi

: > "${RESULTS}"
echo "=== smoke matrix: ${#PLAN[@]} run(s), level=${LEVEL} ==="

pass=0; fail=0; skip=0
for line in "${PLAN[@]}"; do
  # Unit-separated: label, skip reason (empty when runnable), then the argv.
  # Not tabs — bash collapses runs of IFS whitespace and would drop the empty
  # reason field, shifting every argument left by one.
  IFS=$'\x1f' read -r -a fields <<< "${line}"
  label="${fields[0]}"
  reason="${fields[1]}"
  args=("${fields[@]:2}")
  # Tiny smoke configs must never invalidate or overwrite benchmark
  # checkpoints. Every row is routed to this run's isolated artifact tree.
  args+=(--workspace "${WORKSPACE_ROOT}" --checkpoints "${ARTIFACT_ROOT}/checkpoints" --outputs "${ARTIFACT_ROOT}/outputs")

  if [[ -n "${reason}" ]]; then
    echo "SKIP  ${label} (${reason})" | tee -a "${RESULTS}"
    skip=$((skip + 1))
    continue
  fi
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry] ${label} :: ${PYTHON} -m otbench.run ${args[*]}"
    continue
  fi

  log="${LOG_DIR}/${label//[ \/]/_}.log"
  rc=0
  (cd "${PARENT_ROOT}" && timeout "${TIMEOUT}" "${PYTHON}" -m otbench.run "${args[@]}") \
      > "${log}" 2>&1 || rc=$?
  if [[ "${rc}" -eq 0 ]]; then
    echo "PASS  ${label}" | tee -a "${RESULTS}"
    pass=$((pass + 1))
  elif [[ "${rc}" -eq 3 ]]; then
    # The runner checked the registry and declared the combination out of
    # scope before fitting anything; that is a skip, not a failure.
    echo "SKIP  ${label} ($(tail -1 "${log}"))" | tee -a "${RESULTS}"
    skip=$((skip + 1))
  else
    [[ "${rc}" -eq 124 ]] && echo "      (timed out after ${TIMEOUT}s)" >> "${log}"
    echo "FAIL  ${label} (log: ${log})" | tee -a "${RESULTS}"
    tail -4 "${log}" | sed 's/^/      /' | tee -a "${RESULTS}"
    fail=$((fail + 1))
  fi
done

echo "=== DONE: ${pass} passed, ${fail} failed, ${skip} skipped ===" | tee -a "${RESULTS}"
echo "    results: ${RESULTS}"
echo "    logs:    ${LOG_DIR}"
echo "    artifacts: ${ARTIFACT_ROOT}"
[[ "${fail}" -eq 0 ]]
