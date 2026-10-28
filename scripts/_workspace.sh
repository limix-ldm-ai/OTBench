#!/usr/bin/env bash
# Preserve the caller's workspace when batch workers change directory.
PYTHON="${PYTHON:-python}"
WORKSPACE_ROOT="$("${PYTHON}" -c 'import os; from pathlib import Path; print(Path(os.environ.get("OTBENCH_WORKSPACE") or Path.cwd()).expanduser().resolve())')"
export OTBENCH_WORKSPACE="${WORKSPACE_ROOT}"

workspace_path() {
  "${PYTHON}" -c 'import sys; from pathlib import Path; print(Path(sys.argv[1]).expanduser().resolve())' "$1"
}
