#!/usr/bin/env python3
# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Torchrun worker for LimiX2 target-observed imputation.

The scheduler starts one ``torchrun`` agent per node.  Torchrun then starts one
copy of this worker per local GPU.  Work is partitioned globally as
``dataset x seed`` units; every unit keeps its complete mechanism/rate grid in
one ``run_imputation_all.sh`` call so the LimiX2 model and fitted state are
reused across all settings.

This is an embarrassingly parallel benchmark launcher, not distributed model
training.  Distributed environment variables are removed before invoking the
ordinary task driver so LimiX2 never tries to join a process group.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import tomllib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable
from typing import Any


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from protocol_catalog import resolve_datasets  # noqa: E402
from shard_plan import dataset_size  # noqa: E402
from otbench.paths import workspace_root  # noqa: E402


METHOD = "limix2"
TASK = "imputation"
DEFAULT_SCENARIOS = ("MCAR", "MAR", "MNAR", "MIXED_MCAR_MAR_MNAR")
DEFAULT_RATES = ("0.1", "0.3", "0.5")
DEFAULT_SEEDS = (0, 1, 2)
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
DIST_ENV_KEYS = {
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "GROUP_WORLD_SIZE",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "NODE_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_ERROR_FILE",
}


@dataclass(frozen=True)
class WorkUnit:
    dataset: str
    seed: int
    cost: int
    dataset_type: str
    task_type: str
    n_train: int
    n_test: int
    n_features: int
    n_settings: int

    @property
    def key(self) -> str:
        return f"{self.dataset}__seed{self.seed}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def plan_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def parse_rank_env() -> tuple[int, int, int, int]:
    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    missing = [key for key in required if key not in os.environ]
    if missing:
        raise RuntimeError("torchrun environment is incomplete: " + ", ".join(missing))
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    if not (0 <= rank < world_size):
        raise ValueError(f"invalid global rank {rank} for world size {world_size}")
    if not (0 <= local_rank < local_world_size):
        raise ValueError(
            f"invalid local rank {local_rank} for local world size {local_world_size}"
        )
    return rank, local_rank, world_size, local_world_size


def validate_unique_nonnegative(values: list[int], label: str) -> None:
    if not values:
        raise ValueError(f"{label} must not be empty")
    if any(value < 0 for value in values):
        raise ValueError(f"{label} must contain non-negative integers")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must not contain duplicates")


def validate_protocol(args: argparse.Namespace) -> tuple[Path, str]:
    config_path = ROOT / "methods" / METHOD / "config.toml"
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    target_access = str(config.get("impute", {}).get("target_access", "")).strip()
    if target_access != "target_observed":
        raise RuntimeError(
            f"{config_path} must set impute.target_access='target_observed'; "
            f"found {target_access!r}"
        )
    validate_unique_nonnegative(args.seeds, "seeds")
    if len(args.scenarios) != len(set(args.scenarios)):
        raise ValueError("scenarios must not contain duplicates")
    if len(args.rates) != len(set(args.rates)):
        raise ValueError("rates must not contain duplicates")
    if not args.scenarios or not args.rates:
        raise ValueError("scenarios and rates must not be empty")
    for raw in args.rates:
        value = float(raw)
        if not 0.0 < value < 1.0:
            raise ValueError(f"rate must lie in (0, 1), got {raw!r}")
    return config_path, target_access


def build_units(args: argparse.Namespace) -> tuple[list[str], list[WorkUnit]]:
    store_root = args.store.resolve()
    datasets = resolve_datasets(
        METHOD,
        TASK,
        store_root,
        dataset_type=args.dataset_type,
        requested=args.datasets or None,
    )
    n_settings = len(args.scenarios) * len(args.rates)
    units: list[WorkUnit] = []
    for name in datasets:
        size = dataset_size(store_root, name)
        # Scenario/rate count is explicit in the cost even though it is common
        # to every unit.  This keeps plan accounting truthful if a reduced grid
        # is requested for a test launch.
        unit_cost = int(size["cost"]) * n_settings
        for seed in args.seeds:
            units.append(
                WorkUnit(
                    dataset=name,
                    seed=seed,
                    cost=unit_cost,
                    dataset_type=str(size["kind"]),
                    task_type=str(size["task_type"]),
                    n_train=int(size["n_train"]),
                    n_test=int(size["n_test"]),
                    n_features=int(size["n_features"]),
                    n_settings=n_settings,
                )
            )
    return datasets, units


def pack_units(units: list[WorkUnit], world_size: int) -> list[list[WorkUnit]]:
    """Deterministic LPT packing of dataset-seed units over global ranks."""
    if world_size < 1:
        raise ValueError("world_size must be positive")
    bins: list[list[WorkUnit]] = [[] for _ in range(world_size)]
    costs = [0] * world_size
    ordered = sorted(units, key=lambda unit: (-unit.cost, unit.dataset.casefold(), unit.seed))
    for unit in ordered:
        rank = min(range(world_size), key=lambda item: (costs[item], len(bins[item]), item))
        bins[rank].append(unit)
        costs[rank] += unit.cost
    return bins


def make_plan(
    args: argparse.Namespace,
    world_size: int,
    config_path: Path,
    target_access: str,
) -> dict[str, Any]:
    datasets, units = build_units(args)
    packed = pack_units(units, world_size)
    total_cost = sum(unit.cost for unit in units)
    ideal_cost = total_cost / world_size if world_size else 0.0
    ranks = []
    for rank, rank_units in enumerate(packed):
        rank_cost = sum(unit.cost for unit in rank_units)
        ranks.append(
            {
                "rank": rank,
                "n_units": len(rank_units),
                "n_settings": sum(unit.n_settings for unit in rank_units),
                "cost": rank_cost,
                "cost_over_ideal": rank_cost / ideal_cost if ideal_cost else 0.0,
                "units": [asdict(unit) | {"key": unit.key} for unit in rank_units],
            }
        )
    deterministic = {
        "schema_version": 1,
        "method": METHOD,
        "task": TASK,
        "information_contract": target_access,
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "dataset_type": args.dataset_type,
        "store": str(args.store.resolve()),
        "requested_datasets": args.datasets,
        "datasets": datasets,
        "seeds": args.seeds,
        "scenarios": args.scenarios,
        "rates": args.rates,
        "settings_per_dataset_seed": len(args.scenarios) * len(args.rates),
        "n_datasets": len(datasets),
        "n_units": len(units),
        "n_logical_settings": len(units) * len(args.scenarios) * len(args.rates),
        "world_size": world_size,
        "packing": "deterministic LPT over dataset-seed units",
        "cost_proxy": "(n_train+n_test)*n_features*n_scenarios*n_rates",
        "ranks": ranks,
    }
    deterministic["plan_fingerprint"] = plan_fingerprint(deterministic)
    return deterministic


def wait_for_file(path: Path, timeout_seconds: int, label: str) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if path.is_file():
            return
        time.sleep(1.0)
    raise TimeoutError(f"timed out waiting for {label}: {path}")


def initialize_run_root(
    run_root: Path,
    plan: dict[str, Any],
    rank: int,
    timeout_seconds: int,
) -> None:
    ready = run_root / "PLAN_READY"
    if rank == 0:
        if run_root.exists():
            raise FileExistsError(
                f"run directory already exists: {run_root}; choose a new RUN_ID "
                "(completed result reuse is handled by SKIP_EXISTING)"
            )
        for directory in (run_root / "status", run_root / "done", run_root / "logs"):
            directory.mkdir(parents=True, exist_ok=False)
        atomic_json(run_root / "plan.json", plan | {"created_at_utc": utc_now()})
        ready.write_text(plan["plan_fingerprint"] + "\n", encoding="utf-8")
    else:
        wait_for_file(ready, timeout_seconds, "rank-0 plan")

    wait_for_file(run_root / "plan.json", timeout_seconds, "plan.json")
    recorded = json.loads((run_root / "plan.json").read_text(encoding="utf-8"))
    if recorded.get("plan_fingerprint") != plan["plan_fingerprint"]:
        raise RuntimeError(
            "local plan does not match rank-0 plan; check shared dataset store, "
            "topology, arguments, and code/config versions"
        )


def select_gpu(local_rank: int, local_world_size: int) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible not in {"-1", "NoDevFiles"}:
        tokens = [token.strip() for token in visible.split(",") if token.strip()]
        if len(tokens) < local_world_size:
            raise RuntimeError(
                f"CUDA_VISIBLE_DEVICES exposes {len(tokens)} device(s), but torchrun "
                f"created {local_world_size} local workers"
            )
        return tokens[local_rank]
    return str(local_rank)


def child_environment(args: argparse.Namespace, seed: int, gpu: str) -> dict[str, str]:
    env = dict(os.environ)
    for key in DIST_ENV_KEYS:
        env.pop(key, None)
    env.update(
        {
            "OTBENCH_GPUS": gpu,
            "SEEDS": str(seed),
            "SCENARIOS": " ".join(args.scenarios),
            "RATES": " ".join(args.rates),
            "TRIALS": "1",
            "SKIP_EXISTING": "1" if args.skip_existing else "0",
            "DRY_RUN": "1" if args.dry_run else "0",
            "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", "1"),
            "MKL_NUM_THREADS": env.get("MKL_NUM_THREADS", "1"),
            "OPENBLAS_NUM_THREADS": env.get("OPENBLAS_NUM_THREADS", "1"),
            "NUMEXPR_NUM_THREADS": env.get("NUMEXPR_NUM_THREADS", "1"),
            "LOKY_MAX_CPU_COUNT": env.get("LOKY_MAX_CPU_COUNT", "1"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    env["OTBENCH_WORKSPACE"] = str(workspace_root(getattr(args, "workspace", None)))
    env["PYTHON"] = sys.executable
    env["STORE"] = str(args.store.resolve())
    return env


def status_payload(
    *,
    args: argparse.Namespace,
    rank: int,
    local_rank: int,
    world_size: int,
    state: str,
    assigned: list[dict[str, Any]],
    completed: list[str],
    failures: list[dict[str, Any]],
    current: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    return {
        "run_id": args.run_id,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "hostname": os.uname().nodename,
        "state": state,
        "current_unit": current,
        "assigned_units": len(assigned),
        "completed_units": len(completed),
        "failed_units": len(failures),
        "completed": completed,
        "failures": failures,
        "message": message,
        "updated_at_utc": utc_now(),
    }


_ACTIVE_PROCESS: subprocess.Popen[Any] | None = None


def _forward_signal(signum: int, _frame: Any) -> None:
    global _ACTIVE_PROCESS
    if _ACTIVE_PROCESS is not None and _ACTIVE_PROCESS.poll() is None:
        try:
            os.killpg(_ACTIVE_PROCESS.pid, signum)
        except ProcessLookupError:
            pass
    raise SystemExit(128 + signum)


def run_unit(
    args: argparse.Namespace,
    unit: dict[str, Any],
    gpu: str,
    log_path: Path,
    heartbeat: Callable[[], None],
) -> int:
    global _ACTIVE_PROCESS
    command = ["bash", str(SCRIPTS / "run_imputation_all.sh"), METHOD, unit["dataset"]]
    env = child_environment(args, int(unit["seed"]), gpu)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"# started_at_utc={utc_now()}\n")
        log.write(f"# unit={unit['key']} gpu={gpu}\n")
        log.write("# command=" + " ".join(command) + "\n")
        log.flush()
        _ACTIVE_PROCESS = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while True:
            rc = _ACTIVE_PROCESS.poll()
            if rc is not None:
                break
            heartbeat()
            time.sleep(args.heartbeat_seconds)
        log.write(f"# finished_at_utc={utc_now()} exit_code={rc}\n")
        log.flush()
    _ACTIVE_PROCESS = None
    return int(rc)


def wait_for_all_ranks(
    run_root: Path,
    world_size: int,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    expected = [run_root / "done" / f"rank_{rank:03d}.json" for rank in range(world_size)]
    while time.monotonic() < deadline:
        if all(path.is_file() for path in expected):
            return [json.loads(path.read_text(encoding="utf-8")) for path in expected]
        time.sleep(5.0)
    missing = [path.name for path in expected if not path.is_file()]
    raise TimeoutError("timed out waiting for rank completion: " + ", ".join(missing))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset-type", choices=("real", "scm", "all"), default="all")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--store", type=Path)
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--rates", nargs="+", default=list(DEFAULT_RATES))
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--heartbeat-seconds", type=int, default=60)
    parser.add_argument("--plan-timeout-seconds", type=int, default=300)
    parser.add_argument("--barrier-timeout-seconds", type=int, default=172800)
    parser.add_argument(
        "--control-root",
        type=Path,
        default=None,
        help="directory for plans, rank status, and launcher logs (not metric outputs)",
    )
    args = parser.parse_args(argv)
    args.workspace = workspace_root(args.workspace)
    args.store = (args.store or args.workspace / "tabdata" / "store").expanduser().resolve()
    args.control_root = (args.control_root or args.workspace / "outputs" / "multinode" / TASK / METHOD).expanduser().resolve()
    if not RUN_ID_RE.fullmatch(args.run_id) or args.run_id in {".", ".."}:
        parser.error("--run-id may contain only letters, digits, dot, underscore, and dash")
    if args.heartbeat_seconds < 5:
        parser.error("--heartbeat-seconds must be >= 5")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rank, local_rank, world_size, local_world_size = parse_rank_env()
    config_path, target_access = validate_protocol(args)
    plan = make_plan(args, world_size, config_path, target_access)
    run_root = args.control_root.expanduser().resolve() / args.run_id
    initialize_run_root(run_root, plan, rank, args.plan_timeout_seconds)

    status_file = run_root / "status" / f"rank_{rank:03d}.json"
    done_file = run_root / "done" / f"rank_{rank:03d}.json"
    assigned = plan["ranks"][rank]["units"]
    completed: list[str] = []
    failures: list[dict[str, Any]] = []

    def update(state: str, current: str | None = None, message: str | None = None) -> None:
        atomic_json(
            status_file,
            status_payload(
                args=args,
                rank=rank,
                local_rank=local_rank,
                world_size=world_size,
                state=state,
                assigned=assigned,
                completed=completed,
                failures=failures,
                current=current,
                message=message,
            ),
        )

    print(
        f"[rank {rank:03d}/{world_size}] local_rank={local_rank} "
        f"units={len(assigned)} settings={sum(unit['n_settings'] for unit in assigned)}",
        flush=True,
    )
    if args.plan_only:
        update("planned", message="plan-only validation")
    else:
        gpu = select_gpu(local_rank, local_world_size)
        update("running")
        for index, unit in enumerate(assigned):
            key = str(unit["key"])
            log_path = run_root / "logs" / f"rank_{rank:03d}" / f"{index:03d}_{key}.log"
            print(
                f"[rank {rank:03d}] start {index + 1}/{len(assigned)} {key} gpu={gpu}",
                flush=True,
            )

            def heartbeat() -> None:
                update("running", current=key, message="heartbeat")

            rc = run_unit(args, unit, gpu, log_path, heartbeat)
            if rc == 0:
                completed.append(key)
                print(f"[rank {rank:03d}] done {key}", flush=True)
            else:
                failures.append({"unit": key, "exit_code": rc, "log": str(log_path)})
                print(f"[rank {rank:03d}] FAIL {key} exit={rc} log={log_path}", flush=True)
            update("running", message="unit finished")
        update("completed" if not failures else "completed_with_failures")

    rank_done = status_payload(
        args=args,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        state="planned" if args.plan_only else ("succeeded" if not failures else "failed"),
        assigned=assigned,
        completed=completed,
        failures=failures,
        message="rank reached global completion barrier",
    )
    atomic_json(done_file, rank_done)
    all_done = wait_for_all_ranks(run_root, world_size, args.barrier_timeout_seconds)

    summary_file = run_root / "SUMMARY.json"
    if rank == 0:
        summary = {
            "run_id": args.run_id,
            "plan_fingerprint": plan["plan_fingerprint"],
            "world_size": world_size,
            "n_datasets": plan["n_datasets"],
            "n_units": plan["n_units"],
            "n_logical_settings": plan["n_logical_settings"],
            "completed_units": sum(item["completed_units"] for item in all_done),
            "failed_units": sum(item["failed_units"] for item in all_done),
            "rank_states": {str(item["rank"]): item["state"] for item in all_done},
            "finished_at_utc": utc_now(),
            "plan_only": args.plan_only,
        }
        atomic_json(summary_file, summary)
    else:
        wait_for_file(summary_file, args.plan_timeout_seconds, "SUMMARY.json")
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    if int(summary["failed_units"]) > 0:
        return 1
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _forward_signal)
    signal.signal(signal.SIGINT, _forward_signal)
    raise SystemExit(main())
