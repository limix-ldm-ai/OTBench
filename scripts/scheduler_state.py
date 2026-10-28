"""Frozen scheduler plans and identity-bound status records (standard library only)."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import sys
import tempfile
import time

SCHEMA_VERSION = 1
STATES = {"running", "planned", "succeeded", "failed", "skipped", "idle"}
SETTINGS_ENV = (
    "OTBENCH_WORKSPACE", "STORE", "OUTPUTS", "CHECKPOINTS", "EXTRA",
    "DATASET_TYPE", "DATASETS", "SEEDS", "N_TRIALS",
    "TRAIN_SEED", "TRIALS", "SCENARIOS", "RATES", "METRIC",
    "SKIP_EXISTING", "DRY_RUN", "SCHEDULER_PLAN_ONLY",
)


def validate_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or value in {".", ".."}:
        raise ValueError("run id must contain only letters, digits, dot, underscore, or dash")
    return value


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def plan_hash(payload: dict) -> str:
    identity = {key: value for key, value in payload.items() if key not in {"plan_id", "created_at"}}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def seal_plan(payload: dict) -> dict:
    payload = {"scheduler_schema_version": SCHEMA_VERSION, **payload}
    return {**payload, "plan_id": plan_hash(payload), "created_at": datetime.now(timezone.utc).isoformat()}


def load_plan(path: Path, kind: str | None = None) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("scheduler_schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported scheduler plan schema")
    if kind and payload.get("kind") != kind:
        raise ValueError(f"expected a {kind} scheduler plan")
    validate_run_id(payload["run_id"])
    if payload.get("plan_id") != plan_hash(payload):
        raise ValueError("scheduler plan fingerprint is corrupt")
    if payload.get("kind") == "rank":
        assignments = payload["assignments"]
        if not isinstance(assignments, list) or not assignments:
            raise ValueError("rank plan has no assignments")
        if [row["rank"] for row in assignments] != list(range(len(assignments))):
            raise ValueError("rank assignments must be contiguous and unique")
        if int(payload["world_size"]) < len(assignments):
            raise ValueError("world_size does not cover the frozen assignments")
        pairs = [(row["method"], row["task"]) for row in assignments]
        if len(pairs) != len(set(pairs)):
            raise ValueError("rank plan has duplicate method/task assignments")
    elif payload.get("kind") == "sharded":
        shards = payload["shards"]
        ids = [row["shard"] for row in shards]
        selected = payload["selected_shards"]
        if ids != list(range(int(payload["n_shards"]))):
            raise ValueError("shard ids do not cover the frozen plan")
        if not selected or len(selected) != len(set(selected)) or not set(selected) <= set(ids):
            raise ValueError("invalid selected_shards in frozen plan")
    else:
        raise ValueError("unknown scheduler plan kind")
    return payload


@contextmanager
def _plan_lock(run_root: Path):
    run_root.mkdir(parents=True, exist_ok=True)
    lock = run_root / ".plan.lock"
    deadline = time.monotonic() + 30
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for scheduler plan lock: {lock}")
            time.sleep(0.05)
    try:
        yield
    finally:
        lock.rmdir()


def freeze_rank_plan(run_root: Path, run_id: str, world_size: int,
                     assignments: list[tuple[str, str]], settings: dict | None = None) -> dict:
    payload = seal_plan({
        "kind": "rank", "run_id": validate_run_id(run_id), "world_size": world_size,
        "assignments": [{"rank": rank, "method": method, "task": task}
                        for rank, (method, task) in enumerate(assignments)],
        "settings": settings or {},
    })
    with _plan_lock(run_root):
        path = run_root / "plan.json"
        if path.exists():
            recorded = load_plan(path, "rank")
            if recorded["plan_id"] != payload["plan_id"]:
                raise ValueError("RUN_ID already belongs to a different frozen plan; choose a new RUN_ID")
            return recorded
        if any((run_root / "status").glob("rank_*")):
            raise ValueError("RUN_ID has statuses without a frozen plan; choose a new RUN_ID")
        atomic_json(path, payload)
    return load_plan(path, "rank")


def freeze_sharded_plan(run_root: Path, run_id: str, plan: dict,
                        selected: list[int], settings: dict | None = None) -> dict:
    payload = seal_plan({**plan, "kind": "sharded", "run_id": validate_run_id(run_id),
                         "selected_shards": selected, "settings": settings or {}})
    # One invocation owns this directory. A second invocation must never clear
    # or overwrite another subset's statuses, even when its RUN_ID was reused.
    run_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_root.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(f"run directory already exists: {run_root}; choose a new RUN_ID (no statuses were removed)") from exc
    atomic_json(run_root / "plan.json", payload)
    return load_plan(run_root / "plan.json", "sharded")


def write_status(plan_path: Path, slot: int, state: str, *, exit_code: int | None = None,
                 reason: str = "", local_rank: int | None = None) -> dict:
    if state not in STATES:
        raise ValueError(f"unknown scheduler state: {state}")
    plan = load_plan(plan_path)
    if plan["kind"] == "rank":
        if not 0 <= slot < int(plan["world_size"]):
            raise ValueError("rank is outside the frozen world_size")
        assignment = plan["assignments"][slot] if slot < len(plan["assignments"]) else {}
        method, task = assignment.get("method"), assignment.get("task")
        identity = {"rank": slot, "world_size": plan["world_size"], "local_rank": local_rank}
        name = f"rank_{slot:02d}.json"
    else:
        if slot not in plan["selected_shards"]:
            raise ValueError("shard is outside the frozen selected_shards")
        method, task = plan["method"], plan["task"]
        identity = {"shard": slot}
        name = f"shard_{slot:02d}.json"
    payload = {
        "scheduler_schema_version": SCHEMA_VERSION, "plan_id": plan["plan_id"],
        "run_id": plan["run_id"], "method": method, "task": task, **identity,
        "hostname": socket.gethostname(), "state": state, "exit_code": exit_code,
        "reason": reason or None, "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(plan_path.parent / "status" / name, payload)
    return payload


def validate_status(payload: dict, plan: dict, slot: int) -> bool:
    if not isinstance(payload, dict) or payload.get("state") not in STATES:
        return False
    expected = {"scheduler_schema_version": SCHEMA_VERSION, "plan_id": plan["plan_id"],
                "run_id": plan["run_id"]}
    if plan["kind"] == "rank":
        assignment = plan["assignments"][slot] if slot < len(plan["assignments"]) else {}
        expected.update(rank=slot, world_size=plan["world_size"],
                        method=assignment.get("method"), task=assignment.get("task"))
    else:
        expected.update(shard=slot, method=plan["method"], task=plan["task"])
    if any(payload.get(key) != value for key, value in expected.items()):
        return False
    if payload["state"] in {"succeeded", "idle"} and payload.get("exit_code") != 0:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    rank = sub.add_parser("freeze-rank")
    rank.add_argument("--run-root", type=Path, required=True)
    rank.add_argument("--run-id", required=True)
    rank.add_argument("--world-size", type=int, required=True)
    sharded = sub.add_parser("freeze-sharded")
    sharded.add_argument("--run-root", type=Path, required=True)
    sharded.add_argument("--run-id", required=True)
    sharded.add_argument("--plan-file", type=Path, required=True)
    sharded.add_argument("--selected", type=int, nargs="+", required=True)
    sharded.add_argument("--settings", default="")
    assignment = sub.add_parser("rank-assignment")
    assignment.add_argument("--plan", type=Path, required=True)
    assignment.add_argument("--rank", type=int)
    assignment.add_argument("--count", action="store_true")
    assignment.add_argument("--run-id")
    assignment.add_argument("--world-size", type=int)
    status = sub.add_parser("write-status")
    status.add_argument("--plan", type=Path, required=True)
    status.add_argument("--slot", type=int, required=True)
    status.add_argument("--state", required=True)
    status.add_argument("--exit-code", type=int)
    status.add_argument("--local-rank", type=int)
    status.add_argument("--reason", default="")
    heartbeat = sub.add_parser("heartbeat")
    heartbeat.add_argument("--plan", type=Path, required=True)
    heartbeat.add_argument("--slot", type=int, required=True)
    heartbeat.add_argument("--pid", type=int, required=True)
    heartbeat.add_argument("--interval", type=float, default=60)
    heartbeat.add_argument("--local-rank", type=int)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze-rank":
            from protocol_catalog import method_task_plan
            settings = {key: os.environ[key] for key in SETTINGS_ENV if key in os.environ}
            result = freeze_rank_plan(args.run_root, args.run_id, args.world_size, method_task_plan(), settings)
            print(result["plan_id"])
        elif args.command == "freeze-sharded":
            configured = dict(token.split("=", 1) for token in shlex.split(args.settings))
            settings = {
                "config": {key.removeprefix("CFG_").lower(): value for key, value in configured.items()},
                "environment": {key: os.environ[key] for key in SETTINGS_ENV if key in os.environ},
            }
            result = freeze_sharded_plan(args.run_root, args.run_id,
                json.loads(args.plan_file.read_text()), args.selected, settings)
            print(result["plan_id"])
        elif args.command == "rank-assignment":
            plan = load_plan(args.plan, "rank")
            if args.run_id is not None and validate_run_id(args.run_id) != plan["run_id"]:
                raise ValueError("worker run_id differs from the frozen plan")
            if args.world_size is not None and args.world_size != plan["world_size"]:
                raise ValueError("worker world_size differs from the frozen plan")
            if args.count:
                print(len(plan["assignments"]))
            elif args.rank is not None and 0 <= args.rank < len(plan["assignments"]):
                row = plan["assignments"][args.rank]
                print(f"{row['method']}\t{row['task']}")
            else:
                raise ValueError("rank is outside the frozen assignments")
        elif args.command == "heartbeat":
            if args.interval < 1:
                raise ValueError("heartbeat interval must be at least one second")
            while True:
                time.sleep(args.interval)
                try:
                    os.kill(args.pid, 0)
                except ProcessLookupError:
                    break
                write_status(args.plan, args.slot, "running", reason="heartbeat", local_rank=args.local_rank)
        else:
            write_status(args.plan, args.slot, args.state, exit_code=args.exit_code,
                         reason=args.reason, local_rank=args.local_rank)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
