"""Summarize status files from an 8-node method-task rank run."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from scheduler_state import load_plan, validate_run_id, validate_status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("OTBENCH_WORKSPACE") or Path.cwd()).expanduser() / "outputs" / "rank_scheduler")
    parser.add_argument("--strict", action="store_true", help="require every frozen assignment to have succeeded")
    parser.add_argument(
        "--stale-after-seconds", type=int, default=300,
        help="report a running rank as stale when its heartbeat is older than this",
    )
    args = parser.parse_args(argv)

    try:
        validate_run_id(args.run_id)
    except ValueError as exc:
        parser.error(str(exc))
    run_root = args.root / args.run_id
    try:
        plan = load_plan(run_root / "plan.json", "rank")
        if plan["run_id"] != args.run_id:
            raise ValueError("frozen plan run_id differs from the requested run")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        state = "missing" if not run_root.exists() else "invalid_plan"
        print(f"run_id\tstate\treason\n{args.run_id}\t{state}\t{exc}")
        print(f"summary\t{state}=1")
        return 1 if args.strict else 0

    run_dir = run_root / "status"
    actual: dict[int, dict] = {}
    malformed = 0
    for path in sorted(run_dir.glob("rank_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("rank status is not a JSON object")
            rank = int(path.stem.removeprefix("rank_"))
            if rank in actual:
                raise ValueError("duplicate rank status files")
            actual[rank] = payload
        except (OSError, ValueError, KeyError, TypeError) as exc:
            malformed += 1
            print(f"WARN: cannot read {path}: {exc}", file=sys.stderr)

    rows = []
    assigned_ranks = set(range(len(plan["assignments"])))
    for assignment in plan["assignments"]:
        rank, method, task = assignment["rank"], assignment["method"], assignment["task"]
        payload = actual.get(rank, {})
        state = _validated_state(payload, plan, rank, args.stale_after_seconds)
        rows.append((rank, method, task, state, payload.get("hostname", "-")))
    # Include reserve ranks (normally 62 and 63) once their node has reported
    # them, so the same table is also a truthful 64-rank allocation view.
    for rank in sorted(set(actual) - assigned_ranks):
        payload = actual[rank]
        state = (_validated_state(payload, plan, rank, args.stale_after_seconds)
                 if 0 <= rank < int(plan["world_size"]) else "identity_mismatch")
        if state == "succeeded":
            state = "identity_mismatch"
        rows.append((rank, "-", "-", state, payload.get("hostname", "-")))
    print(f"run_id\t{plan['run_id']}\tplan_id\t{plan['plan_id']}")
    print("rank\tmethod\ttask\tstate\thost")
    for row in rows:
        print("\t".join(map(str, row)))
    counts = Counter(row[3] for row in rows)
    if malformed:
        counts["malformed"] += malformed
    print("summary\t" + " ".join(f"{state}={count}" for state, count in sorted(counts.items())))
    assignments_complete = all(row[3] == "succeeded" for row in rows if row[0] in assigned_ranks)
    reserves_valid = all(row[3] == "idle" for row in rows if row[0] not in assigned_ranks)
    if args.strict and (not assignments_complete or not reserves_valid or malformed):
        return 1
    return 0


def _validated_state(payload: dict, plan: dict, rank: int, stale_after_seconds: int) -> str:
    if not payload:
        return "missing"
    if not validate_status(payload, plan, rank):
        return "identity_mismatch"
    if rank < len(plan["assignments"]) and payload["state"] == "idle":
        return "identity_mismatch"
    return _state_with_staleness(payload, stale_after_seconds)


def _state_with_staleness(payload: dict, stale_after_seconds: int) -> str:
    state = payload.get("state", "missing")
    if state != "running" or stale_after_seconds <= 0:
        return state
    raw = payload.get("updated_at")
    if not raw:
        return "stale"
    try:
        updated = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
    except ValueError:
        return "stale"
    age = (datetime.now(timezone.utc) - updated.astimezone(timezone.utc)).total_seconds()
    return "stale" if age > stale_after_seconds else state


if __name__ == "__main__":
    raise SystemExit(main())
