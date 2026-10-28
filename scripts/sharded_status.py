"""Verify completion of every selected shard in a frozen launch plan."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

from rank_status import _state_with_staleness
from scheduler_state import load_plan, validate_run_id, validate_status


def summarize_run(run_root: Path, stale_after_seconds: int = 300) -> dict:
    try:
        plan = load_plan(run_root / "plan.json", "sharded")
        if plan["run_id"] != run_root.name:
            raise ValueError("run directory differs from the frozen run_id")
        if plan["method"] != run_root.parent.name or plan["task"] != run_root.parent.parent.name:
            raise ValueError("run directory differs from the frozen method/task")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"run_id": run_root.name, "method": "-", "task": "-",
                "state": "invalid_plan", "selection": "?", "completed": 0,
                "problem_shards": str(exc)}
    counts = Counter()
    problems = []
    expected_names = {f"shard_{slot:02d}.json" for slot in plan["selected_shards"]}
    extra = [path.name for path in (run_root / "status").glob("shard_*")
             if path.name not in expected_names]
    if extra:
        counts["identity_mismatch"] += len(extra)
        problems.extend(extra)
    for slot in plan["selected_shards"]:
        path = run_root / "status" / f"shard_{slot:02d}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            state = (_state_with_staleness(payload, stale_after_seconds)
                     if validate_status(payload, plan, slot) else "identity_mismatch")
            if state == "idle":
                state = "identity_mismatch"
        except FileNotFoundError:
            state = "missing"
        except (OSError, ValueError, KeyError, TypeError):
            state = "malformed"
        counts[state] += 1
        if state != "succeeded":
            problems.append(f"{slot}:{state}")
    state = next((name for name in ("failed", "malformed", "identity_mismatch", "stale", "missing",
                                    "running", "planned", "skipped") if counts[name]), "succeeded")
    return {"run_id": plan["run_id"], "method": plan["method"], "task": plan["task"],
            "state": state, "selection": f"{len(plan['selected_shards'])}/{plan['n_shards']}",
            "completed": counts["succeeded"], "problem_shards": ",".join(problems) or "-"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
        default=Path(os.environ.get("OTBENCH_WORKSPACE") or Path.cwd()).expanduser() / "outputs" / "sharded")
    parser.add_argument("--run-id", help="inspect one launch identity across method/task directories")
    parser.add_argument("--method")
    parser.add_argument("--task", choices=("generation", "imputation", "prediction"))
    parser.add_argument("--strict", action="store_true", help="require all selected shards to have succeeded")
    parser.add_argument("--stale-after-seconds", type=int, default=300)
    args = parser.parse_args(argv)
    if args.run_id:
        try:
            validate_run_id(args.run_id)
        except ValueError as exc:
            parser.error(str(exc))
    rows = []
    for path in sorted(args.root.glob("*/*/*/plan.json")):
        run_root = path.parent
        if args.run_id and run_root.name != args.run_id:
            continue
        if args.method and run_root.parent.name != args.method:
            continue
        if args.task and run_root.parent.parent.name != args.task:
            continue
        rows.append(summarize_run(run_root, args.stale_after_seconds))
    if not rows:
        rows.append({"run_id": args.run_id or "-", "method": args.method or "-",
                     "task": args.task or "-", "state": "missing", "selection": "?",
                     "completed": 0, "problem_shards": "no matching frozen run"})
    fields = ("run_id", "method", "task", "state", "selection", "completed", "problem_shards")
    print("\t".join(fields))
    for row in rows:
        print("\t".join(str(row[field]) for field in fields))
    counts = Counter(row["state"] for row in rows)
    print("summary\t" + " ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    return int(args.strict and any(row["state"] != "succeeded" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
