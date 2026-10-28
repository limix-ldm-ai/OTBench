# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from otbench.metrics.catalog import CATALOG, METRIC_SCHEMA_VERSION, MetricResult


def build_report(
    meta: dict[str, Any],
    per_run: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate several evaluation runs of the same configuration.

    Only ``ok`` values enter the mean, and the per-metric record keeps the count
    of runs that produced one. A metric that succeeded once out of five is not
    the same as one that succeeded five times, and the aggregate has to say so.
    """
    keys = sorted({key for run in per_run for key in run.get("metrics", {})})
    aggregate: dict[str, dict[str, Any]] = {}
    for key in keys:
        raw = [run.get("metrics", {}).get(key) for run in per_run]
        values = np.array([v for v in raw if v is not None], dtype=float)
        values = values[np.isfinite(values)]
        spec = CATALOG.get(key)
        aggregate[key] = {
            "mean": float(values.mean()) if len(values) else None,
            "std": float(values.std()) if len(values) else None,
            "n_ok": int(len(values)),
            "n_runs": len(per_run),
            "direction": spec.direction if spec else None,
            "tier": spec.tier if spec else None,
            "values": [_json_safe(value) for value in raw],
        }

    status_counts: dict[str, int] = {}
    for run in per_run:
        for record in run.get("results", []):
            status_counts[record["status"]] = status_counts.get(record["status"], 0) + 1

    return {
        "schema_version": METRIC_SCHEMA_VERSION,
        "meta": meta,
        "metrics": {key: value["mean"] for key, value in aggregate.items()},
        "aggregate": aggregate,
        "status_counts": status_counts,
        "per_run": per_run,
    }


def run_payload(results: list[MetricResult], gen_path: str, **extra: Any) -> dict[str, Any]:
    """One run's contribution to a report, in the shared result protocol."""
    records = [r.to_dict() for r in results]
    return {
        "gen_path": gen_path,
        "metrics": {r["name"]: r["value"] for r in records if r["status"] == "ok"},
        "results": records,
        "skipped": {r["name"]: r.get("reason") for r in records if r["status"] != "ok"},
        **extra,
    }


def save_report(report: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(report), f, ensure_ascii=False, indent=2)


def print_summary(report: dict[str, Any]) -> None:
    print("\n=== Aggregate (mean +/- std) ===")
    for key, value in report["aggregate"].items():
        if value["mean"] is None:
            print(f"{key:56s}: -")
            continue
        arrow = {"higher_better": "^", "lower_better": "v", "target_zero": "->0",
                 "target_half": "->.5"}.get(value.get("direction") or "", "")
        print(f"{key:56s}: {value['mean']:.6f} +/- {value['std']:.6f}  {arrow}")
    if report.get("status_counts"):
        print(f"--- statuses: {report['status_counts']}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
