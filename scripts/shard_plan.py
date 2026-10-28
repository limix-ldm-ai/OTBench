#!/usr/bin/env python3
# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Cost-balanced dataset shards for the single-node sharded launcher.

``run_sharded.sh`` runs ONE method/task over the whole dataset store on several
local GPUs by giving each GPU a disjoint subset of datasets.  Splitting by
dataset count would be badly unbalanced -- the store spans three orders of
magnitude in size -- so shards are packed by a cost proxy read from each
dataset's ``meta.json``:

    cost = (n_train + n_test) * n_features

The proxy is deliberately task-independent: imputation multiplies every dataset
by the same 12 settings and 3 seeds, and generation by the same 3 seeds, so
those factors cannot change the relative balance.

Datasets the method/task cannot run on are dropped BEFORE packing (the registry
decides, via ``protocol_catalog``), so a regression-only or classification-only
method still gets balanced shards instead of shards full of skipped work.

This module also resolves the launcher's run settings. ``run_sharded.sh`` can be
driven by a TOML file, by command-line flags, or by a file plus flags that
override it; all three end up in ``load_config``, so the schema and the
validation rules have exactly one implementation.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path

from protocol_catalog import TASKS, resolve_datasets

KINDS = ("real", "scm")
DEFAULT_SHARDS = 8

# Config keys and how each is rendered for the shell.  A key is set either by
# the TOML file (native types) or by a --set override (always a string), so
# every kind below accepts both.  "words" takes a TOML array or an
# already-space-separated string and always emits one string, because every task
# driver consumes these as space-separated env vars.
CONFIG_SCHEMA: dict[str, str] = {
    "method": "str",
    "task": "str",
    "shards": "int",
    "run_shards": "words",
    "dataset_type": "str",
    "store": "str",
    "workspace": "str",
    "run_id": "str",
    "gpus": "words",
    "max_workers": "int",
    "skip_existing": "bool",
    "dry_run": "bool",
    "extra": "str",
    "seeds": "words",
    "n_trials": "int",
    "metric": "str",
    "scenarios": "words",
    "rates": "words",
    "train_seed": "int",
}


# --------------------------------------------------------------------------- #
# dataset sizes
def dataset_dir(store_root: Path, name: str) -> Path:
    for kind in KINDS:
        if (store_root / kind / name).is_dir():
            return store_root / kind / name
    raise FileNotFoundError(f"dataset {name!r} is absent from {store_root}/{{real,scm}}")


def dataset_size(store_root: Path, name: str) -> dict:
    """Read one dataset's shape from meta.json (no CSV parsing)."""
    directory = dataset_dir(store_root, name)
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    n_rows = int(meta["n_train"]) + int(meta["n_test"])
    n_features = int(meta["n_features"])
    return {
        "dataset": name,
        "kind": directory.parent.name,
        "task_type": meta["task_type"],
        "n_train": int(meta["n_train"]),
        "n_test": int(meta["n_test"]),
        "n_rows": n_rows,
        "n_features": n_features,
        "cost": n_rows * n_features,
    }


def store_sizes(store_root: Path, dataset_type: str = "all") -> list[dict]:
    kinds = KINDS if dataset_type == "all" else (dataset_type,)
    names = [
        path.name
        for kind in kinds
        if (store_root / kind).is_dir()
        for path in sorted((store_root / kind).iterdir())
        if path.is_dir()
    ]
    return [dataset_size(store_root, name) for name in names]


# --------------------------------------------------------------------------- #
# packing
def pack_shards(sizes: list[dict], n_shards: int) -> list[list[dict]]:
    """Longest-processing-time-first packing; deterministic for a given input.

    LPT is the standard greedy for minimizing makespan on identical machines and
    is within 4/3 of optimal.  The real bound here is the largest single dataset:
    a dataset is never split, so a method whose compatible set is dominated by
    one huge dataset cannot balance better than that dataset's cost.
    """
    shards: list[list[dict]] = [[] for _ in range(n_shards)]
    loads = [0] * n_shards
    for item in sorted(sizes, key=lambda size: (-size["cost"], size["dataset"])):
        target = min(range(n_shards), key=lambda i: (loads[i], i))
        shards[target].append(item)
        loads[target] += item["cost"]
    return shards


def build_plan(
    method: str,
    task: str,
    store_root: Path,
    dataset_type: str,
    n_shards: int,
) -> dict:
    if n_shards < 1:
        raise ValueError("shards must be >= 1")
    compatible = resolve_datasets(method, task, store_root, dataset_type)
    sizes = [dataset_size(store_root, name) for name in compatible]
    shards = pack_shards(sizes, n_shards)
    loads = [sum(item["cost"] for item in shard) for shard in shards]
    total = sum(loads)
    ideal = total / n_shards if n_shards else 0
    return {
        "method": method,
        "task": task,
        "store_root": str(store_root),
        "dataset_type": dataset_type,
        "cost_model": "(n_train + n_test) * n_features",
        "n_shards": n_shards,
        "n_datasets": len(sizes),
        "total_cost": total,
        "ideal_cost_per_shard": ideal,
        "balance_makespan_over_ideal": (max(loads) / ideal) if ideal else None,
        "shards": [
            {
                "shard": i,
                "n_datasets": len(shard),
                "cost": loads[i],
                "datasets": [item["dataset"] for item in shard],
                "detail": shard,
            }
            for i, shard in enumerate(shards)
        ],
    }


# --------------------------------------------------------------------------- #
# run config
def _render_words(key: str, value) -> str:
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, (list, tuple)):
        return " ".join(_render_scalar(key, item) for item in value)
    return _render_scalar(key, value)


def _render_scalar(key: str, value) -> str:
    if isinstance(value, bool):
        raise ValueError(f"{key}: booleans are not valid here")
    if isinstance(value, float):
        return repr(value)
    return str(value)


TRUE_WORDS = ("true", "1", "yes", "on")
FALSE_WORDS = ("false", "0", "no", "off")


def _coerce(key: str, kind: str, value) -> str:
    """Normalize one TOML-native or command-line value to its shell string."""
    if kind == "bool":
        if isinstance(value, bool):
            return "1" if value else "0"
        word = value.strip().lower() if isinstance(value, str) else None
        if word in TRUE_WORDS:
            return "1"
        if word in FALSE_WORDS:
            return "0"
        raise ValueError(f"{key} must be true or false; got {value!r}")
    if kind == "int":
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        if isinstance(value, str) and value.strip().lstrip("+-").isdigit():
            return str(int(value))
        raise ValueError(f"{key} must be an integer; got {value!r}")
    if kind == "str":
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string; got {value!r}")
        return value.strip()
    return _render_words(key, value)


def load_config(path: Path | None, overrides: Sequence[str] = ()) -> dict[str, str]:
    """Merge the TOML run config with ``key=value`` overrides and validate it.

    ``path`` may be None (command-line-only launch) and overrides win over the
    file, so one shared config can be reused by many machines that each pass a
    different method.
    """
    raw: dict = {}
    if path is not None:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
        nested = sorted(key for key, value in raw.items() if isinstance(value, dict))
        if nested:
            raise ValueError(f"config must be a flat table; got section(s): {', '.join(nested)}")
    for override in overrides:
        key, separator, value = override.partition("=")
        if not separator:
            raise ValueError(f"override must be key=value; got {override!r}")
        raw[key.strip()] = value

    unknown = sorted(set(raw) - set(CONFIG_SCHEMA))
    if unknown:
        raise ValueError(
            f"unknown key(s): {', '.join(unknown)}; valid keys (command-line flags are the "
            f"same names with dashes): {', '.join(sorted(CONFIG_SCHEMA))}"
        )

    config: dict[str, str] = {key: "" for key in CONFIG_SCHEMA}
    for key, kind in CONFIG_SCHEMA.items():
        if key in raw:
            config[key] = _coerce(key, kind, raw[key])

    if not config["method"]:
        raise ValueError("method is required (config key 'method' or --method)")
    if config["task"] not in TASKS:
        raise ValueError(
            f"task (config key 'task' or --task) must be one of {', '.join(TASKS)}; "
            f"got {config['task']!r}"
        )
    if not config["shards"]:
        config["shards"] = str(DEFAULT_SHARDS)
    if int(config["shards"]) < 1:
        raise ValueError("shards must be >= 1")
    if config["max_workers"] and int(config["max_workers"]) < 1:
        raise ValueError("max_workers must be >= 1")
    if not config["dataset_type"]:
        config["dataset_type"] = "all"
    if config["dataset_type"] not in ("real", "scm", "all"):
        raise ValueError("dataset_type must be real, scm or all")

    if config["seeds"]:
        if config["task"] == "prediction":
            raise ValueError("seeds only applies to generation or imputation; prediction uses train_seed")
        tokens = config["seeds"].split()
        if any(not token.isdigit() or str(int(token)) != token for token in tokens):
            raise ValueError("seeds must contain non-negative integers")
        if len(tokens) != len(set(tokens)):
            raise ValueError("seeds must not contain duplicates")
    if config["n_trials"] and config["task"] != "generation":
        raise ValueError("n_trials only applies to task='generation'")
    if config["task"] == "generation":
        if config["seeds"] and config["n_trials"]:
            raise ValueError("seeds cannot be combined with n_trials")
        if not config["seeds"]:
            config["n_trials"] = config["n_trials"] or "3"
        if config["n_trials"] and int(config["n_trials"]) < 1:
            raise ValueError("n_trials must be >= 1")
    if config["train_seed"] and config["task"] != "prediction":
        raise ValueError("train_seed only applies to prediction; imputation uses seeds")
    if config["task"] in ("generation", "imputation") and not config["skip_existing"]:
        config["skip_existing"] = "1"

    if not config["run_shards"]:
        config["run_shards"] = "all"
    if config["run_shards"] != "all":
        if len(config["run_shards"].split()) != len(set(config["run_shards"].split())):
            raise ValueError("run_shards must not contain duplicates")
        for token in config["run_shards"].split():
            if not token.isdigit() or int(token) >= int(config["shards"]):
                raise ValueError(
                    f"run_shards must be 'all' or shard ids in [0, {int(config['shards']) - 1}]"
                )
    return config


# --------------------------------------------------------------------------- #
def _resolved_store_root(value: Path | None) -> Path:
    workspace = Path(os.environ.get("OTBENCH_WORKSPACE") or Path.cwd()).expanduser().resolve()
    root = Path(value).expanduser().resolve() if value else workspace / "tabdata" / "store"
    if not root.is_dir():
        raise FileNotFoundError(f"dataset store does not exist: {root}")
    return root


def _cmd_sizes(args: argparse.Namespace) -> None:
    sizes = store_sizes(_resolved_store_root(args.store_root), args.dataset_type)
    sizes.sort(key=lambda size: -size["cost"])
    total = sum(size["cost"] for size in sizes)
    print(f"{'kind':5} {'dataset':40} {'rows':>8} {'feat':>5} {'cost':>10} {'share%':>7}  task")
    for size in sizes:
        share = 100 * size["cost"] / total if total else 0
        print(
            f"{size['kind']:5} {size['dataset']:40} {size['n_rows']:8d} "
            f"{size['n_features']:5d} {size['cost']:10d} {share:7.2f}  {size['task_type']}"
        )
    print(f"\n{len(sizes)} dataset(s), total cost {total} ((n_train + n_test) * n_features)")


def _cmd_plan(args: argparse.Namespace) -> None:
    plan = build_plan(
        args.method,
        args.task,
        _resolved_store_root(args.store_root),
        args.dataset_type,
        args.shards,
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    for shard in plan["shards"]:
        print(
            f"{shard['shard']}\t{shard['n_datasets']}\t{shard['cost']}\t"
            f"{' '.join(shard['datasets'])}"
        )


def _cmd_config(args: argparse.Namespace) -> None:
    config = load_config(args.file, args.set)
    for key, value in config.items():
        print(f"CFG_{key.upper()}={shlex.quote(value)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sizes = subparsers.add_parser("sizes", help="print every dataset's shape and cost")
    sizes.add_argument("--store-root", type=Path)
    sizes.add_argument("--dataset-type", choices=("real", "scm", "all"), default="all")

    plan = subparsers.add_parser("plan", help="print cost-balanced shards as TSV")
    plan.add_argument("--method", required=True)
    plan.add_argument("--task", required=True, choices=TASKS)
    plan.add_argument("--shards", type=int, default=DEFAULT_SHARDS)
    plan.add_argument("--store-root", type=Path)
    plan.add_argument("--dataset-type", choices=("real", "scm", "all"), default="all")
    plan.add_argument("--json", type=Path, help="also write the full plan as JSON")

    config = subparsers.add_parser("config", help="validate run settings as shell assignments")
    config.add_argument("--file", type=Path, help="TOML baseline (optional)")
    config.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="override one key; repeatable; wins over --file",
    )

    args = parser.parse_args(argv)
    handlers = {"sizes": _cmd_sizes, "plan": _cmd_plan, "config": _cmd_config}
    try:
        handlers[args.command](args)
    except (FileNotFoundError, KeyError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
