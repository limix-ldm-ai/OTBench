# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Script-facing catalog derived from the method registry and dataset schemas.

This module deliberately owns no method or dataset list.  It gives shell
drivers a stable way to resolve valid method/task/dataset combinations and
assign independent evaluation workers from the same registry.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if "otbench" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "otbench", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot bootstrap OTBench from {ROOT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["otbench"] = module
    spec.loader.exec_module(module)

from otbench.methods.common.base import Task  # noqa: E402
from otbench.methods.common.capabilities import unsupported_reason  # noqa: E402
from otbench.methods.common.registry import METHODS, get_spec  # noqa: E402
from otbench.tabdata.schema import EvalSchema, load_schema  # noqa: E402
from otbench.tabdata.io import read_frame_csv  # noqa: E402

TASKS = ("generation", "imputation", "prediction")


def method_task_plan() -> list[tuple[str, str]]:
    """Return distributed implementations' rank plan, task-major then method name."""
    return [
        (name, task)
        for task in TASKS
        for name in sorted(METHODS)
        if Task(task) in METHODS[name].tasks
        and METHODS[name].capabilities.status != "external_only"
    ]


def resolve_datasets(
    method: str,
    task: str,
    store_root: str | Path,
    dataset_type: str = "real",
    requested: list[str] | None = None,
) -> list[str]:
    """Resolve valid datasets, rejecting invalid explicit requests."""
    spec = get_spec(method)
    if Task(task) not in spec.tasks:
        supported = sorted(value.value for value in spec.tasks)
        raise ValueError(f"method {method!r} does not support {task!r}; supported={supported}")

    root = Path(store_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset store does not exist: {root}")

    explicit = bool(requested)
    if explicit:
        dirs = []
        seen = set()
        for name in requested or []:
            if name in seen:
                continue
            seen.add(name)
            matches = [root / kind / name for kind in ("real", "scm") if (root / kind / name).is_dir()]
            if not matches:
                raise FileNotFoundError(f"dataset {name!r} is not present under {root}/{{real,scm}}")
            dirs.append(matches[0])
    else:
        kinds = ("real", "scm") if dataset_type == "all" else (dataset_type,)
        dirs = sorted(
            (path for kind in kinds if (root / kind).is_dir()
             for path in (root / kind).iterdir() if path.is_dir()),
            key=lambda path: path.name.casefold(),
        )

    valid: list[str] = []
    rejected: list[str] = []
    for directory in dirs:
        schema = _catalog_schema(directory)
        reason = unsupported_reason(
            spec.capabilities,
            schema,
            task,
            mask_granularity="cell" if task == "imputation" else None,
        )
        if reason is None:
            valid.append(directory.name)
        else:
            rejected.append(f"{directory.name}: {reason}")

    if explicit and rejected:
        raise ValueError("unsupported explicit dataset selection: " + "; ".join(rejected))
    if not valid:
        detail = "; ".join(rejected) if rejected else "no materialized datasets found"
        raise ValueError(f"no compatible datasets for {method}/{task}: {detail}")
    return valid


def _catalog_schema(directory: Path) -> EvalSchema:
    """Load declared schema and restore the training target's exact category domain."""
    schema_path = directory / "schema.json"
    if not schema_path.is_file():
        raise FileNotFoundError(f"missing schema.json: {directory}")
    schema = load_schema(schema_path)
    if schema.task_type != "classification" or schema.category_domain(schema.target_col) is not None:
        return schema
    train_path = directory / f"{directory.name}_train.csv"
    if not train_path.is_file():
        raise FileNotFoundError(f"missing train CSV: {train_path}")
    target = read_frame_csv(train_path, schema, usecols=[schema.target_col], strict=True)[schema.target_col]
    domains = dict(schema.category_domains)
    domains[schema.target_col] = [_python_scalar(value) for value in target.dropna().unique().tolist()]
    return replace(schema, category_domains=domains)


def _python_scalar(value):
    return value.item() if hasattr(value, "item") else value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    datasets = subparsers.add_parser("datasets", help="resolve compatible materialized datasets")
    datasets.add_argument("--method", required=True)
    datasets.add_argument("--task", required=True, choices=TASKS)
    datasets.add_argument("--store-root", type=Path, required=True)
    datasets.add_argument("--dataset-type", choices=("real", "scm", "all"), default="real")
    datasets.add_argument("dataset", nargs="*")

    rank_plan = subparsers.add_parser(
        "rank-plan", help="render the canonical one-rank-per-method-task plan"
    )
    rank_mode = rank_plan.add_mutually_exclusive_group()
    rank_mode.add_argument("--rank", type=int, help="print one zero-based rank assignment")
    rank_mode.add_argument("--count", action="store_true", help="print the number of assignments")
    rank_plan.add_argument("--header", action="store_true", help="include a TSV header")

    args = parser.parse_args(argv)
    try:
        if args.command == "datasets":
            for name in resolve_datasets(
                args.method, args.task, args.store_root, args.dataset_type, args.dataset
            ):
                print(name)
        else:
            plan = method_task_plan()
            if args.count:
                print(len(plan))
            elif args.rank is not None:
                if args.rank < 0 or args.rank >= len(plan):
                    raise ValueError(f"rank must be in [0, {len(plan) - 1}]")
                print("\t".join(plan[args.rank]))
            else:
                if args.header:
                    print("rank\tmethod\ttask")
                for rank, (method, task) in enumerate(plan):
                    print(f"{rank}\t{method}\t{task}")
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
