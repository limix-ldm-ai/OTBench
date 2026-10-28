# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Run one benchmark task through the OTBench command."""
from __future__ import annotations

# A source checkout may have any directory name.
if __package__ in (None, ""):
    import importlib.util
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parent
    _spec = importlib.util.spec_from_file_location(
        "otbench", _root / "__init__.py", submodule_search_locations=[str(_root)]
    )
    if _spec is None or _spec.loader is None:
        raise RuntimeError(f"Cannot bootstrap OTBench from {_root}")
    _package = importlib.util.module_from_spec(_spec)
    sys.modules["otbench"] = _package
    _spec.loader.exec_module(_package)
    from otbench.run import main as _main

    raise SystemExit(_main())

import argparse
import json
import sys

from otbench.methods.common.config import parse_overrides
from otbench.methods.common.registry import available_methods, get_spec
from otbench.methods.common.resources import workspace_root

TASKS = ("generation", "imputation", "prediction")
SCENARIOS = ("MCAR", "MAR", "MNAR", "MIXED_MCAR_MAR_MNAR")
UNSUPPORTED_EXIT_CODE = 3

# Reject explicit task-specific options even when their value equals a default.
_OPTION_TASKS = {
    "--n-samples": {"generation", "prediction"},
    "--eval": {"generation"},
    "--metric": set(TASKS),
    "--trials": {"imputation", "prediction"},
    **{flag: {"imputation"} for flag in (
        "--scenario", "--rate", "--selection-seed", "--mask-seed", "--impute-seed",
        "--no-direct-eval", "--no-downstream-eval", "--scientific-model", "--stability-protocol",
    )},
}


class TaskArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        argv = list(sys.argv[1:] if args is None else args)
        parsed = super().parse_args(argv, namespace)
        if parsed.list_methods:
            return parsed
        for field in ("task", "method", "dataset"):
            if getattr(parsed, field, None) is None:
                self.error(f"the following argument is required: --{field}")
        for token in argv:
            flag = token.split("=", 1)[0]
            allowed = _OPTION_TASKS.get(flag)
            if allowed is not None and parsed.task not in allowed:
                self.error(f"{flag} does not apply to task {parsed.task!r}")
        return parsed


def _json_or_auto(value: str | None) -> dict | str | None:
    if value is None or value == "auto":
        return value
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"expected JSON or 'auto', got {value!r}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit(f"expected a JSON object or 'auto', got {type(parsed).__name__}")
    return parsed


def add_common_arguments(p, *, train_controls=True, list_methods=True):
    """Arguments shared by single-setting and grid entrypoints."""
    if list_methods:
        p.add_argument("--list-methods", action="store_true", help="List registry methods and supported tasks, then exit.")
    else:
        p.set_defaults(list_methods=False)
    p.add_argument("--method", required=not list_methods, help="Registered method name; see --list-methods.")
    p.add_argument("--dataset", required=not list_methods, help="Materialized dataset name.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-seed", type=int, default=None, help="Checkpoint seed (default: --seed).")
    p.add_argument("--workspace", default=None, help="Artifact workspace (default: OTBENCH_WORKSPACE or current directory).")
    p.add_argument("--store", default=None, help="Override workspace/tabdata/store.")
    p.add_argument("--outputs", default=None, help="Override workspace/outputs/<task>.")
    p.add_argument("--checkpoints", default=None, help="Override workspace/checkpoints.")
    p.add_argument("--config", default=None, help="Alternative method config.toml.")
    p.add_argument("--override", action="append", default=[], help="Repeatable section.key=value config override.")
    p.add_argument("--device", default=None)
    if train_controls:
        p.add_argument("--train-only", action="store_true", help="Train/validate the checkpoint and skip inference.")
        p.add_argument("--force-retrain", action="store_true")
    else:
        p.set_defaults(train_only=False, force_retrain=False)
    p.add_argument("--skip-existing", action="store_true", help="Reuse provenance- and artifact-validated completed results.")
    return p


def build_parser():
    """Build the command-line parser for all tasks."""
    p = TaskArgumentParser(
        prog="otbench",
        description="Tabular generation, imputation, and prediction benchmark.",
        allow_abbrev=False,
    )
    p.add_argument("--task", choices=TASKS)
    add_common_arguments(p)
    p.add_argument("--metric", default="all", help="Evaluation metric selector (default: all).")
    p.add_argument("--n-samples", type=int, default=None, help="Generated rows, or prediction sample budget.")
    p.add_argument("--eval", action="store_true", help="Evaluate generated samples.")
    imp = p.add_argument_group("imputation")
    imp.add_argument("--scenario", default="MCAR", choices=SCENARIOS)
    imp.add_argument("--rate", type=float, default=0.3)
    imp.add_argument("--selection-seed", type=int, default=None)
    imp.add_argument("--mask-seed", type=int, default=None)
    imp.add_argument("--impute-seed", type=int, default=None)
    imp.add_argument("--no-direct-eval", action="store_true")
    imp.add_argument("--no-downstream-eval", action="store_true")
    imp.add_argument("--scientific-model", default=None, help="Pre-registered Rubin estimand as a JSON object or 'auto'.")
    imp.add_argument("--stability-protocol", default=None, help="Pre-registered stability protocol as a JSON object or 'auto'.")
    p.add_argument("--trials", type=int, default=None, help="Stochastic draws per setting (default: method config).")
    return p


def resource_arguments(args: argparse.Namespace) -> dict:
    """Resolve CLI roots once; explicit root flags take precedence."""
    root = workspace_root(getattr(args, "workspace", None))
    return {
        "store_root": args.store or root / "tabdata" / "store",
        "outputs_root": args.outputs or root / "outputs" / args.task,
        "checkpoints_root": args.checkpoints or root / "checkpoints",
    }


def run_from_args(args: argparse.Namespace) -> dict:
    spec = get_spec(args.method)
    if spec.capabilities.status == "external_only":
        from otbench.methods.common.capabilities import UnsupportedRun, unsupported_reason
        raise UnsupportedRun(unsupported_reason(spec.capabilities, None, args.task))
    if args.task not in {task.value for task in spec.tasks}:
        supported = sorted(task.value for task in spec.tasks)
        raise SystemExit(f"Method {args.method!r} does not support task {args.task!r} (supported: {supported}).")
    common = dict(
        method=args.method, dataset=args.dataset, seed=args.seed, train_seed=args.train_seed,
        config_file=args.config, overrides=parse_overrides(args.override), device=args.device,
        train_only=args.train_only, force_retrain=args.force_retrain,
        skip_existing=args.skip_existing,
        **resource_arguments(args),
    )
    if args.task == "generation":
        from otbench.generation.runner import run_generation
        return run_generation(n_samples=args.n_samples, evaluate=args.eval, metric=args.metric, **common)
    if args.task == "imputation":
        from otbench.imputation.runner import run_imputation
        return run_imputation(
            scenario=args.scenario, rate=args.rate, selection_seed=args.selection_seed,
            mask_seed=args.mask_seed, impute_seed=args.impute_seed, trials=args.trials,
            eval_direct=not args.no_direct_eval, eval_downstream=not args.no_downstream_eval,
            scientific_model=_json_or_auto(args.scientific_model),
            stability_protocol=_json_or_auto(args.stability_protocol), metric=args.metric, **common,
        )
    from otbench.prediction.runner import run_prediction
    return run_prediction(trials=args.trials, n_samples=args.n_samples, metric=args.metric, **common)


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_methods:
        print("method\ttasks\ttrainable\tavailability")
        for name in available_methods():
            spec = get_spec(name)
            print(f"{name}\t{','.join(sorted(t.value for t in spec.tasks))}\t{str(spec.trainable).lower()}\t{spec.capabilities.status}")
        return 0
    if args.method not in available_methods():
        parser.error(f"unknown method {args.method!r}; use --list-methods")
    from otbench.methods.common.capabilities import UnsupportedRun
    try:
        result = run_from_args(args)
    except UnsupportedRun as exc:
        print(f"SKIP: {exc}", file=sys.stderr)
        return UNSUPPORTED_EXIT_CODE
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
