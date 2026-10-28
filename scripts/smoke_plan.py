# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Build registry-derived smoke runs with small training budgets.

Quick mode selects one supported data regime per method; full mode covers all
supported regimes and declared tasks. Missing optional packages or pretrained
resources produce explicit skips. Available but broken backends remain failures.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Field separator for the emitted plan. ASCII unit separator, so that an empty
#: field survives the shell's word splitting (see :func:`main`).
SEP = "\x1f"

if "otbench" not in sys.modules:  # allow running as a plain script
    _spec = importlib.util.spec_from_file_location(
        "otbench", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    _module = importlib.util.module_from_spec(_spec)
    sys.modules["otbench"] = _module
    _spec.loader.exec_module(_module)

from otbench.methods.common.readiness import method_config, missing_requirements
from otbench.methods.common.resources import pretrained_resources, resolve_resource  # noqa: E402
from otbench.methods.common.registry import METHODS  # noqa: E402

# Case -> dataset, in the order LEVEL=quick prefers them.
DATASETS: dict[str, str] = {
    "binclass_mixed": "auto_price",
    "regression_mixed": "healthcare_insurance_expenses",
    "regression_numeric": "SolarPower",
    "multiclass_categorical": "website_phishing",
    "binclass_numeric": "appendicitis_test",
    "multiclass_numeric": "WineDataset",
}

CASE_TASK_TYPES: dict[str, str] = {
    "binclass_mixed": "binclass",
    "regression_mixed": "regression",
    "regression_numeric": "regression",
    "multiclass_categorical": "multiclass",
    "binclass_numeric": "binclass",
    "multiclass_numeric": "multiclass",
}

CASE_COLUMN_TYPES: dict[str, frozenset[str]] = {
    "binclass_mixed": frozenset({"numerical", "categorical"}),
    "regression_mixed": frozenset({"numerical", "categorical"}),
    "regression_numeric": frozenset({"numerical"}),
    "multiclass_categorical": frozenset({"categorical"}),
    "binclass_numeric": frozenset({"numerical"}),
    "multiclass_numeric": frozenset({"numerical"}),
}

# Training truncated to a wiring check. Values are `--override` arguments.
BUDGETS: dict[str, list[str]] = {
    "arf": ["model.num_trees=8", "train.max_iters=1"],
    "bn": ["train.struct_learning_n_iter=20"],
    "bn_original_units": ["train.struct_learning_n_iter=20"],
    "catboost": ["model.iterations=20"],
    "ctgan": ["train.n_iter=2"],
    "diffputer": ["train.max_epochs=2", "em.em_iterations=1", "em.num_trials=2",
                  "em.num_steps=4", "impute.trials=2"],
    "forest_vp": ["model.n_t=5", "model.duplicate_K=2", "model.n_estimators=10",
                  "model.max_depth=3", "model.n_jobs=1", "impute.repaint_r=2"],
    "forest_vp_joint": ["model.n_t=5", "model.duplicate_K=2", "model.n_estimators=10",
                         "model.max_depth=3", "model.n_jobs=1", "impute.repaint_r=2"],
    "ft_transformer": ["train.epochs=2", "train.batch_size=256"],
    "great": ["train.n_iter=1", "impute.max_retries=2"],
    "goggle": ["train.n_iter=2", "model.encoder_dim=16", "model.decoder_dim=16"],
    "hyperimpute": ["model.n_inner_iter=1"],
    "lightgbm": ["model.n_estimators=20"],
    "lightgbm_quantile": ["model.n_estimators=20"],
    "mlp": ["train.max_iter=20", "train.n_iter_no_change=5"],
    "nflow": ["train.n_iter=5", "model.n_units_hidden=16"],
    "ngboost": ["model.n_estimators=20", "train.early_stopping_rounds=5"],
    # Below ~60 trees the sampler stops placing mass on the rarer classes and
    # the adapter (correctly) refuses to return a short frame.
    "nrgboost": ["model.num_trees=60"],
    "random_forest": ["model.n_estimators=20"],
    "tabcsdi": ["train.epochs=2", "diffusion.channels=16", "diffusion.num_steps=10",
                "impute.trials=2"],
    "tabddpm": ["model.d_layers=[64,64]", "model.num_timesteps=8", "train.steps=20",
                "train.batch_size=1024"],
    "tabdiff": ["model.num_timesteps=4", "model.dim_t=64", "train.steps=2",
                "train.check_val_every=100", "sample.batch_size=2000"],
    "tabebm": ["model.sgld_steps=20", "model.max_data_size=2000"],
    # V2 attends over the whole matrix in one pass, so the context budget is
    # the memory knob: 2048 rows of a wide one-hot block asks for >14 GiB.
    "tabimpute": ["impute.max_context_rows=256"],
    "tabm": ["train.epochs=2", "train.batch_size=256"],
    "tabsyn": ["train.vae_epochs=2", "train.diff_epochs=2", "train.batch_size=2048",
               "sample.num_steps=4", "impute.num_steps=4", "impute.resample_rounds=2"],
    "tvae": ["train.n_iter=2"],
    "xgboost": ["model.n_estimators=20"],
}

# Task-level arguments that are not method-specific.
TASK_ARGS: dict[str, list[str]] = {
    "generation": ["--n-samples", "200", "--eval", "--metric", "density"],
    "imputation": ["--scenario", "MCAR", "--rate", "0.3", "--trials", "1"],
    "prediction": ["--trials", "1", "--n-samples", "8"],
}


class Row:
    __slots__ = ("method", "task", "dataset", "argv", "skip")

    def __init__(self, method: str, task: str, dataset: str,
                 argv: list[str], skip: str | None) -> None:
        self.method, self.task, self.dataset = method, task, dataset
        self.argv, self.skip = argv, skip

    @property
    def label(self) -> str:
        return f"{self.method} {self.task[:3]} {self.dataset}"


def skip_reason(name: str) -> str | None:
    """Why this method cannot run here at all, or None."""
    return "; ".join(missing_requirements(name, checkpoints_root=os.environ.get("CHECKPOINTS"))) or None


def regimes_for(name: str, level: str) -> list[str]:
    caps = METHODS[name].capabilities
    supported = [
        case for case in DATASETS
        if CASE_TASK_TYPES[case] in caps.task_types
        and CASE_COLUMN_TYPES[case] <= caps.column_types
        and caps.requires_column_types <= CASE_COLUMN_TYPES[case]
    ]
    if not supported:
        return []
    return supported if level == "full" else supported[:1]


def build(level: str = "quick", only: list[str] | None = None) -> list[Row]:
    rows: list[Row] = []
    for name in sorted(METHODS):
        if only and name not in only:
            continue
        reason = skip_reason(name)
        overrides: list[str] = []
        for value in BUDGETS.get(name, []):
            overrides += ["--override", value]
        resources = pretrained_resources(name)
        if resources:
            config = method_config(name)
            for resource in resources:
                path = resolve_resource(name, config, resource, os.environ.get("CHECKPOINTS"), require_exists=False)
                # Keep training artifacts isolated while reusing the configured
                # pretrained resource, including paths containing spaces.
                overrides += ["--override", f"model.{resource.key}={json.dumps(str(path), ensure_ascii=False)}"]
        for task in sorted(t.value for t in METHODS[name].tasks):
            for regime in regimes_for(name, level):
                dataset = DATASETS[regime]
                argv = ["--task", task, "--method", name, "--dataset", dataset]
                argv += TASK_ARGS[task] + overrides
                rows.append(Row(name, task, dataset, argv, reason))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=("quick", "full"), default="quick")
    parser.add_argument("--only", nargs="*", default=None,
                        help="restrict to these method names")
    parser.add_argument(
        "--readiness", metavar="METHOD",
        help="print READY or the checkout-level skip reason for one method",
    )
    args = parser.parse_args(argv)

    if args.readiness is not None:
        if args.readiness not in METHODS:
            parser.error(f"unknown method: {args.readiness}")
        print(skip_reason(args.readiness) or "READY")
        return 0

    # Unit-separated, not tab-separated: bash treats tabs as IFS whitespace and
    # collapses runs of them, which would swallow the empty "no skip reason"
    # field and shift every argument left by one.
    for row in build(args.level, args.only):
        print(SEP.join([row.label, row.skip or "", *row.argv]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
