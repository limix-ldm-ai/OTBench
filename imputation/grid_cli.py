# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Dataset-level imputation grid entry.

Unlike the ordinary one-setting CLI, this keeps one method instance alive for
all requested missingness settings so conditional imputers can fit each masked
column once and reuse it across the grid.
"""
from __future__ import annotations

import argparse
import json
import sys

from otbench.methods.common.capabilities import UnsupportedRun
from otbench.methods.common.config import parse_overrides
from otbench.run import SCENARIOS, _json_or_auto, add_common_arguments, resource_arguments

from .runner import run_imputation_grid

UNSUPPORTED_EXIT_CODE = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("OTBench imputation grid")
    add_common_arguments(parser, train_controls=False, list_methods=False)
    parser.set_defaults(task="imputation")
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, required=True)
    parser.add_argument("--rates", nargs="+", type=float, required=True)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--no-direct-eval", action="store_true")
    parser.add_argument("--no-downstream-eval", action="store_true")
    parser.add_argument("--scientific-model", default=None)
    parser.add_argument("--stability-protocol", default=None)
    parser.add_argument("--metric", default="all", help="Evaluation metric selector (default: all).")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_imputation_grid(
            method=args.method,
            dataset=args.dataset,
            scenarios=args.scenarios,
            rates=args.rates,
            seed=args.seed,
            train_seed=args.train_seed,
            trials=args.trials,
            **resource_arguments(args),
            config_file=args.config,
            overrides=parse_overrides(args.override),
            device=args.device,
            eval_direct=not args.no_direct_eval,
            eval_downstream=not args.no_downstream_eval,
            scientific_model=_json_or_auto(args.scientific_model),
            stability_protocol=_json_or_auto(args.stability_protocol),
            metric=args.metric,
            skip_existing=args.skip_existing,
        )
    except UnsupportedRun as exc:
        print(f"SKIP: {exc}", file=sys.stderr)
        return UNSUPPORTED_EXIT_CODE
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
