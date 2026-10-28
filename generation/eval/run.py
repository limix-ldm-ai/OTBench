# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
from __future__ import annotations

import argparse
import time
from pathlib import Path

from otbench.generation.eval.data import list_gen_files, load_eval_frames, read_csv
from otbench.generation.eval.metrics.ci import MAX_CI_STATEMENTS
from otbench.generation.eval.registry import EvalOptions, evaluate_selected, resolve_metrics
from otbench.generation.eval.report import build_report, print_summary, run_payload, save_report
from otbench.tabdata.schema import EvalSchema, load_schema
from otbench.generation.eval.schema_inference import infer_schema_from_frames
from otbench.tabdata.schema import enrich_schema
from otbench.paths import workspace_root


def default_store() -> Path:
    """Resolve the dataset store in the selected workspace."""
    return workspace_root() / "tabdata" / "store"


DEFAULT_STORE = default_store()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Unified tabular evaluation from train/test/gen CSVs")

    # === Convenience entry: resolve everything from the local dataset store ===
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name in the store (e.g. asia, adult). Auto-resolves "
        "train/test/schema (+graph.json -> mode=scm). Overrides --train/--test/--schema/--scm.",
    )
    parser.add_argument(
        "--store",
        default=str(DEFAULT_STORE),
        help="Root of the dataset store (default: workspace/tabdata/store).",
    )

    # Explicit CSV inputs.
    parser.add_argument("--train", default=None, help="Training/reference CSV path.")
    parser.add_argument("--test", default=None, help="Test/evaluation CSV path.")
    parser.add_argument("--gen", required=True, help="Generated CSV path, or a directory of CSV files.")
    parser.add_argument("--schema", default=None, help="Schema JSON path.")
    parser.add_argument(
        "--infer-schema",
        action="store_true",
        help="Infer schema when --schema is not provided. The last column is used as target.",
    )
    parser.add_argument("--mode", choices=["real", "scm"], default=None)
    parser.add_argument("--scm", default=None, help="SCM JSON path for CI metrics in scm mode.")
    parser.add_argument(
        "--metric",
        default="all",
        help="Metric selector: 'all', a family (validity, fidelity, support, memorization, "
        "distinguishability, utility, query, privacy, disclosure, structural, cost) or a "
        "fully qualified name such as generation.privacy.mean_min_euclidean_dcr.",
    )
    parser.add_argument("--out", default="metrics.json")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--reference-split", choices=["train", "test"], default="test")
    parser.add_argument("--privacy-max-samples", type=int, default=None)
    parser.add_argument("--n-queries", type=int, default=300,
                        help="Number of sampled workload queries per arity for Query Error.")
    parser.add_argument("--ci-alpha", type=float, default=0.01)
    parser.add_argument("--ci-bins", type=int, default=5)
    parser.add_argument(
        "--ci-max-statements",
        type=int,
        default=MAX_CI_STATEMENTS,
        help="Hard cap for the deterministic stratified SCM CI workload.",
    )
    parser.add_argument("--gen-raw", default=None,
                        help="Unaligned generator output; validity, memorization and privacy read it.")
    return parser.parse_args()


def resolve_dataset_dir(store: str, name: str) -> Path:
    """Find <store>/{scm,real}/<name> for the given dataset name."""
    root = Path(store)
    for dataset_type in ("scm", "real"):
        candidate = root / dataset_type / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Dataset {name!r} not found under {root}/scm/ or {root}/real/. "
        f"Build it first with `python -m otbench.tabdata.build_datasets {name}`."
    )


def apply_dataset_defaults(args: argparse.Namespace) -> None:
    """Fill train/test/schema/scm/mode from the store when --dataset is given.

    - <name>_train.csv / <name>_test.csv / schema.json are taken from the
      dataset dir (store files are named ``<name>_train.csv``; the old
      ``train.csv`` lookup never matched anything).
    - If graph.json exists, mode defaults to ``scm`` and --scm points at it.
    - Otherwise mode defaults to ``real``.
    Explicit flags always win over auto-resolved values.
    """
    if args.dataset is None:
        if args.mode is None:
            args.mode = "real"
        return

    dataset_dir = resolve_dataset_dir(args.store, args.dataset)
    graph_path = dataset_dir / "graph.json"

    args.train = args.train or str(dataset_dir / f"{args.dataset}_train.csv")
    args.test = args.test or str(dataset_dir / f"{args.dataset}_test.csv")
    args.schema = args.schema or str(dataset_dir / "schema.json")

    if graph_path.exists():
        # SCM dataset: enable CI metrics by default.
        if args.mode is None:
            args.mode = "scm"
        if args.mode == "scm":
            args.scm = args.scm or str(graph_path)
    else:
        if args.mode is None:
            args.mode = "real"


def check_gen_alignment(schema: EvalSchema, gen_path: str) -> None:
    """Emit a clear, actionable warning if a gen CSV is missing schema columns.

    eval.data.align_columns raises a hard ValueError on missing columns; this
    surfaces missing columns or graph nodes at the evaluation boundary.
    """
    gen_cols = set(read_csv(gen_path).columns)
    required = schema.all_columns
    missing = [c for c in required if c not in gen_cols]
    if missing:
        print(
            f"[align-check] WARNING: generated file is missing {len(missing)} schema column(s): "
            f"{missing}\n"
            f"             gen={gen_path}\n"
            f"             This usually means the generator's preprocessing dropped them "
            f"(e.g. constant features). Run generation through otbench.generation (the "
            f"runner back-fills these) so SCM graph nodes stay aligned."
        )


def main() -> None:
    args = parse_args()
    start = time.time()

    apply_dataset_defaults(args)
    if not args.train or not args.test:
        raise ValueError("Provide --dataset <name>, or both --train and --test explicitly.")

    selected = resolve_metrics(args.metric)
    gen_files = list_gen_files(args.gen)
    schema, schema_source = resolve_schema(args, gen_files[0])

    for gen_path in gen_files:
        check_gen_alignment(schema, gen_path)

    options = EvalOptions(
        mode=args.mode,
        scm_path=args.scm,
        reference_split=args.reference_split,
        privacy_max_samples=args.privacy_max_samples,
        ci_alpha=args.ci_alpha,
        ci_bins=args.ci_bins,
        ci_max_statements=args.ci_max_statements,
        seed=args.seed,
        n_queries=args.n_queries,
    )

    per_run = []
    for gen_path in gen_files:
        frames = load_eval_frames(
            args.train, args.test, gen_path, schema, gen_raw_path=args.gen_raw
        )
        per_run.append(run_payload(evaluate_selected(frames, selected, options), gen_path))

    meta = {
        "dataset": args.dataset,
        "train": str(Path(args.train)),
        "test": str(Path(args.test)),
        "gen": str(Path(args.gen)),
        "schema": str(Path(args.schema)) if args.schema else None,
        "schema_source": schema_source,
        "mode": args.mode,
        "scm": str(Path(args.scm)) if args.scm else None,
        "metric_arg": args.metric,
        "selected_metrics": sorted(selected),
        "seed": args.seed,
        "ci_test": schema.ci_test,
        "ci_max_statements": args.ci_max_statements,
        "target_col": schema.target_col,
        "numerical_cols": schema.numerical_cols,
        "categorical_cols": schema.categorical_cols,
        "id_cols": schema.id_cols,
        "elapsed_sec": time.time() - start,
    }
    report = build_report(meta, per_run)
    save_report(report, args.out)
    print_summary(report)
    print(f"\nSaved to: {args.out}")


def resolve_schema(args: argparse.Namespace, first_gen_path: str):
    if args.schema:
        declared = load_schema(args.schema)
        train_df = read_csv(args.train, declared)
        return enrich_schema(train_df, declared), "file+train_enriched"
    if not args.infer_schema:
        raise ValueError("Either provide --schema schema.json / --dataset, or pass --infer-schema.")

    train_df = read_csv(args.train)
    test_df = read_csv(args.test)
    gen_df = read_csv(first_gen_path)
    return enrich_schema(
        train_df, infer_schema_from_frames(train_df, test_df, gen_df)
    ), "inferred+train_enriched"


if __name__ == "__main__":
    main()
