# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Generation runner: checkpoint-aware sampling + optional evaluation."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from otbench.methods.common import checkpoints as ckpt_utils
from otbench.methods.common import data as data_utils
from otbench.methods.common import get_method
from otbench.methods.common.base import GenerateContext, MethodContext, Task
from otbench.methods.common.capabilities import UnsupportedRun, unsupported_reason
from otbench.methods.common.profiling import PhaseProfiler, checkpoint_model_profile
from otbench.methods.common.result_cache import (
    attach_result_cache,
    build_run_fingerprint,
    fingerprint_tag,
    validate_result_cache,
    write_csv_atomic,
    write_json_atomic,
)
from otbench.methods.common.runtime import ensure_trained
from otbench.tabdata import load_dataset
from otbench.paths import workspace_root
from otbench.tabdata.io import csv_text

from .alignment import align_generated

OUTPUTS_ROOT = workspace_root() / "outputs" / "generation"


def gen_path_for(
    method: str, dataset_type: str, dataset: str, seed: int,
    outputs_root, run_tag: str,
) -> Path:
    root = Path(outputs_root) if outputs_root else OUTPUTS_ROOT
    suffix = f"_run{run_tag}"
    return root / method / "gen" / f"{method}_{dataset_type}_{dataset}_seed{seed}{suffix}_gen.csv"


def raw_gen_path_for(
    method: str, dataset_type: str, dataset: str, seed: int,
    outputs_root, run_tag: str,
) -> Path:
    """The immutable, unrepaired generator output (validity/privacy metrics read this)."""
    root = Path(outputs_root) if outputs_root else OUTPUTS_ROOT
    suffix = f"_run{run_tag}"
    return root / method / "gen_raw" / f"{method}_{dataset_type}_{dataset}_seed{seed}{suffix}_gen_raw.csv"


def metrics_path_for(
    method: str, dataset_type: str, dataset: str, seed: int,
    outputs_root, run_tag: str,
) -> Path:
    root = Path(outputs_root) if outputs_root else OUTPUTS_ROOT
    suffix = f"_run{run_tag}"
    return root / method / "metrics" / f"{method}_{dataset_type}_{dataset}_seed{seed}{suffix}.json"


def generation_manifest_path_for(
    method: str, dataset_type: str, dataset: str, seed: int,
    outputs_root, run_tag: str,
) -> Path:
    """Commit fit/sampling provenance before the fallible metric phase starts."""
    root = Path(outputs_root) if outputs_root else OUTPUTS_ROOT
    suffix = f"_run{run_tag}"
    return root / method / "generated" / f"{method}_{dataset_type}_{dataset}_seed{seed}{suffix}.json"


def run_generation(
    method: str,
    dataset: str,
    seed: int = 0,
    train_seed: Optional[int] = None,
    n_samples: Optional[int] = None,
    store_root=None,
    outputs_root=None,
    checkpoints_root=None,
    config_file=None,
    overrides: Optional[dict] = None,
    device: Optional[str] = None,
    train_only: bool = False,
    force_retrain: bool = False,
    evaluate: bool = False,
    metric: str = "all",
    shadow_audit: dict | None = None,
    method_metadata: dict | None = None,
    skip_existing: bool = False,
) -> dict:
    start = time.time()
    if evaluate:
        from .eval.registry import resolve_metrics
        resolve_metrics(metric)
    train_seed = seed if train_seed is None else train_seed
    bundle = load_dataset(dataset, store_root)
    m = get_method(method, config_file=config_file, overrides=overrides,
                   checkpoints_root=checkpoints_root)
    if not m.supports(Task.GENERATION):
        raise ValueError(f"Method {method!r} does not support generation.")
    reason = unsupported_reason(m.capabilities, bundle.schema, "generation")
    if reason:
        raise UnsupportedRun(f"{method} cannot run generation on {dataset}: {reason}")

    ckpt_dir = ckpt_utils.checkpoint_dir_for_method(m, dataset, train_seed, checkpoints_root)
    requested = n_samples if n_samples is not None else len(bundle.train)
    if requested < 1:
        raise ValueError("generation n_samples must be >= 1")
    cache_parameters = {
        "seed": seed,
        "train_seed": train_seed,
        "n_samples": requested,
        "evaluate": evaluate,
        "metric": metric,
        "shadow_audit": shadow_audit,
        "method_metadata": method_metadata,
        "device": device,
    }
    cache_fingerprint = None
    checkpoint_ready = m.is_trained(MethodContext(
        bundle=bundle, config=m.config, checkpoint_dir=ckpt_dir,
        train_seed=train_seed, seed=seed, device=device,
    ))
    if (skip_existing and evaluate and not train_only
            and not force_retrain and checkpoint_ready):
        cache_fingerprint = build_run_fingerprint(
            "generation", m, bundle, ckpt_dir, cache_parameters,
        )
        run_tag = fingerprint_tag(cache_fingerprint)
        raw_path = raw_gen_path_for(
            method, bundle.dataset_type, dataset, seed, outputs_root, run_tag,
        )
        out_path = gen_path_for(
            method, bundle.dataset_type, dataset, seed, outputs_root, run_tag,
        )
        metrics_path = metrics_path_for(
            method, bundle.dataset_type, dataset, seed, outputs_root, run_tag,
        )
        generated_path = generation_manifest_path_for(
            method, bundle.dataset_type, dataset, seed, outputs_root, run_tag,
        )
        cached = validate_result_cache(
            metrics_path, cache_fingerprint,
            expected_artifacts=[raw_path, out_path],
        )
        if cached.valid:
            print(f"[generation] cache hit: {metrics_path} ({cached.reason})")
            return {
                "task": "generation", "method": method, "dataset": dataset,
                "dataset_type": bundle.dataset_type, "seed": seed,
                "train_seed": train_seed, "checkpoint_dir": str(ckpt_dir),
                "trained": False, "cache_hit": True,
                "gen_path": str(out_path), "gen_raw_path": str(raw_path),
                "metrics_path": str(metrics_path),
                "method_capabilities": m.capabilities.to_dict(),
                "elapsed_sec": time.time() - start,
            }
        print(f"[generation] cache miss: {metrics_path} ({cached.reason})")
        generated = validate_result_cache(
            generated_path, cache_fingerprint,
            expected_artifacts=[raw_path, out_path],
        )
        if generated.valid and (generated.payload or {}).get("kind") == "generation_artifacts":
            print(f"[generation] generated-artifact cache hit: {generated_path}")
            return _resume_evaluation(
                bundle=bundle, method_obj=m, checkpoint_dir=ckpt_dir,
                seed=seed, train_seed=train_seed, device=device,
                requested=requested, metric=metric, raw_path=raw_path,
                out_path=out_path, metrics_path=metrics_path,
                fingerprint=cache_fingerprint, manifest=generated.payload or {},
                shadow_audit=shadow_audit,
            )

    profiler = PhaseProfiler(device=device)
    with profiler.phase("fit", rows=len(bundle.train)) as fit_phase:
        trained = ensure_trained(m, bundle, ckpt_dir, train_seed, device, force_retrain)
        fit_phase.extra["executed"] = trained
    profiler.record_model(**checkpoint_model_profile(m, ckpt_dir))
    result = {
        "task": "generation",
        "method": method,
        "dataset": dataset,
        "dataset_type": bundle.dataset_type,
        "seed": seed,
        "train_seed": train_seed,
        "checkpoint_dir": str(ckpt_dir),
        "trained": trained,
        "method_capabilities": m.capabilities.to_dict(),
    }
    if train_only:
        result["profile"] = profiler.to_dict()
        result["elapsed_sec"] = time.time() - start
        return result

    # New artifacts are content-addressed by the complete run identity. This
    # prevents config/trial/code/checkpoint variants from overwriting one
    # another even though their human-readable dataset/seed prefix is shared.
    cache_fingerprint = build_run_fingerprint(
        "generation", m, bundle, ckpt_dir, cache_parameters,
    )
    run_tag = fingerprint_tag(cache_fingerprint)
    raw_path = raw_gen_path_for(
        method, bundle.dataset_type, dataset, seed, outputs_root, run_tag,
    )
    out_path = gen_path_for(
        method, bundle.dataset_type, dataset, seed, outputs_root, run_tag,
    )
    metrics_path = metrics_path_for(
        method, bundle.dataset_type, dataset, seed, outputs_root, run_tag,
    )
    generated_path = generation_manifest_path_for(
        method, bundle.dataset_type, dataset, seed, outputs_root, run_tag,
    )

    gctx = GenerateContext(
        bundle=bundle, config=m.config, checkpoint_dir=ckpt_dir,
        train_seed=train_seed, seed=seed, device=device, n_samples=n_samples,
    )
    with profiler.phase("inference", rows=requested):
        raw = m.generate(gctx).reset_index(drop=True)
    audit_metadata = dict(method_metadata or m.evaluation_metadata(gctx))
    result["method_metadata"] = audit_metadata

    # The raw frame is written first and never touched again — every validity,
    # missingness and exact-copy signal has to survive to the metric layer.
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    _write_immutable_raw(raw, raw_path, replace_conflict=skip_existing or force_retrain,
                         schema=bundle.schema)

    fill = data_utils.fill_values_from_train(bundle.train, bundle.schema)
    gen, alignment = align_generated(raw, bundle.schema, fill)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(gen, out_path, schema=bundle.schema)
    print(f"[generation] wrote {len(gen)} rows -> {out_path} ({alignment.n_repairs} repairs)")
    result["gen_path"] = str(out_path)
    result["gen_raw_path"] = str(raw_path)
    result["alignment"] = alignment.to_dict()
    result["n_rows"] = int(len(gen))

    manifest = attach_result_cache({
        "kind": "generation_artifacts",
        "status": "generated",
        "alignment": alignment.to_dict(),
        "n_rows": int(len(gen)),
        "profile": profiler.to_dict(),
        "method_metadata": audit_metadata,
    }, cache_fingerprint, [raw_path, out_path])
    write_json_atomic(generated_path, manifest)
    result["generation_manifest_path"] = str(generated_path)

    if evaluate:
        report_path = run_eval(
            bundle, str(out_path), metrics_path, metric,
            seed=seed, gen_raw_path=str(raw_path), alignment=alignment.to_dict(),
            profiler=profiler, shadow_audit=shadow_audit,
            method_metadata=audit_metadata,
            method_capabilities=m.capabilities.to_dict(),
        )
        report = attach_result_cache(
            load_report(report_path), cache_fingerprint, [raw_path, out_path],
        )
        write_json_atomic(report_path, report)
        result["metrics_path"] = str(report_path)
    result["profile"] = profiler.to_dict()
    result["elapsed_sec"] = time.time() - start
    return result


def _resume_evaluation(
    *, bundle, method_obj, checkpoint_dir: Path, seed: int, train_seed: int,
    device: str | None, requested: int, metric: str, raw_path: Path,
    out_path: Path, metrics_path: Path, fingerprint: dict, manifest: dict,
    shadow_audit: dict | None,
) -> dict:
    """Run only metrics from a validated, atomically committed sample pair."""
    start = time.time()
    profiler = PhaseProfiler.from_dict(manifest.get("profile"), device=device)
    method_metadata = dict(manifest.get("method_metadata") or {})
    report_path = run_eval(
        bundle, str(out_path), metrics_path, metric,
        seed=seed, gen_raw_path=str(raw_path),
        alignment=manifest.get("alignment"), profiler=profiler,
        shadow_audit=shadow_audit, method_metadata=method_metadata,
        method_capabilities=method_obj.capabilities.to_dict(),
    )
    report = load_report(report_path)
    report = attach_result_cache(report, fingerprint, [raw_path, out_path])
    write_json_atomic(report_path, report)
    return {
        "task": "generation", "method": method_obj.name,
        "dataset": bundle.name, "dataset_type": bundle.dataset_type,
        "seed": seed, "train_seed": train_seed,
        "checkpoint_dir": str(checkpoint_dir), "trained": False,
        "cache_hit": False, "generation_cache_hit": True,
        "provenance_status": "verified",
        "gen_path": str(out_path), "gen_raw_path": str(raw_path),
        "metrics_path": str(report_path),
        "n_rows": int(manifest.get("n_rows", requested)),
        "alignment": manifest.get("alignment"),
        "method_metadata": method_metadata,
        "method_capabilities": method_obj.capabilities.to_dict(),
        "profile": profiler.to_dict(), "elapsed_sec": time.time() - start,
    }




def run_eval(
    bundle,
    gen_path: str,
    out_path: Path,
    metric: str = "all",
    seed: int = 0,
    gen_raw_path: str | None = None,
    alignment: dict | None = None,
    profiler: PhaseProfiler | None = None,
    shadow_audit: dict | None = None,
    method_metadata: dict | None = None,
    method_capabilities: dict | None = None,
) -> Path:
    """Evaluate one generated CSV against the dataset's train/test splits."""
    from otbench.generation.eval.data import load_eval_frames
    from otbench.generation.eval.registry import EvalOptions, evaluate_selected, resolve_metrics
    from otbench.generation.eval.report import build_report, print_summary, run_payload, save_report
    from otbench.metrics.cost import finalize_cost_results

    start = time.time()
    train_csv = str(bundle.dir / f"{bundle.name}_train.csv")
    test_csv = str(bundle.dir / f"{bundle.name}_test.csv")
    mode = "scm" if bundle.is_scm else "real"
    scm_path = str(bundle.dir / "graph.json") if bundle.is_scm else None

    selected = resolve_metrics(metric)
    frames = load_eval_frames(
        train_csv, test_csv, gen_path, bundle.schema, gen_raw_path=gen_raw_path
    )
    profiler = profiler or PhaseProfiler()
    with profiler.phase("eval", rows=len(frames.gen)):
        # The COST family reads the profile, so it is passed in already built;
        # the eval phase's own duration lands in the profile written to `meta`.
        options = EvalOptions(
            mode=mode, scm_path=scm_path, seed=seed, profile=profiler.to_dict(),
            shadow_audit=shadow_audit, method_metadata=dict(method_metadata or {}),
        )
        results = evaluate_selected(frames, selected, options)
    results = finalize_cost_results("generation", results, profiler.to_dict())
    per_run = [run_payload(results, gen_path, alignment=alignment)]

    meta = {
        "dataset": bundle.name,
        "train": train_csv,
        "test": test_csv,
        "gen": gen_path,
        "gen_raw": gen_raw_path,
        "alignment": alignment,
        "schema": str(bundle.dir / "schema.json"),
        "schema_source": "file",
        "mode": mode,
        "scm": scm_path,
        "metric_arg": metric,
        "selected_metrics": sorted(selected),
        "eval_seed": seed,
        "ci_test": bundle.schema.ci_test,
        "ci_max_statements": options.ci_max_statements,
        "target_col": bundle.schema.target_col,
        "numerical_cols": bundle.schema.numerical_cols,
        "categorical_cols": bundle.schema.categorical_cols,
        "id_cols": bundle.schema.id_cols,
        "profile": profiler.to_dict(),
        "method_metadata": dict(method_metadata or {}),
        "method_capabilities": dict(method_capabilities or {}),
        "elapsed_sec": time.time() - start,
    }
    report = build_report(meta, per_run)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_report(report, str(out_path))
    print_summary(report)
    print(f"[generation] metrics -> {out_path}")
    return out_path


def _write_immutable_raw(frame, path: Path, replace_conflict: bool = False, schema=None) -> None:
    """Create raw output once, unless replacing a provenance-invalid cache slot."""
    payload = csv_text(frame, schema)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            if replace_conflict:
                write_csv_atomic(frame, path, schema=schema)
                return
            raise FileExistsError(
                f"immutable raw artifact already exists with different content: {path}"
            )
        return
    write_csv_atomic(frame, path, schema=schema)


def load_report(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
