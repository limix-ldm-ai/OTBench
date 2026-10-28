# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Prediction runner: (train-or-load) -> predict -> metrics -> outputs."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from otbench.methods.common import checkpoints as ckpt_utils
from otbench.methods.common import get_method
from otbench.methods.common.base import MethodContext, PredictContext, PredictionOutput, Task
from otbench.methods.common.capabilities import UnsupportedRun, unsupported_reason
from otbench.methods.common.capabilities import dataset_task_type
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

from .eval import PredictionEvalOptions, encode_labels, evaluate_prediction, summarize

OUTPUTS_ROOT = workspace_root() / "outputs" / "prediction"

DEFAULT_TRIALS = {"tabddpm": 5, "tabsyn": 5, "tabdiff": 5}


def _prediction_result_paths(
    outputs: Path,
    method: str,
    base_run_key: str,
    run_tag: str,
) -> tuple[Path, Path]:
    run_key = f"{base_run_key}_run{run_tag}"
    return (
        outputs / method / "predictions" / f"{run_key}.csv",
        outputs / method / "metrics" / f"{run_key}_metrics.json",
    )


def run_prediction(
    method: str,
    dataset: str,
    seed: int = 0,
    train_seed: Optional[int] = None,
    trials: Optional[int] = None,
    n_samples: Optional[int] = None,
    store_root=None,
    outputs_root=None,
    checkpoints_root=None,
    config_file=None,
    overrides: Optional[dict] = None,
    device: Optional[str] = None,
    train_only: bool = False,
    force_retrain: bool = False,
    metric: str = "all",
    ood: Optional[dict] = None,
    attack_artifact: Optional[dict] = None,
    skip_existing: bool = False,
) -> dict:
    start = time.time()
    from .eval import resolve_metrics
    resolve_metrics(metric)
    train_seed = seed if train_seed is None else train_seed
    outputs = Path(outputs_root) if outputs_root else OUTPUTS_ROOT

    bundle = load_dataset(dataset, store_root)
    m = get_method(method, config_file=config_file, overrides=overrides,
                   checkpoints_root=checkpoints_root)
    if not m.supports(Task.PREDICTION):
        raise ValueError(f"Method {method!r} does not support prediction.")
    reason = unsupported_reason(m.capabilities, bundle.schema, "prediction")
    if reason:
        raise UnsupportedRun(f"{method} cannot run prediction on {dataset}: {reason}")
    n_trials = trials if trials is not None else int(
        m.cfg("predict", "trials", DEFAULT_TRIALS.get(method, 1))
    )
    sample_budget = n_samples if n_samples is not None else int(
        m.cfg("predict", "n_samples", 100)
    )
    if n_trials < 1:
        raise ValueError("prediction trials must be >= 1")
    if sample_budget < 1:
        raise ValueError("prediction n_samples must be >= 1")

    ckpt_dir = ckpt_utils.checkpoint_dir_for_method(m, dataset, train_seed, checkpoints_root)
    base_run_key = f"{method}_{dataset}_seed{seed}_train_seed{train_seed}"
    cache_parameters = {
        "seed": seed, "train_seed": train_seed, "trials": n_trials,
        "n_samples": sample_budget, "metric": metric, "ood": ood,
        "attack_artifact": attack_artifact, "device": device,
    }
    cache_fingerprint = None
    checkpoint_ready = m.is_trained(MethodContext(
        bundle=bundle, config=m.config, checkpoint_dir=ckpt_dir,
        train_seed=train_seed, seed=seed, device=device,
    ))
    if skip_existing and not train_only and not force_retrain and checkpoint_ready:
        cache_fingerprint = build_run_fingerprint(
            "prediction", m, bundle, ckpt_dir, cache_parameters,
        )
        run_tag = fingerprint_tag(cache_fingerprint)
        predictions_path, metrics_path = _prediction_result_paths(
            outputs, method, base_run_key, run_tag,
        )
        cached = validate_result_cache(
            metrics_path, cache_fingerprint, expected_artifacts=[predictions_path],
        )
        if cached.valid:
            payload = cached.payload or {}
            print(f"[prediction] cache hit: {metrics_path} ({cached.reason})")
            return {
                "task": "prediction", "method": method, "dataset": dataset,
                "dataset_type": bundle.dataset_type, "seed": seed,
                "train_seed": train_seed, "trials": n_trials,
                "n_samples": sample_budget, "checkpoint_dir": str(ckpt_dir),
                "trained": False, "cache_hit": True,
                "predictions_path": str(predictions_path),
                "metrics_path": str(metrics_path),
                "metrics": payload.get("metrics", {}),
                "method_capabilities": m.capabilities.to_dict(),
                "elapsed_sec": time.time() - start,
            }
        print(f"[prediction] cache miss: {metrics_path} ({cached.reason})")

    profiler = PhaseProfiler(device=device)
    with profiler.phase("fit", rows=len(bundle.train)) as fit_phase:
        trained = ensure_trained(m, bundle, ckpt_dir, train_seed, device, force_retrain)
        fit_phase.extra["executed"] = trained
    profiler.record_model(**checkpoint_model_profile(m, ckpt_dir))
    supervised_task_type = dataset_task_type(bundle.schema)
    result: dict[str, Any] = {
        "task": "prediction",
        "method": method,
        "dataset": dataset,
        "seed": seed,
        "train_seed": train_seed,
        "trials": n_trials,
        "n_samples": sample_budget,
        "checkpoint_dir": str(ckpt_dir),
        "trained": trained,
        "method_capabilities": m.capabilities.to_dict(supervised_task_type),
    }
    if train_only:
        result["profile"] = profiler.to_dict()
        result["elapsed_sec"] = time.time() - start
        return result

    cache_fingerprint = build_run_fingerprint(
        "prediction", m, bundle, ckpt_dir, cache_parameters,
    )
    run_tag = fingerprint_tag(cache_fingerprint)
    predictions_path, metrics_path = _prediction_result_paths(
        outputs, method, base_run_key, run_tag,
    )

    pctx = PredictContext(
        bundle=bundle, config=m.config, checkpoint_dir=ckpt_dir,
        train_seed=train_seed, seed=seed, device=device, trials=n_trials,
        n_samples=sample_budget,
    )
    with profiler.phase("inference", rows=len(bundle.test)):
        output = m.predict(pctx)

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    y_true = bundle.test[bundle.schema.target_col].reset_index(drop=True)
    is_regression = supervised_task_type == "regression"

    ood_payload = dict(ood) if ood is not None else None
    if ood_payload is not None and bundle.schema.ood_split:
        ood_payload.setdefault("split_source", "schema.ood_split")
        ood_payload.setdefault("split_declaration", bundle.schema.ood_split)

    with profiler.phase("eval", rows=len(bundle.test)):
        options = PredictionEvalOptions(
            metric=metric, profile=profiler.to_dict(), ood=ood_payload,
            attack_artifact=attack_artifact
        )
        results = evaluate_prediction(y_true, output, bundle.schema, bundle.train, options)
    from otbench.metrics.cost import finalize_cost_results
    results = finalize_cost_results("prediction", results, profiler.to_dict())

    frame = _prediction_frame(y_true, output, bundle, is_regression)
    write_csv_atomic(frame, predictions_path)
    payload = {
        "task": "prediction",
        "method": method,
        "dataset": dataset,
        "dataset_type": bundle.dataset_type,
        "task_type": "regression" if is_regression else "classification",
        "seed": seed,
        "train_seed": train_seed,
        "trials": n_trials,
        "n_samples": sample_budget,
        "prediction_metadata": output.metadata,
        "predictions_path": str(predictions_path),
        "profile": profiler.to_dict(),
        "method_capabilities": m.capabilities.to_dict(supervised_task_type),
        **summarize(results),
    }
    payload = attach_result_cache(payload, cache_fingerprint, [predictions_path])
    write_json_atomic(metrics_path, payload)

    n_ok = sum(1 for r in results if r.status == "ok")
    print(f"[prediction] {n_ok}/{len(results)} metrics computed -> {metrics_path}")
    result.update({
        "metrics": payload["metrics"],
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "profile": profiler.to_dict(),
        "elapsed_sec": time.time() - start,
    })
    return result


def _prediction_frame(y_true: pd.Series, output: PredictionOutput, bundle,
                      is_regression: bool) -> pd.DataFrame:
    """The per-row prediction artifact, kept alongside the metrics.

    Classification records train-domain membership. The evaluation protocol
    marks the split unsupported if it includes unknown true labels.
    """
    if is_regression:
        frame = pd.DataFrame({
            "y_true": y_true,
            "y_pred": np.asarray(output.y_pred, dtype=np.float64).reshape(-1),
        })
        if output.quantiles is not None and output.quantile_levels:
            quantiles = np.asarray(output.quantiles, dtype=float)
            for index, level in enumerate(output.quantile_levels):
                if index < quantiles.shape[1]:
                    frame[f"q_{level}"] = quantiles[:, index]
        if output.y_samples is not None:
            samples = np.asarray(output.y_samples, dtype=float)
            frame["sample_mean"] = samples.mean(axis=1)
            frame["sample_std"] = samples.std(axis=1)
        return frame

    encoded = encode_labels(y_true, output, bundle.train, bundle.schema)
    frame = pd.DataFrame({
        "y_true": y_true,
        "y_pred": np.asarray(output.y_pred, dtype=object).reshape(-1),
        "scored": encoded["known"],
    })
    if output.y_proba is not None and output.labels is not None:
        proba = np.asarray(output.y_proba, dtype=np.float64)
        for index, label in enumerate(output.labels):
            if index < proba.shape[1]:
                frame[f"proba_{label}"] = proba[:, index]
    elif output.y_score is not None:
        frame["proba_positive"] = np.asarray(output.y_score, dtype=np.float64).reshape(-1)
    return frame
