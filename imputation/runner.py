# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Imputation runner: mask -> (train-or-load) -> impute trials -> metrics."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from otbench.methods.common import checkpoints as ckpt_utils
from otbench.methods.common import get_method
from otbench.methods.common.base import ImputeContext, MethodContext, Task
from otbench.methods.common.base import ImputationOutput
from otbench.paths import workspace_root
from otbench.tabdata.io import read_frame_csv
from otbench.methods.common.capabilities import UnsupportedRun, unsupported_reason
from otbench.methods.common.profiling import (
    PhaseProfiler,
    PhaseRecord,
    checkpoint_model_profile,
)
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

from .eval import (
    ImputationEvalOptions,
    ImputationEvaluationPlan,
    build_evaluation_plan,
    PreparedImputationEvaluators,
    evaluate_imputation,
    prepare_imputation_evaluators,
    summarize,
)
from .masking import SCENARIOS, make_mask_bundle, mask_bundle_dir, save_mask_bundle, select_columns
from .validation import validate_completion, normalize_completion

OUTPUTS_ROOT = workspace_root() / "outputs" / "imputation"

# default stochastic-imputation repeats: diffusion methods are sampled several
# times, deterministic column-wise foundation imputers once.
DEFAULT_TRIALS = {"tabddpm": 5, "tabsyn": 5, "tabdiff": 5}


def _rate_str(rate: float) -> str:
    return ("%g" % float(rate)).replace(".", "p")


def _default_estimand(schema, mask) -> Optional[dict]:
    """The estimand Rubin's rules are validated against under ``--scientific-model auto``.

    The mean of the first numerical feature the run actually hides, in schema
    order. The mask matters: only a handful of columns are made missing, and an
    unmasked column is identical in every draw, so its between-draw variance is
    exactly zero and Rubin's rules would report a flawless zero bias for every
    method. Pre-registration is preserved because the rule lives here and the
    mask is fixed before imputation rather than chosen from the completions; the
    column it resolves to is recorded with the metric. Datasets whose masked
    columns are all categorical get no estimand and the metric skips.
    """
    for col in schema.numerical_cols:
        if col != schema.target_col and col in mask.columns and bool(mask[col].any()):
            return {"kind": "mean", "column": col}
    return None


def _imputation_result_paths(
    outputs: Path,
    method: str,
    dataset: str,
    base_run_key: str,
    n_trials: int,
    run_tag: str,
) -> tuple[str, Path, Path, Path, list[Path]]:
    run_key = f"{base_run_key}_run{run_tag}"
    eval_dir = outputs / method / "metrics" / dataset / base_run_key / f"run_{run_tag}"
    metrics_path = eval_dir / "metrics.json"
    run_state_path = eval_dir / "run_state.json"
    imputed_paths = [
        outputs / method / "imputed" / f"{method}_{dataset}_{run_key}_trial{trial}.csv"
        for trial in range(n_trials)
    ]
    return run_key, eval_dir, metrics_path, run_state_path, imputed_paths


@dataclass
class _PreparedImputationRun:
    start: float
    method: str
    dataset: str
    scenario: str
    rate: float
    train_seed: int
    n_trials: int
    outputs: Path
    bundle: Any
    method_obj: Any
    mask_bundle: Any
    ictx: ImputeContext
    profiler: PhaseProfiler
    ckpt_dir: Path
    cache_fingerprint: dict[str, Any]
    run_key: str
    eval_dir: Path
    metrics_path: Path
    run_state_path: Path
    result: dict[str, Any]
    eval_direct: bool
    eval_downstream: bool
    metric: str
    scientific_model: Optional[dict | str]
    matched_runs: Optional[list[dict]]
    stability_protocol: Optional[dict | str]
    evaluation_plan: ImputationEvaluationPlan
    skip_existing: bool = False
    resume_output: Any = None
    inference_committed: bool = False


def _cache_result(
    *,
    start: float,
    method: str,
    dataset: str,
    scenario: str,
    rate: float,
    seed: int,
    train_seed: int,
    ckpt_dir: Path,
    mask_dir: Path,
    eval_dir: Path,
    metrics_path: Path,
    imputed_paths: list[Path],
    n_trials: int,
    method_obj: Any,
    payload: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "task": "imputation", "method": method, "dataset": dataset,
        "scenario": scenario, "rate": rate, "seed": seed,
        "train_seed": train_seed, "checkpoint_dir": str(ckpt_dir),
        "trained": False, "cache_hit": True,
        "mask_dir": str(mask_dir), "eval_dir": str(eval_dir),
        "metrics_path": str(metrics_path),
        "metrics": payload.get("metrics", {}),
        "imputed_paths": [str(path) for path in imputed_paths],
        "n_draws": n_trials,
        "method_capabilities": method_obj.capabilities.to_dict(),
        "elapsed_sec": time.time() - start,
    }
    return result


def _prepare_imputation_run(
    method: str,
    dataset: str,
    scenario: str = "MCAR",
    rate: float = 0.3,
    seed: int = 0,
    train_seed: Optional[int] = None,
    selection_seed: Optional[int] = None,
    mask_seed: Optional[int] = None,
    impute_seed: Optional[int] = None,
    trials: Optional[int] = None,
    store_root=None,
    outputs_root=None,
    checkpoints_root=None,
    config_file=None,
    overrides: Optional[dict] = None,
    device: Optional[str] = None,
    train_only: bool = False,
    force_retrain: bool = False,
    eval_direct: bool = True,
    eval_downstream: bool = True,
    metric: str = "all",
    scientific_model: Optional[dict | str] = None,
    matched_runs: Optional[list[dict]] = None,
    stability_protocol: Optional[dict | str] = None,
    skip_existing: bool = False,
    *,
    method_obj=None,
    bundle_obj=None,
    selection_obj=None,
) -> dict[str, Any] | _PreparedImputationRun:
    start = time.time()
    evaluation_plan = build_evaluation_plan(
        metric, include_direct=eval_direct, include_downstream=eval_downstream,
    )
    scenario = scenario.upper()
    if scenario not in SCENARIOS or not 0 <= rate <= 1:
        raise ValueError("imputation needs a declared scenario and a rate in [0, 1]")
    train_seed = seed if train_seed is None else train_seed
    selection_seed = seed if selection_seed is None else selection_seed
    mask_seed = seed if mask_seed is None else mask_seed
    impute_seed = seed if impute_seed is None else impute_seed
    outputs = Path(outputs_root) if outputs_root else OUTPUTS_ROOT

    bundle = bundle_obj if bundle_obj is not None else load_dataset(dataset, store_root)
    m = method_obj or get_method(method, config_file=config_file, overrides=overrides,
                                checkpoints_root=checkpoints_root)
    if m.name != method:
        raise ValueError(
            f"provided method instance is {m.name!r}, requested {method!r}"
        )
    if not m.supports(Task.IMPUTATION):
        raise ValueError(f"Method {method!r} does not support imputation.")
    reason = unsupported_reason(
        m.capabilities, bundle.schema, "imputation", mask_granularity="cell"
    )
    if reason:
        raise UnsupportedRun(f"{method} cannot run imputation on {dataset}: {reason}")
    n_trials = trials if trials is not None else int(
        m.cfg("impute", "trials", DEFAULT_TRIALS.get(method, 1))
    )
    if n_trials < 1:
        raise ValueError("imputation trials must be >= 1")
    if stability_protocol == "auto":
        stability_protocol = {
            "pre_registered": True, "model": "logistic_regression", "n_draws": n_trials,
            "declared_by": "--stability-protocol auto",
        }

    base_run_key = (
        f"{scenario}_rate{_rate_str(rate)}_train_seed{train_seed}"
        f"_selection_seed{selection_seed}_mask_seed{mask_seed}_impute_seed{impute_seed}"
    )
    mask_root = (
        outputs / "masks" / dataset
        / f"{scenario}_rate{_rate_str(rate)}_selection_seed{selection_seed}_mask_seed{mask_seed}"
    )
    mask_dir = mask_root
    ckpt_dir = ckpt_utils.checkpoint_dir_for_method(
        m, dataset, train_seed, checkpoints_root
    )
    cache_parameters = {
        "scenario": scenario, "rate": rate, "seed": seed,
        "train_seed": train_seed, "selection_seed": selection_seed,
        "mask_seed": mask_seed, "impute_seed": impute_seed,
        "trials": n_trials, "eval_direct": eval_direct,
        "eval_downstream": eval_downstream, "metric": metric,
        "scientific_model": scientific_model, "matched_runs": matched_runs,
        "stability_protocol": stability_protocol, "device": device,
    }
    checkpoint_ready = m.is_trained(MethodContext(
        bundle=bundle, config=m.config, checkpoint_dir=ckpt_dir,
        train_seed=train_seed, seed=seed, device=device,
    ))
    if (skip_existing and not train_only
            and not force_retrain and checkpoint_ready):
        requested_fingerprint = build_run_fingerprint(
            "imputation", m, bundle, ckpt_dir, cache_parameters,
        )
        run_tag = fingerprint_tag(requested_fingerprint)
        _, eval_dir, metrics_path, run_state_path, expected_imputed_paths = (
            _imputation_result_paths(
                outputs, method, dataset, base_run_key, n_trials, run_tag,
            )
        )
        cached = validate_result_cache(
            metrics_path, requested_fingerprint,
        )
        if cached.valid:
            recorded = (cached.payload or {}).get("meta", {}).get("mask_dir")
            candidate = Path(recorded) if isinstance(recorded, str) else mask_root
            if candidate.resolve().is_relative_to((outputs / "masks" / dataset).resolve()):
                mask_dir = candidate
            expected_artifacts = [*expected_imputed_paths, *(mask_dir / name for name in
                                  ("selection.json", "mask.json", "masked.csv", "bundle.json")),
                                  run_state_path]
            cached = validate_result_cache(metrics_path, requested_fingerprint,
                                           expected_artifacts=expected_artifacts)
        if cached.valid:
            print(f"[imputation] cache hit: {metrics_path} ({cached.reason})")
            return _cache_result(
                start=start, method=method, dataset=dataset, scenario=scenario,
                rate=rate, seed=seed, train_seed=train_seed, ckpt_dir=ckpt_dir,
                mask_dir=mask_dir, eval_dir=eval_dir, metrics_path=metrics_path,
                imputed_paths=expected_imputed_paths, n_trials=n_trials,
                method_obj=m, payload=cached.payload or {},
            )
        print(f"[imputation] cache miss: {metrics_path} ({cached.reason})")

    profiler = PhaseProfiler(device=device)
    with profiler.phase("fit", rows=len(bundle.train)) as fit_phase:
        trained = ensure_trained(m, bundle, ckpt_dir, train_seed, device, force_retrain)
        fit_phase.extra["executed"] = trained
    profiler.record_model(**checkpoint_model_profile(m, ckpt_dir))
    result: dict[str, Any] = {
        "task": "imputation", "method": method, "dataset": dataset,
        "scenario": scenario, "rate": rate, "seed": seed,
        "train_seed": train_seed, "checkpoint_dir": str(ckpt_dir),
        "trained": trained, "method_capabilities": m.capabilities.to_dict(),
    }
    if train_only:
        result["profile"] = profiler.to_dict()
        result["elapsed_sec"] = time.time() - start
        return result

    cache_fingerprint = build_run_fingerprint(
        "imputation", m, bundle, ckpt_dir, cache_parameters,
    )
    run_tag = fingerprint_tag(cache_fingerprint)
    run_key, eval_dir, metrics_path, run_state_path, _ = _imputation_result_paths(
        outputs, method, dataset, base_run_key, n_trials, run_tag,
    )

    selection = selection_obj or select_columns(bundle.train, bundle.schema, seed=selection_seed)
    mask_bundle = make_mask_bundle(dataset, bundle.test, selection, scenario, rate,
                                   mask_seed, schema=bundle.schema)
    mask_dir = save_mask_bundle(outputs, mask_bundle)
    result["mask_dir"] = str(mask_dir)
    if scientific_model == "auto":
        scientific_model = _default_estimand(bundle.schema, mask_bundle.mask)

    eval_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_state_path, {"status": "running", "mask_dir": str(mask_dir)})
    ictx = ImputeContext(
        bundle=bundle, config=m.config, checkpoint_dir=ckpt_dir,
        train_seed=train_seed, seed=impute_seed, device=device,
        masked_test=mask_bundle.masked, mask=mask_bundle.mask, trials=n_trials,
    )
    plan = _PreparedImputationRun(
        start=start, method=method, dataset=dataset, scenario=scenario, rate=rate,
        train_seed=train_seed, n_trials=n_trials, outputs=outputs, bundle=bundle,
        method_obj=m, mask_bundle=mask_bundle, ictx=ictx, profiler=profiler,
        ckpt_dir=ckpt_dir, cache_fingerprint=cache_fingerprint, run_key=run_key,
        eval_dir=eval_dir, metrics_path=metrics_path,
        run_state_path=run_state_path, result=result, eval_direct=eval_direct,
        eval_downstream=eval_downstream, metric=metric,
        scientific_model=scientific_model, matched_runs=matched_runs,
        stability_protocol=stability_protocol,
        evaluation_plan=evaluation_plan, skip_existing=skip_existing,
    )
    if skip_existing:
        cached = validate_result_cache(plan.eval_dir / "inference.json", cache_fingerprint,
                                       expected_artifacts=_inference_artifacts(plan))
        if cached.valid:
            payload = cached.payload or {}
            plan.resume_output = ImputationOutput.from_frames(
                [read_frame_csv(path, bundle.schema) for path in _imputed_paths(plan)],
                metadata=payload.get("draw_metadata", {}),
            )
            plan.profiler = PhaseProfiler.from_dict(payload.get("profile"), device=device)
            plan.inference_committed = True
            plan.result["inference_cache_hit"] = True
            plan.result["imputed_paths"] = [str(path) for path in _imputed_paths(plan)]
    return plan


def _mark_failed(
    plan: _PreparedImputationRun,
    exc: BaseException,
    imputed_paths: Optional[list[str]] = None,
) -> None:
    _write_json(plan.run_state_path, {
        "status": "failed",
        "mask_dir": str(plan.result["mask_dir"]),
        "imputed_paths": list(imputed_paths if imputed_paths is not None
                              else plan.result.get("imputed_paths", [])),
        "error": f"{type(exc).__name__}: {exc}",
    })


def _eval_options(
    plan: _PreparedImputationRun,
    prepared: PreparedImputationEvaluators | None = None,
) -> ImputationEvalOptions:
    return ImputationEvalOptions(
        random_state=plan.train_seed,
        native_draws=plan.method_obj.capabilities.can_emit("draws", "imputation"),
        profile=plan.profiler.to_dict(),
        scientific_model=plan.scientific_model,
        matched_runs=plan.matched_runs,
        stability_protocol=plan.stability_protocol,
        include_downstream=plan.eval_downstream,
        include_direct=plan.eval_direct,
        metric=plan.metric,
        plan=plan.evaluation_plan,
        prepared_evaluators=prepared,
    )


def _amortize_eval_setup(
    record: PhaseRecord,
    setup: PhaseRecord,
    grid_size: int,
) -> None:
    setup_share = setup.wall_clock_sec / grid_size
    record.wall_clock_sec += setup_share
    if record.rows:
        record.sec_per_1k_rows = record.wall_clock_sec * 1000.0 / record.rows
    if record.cells:
        record.sec_per_1k_cells = record.wall_clock_sec * 1000.0 / record.cells
    if setup.peak_ram_mb is not None:
        record.peak_ram_mb = max(record.peak_ram_mb or 0.0, setup.peak_ram_mb)
    if setup.peak_vram_mb is not None:
        record.peak_vram_mb = max(record.peak_vram_mb or 0.0, setup.peak_vram_mb)
    record.extra.update({
        "shared_eval_setup_grid_size": grid_size,
        "shared_eval_setup_total_wall_clock_sec": setup.wall_clock_sec,
        "shared_eval_setup_amortized_wall_clock_sec": setup_share,
    })


def _imputed_paths(plan: _PreparedImputationRun) -> list[Path]:
    return [plan.outputs / plan.method / "imputed"
            / f"{plan.method}_{plan.dataset}_{plan.run_key}_trial{trial}.csv"
            for trial in range(plan.n_trials)]


def _inference_artifacts(plan: _PreparedImputationRun) -> list[Path]:
    mask_dir = Path(plan.result["mask_dir"])
    return [*_imputed_paths(plan), *(mask_dir / name for name in
            ("selection.json", "mask.json", "masked.csv", "bundle.json"))]


def _commit_inference(plan: _PreparedImputationRun, output: ImputationOutput) -> None:
    if output.n_draws != plan.n_trials:
        raise ValueError(f"{plan.method} returned {output.n_draws} draws; expected {plan.n_trials}")
    output.draws = [normalize_completion(plan.bundle.test, draw, plan.bundle.schema)
                    for draw in output.draws]
    for draw in output.draws:
        validate_completion(plan.bundle.test, draw, plan.mask_bundle.mask, plan.bundle.schema)
    if plan.inference_committed:
        return
    paths = _imputed_paths(plan)
    for path, draw in zip(paths, output.draws):
        write_csv_atomic(draw, path, schema=plan.bundle.schema)
    payload = attach_result_cache({
        "kind": "imputation_inference", "n_draws": output.n_draws,
        "draw_metadata": output.metadata, "profile": plan.profiler.to_dict(),
    }, plan.cache_fingerprint, _inference_artifacts(plan))
    write_json_atomic(plan.eval_dir / "inference.json", payload)
    plan.inference_committed = True
    plan.result["imputed_paths"] = [str(path) for path in paths]
    _write_json(plan.run_state_path, {"status": "inferred", "mask_dir": plan.result["mask_dir"],
                                     "imputed_paths": plan.result["imputed_paths"]})


def _execute_imputation_run(
    plan: _PreparedImputationRun,
    *,
    draws_output=None,
    inference_record: Optional[PhaseRecord] = None,
    prepared_evaluators: PreparedImputationEvaluators | None = None,
    eval_setup_record: Optional[PhaseRecord] = None,
    eval_setup_grid_size: int = 1,
) -> dict[str, Any]:
    m = plan.method_obj
    n_cells = int(plan.mask_bundle.mask.to_numpy(dtype=bool).sum())
    imputed_paths: list[str] = []
    metrics_payload: dict[str, Any] | None = None
    shared_inference = draws_output is not None or plan.resume_output is not None
    try:
        if draws_output is None:
            draws_output = plan.resume_output
        if draws_output is None:
            with plan.profiler.phase(
                "inference", rows=len(plan.bundle.test), cells=n_cells,
                n_draws=plan.n_trials,
            ):
                draws_output = m.impute_draws(plan.ictx)
        elif inference_record is None and not plan.inference_committed:
            raise ValueError("precomputed imputation output needs an inference record")
        elif inference_record is not None and not any(record is inference_record for record in plan.profiler.records):
            plan.profiler.records.append(inference_record)

        _commit_inference(plan, draws_output)
        imputed_paths = list(plan.result["imputed_paths"])

        if plan.eval_direct or plan.eval_downstream:
            with plan.profiler.phase("eval", cells=n_cells):
                options = _eval_options(plan, prepared_evaluators)
                results = evaluate_imputation(
                    plan.bundle.test, draws_output, plan.mask_bundle,
                    plan.bundle.schema, plan.bundle.train, options,
                )
            if eval_setup_record is not None:
                _amortize_eval_setup(
                    plan.profiler.records[-1],
                    eval_setup_record,
                    eval_setup_grid_size,
                )
            from otbench.metrics.cost import finalize_cost_results
            results = finalize_cost_results(
                "imputation", results, plan.profiler.to_dict()
            )
            metrics_payload = {
                "meta": {
                    "method": plan.method, "dataset": plan.dataset,
                    "scenario": plan.scenario, "rate": plan.rate,
                    "n_draws": draws_output.n_draws,
                    "native_draws": options.native_draws,
                    "draw_metadata": draws_output.metadata,
                    "mask_dir": str(plan.result["mask_dir"]),
                    "profile": plan.profiler.to_dict(),
                    "method_capabilities": m.capabilities.to_dict(),
                    "evaluation_protocol_version": 2,
                    "selected_metrics": sorted(plan.evaluation_plan.enabled),
                },
                **summarize(results),
            }
            plan.result["metrics_path"] = str(plan.metrics_path)
            plan.result["metrics"] = {
                result.name: result.value for result in results
                if result.status == "ok"
            }

        _write_json(plan.run_state_path, {
            "status": "succeeded", "mask_dir": str(plan.result["mask_dir"]),
            "imputed_paths": imputed_paths,
        })
        if metrics_payload is not None:
            mask_dir = Path(plan.result["mask_dir"])
            metrics_payload = attach_result_cache(
                metrics_payload, plan.cache_fingerprint,
                [
                    *_inference_artifacts(plan), plan.run_state_path,
                ],
            )
            write_json_atomic(plan.metrics_path, metrics_payload)
        elapsed = (
            sum(record.wall_clock_sec for record in plan.profiler.records)
            if shared_inference else time.time() - plan.start
        )
        plan.result.update({
            "imputed_paths": imputed_paths, "eval_dir": str(plan.eval_dir),
            "n_draws": draws_output.n_draws,
            "draw_metadata": draws_output.metadata,
            "profile": plan.profiler.to_dict(), "elapsed_sec": elapsed,
        })
        return plan.result
    except BaseException as exc:
        _mark_failed(plan, exc, imputed_paths)
        raise


def run_imputation(
    method: str,
    dataset: str,
    scenario: str = "MCAR",
    rate: float = 0.3,
    seed: int = 0,
    train_seed: Optional[int] = None,
    selection_seed: Optional[int] = None,
    mask_seed: Optional[int] = None,
    impute_seed: Optional[int] = None,
    trials: Optional[int] = None,
    store_root=None,
    outputs_root=None,
    checkpoints_root=None,
    config_file=None,
    overrides: Optional[dict] = None,
    device: Optional[str] = None,
    train_only: bool = False,
    force_retrain: bool = False,
    eval_direct: bool = True,
    eval_downstream: bool = True,
    metric: str = "all",
    scientific_model: Optional[dict | str] = None,
    matched_runs: Optional[list[dict]] = None,
    stability_protocol: Optional[dict | str] = None,
    skip_existing: bool = False,
) -> dict:
    prepared = _prepare_imputation_run(
        method=method, dataset=dataset, scenario=scenario, rate=rate, seed=seed,
        train_seed=train_seed, selection_seed=selection_seed,
        mask_seed=mask_seed, impute_seed=impute_seed, trials=trials,
        store_root=store_root, outputs_root=outputs_root,
        checkpoints_root=checkpoints_root, config_file=config_file,
        overrides=overrides, device=device, train_only=train_only,
        force_retrain=force_retrain, eval_direct=eval_direct,
        eval_downstream=eval_downstream, metric=metric,
        scientific_model=scientific_model, matched_runs=matched_runs,
        stability_protocol=stability_protocol, skip_existing=skip_existing,
    )
    if isinstance(prepared, dict):
        return prepared
    return _execute_imputation_run(prepared)


def _amortized_inference_record(
    shared: PhaseRecord,
    plan: _PreparedImputationRun,
    grid_size: int,
) -> PhaseRecord:
    cells = int(plan.mask_bundle.mask.to_numpy(dtype=bool).sum())
    wall = shared.wall_clock_sec / grid_size
    rows = len(plan.bundle.test)
    return PhaseRecord(
        name="inference", wall_clock_sec=wall, status=shared.status,
        error=shared.error, rows=rows, cells=cells,
        sec_per_1k_rows=(wall * 1000.0 / rows) if rows else None,
        sec_per_1k_cells=(wall * 1000.0 / cells) if cells else None,
        peak_ram_mb=shared.peak_ram_mb, peak_vram_mb=shared.peak_vram_mb,
        extra={
            "n_draws": plan.n_trials,
            "execution_mode": "dataset_setting_grid",
            "shared_grid_size": grid_size,
            "shared_grid_total_wall_clock_sec": shared.wall_clock_sec,
            "wall_clock_allocation": "equal_amortized_per_active_setting",
        },
    )


def run_imputation_grid(
    method: str,
    dataset: str,
    scenarios: list[str],
    rates: list[float],
    seed: int = 0,
    train_seed: Optional[int] = None,
    trials: Optional[int] = None,
    store_root=None,
    outputs_root=None,
    checkpoints_root=None,
    config_file=None,
    overrides: Optional[dict] = None,
    device: Optional[str] = None,
    eval_direct: bool = True,
    eval_downstream: bool = True,
    metric: str = "all",
    scientific_model: Optional[dict | str] = None,
    stability_protocol: Optional[dict | str] = None,
    skip_existing: bool = False,
) -> dict[str, Any]:
    """Run one dataset's missingness grid in one process.

    This is the execution contract needed by conditional foundation imputers:
    their ``impute_draws_many`` implementation may fit each conditional once and
    infer every requested setting before that fitted state is replaced.
    """

    if not scenarios or not rates:
        raise ValueError("imputation grid needs at least one scenario and rate")
    build_evaluation_plan(metric, include_direct=eval_direct, include_downstream=eval_downstream)
    requested = [(scenario.upper(), float(rate)) for scenario in scenarios for rate in rates]
    if any(scenario not in SCENARIOS or not 0 <= rate <= 1 for scenario, rate in requested):
        raise ValueError("imputation needs declared scenarios and rates in [0, 1]")
    if len(requested) != len(set(requested)):
        raise ValueError("imputation grid settings must be unique")
    m = get_method(method, config_file=config_file, overrides=overrides,
                   checkpoints_root=checkpoints_root)
    bundle = load_dataset(dataset, store_root)
    plans: list[_PreparedImputationRun] = []
    results: list[dict[str, Any]] = []
    selection = None
    try:
        for scenario, rate in requested:
            prepared = _prepare_imputation_run(
                method=method, dataset=dataset, scenario=scenario, rate=rate,
                seed=seed, train_seed=train_seed, selection_seed=seed,
                mask_seed=seed, impute_seed=seed, trials=trials,
                store_root=store_root, outputs_root=outputs_root,
                checkpoints_root=checkpoints_root, config_file=config_file,
                overrides=overrides, device=device, eval_direct=eval_direct,
                eval_downstream=eval_downstream, metric=metric,
                scientific_model=scientific_model, stability_protocol=stability_protocol,
                skip_existing=skip_existing,
                method_obj=m, bundle_obj=bundle, selection_obj=selection,
            )
            if isinstance(prepared, dict):
                results.append(prepared)
            else:
                plans.append(prepared)
                if selection is None:
                    selection = prepared.mask_bundle.selection
    except BaseException as exc:
        for plan in plans:
            _mark_failed(plan, exc)
        raise

    shared_profile = None
    shared_eval_setup_profile = None
    failures: list[str] = []
    ready = [(plan, plan.resume_output) for plan in plans if plan.resume_output is not None]
    fresh = [plan for plan in plans if plan.resume_output is None]
    if fresh:
        profiler = PhaseProfiler(device=device)
        try:
            total_cells = sum(int(plan.mask_bundle.mask.to_numpy(dtype=bool).sum()) for plan in fresh)
            with profiler.phase(
                "inference", rows=len(bundle.test) * len(fresh), cells=total_cells,
                n_settings=len(fresh), total_draws=sum(plan.n_trials for plan in fresh),
            ):
                outputs = m.impute_draws_many([plan.ictx for plan in fresh])
            if len(outputs) != len(fresh):
                raise ValueError(f"{method}.impute_draws_many returned {len(outputs)} outputs "
                                 f"for {len(fresh)} settings")
        except BaseException as exc:
            for plan in plans:
                _mark_failed(plan, exc)
            raise
        shared = profiler.records[-1]
        shared_profile = shared.to_dict()
        # Commit each valid completion before shared evaluator setup can fail.
        for plan, output in zip(fresh, outputs):
            plan.profiler.records.append(_amortized_inference_record(shared, plan, len(fresh)))
            try:
                _commit_inference(plan, output)
                ready.append((plan, output))
            except Exception as exc:
                _mark_failed(plan, exc)
                failures.append(f"{plan.scenario}/rate{plan.rate:g}: {type(exc).__name__}: {exc}")

    prepared_evaluators = None
    eval_setup_record = None
    if ready and (eval_direct or eval_downstream):
        eval_profiler = PhaseProfiler(device=device)
        try:
            with eval_profiler.phase("eval_setup", rows=len(bundle.train), n_settings=len(ready)):
                first_plan = ready[0][0]
                prepared_evaluators = prepare_imputation_evaluators(
                    bundle.test, bundle.train, bundle.schema,
                    _eval_options(first_plan), first_plan.n_trials,
                )
            eval_setup_record = eval_profiler.records[-1]
            shared_eval_setup_profile = eval_setup_record.to_dict()
        except BaseException as exc:
            for plan, _ in ready:
                _mark_failed(plan, exc)
            raise
    for plan, output in ready:
        try:
            results.append(_execute_imputation_run(
                plan, draws_output=output, prepared_evaluators=prepared_evaluators,
                eval_setup_record=eval_setup_record, eval_setup_grid_size=len(ready),
            ))
        except Exception as exc:
            failures.append(f"{plan.scenario}/rate{plan.rate:g}: {type(exc).__name__}: {exc}")
    if failures:
        raise RuntimeError(f"{len(failures)} imputation grid setting(s) failed: " + "; ".join(failures))

    results.sort(key=lambda item: requested.index((item["scenario"], float(item["rate"]))))
    return {
        "task": "imputation", "method": method, "dataset": dataset,
        "requested_settings": len(requested),
        "computed_settings": len(plans),
        "cache_hits": sum(bool(item.get("cache_hit")) for item in results),
        "shared_inference_profile": shared_profile,
        "shared_evaluation_setup_profile": shared_eval_setup_profile,
        "results": results,
    }


def _write_json(path: Path, payload: dict) -> None:
    write_json_atomic(path, payload)
