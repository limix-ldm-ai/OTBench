"""G27 / I09 / P25: the COST family, derived from the shared phase profiler.

Every task reports the same seven quantities under a task-prefixed name, so a
cross-task compute comparison does not have to reconcile three conventions.

Two conventions worth stating, because they change how the numbers read:

* ``fit_seconds`` is 0 when a valid checkpoint was reused. That is the honest
  number for *this* run; the training cost is attributed to the run that paid it.
* Pretrained/foundation methods report their local inference cost only. The
  upstream pretraining cost is not amortised into it, and saying so is part of
  the metric definition rather than a footnote.
"""
from __future__ import annotations

from typing import Any

from .catalog import MetricResult

# task -> (metric leaf, profiler phase, throughput unit)
_THROUGHPUT = {
    "generation": ("sampling_seconds_per_1k_rows", "inference", "sec_per_1k_rows"),
    "imputation": ("seconds_per_1k_missing_cells", "inference", "sec_per_1k_cells"),
    "prediction": ("inference_seconds_per_1k_rows", "inference", "sec_per_1k_rows"),
}


def evaluate_cost(task: str, profile: dict[str, Any] | None) -> list[MetricResult]:
    """Turn one :meth:`PhaseProfiler.to_dict` payload into COST metric results."""
    leaf, phase_name, throughput_key = _THROUGHPUT[task]
    names = [
        f"{task}.cost.fit_seconds",
        f"{task}.cost.{leaf}",
        f"{task}.cost.eval_seconds",
        f"{task}.cost.peak_ram_mb",
        f"{task}.cost.peak_vram_mb",
        f"{task}.cost.n_parameters",
        f"{task}.cost.n_auxiliary_models",
    ]
    if not profile:
        return [MetricResult.skipped(name, "no profiling record for this run") for name in names]

    phases = {record["name"]: record for record in profile.get("phases", [])}
    totals = profile.get("totals_sec", {})
    model = profile.get("model", {})
    inference = phases.get(phase_name, {})

    fit = phases.get("fit", {})
    checkpoint_reused = not fit.get("extra", {}).get("executed", True)
    fit_seconds = 0.0 if fit and checkpoint_reused else totals.get("fit")
    results = [
        _scalar(names[0], fit_seconds, "the run did not include a fit phase",
                extra={"checkpoint_reused": checkpoint_reused}),
        _scalar(names[1], inference.get(throughput_key),
                "the inference phase recorded no workload size"),
        _scalar(names[2], totals.get("eval"), "the run did not include an eval phase"),
        _scalar(names[3], _peak(phases, "peak_ram_mb"), "host memory tracking was disabled"),
        _scalar(names[4], _peak(phases, "peak_vram_mb"), "no CUDA device was in use"),
        _scalar(names[5], model.get("n_parameters"),
                "the backend does not expose a parameter count"),
        _scalar(names[6], model.get("n_auxiliary_models"),
                "the method did not report an auxiliary model count"),
    ]
    failures = [r for r in profile.get("phases", []) if r.get("status") == "failed"]
    if failures:
        for result in results:
            result.extra["phase_failures"] = [
                {"phase": r["name"], "error": r.get("error")} for r in failures
            ]
    return results


def finalize_cost_results(
    task: str,
    results: list[MetricResult],
    profile: dict[str, Any],
) -> list[MetricResult]:
    """Replace provisional COST results after the eval phase has closed.

    Evaluators run *inside* the timed eval context, so a profile captured there
    cannot yet contain eval duration.  Runners call this once outside the
    context; only already-selected cost results are replaced.
    """
    final = {result.name: result for result in evaluate_cost(task, profile)}
    return [final.get(result.name, result) for result in results]


def _scalar(name: str, value, reason: str, extra: dict | None = None) -> MetricResult:
    if value is None:
        return MetricResult.skipped(name, reason)
    return MetricResult.ok(name, float(value), extra=dict(extra or {}))


def _peak(phases: dict[str, dict], key: str) -> float | None:
    values = [record.get(key) for record in phases.values() if record.get(key) is not None]
    return max(values) if values else None
