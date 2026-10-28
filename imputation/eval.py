# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Imputation evaluation dispatcher: one call, one result per catalog metric.

The split between what runs per draw and what runs once is deliberate. Point
and structure metrics (I01-I07) describe a single completed table, so they are
computed per draw and averaged with the spread kept. The probabilistic metrics
(I10-I17) describe the *set* of draws and are computed once over all of them.
Computing the second kind per draw and averaging would silently turn a
calibration metric into a point metric.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from otbench.methods.common.base import ImputationOutput
from otbench.metrics import load_all_specs
from otbench.metrics.catalog import CATALOG, MetricResult
from otbench.metrics.cost import evaluate_cost

from .conditional import (
    GROUP_SOURCE_METRICS,
    PreparedLabelStability,
    evaluate_group_gap,
    evaluate_label_stability,
    evaluate_rubin_validity,
    evaluate_shift_degradation,
    prepare_label_stability,
)
from .downstream import (
    PreparedDownstreamUtility,
    evaluate_downstream_utility,
    prepare_downstream_utility,
)
from .draw_metrics import evaluate_draw_metrics
from .masking import MaskBundle
from .metrics import aggregate_results, evaluate_point_metrics
from .validation import validate_completion

TASK = "imputation"


@dataclass(frozen=True)
class ImputationEvaluationPlan:
    requested: frozenset[str]
    enabled: frozenset[str]

    @property
    def needs_point(self) -> bool:
        return any(CATALOG.require(name).family in {"point", "structure"}
                   or name == "imputation.fairness.group_gap" for name in self.enabled)

    @property
    def needs_downstream(self) -> bool:
        return any(CATALOG.require(name).family == "downstream"
                   or name == "imputation.fairness.downstream_gap" for name in self.enabled)


def build_evaluation_plan(metric: str = "all", *, include_direct: bool = True,
                          include_downstream: bool = True) -> ImputationEvaluationPlan:
    requested = frozenset(resolve_metrics(metric))
    if not include_direct and not include_downstream:
        return ImputationEvaluationPlan(requested, frozenset())
    enabled = set()
    for name in requested:
        family = CATALOG.require(name).family
        downstream = family == "downstream" or name == "imputation.fairness.downstream_gap"
        if family == "cost" or (include_downstream if downstream else include_direct):
            enabled.add(name)
    return ImputationEvaluationPlan(requested, frozenset(enabled))


@dataclass(frozen=True)
class PreparedImputationEvaluators:
    """Train-fitted evaluators reusable across one dataset/seed grid."""

    downstream: PreparedDownstreamUtility | None = None
    label_stability: PreparedLabelStability | None = None


@dataclass(frozen=True)
class ImputationEvalOptions:
    """What the evaluator is allowed to compute, and with what preconditions."""

    random_state: int = 0
    native_draws: bool = True
    profile: dict | None = None
    scientific_model: dict | None = None
    matched_runs: list[dict] | None = None
    stability_protocol: dict | None = None
    include_downstream: bool = True
    include_direct: bool = True
    metric: str = "all"
    plan: ImputationEvaluationPlan | None = None
    prepared_evaluators: PreparedImputationEvaluators | None = field(
        default=None, repr=False, compare=False
    )


def resolve_metrics(metric_arg: str) -> set[str]:
    load_all_specs()
    available = set(CATALOG.names(TASK))
    selected: set[str] = set()
    unknown: list[str] = []
    for token in (part.strip() for part in metric_arg.split(",")):
        if not token:
            continue
        if token == "all":
            selected |= available
            continue
        try:
            selected |= CATALOG.select(token, TASK)
        except ValueError:
            unknown.append(token)
    if unknown:
        raise ValueError(
            f"Unknown metric selector(s) {unknown}. Families: {CATALOG.families(TASK)}"
        )
    return selected or available


def prepare_imputation_evaluators(
    clean_test: pd.DataFrame,
    train: pd.DataFrame,
    schema,
    options: ImputationEvalOptions,
    n_draws: int,
) -> PreparedImputationEvaluators:
    """Prepare only expensive evaluators whose selected metrics can run."""

    plan = options.plan or build_evaluation_plan(
        options.metric, include_direct=options.include_direct,
        include_downstream=options.include_downstream,
    )
    wanted = plan.enabled
    families = {CATALOG.require(name).family for name in wanted}
    downstream = None
    if plan.needs_downstream:
        downstream = prepare_downstream_utility(
            train, clean_test, schema, options.random_state
        )

    stability = None
    protocol = options.stability_protocol
    task = schema.task_type or (
        "regression" if schema.target_col in schema.numerical_cols else "classification"
    )
    if (
        "stability" in families
        and options.native_draws
        and n_draws >= 2
        and isinstance(protocol, dict)
        and protocol.get("pre_registered")
        and task == "classification"
    ):
        stability = prepare_label_stability(train, schema, options.random_state)

    return PreparedImputationEvaluators(
        downstream=downstream, label_stability=stability
    )


def evaluate_imputation(
    clean_test: pd.DataFrame,
    draws_output: ImputationOutput,
    mask_bundle: MaskBundle,
    schema,
    train: pd.DataFrame,
    options: ImputationEvalOptions = ImputationEvalOptions(),
) -> list[MetricResult]:
    load_all_specs()
    plan = options.plan or build_evaluation_plan(
        options.metric, include_direct=options.include_direct,
        include_downstream=options.include_downstream,
    )
    wanted = plan.enabled
    results = [MetricResult.skipped(name, "disabled by the explicit evaluation switches")
               for name in sorted(plan.requested - wanted)]
    draws = list(draws_output.draws)
    for draw in draws:
        validate_completion(clean_test, draw, mask_bundle.mask, schema)

    def run(families: tuple[str, ...], compute, *, owned_names=None) -> list[MetricResult]:
        names = ({n for n in wanted if CATALOG.require(n).family in families}
                 if owned_names is None else set(owned_names) & wanted)
        if not names:
            return []
        try:
            produced = compute()
        except Exception as exc:  # noqa: BLE001 — one family must not sink the report
            produced = [MetricResult.failed(name, exc) for name in sorted(names)]
        kept = [r for r in produced if r.name in names]
        results.extend(kept)
        return produced

    # --- I01-I07: per draw, then averaged (the spread goes in the details) ---
    point_results = []
    if plan.needs_point:
        point_names = {name for name in wanted if CATALOG.require(name).family in {"point", "structure"}}
        if "imputation.fairness.group_gap" in wanted:
            point_names |= set(GROUP_SOURCE_METRICS)
        try:
            point_results = _average_over_draws([
                evaluate_point_metrics(clean_test, draw, mask_bundle, schema, train,
                                       selected=point_names) for draw in draws
            ])
        except Exception as exc:
            point_results = [MetricResult.failed(name, exc) for name in sorted(point_names)]
        results.extend(r for r in point_results if r.name in wanted)

    point_estimate = None
    if plan.needs_downstream or "imputation.fairness.group_gap" in wanted:
        point_estimate = draws_output.point_estimate(schema)

    # --- I08, I19: downstream, on the protocol point estimate --------------
    if plan.needs_downstream and draws:
        prepared = options.prepared_evaluators
        run(("downstream", "fairness"), lambda: evaluate_downstream_utility(
            train, point_estimate, clean_test, schema,
            random_state=options.random_state, group_cols=list(schema.group_cols),
            prepared=prepared.downstream if prepared else None,
        ), owned_names={"imputation.downstream.utility_gap", "imputation.fairness.downstream_gap"})

    # --- I18: subgroup gap of the point metrics -----------------------------
    if draws and "imputation.fairness.group_gap" in wanted:
        run(("fairness",), lambda: [evaluate_group_gap(
            clean_test, point_estimate, mask_bundle, schema, train, point_results
        )], owned_names={"imputation.fairness.group_gap"})

    # --- I10-I17: once over the whole set of draws --------------------------
    run(("draws",), lambda: evaluate_draw_metrics(
        clean_test, draws_output, mask_bundle, schema, train,
        native_draws=options.native_draws,
    ))

    # --- I20-I22: conditional --------------------------------------------
    prepared = options.prepared_evaluators
    run(("stability",), lambda: [evaluate_label_stability(
        train, draws, schema, random_state=options.random_state,
        native_draws=options.native_draws,
        protocol=options.stability_protocol,
        prepared=prepared.label_stability if prepared else None,
    )])
    run(("robustness",), lambda: [evaluate_shift_degradation(options.matched_runs)])
    run(("inference",), lambda: [evaluate_rubin_validity(
        draws, options.scientific_model, truth=clean_test,
        native_draws=options.native_draws,
    )])
    run(("cost",), lambda: evaluate_cost(TASK, options.profile))

    produced = {r.name for r in results}
    for name in sorted(wanted - produced):
        results.append(MetricResult.skipped(name, "no evaluator produced this metric"))
    return sorted(results, key=lambda r: r.name)


def _average_over_draws(per_draw: list[list[MetricResult]]) -> list[MetricResult]:
    """Mean of a per-draw metric, carrying the across-draw spread and statuses."""
    if not per_draw:
        return []
    if len(per_draw) == 1:
        return per_draw[0]

    aggregate = aggregate_results(per_draw)
    by_name = {r.name: r for draw in reversed(per_draw) for r in draw}
    for draw in per_draw:
        by_name.update({r.name: r for r in draw if r.status == "ok"})
    out: list[MetricResult] = []
    for name, record in aggregate.items():
        template = by_name.get(name)
        if record["mean"] is None:
            reason = (template.reason if template and template.reason
                      else "no draw produced this metric")
            statuses = record["statuses"]
            status = next((s for s in ("oom", "timeout", "failed", "unsupported", "skipped")
                           if s in statuses), "skipped")
            out.append(MetricResult(name, status=status, reason=reason,
                                    extra={"draw_statuses": statuses, "n_draws": record["n_trials"]}))
            continue
        extra = dict(template.extra) if template else {}
        extra.update({
            "n_draws_scored": record["n_ok"],
            "n_draws": record["n_trials"],
            "std_over_draws": record["std"],
            "aggregation_over_draws": "mean of the per-draw value",
            "draw_statuses": record["statuses"],
        })
        out.append(MetricResult.ok(
            name, record["mean"], n=template.n if template else None,
            details=template.details if template else None, extra=extra,
        ))
    return out


def summarize(results: list[MetricResult]) -> dict[str, Any]:
    return {
        "metrics": {r.name: r.value for r in results if r.status == "ok"},
        "results": [r.to_dict() for r in results],
        "skipped": {r.name: r.reason for r in results if r.status != "ok"},
    }
