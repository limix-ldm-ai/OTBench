# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Generation evaluation dispatcher: selector -> metric modules -> results.

Selection goes through the shared catalog, so ``--metric`` accepts a family
name (``validity``, ``fidelity``, ``privacy``, ...) or any fully qualified
metric name, and an unknown selector fails immediately instead of silently
evaluating nothing.

Every metric family is evaluated inside a guard that converts an exception into
a ``failed`` result. One broken metric must not take the other forty-seven with
it, and a crash must stay distinguishable from a legitimate skip.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from otbench.generation.eval.data import EvalFrames
from otbench.generation.eval.metrics.ci import MAX_CI_STATEMENTS
from otbench.metrics import load_all_specs
from otbench.metrics.catalog import CATALOG, MetricResult
from otbench.metrics.cost import evaluate_cost

TASK = "generation"

@dataclass(frozen=True)
class EvalOptions:
    mode: str = "real"
    scm_path: str | None = None
    reference_split: str = "test"
    privacy_max_samples: int | None = None
    ci_alpha: float = 0.01
    ci_bins: int = 5
    ci_max_statements: int = MAX_CI_STATEMENTS
    seed: int = 0
    n_queries: int = 300
    method_metadata: dict = field(default_factory=dict)
    shadow_audit: dict | None = None
    profile: dict | None = None


def all_metrics() -> set[str]:
    load_all_specs()
    return set(CATALOG.names(TASK))


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
        else:
            try:
                selected |= CATALOG.select(token, TASK)
            except ValueError:
                unknown.append(token)
    if unknown:
        raise ValueError(
            f"Unknown metric selector(s) {unknown}. Families: {CATALOG.families(TASK)}"
        )
    return selected or available


def evaluate_selected(
    frames: EvalFrames,
    selected: set[str],
    options: EvalOptions,
) -> list[MetricResult]:
    """Evaluate every selected metric and return one result per metric."""
    load_all_specs()
    wanted = set(selected)
    results: list[MetricResult] = []

    def run(families: tuple[str, ...], compute) -> None:
        names = {n for n in wanted if CATALOG.require(n).family in families}
        if not names:
            return
        try:
            produced = compute()
        except Exception as exc:  # noqa: BLE001 — one family must not sink the report
            produced = [MetricResult.failed(name, exc) for name in sorted(names)]
        results.extend(r for r in produced if r.name in wanted)

    from otbench.generation.eval.metrics import (
        conditional,
        distinguishability,
        fidelity,
        query,
        support,
        utility,
    )
    from otbench.generation.eval.metrics.ci import evaluate_conditional_independence
    from otbench.generation.eval.metrics.validity import evaluate_validity

    run(("validity",), lambda: evaluate_validity(frames))
    run(("fidelity",), lambda: fidelity.evaluate_fidelity(frames, options.reference_split))
    run(("support",), lambda: support.evaluate_support(frames, options.reference_split))
    run(("memorization",), lambda: support.evaluate_memorization(frames))
    run(("distinguishability",), lambda: distinguishability.evaluate_distinguishability(
        frames, options.reference_split, seed=options.seed))
    run(("utility",), lambda: utility.evaluate_utility(
        frames,
        include_global="generation.utility.global_utility_ratio" in wanted,
        include_tstr="generation.utility.tstr_gap" in wanted,
        seed=options.seed,
    ))
    run(("query",), lambda: query.evaluate_query_error(
        frames, options.reference_split, seed=options.seed, n_queries=options.n_queries))
    run(("privacy",), lambda: _privacy_family(frames, options))
    run(("disclosure",), lambda: conditional.evaluate_attribute_disclosure(
        frames, seed=options.seed))
    run(("structural",), lambda: evaluate_conditional_independence(
        frames, options.scm_path, alpha=options.ci_alpha, n_bins=options.ci_bins,
        mode=options.mode, max_statements=options.ci_max_statements))
    run(("cost",), lambda: evaluate_cost(TASK, options.profile))

    # A metric that no family produced (e.g. because a module short-circuited)
    # still has to appear, otherwise "absent" and "not applicable" merge.
    produced = {r.name for r in results}
    for name in sorted(wanted - produced):
        results.append(MetricResult.skipped(name, "no evaluator produced this metric"))
    return sorted(results, key=lambda r: r.name)


def _privacy_family(frames: EvalFrames, options: EvalOptions) -> list[MetricResult]:
    from otbench.generation.eval.metrics import conditional, privacy as privacy_module
    from otbench.generation.privacy_audit import evaluate_shadow_audit

    results = privacy_module.evaluate_privacy(
        frames,
        seed=options.seed,
        max_reference_rows=options.privacy_max_samples or privacy_module.MAX_REFERENCE_ROWS,
    )
    results.extend(evaluate_shadow_audit(options.shadow_audit))
    results.extend(conditional.evaluate_dp_claim(options.method_metadata))
    return results


def summarize(results: list[MetricResult]) -> dict[str, Any]:
    """Group metric values, details, and unavailable-result reasons."""
    metrics = {r.name: r.value for r in results if r.status == "ok"}
    details = {r.name: r.details for r in results if r.details}
    skipped = {r.name: r.reason for r in results if r.status != "ok"}
    return {"metrics": metrics, "details": details, "skipped": skipped}
