"""The single metric catalog: names, directions, sources and result protocol.

Every number this benchmark reports is described by exactly one
:class:`MetricSpec` and carried by exactly one :class:`MetricResult`. The point
is the downstream metric-redundancy study: it needs to know, for each metric,
what it measures, which way is better, what it required as input, and — for
every missing value — *why* it is missing.

Two rules the rest of the codebase relies on:

1. **A missing value is never just ``None``.** ``unsupported`` (the method
   cannot produce the required input), ``skipped`` (a precondition such as an
   OOD split was not declared), ``failed`` (the computation raised), ``timeout``
   and ``oom`` are distinct statuses. Collapsing them into one NA makes
   "generator has no privacy leak" indistinguishable from "the privacy metric
   crashed".
2. **Names are unambiguous.** The catalog forbids bare ``dcr``, bare ``coverage``,
   bare ``ci`` and bare ``auc``: each has at least two incompatible meanings in
   this literature, so the family has to be part of the name.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

METRIC_SCHEMA_VERSION = "1.1.0"

# --- statuses --------------------------------------------------------------
OK = "ok"
SKIPPED = "skipped"          # precondition not declared (no OOD split, no groups, ...)
UNSUPPORTED = "unsupported"  # the method cannot emit the required output
FAILED = "failed"            # the computation raised
TIMEOUT = "timeout"
OOM = "oom"
STATUSES = frozenset({OK, SKIPPED, UNSUPPORTED, FAILED, TIMEOUT, OOM})

# --- directions ------------------------------------------------------------
HIGHER = "higher_better"
LOWER = "lower_better"
TARGET_ZERO = "target_zero"    # an error/gap: 0 is ideal, either sign is bad
TARGET_HALF = "target_half"    # a discriminator AUROC: 0.5 is ideal
NEUTRAL = "neutral"            # descriptive, no preferred direction
DIRECTIONS = frozenset({HIGHER, LOWER, TARGET_ZERO, TARGET_HALF, NEUTRAL})

# --- tiers -----------------------------------------------------------------
P0 = "P0"
P0_CONDITIONAL = "P0-conditional"   # P0 when its precondition holds, else skipped
DIAGNOSTIC = "diagnostic"           # descriptive detail outside headline metrics

# --- input requirement tokens ---------------------------------------------
# Checked by the runners against method capabilities and schema declarations.
REQ_RAW_GEN = "raw_gen"                 # the unrepaired generator output
REQ_DRAWS = "draws"                     # K>1 joint imputation draws
REQ_PROBA = "proba"                     # class probabilities
REQ_SAMPLES = "samples"                 # samples from p(y|x)
REQ_DENSITY = "density"                 # evaluable log p(y|x)
REQ_QUANTILES = "quantiles"             # native quantiles of p(y|x)
REQ_SENSITIVE = "sensitive_cols"
REQ_QUASI_IDENTIFIERS = "quasi_identifier_cols"
REQ_GROUPS = "group_cols"
REQ_OOD = "ood_split"
REQ_SCM_GRAPH = "scm_graph"
REQ_SHADOW = "shadow_models"            # membership-inference shadow training runs
REQ_DP_ACCOUNTANT = "dp_accountant"
REQ_SCIENTIFIC_MODEL = "scientific_model"   # a pre-registered estimand for Rubin's rules
REQ_MATCHED_RUNS = "matched_runs"       # paired runs across missingness rates
REQ_NUMERICAL = "numerical_columns"
REQ_CATEGORICAL = "categorical_columns"
REQ_CLASSIFICATION = "classification"
REQ_REGRESSION = "regression"

# --- name hygiene ----------------------------------------------------------
# Bare names with more than one accepted meaning in this literature.
AMBIGUOUS_LEAVES = frozenset({"dcr", "coverage", "ci", "auc", "utility", "distance"})


@dataclass(frozen=True)
class MetricSpec:
    """What a metric is, not how it is computed."""

    name: str                 # "<task>.<family>.<leaf>"
    task: str                 # generation | imputation | prediction | any
    family: str               # validity | fidelity | ... (the middle name segment)
    direction: str
    tier: str = P0
    requires: frozenset = frozenset()
    aggregation: str = "global"     # global | macro_over_columns | weighted_by_cells | ...
    source: str = ""                # citation / upstream definition
    seeded: bool = False            # result depends on an eval RNG seed
    description: str = ""

    def __post_init__(self) -> None:
        if self.direction not in DIRECTIONS:
            raise ValueError(f"{self.name}: direction must be one of {sorted(DIRECTIONS)}")
        parts = self.name.split(".")
        if len(parts) < 3:
            raise ValueError(f"{self.name}: metric names must be '<task>.<family>.<leaf>'")
        if parts[0] != self.task:
            raise ValueError(f"{self.name}: name prefix must match task={self.task!r}")
        if parts[1] != self.family:
            raise ValueError(f"{self.name}: second segment must match family={self.family!r}")
        if parts[-1] in AMBIGUOUS_LEAVES:
            raise ValueError(
                f"{self.name}: {parts[-1]!r} is ambiguous on its own — qualify it "
                f"(e.g. 'mean_min_euclidean_dcr', 'beta_recall_coverage', "
                f"'conditional_independence', 'attack_auroc')"
            )
        if not self.source:
            raise ValueError(f"{self.name}: every metric must record where its definition comes from")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task": self.task,
            "family": self.family,
            "direction": self.direction,
            "tier": self.tier,
            "requires": sorted(self.requires),
            "aggregation": self.aggregation,
            "source": self.source,
            "seeded": self.seeded,
            "description": self.description,
        }


@dataclass
class MetricResult:
    """One metric's outcome, including the reason when there is no value."""

    name: str
    value: Optional[float] = None
    status: str = OK
    reason: Optional[str] = None
    # Per-column / per-group breakdown; kept next to the scalar so a suspicious
    # summary can always be traced back to the column that produced it.
    details: dict = field(default_factory=dict)
    # How many units (rows / cells / columns) the scalar was averaged over.
    n: Optional[int] = None
    seed: Optional[int] = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"{self.name}: unknown status {self.status!r}")
        if self.status == OK and self.value is None:
            raise ValueError(f"{self.name}: status=ok requires a finite scalar value")
        if self.status == OK and self.value is not None:
            try:
                finite = bool(math.isfinite(float(self.value)))
            except (TypeError, ValueError):
                finite = False
            if not finite:
                raise ValueError(f"{self.name}: status=ok requires a finite scalar value")
        if self.status != OK and not self.reason:
            raise ValueError(f"{self.name}: status={self.status} requires a reason")

    @classmethod
    def ok(cls, name: str, value: float | None, **kwargs) -> "MetricResult":
        return cls(name=name, value=value, status=OK, **kwargs)

    @classmethod
    def skipped(cls, name: str, reason: str, **kwargs) -> "MetricResult":
        return cls(name=name, status=SKIPPED, reason=reason, **kwargs)

    @classmethod
    def unsupported(cls, name: str, reason: str, **kwargs) -> "MetricResult":
        return cls(name=name, status=UNSUPPORTED, reason=reason, **kwargs)

    @classmethod
    def failed(cls, name: str, error: BaseException, **kwargs) -> "MetricResult":
        return cls(
            name=name, status=FAILED, reason=f"{type(error).__name__}: {error}", **kwargs
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "value": self.value,
            "status": self.status,
            "schema_version": METRIC_SCHEMA_VERSION,
        }
        spec = CATALOG.get(self.name)
        if spec is not None:
            payload["direction"] = spec.direction
            payload["tier"] = spec.tier
            payload["source"] = spec.source
            payload["aggregation"] = spec.aggregation
            payload["requires"] = sorted(spec.requires)
            payload["seeded"] = spec.seeded
        if self.reason is not None:
            payload["reason"] = self.reason
        if self.n is not None:
            payload["n"] = self.n
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.details:
            payload["details"] = self.details
        if self.extra:
            payload.update(self.extra)
        return payload


class MetricCatalog:
    """Name -> :class:`MetricSpec`, with duplicate-name protection."""

    def __init__(self) -> None:
        self._specs: dict[str, MetricSpec] = {}

    def register(self, spec: MetricSpec) -> MetricSpec:
        if spec.name in self._specs:
            raise ValueError(f"Metric {spec.name!r} is already registered.")
        self._specs[spec.name] = spec
        return spec

    def add(self, **kwargs) -> MetricSpec:
        return self.register(MetricSpec(**kwargs))

    def get(self, name: str) -> MetricSpec | None:
        return self._specs.get(name)

    def require(self, name: str) -> MetricSpec:
        try:
            return self._specs[name]
        except KeyError:
            raise KeyError(f"Metric {name!r} is not in the catalog.") from None

    def names(self, task: str | None = None, tier: str | None = None) -> list[str]:
        return sorted(
            name
            for name, spec in self._specs.items()
            if (task is None or spec.task == task) and (tier is None or spec.tier == tier)
        )

    def specs(self, task: str | None = None) -> list[MetricSpec]:
        return [self._specs[name] for name in self.names(task)]

    def families(self, task: str) -> list[str]:
        return sorted({spec.family for spec in self.specs(task)})

    def select(self, selector: str, task: str) -> set[str]:
        """Resolve a CLI selector (``all``, a family name, or explicit names)."""
        available = set(self.names(task))
        selected: set[str] = set()
        for token in (part.strip() for part in selector.split(",")):
            if not token:
                continue
            if token == "all":
                selected |= available
            elif token in self.families(task):
                selected |= {n for n in available if self._specs[n].family == token}
            elif token in available:
                selected.add(token)
            elif f"{task}.{token}" in available:
                selected.add(f"{task}.{token}")
            else:
                raise ValueError(
                    f"Unknown metric selector {token!r} for task {task!r}. "
                    f"Families: {self.families(task)}"
                )
        return selected or available

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": METRIC_SCHEMA_VERSION,
            "metrics": [spec.to_dict() for spec in self.specs()],
        }


CATALOG = MetricCatalog()


def results_to_payload(
    results: Iterable[MetricResult],
    *,
    seed: int | None = None,
    extra: dict | None = None,
) -> dict[str, Any]:
    """Serialise results into the shared report layout.

    ``metrics`` holds the scalars that were actually computed (convenient for
    tables); ``results`` holds the full record for every requested metric,
    including the ones that were skipped and why.
    """
    records = [r.to_dict() for r in results]
    return {
        "schema_version": METRIC_SCHEMA_VERSION,
        "seed": seed,
        "metrics": {r["name"]: r["value"] for r in records if r["status"] == OK},
        "results": records,
        "status_counts": _status_counts(records),
        **(extra or {}),
    }


def _status_counts(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    return counts
