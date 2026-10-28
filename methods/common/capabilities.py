# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Declarative capability metadata for registered methods.

The registry stores one :class:`MethodCapabilities` per method. Task runners and
metric evaluators read it *before* importing the adapter, so an unsupported
(method, dataset, metric) combination is reported as a structured ``skipped``
result instead of being silently approximated.

Nothing here is inferred at runtime: a capability that is not declared is not
available. Adapters that claim more than they implement are caught by the
registry/adapter consistency check in :mod:`otbench.methods.common.registry`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

# --- vocabularies ----------------------------------------------------------
# Feature-column dtypes a method can model natively. The supervised target is
# governed by task_types: a categorical classification target must not make an
# otherwise all-numerical feature table look mixed (SMOTE is the canonical case).
COLUMN_TYPES = frozenset({"numerical", "categorical"})

# Supervised task types, using the vendored-repo naming.
TASK_TYPES = frozenset({"binclass", "multiclass", "regression"})

# Which missingness patterns a method can impute.
#   cell   - arbitrary per-cell mask (MCAR/MAR/MNAR cell masks)
#   column - whole columns missing for a subset of rows
#   target - only the target column is missing (the prediction-as-imputation case)
MASK_GRANULARITY = frozenset({"cell", "column", "target"})

# What a prediction adapter can emit.
#   point     - a single y_pred per row (every method)
#   proba     - calibrated-ish class probabilities (classification)
#   samples   - draws from p(y|x) (probabilistic regression)
#   density   - an evaluable log p(y|x) (parametric probabilistic regression)
#   quantiles - a fixed quantile grid of p(y|x)
PREDICTION_OUTPUTS = frozenset({"point", "proba", "samples", "density", "quantiles"})

# What an imputation adapter can emit.
#   point - a single completed frame
#   draws - K jointly-sampled completed frames from the same mask (multiple imputation)
IMPUTATION_OUTPUTS = frozenset({"point", "draws"})

# How a method sees the evaluation rows.
#   inductive     - fitted on train only, applied to unseen rows
#   transductive  - the rows being imputed/predicted are part of the fit
LEARNING_MODES = frozenset({"inductive", "transductive"})

# How many models one logical "method run" trains.
AUXILIARY_MODEL_MODES = frozenset(
    {"single", "ensemble", "per_class", "per_column", "per_quantile", "per_draw"}
)

# Implementation maturity and distribution availability, surfaced in result payloads.
STATUSES = frozenset({"supported", "limited", "experimental", "diagnostic", "external_only"})


@dataclass(frozen=True)
class UpstreamRef:
    """Provenance of the algorithm implementation behind an adapter."""

    name: str
    url: str = ""
    version: str = ""          # pip version or vendored commit
    license: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "url": self.url,
            "version": self.version,
            "license": self.license,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class MethodCapabilities:
    """What a method can actually do, declared once in the registry."""

    column_types: frozenset = frozenset(COLUMN_TYPES)
    # Types the method cannot run *without*, as opposed to the ones it merely
    # tolerates. SMOTE-NC, for instance, accepts categoricals but interpolates
    # in numeric space and refuses a purely categorical frame.
    requires_column_types: frozenset = frozenset()
    task_types: frozenset = frozenset(TASK_TYPES)
    mask_granularity: frozenset = frozenset()
    trains_with_missing: bool = False
    learning_mode: str = "inductive"
    prediction_outputs: frozenset = frozenset()
    imputation_outputs: frozenset = frozenset()
    # Methods sharing a checkpoint_group reuse one fitted artifact across tasks.
    # Distinct groups must never reuse each other's checkpoints.
    checkpoint_group: str = ""
    auxiliary_models: str = "single"
    upstream: UpstreamRef = field(default_factory=lambda: UpstreamRef(name="unknown"))
    status: str = "supported"
    notes: str = ""

    def __post_init__(self) -> None:
        _check_subset("column_types", self.column_types, COLUMN_TYPES)
        _check_subset("requires_column_types", self.requires_column_types, COLUMN_TYPES)
        if not self.requires_column_types <= self.column_types:
            raise ValueError(
                "requires_column_types must be a subset of column_types; "
                f"{sorted(self.requires_column_types - self.column_types)} is required but not supported"
            )
        _check_subset("task_types", self.task_types, TASK_TYPES)
        _check_subset("mask_granularity", self.mask_granularity, MASK_GRANULARITY)
        _check_subset("prediction_outputs", self.prediction_outputs, PREDICTION_OUTPUTS)
        _check_subset("imputation_outputs", self.imputation_outputs, IMPUTATION_OUTPUTS)
        if self.learning_mode not in LEARNING_MODES:
            raise ValueError(f"learning_mode must be one of {sorted(LEARNING_MODES)}")
        if self.auxiliary_models not in AUXILIARY_MODEL_MODES:
            raise ValueError(f"auxiliary_models must be one of {sorted(AUXILIARY_MODEL_MODES)}")
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {sorted(STATUSES)}")

    # --- queries used by runners -----------------------------------------
    def supports_column_types(self, types: Iterable[str]) -> bool:
        return set(types).issubset(self.column_types)

    def supports_task_type(self, task_type: str) -> bool:
        return task_type in self.task_types

    def supports_mask(self, granularity: str) -> bool:
        return granularity in self.mask_granularity

    def prediction_outputs_for(self, task_type: str | None = None) -> frozenset:
        """Outputs applicable to one supervised task type.

        ``prediction_outputs`` is the union exposed by an adapter.  Probability
        matrices apply only to classification, while samples/densities/quantiles
        apply only to regression.  Keeping this rule here prevents every runner
        from inventing its own interpretation of the same declaration.
        """
        outputs = set(self.prediction_outputs)
        if task_type == "regression":
            outputs.discard("proba")
        elif task_type in {"binclass", "multiclass", "classification"}:
            outputs -= {"samples", "density", "quantiles"}
        return frozenset(outputs)

    def can_emit(self, output: str, task: str, task_type: str | None = None) -> bool:
        if task == "prediction":
            return output in self.prediction_outputs_for(task_type)
        if task == "imputation":
            return output in self.imputation_outputs
        return False

    def to_dict(self, task_type: str | None = None) -> dict[str, Any]:
        return {
            "column_types": sorted(self.column_types),
            "requires_column_types": sorted(self.requires_column_types),
            "task_types": sorted(self.task_types),
            "mask_granularity": sorted(self.mask_granularity),
            "trains_with_missing": self.trains_with_missing,
            "learning_mode": self.learning_mode,
            "prediction_outputs": sorted(self.prediction_outputs_for(task_type)),
            "imputation_outputs": sorted(self.imputation_outputs),
            "checkpoint_group": self.checkpoint_group,
            "auxiliary_models": self.auxiliary_models,
            "upstream": self.upstream.to_dict(),
            "status": self.status,
            "notes": self.notes,
        }


def dataset_task_type(schema) -> str:
    """Map an :class:`EvalSchema` onto the ``binclass/multiclass/regression`` vocabulary.

    ``schema.task_type`` only distinguishes regression from classification, so the
    binary/multiclass split has to come from the declared target domain.
    """
    if schema.task_type == "regression" or schema.target_col in schema.numerical_cols:
        return "regression"
    domain = schema.category_domain(schema.target_col)
    if domain is not None and len(domain) > 2:
        return "multiclass"
    if domain is not None:
        return "binclass"
    return "multiclass"


def dataset_column_types(schema) -> set[str]:
    types: set[str] = set()
    excluded = set(schema.id_cols) | {schema.target_col}
    if [c for c in schema.numerical_cols if c not in excluded]:
        types.add("numerical")
    if [c for c in schema.categorical_cols if c not in excluded]:
        types.add("categorical")
    return types


class UnsupportedRun(RuntimeError):
    """A method/dataset/task combination the registry declares out of scope.

    Distinct from a crash: the runner raises this *before* fitting anything so
    callers can record a skip rather than a failure.
    """


def unsupported_reason(
    caps: MethodCapabilities,
    schema,
    task: str,
    *,
    mask_granularity: str | None = None,
) -> str | None:
    """Return a human-readable skip reason, or ``None`` when the run is valid."""
    if caps.status == "external_only":
        return (
            f"{caps.upstream.name} source is not distributed: the reviewed upstream "
            f"supplied no redistribution license; obtain it from {caps.upstream.url}. "
            "This release provides an unavailable registry placeholder only."
        )
    task_type = dataset_task_type(schema)
    if not caps.supports_task_type(task_type):
        return f"method does not support task_type={task_type}"
    present_types = dataset_column_types(schema)
    missing_types = present_types - set(caps.column_types)
    if missing_types:
        return f"method does not support column types {sorted(missing_types)}"
    absent_required = set(caps.requires_column_types) - present_types
    if absent_required:
        return f"method requires column types {sorted(absent_required)}, which this dataset has none of"
    if task == "imputation":
        if not caps.mask_granularity:
            return "method declares no imputation mask granularity"
        if mask_granularity and not caps.supports_mask(mask_granularity):
            return f"method does not support mask_granularity={mask_granularity}"
    if task == "prediction" and "point" not in caps.prediction_outputs_for(task_type):
        return "method declares no point prediction output"
    return None


def _check_subset(field_name: str, values: Iterable[str], allowed: frozenset) -> None:
    extra = set(values) - set(allowed)
    if extra:
        raise ValueError(f"{field_name} has unknown entries {sorted(extra)}; allowed={sorted(allowed)}")
