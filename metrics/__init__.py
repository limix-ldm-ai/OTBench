# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Cross-task metric layer: the catalog, the result protocol, and shared math.

Importing this package registers every task's metric specs into the single
:data:`~otbench.metrics.catalog.CATALOG`, so tools that only need metric
*metadata* (the redundancy study, documentation generators, the CLI's
``--list-metrics``) can read it without importing torch or any method backend.
"""
from __future__ import annotations

from .catalog import (
    CATALOG,
    DIAGNOSTIC,
    METRIC_SCHEMA_VERSION,
    P0,
    P0_CONDITIONAL,
    MetricResult,
    MetricSpec,
    results_to_payload,
)


def load_all_specs():
    """Import every task's spec module so the catalog is complete."""
    from otbench.generation.eval import specs as _generation_specs  # noqa: F401
    from otbench.imputation import specs as _imputation_specs  # noqa: F401
    from otbench.prediction import specs as _prediction_specs  # noqa: F401

    return CATALOG


__all__ = [
    "CATALOG",
    "DIAGNOSTIC",
    "METRIC_SCHEMA_VERSION",
    "MetricResult",
    "MetricSpec",
    "P0",
    "P0_CONDITIONAL",
    "load_all_specs",
    "results_to_payload",
]
