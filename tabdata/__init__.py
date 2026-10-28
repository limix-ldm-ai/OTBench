"""Dataset layer: store + loader + schema contract.

Named ``tabdata`` (not ``datasets``) on purpose: a top-level/regular package
called ``datasets`` would shadow HuggingFace ``datasets``, which the GReaT
method (via ``be_great``) imports at runtime.
"""
from .loader import DatasetBundle, find_dataset_dir, load_dataset, schema_to_col2type
from .schema import (
    EvalSchema,
    derive_empirical_constraints,
    enrich_schema,
    load_schema,
    schema_from_dict,
)

__all__ = [
    "DatasetBundle",
    "EvalSchema",
    "derive_empirical_constraints",
    "enrich_schema",
    "find_dataset_dir",
    "load_dataset",
    "load_schema",
    "schema_from_dict",
    "schema_to_col2type",
]
