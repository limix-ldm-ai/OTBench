# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Load datasets from the local store (shared by every layer above).

Store layout:

    store/<dataset_type>/<name>/
        <name>_train.csv   # reference split (D_ref)
        <name>_test.csv    # held-out split
        schema.json        # EvalSchema fields
        graph.json         # SCM only: ground-truth causal graph
        meta.json          # provenance + constant-column record
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .schema import EvalSchema, enrich_schema, load_schema
from .io import read_frame_csv
from otbench.paths import workspace_root

STORE_ROOT = workspace_root() / "tabdata" / "store"


@dataclass
class DatasetBundle:
    name: str
    dataset_type: str               # "scm" | "real"
    dir: Path
    train: pd.DataFrame
    test: pd.DataFrame
    schema: EvalSchema
    graph: Optional[dict] = None    # SCM only
    meta: Optional[dict] = None

    @property
    def is_scm(self) -> bool:
        return self.dataset_type == "scm"


def find_dataset_dir(name: str, store_root: str | Path | None = None) -> Path:
    """Locate store/{scm,real}/<name>."""
    root = Path(store_root) if store_root is not None else STORE_ROOT
    for dataset_type in ("scm", "real"):
        candidate = root / dataset_type / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Dataset {name!r} not found under {root}/scm/ or {root}/real/. "
        f"Build it with `python -m otbench.tabdata.build_datasets {name}`."
    )


def load_dataset(name: str, store_root: str | Path | None = None) -> DatasetBundle:
    dataset_dir = find_dataset_dir(name, store_root)
    dataset_type = dataset_dir.parent.name

    declared = load_schema(dataset_dir / "schema.json")
    train = read_frame_csv(dataset_dir / f"{name}_train.csv", declared, strict=True)
    test = read_frame_csv(dataset_dir / f"{name}_test.csv", declared, strict=True)
    # Empirical domains/ranges are fitted on D_ref only. Keeping enrichment in
    # the bottom-layer loader gives every runner the same leakage-safe schema.
    schema = enrich_schema(train, declared)

    graph = _read_json(dataset_dir / "graph.json")
    meta = _read_json(dataset_dir / "meta.json")

    return DatasetBundle(
        name=name,
        dataset_type=dataset_type,
        dir=dataset_dir,
        train=train,
        test=test,
        schema=schema,
        graph=graph,
        meta=meta,
    )


def schema_to_col2type(schema: EvalSchema) -> dict[str, str]:
    """Map each column to 'numerical'/'categorical' (used by the methods layer)."""
    col2type: dict[str, str] = {}
    for col in schema.numerical_cols:
        col2type[col] = "numerical"
    for col in schema.categorical_cols:
        col2type[col] = "categorical"
    return col2type


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
