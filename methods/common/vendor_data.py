# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Bridge from the CSV store to the original diffusion repos' data layout.

tab-ddpm / tabsyn / TabDiff all consume the same on-disk dataset layout
(produced upstream by their ``process_dataset.py``):

    <data_dir>/
        X_num_{train,val,test}.npy   # RAW numeric features (float32; NaN allowed,
                                     #   the repos' own Transformations fill them)
        X_cat_{train,val,test}.npy   # RAW categorical features (strings)
        y_{train,val,test}.npy       # classification: int codes; regression: float
        train.csv / val.csv / test.csv (original column names & order)
        info.json                    # column indices/mappings + counts + task_type
    <synthetic_dir>/<name>/real.csv, test.csv, val.csv   # eval references

This exporter reproduces that layout **with raw values** (no normalization —
unlike the discarded tabimpute vendor codec) so each vendored original pipeline
applies its own preprocessing (quantile transform, ordinal encoding, ...).

Notes:
- ``val`` is a deterministic 10%% slice of the benchmark train split (rng(0))
  that is **removed from the exported ``train``**. The two are disjoint, so
  epoch selection / early stopping / threshold fitting on ``val`` is unbiased.
  (Before v5 ``val`` was a *copy* of train rows, which made every "validation"
  number an in-sample number.)
- some original pipelines read their "test" split during training (tabsyn's
  VAE selects the best epoch on test CE; TabDiff logs density metrics against
  test). Feeding them the benchmark test would leak it into the
  imputation/prediction benchmarks, so a ``train_view/`` subdir is exported
  whose "test"/"val" files all point at the val subset. TRAINING always runs
  on ``train_view/``; INFERENCE runs on the main layout (test = benchmark
  test). Both views share the identical fit split, so every train-fitted
  transform (quantile/ordinal/target codec) is identical across views.
- classification targets are factorized to int codes 0..K-1 **from the train
  split only**; the original label values are recorded in
  ``info.json["y_classes"]`` for decoding. A test label absent from train is
  exported as code ``-1`` and counted in ``info.json["unknown_test_classes"]``
  — it must never widen the training label space.
- categorical features are exported as strings (NaN -> "nan", mirroring the
  upstream '?' -> 'nan' convention); original per-column values are recorded
  in ``info.json["cat_categories"]`` so decoded frames can restore dtypes.

The export is method- and seed-independent, so it is cached once per dataset
under ``checkpoints/_data_cache/<dataset>/`` and shared by the three methods.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from otbench.tabdata import DatasetBundle

from . import data as data_utils
from .checkpoints import checkpoints_root
from .locking import atomic_output_path, exclusive_directory_lock

VAL_FRACTION = 0.1
# Export contract: disjoint fit/validation rows, train-only vocabularies,
# numeric zero-width empty blocks, and identifiers excluded from model arrays.
VENDOR_DATA_EXPORT_VERSION = 6


def data_cache_dir(dataset: str, root: str | Path | None = None) -> Path:
    # Distinct export contracts use independent cache directories.
    return checkpoints_root(root) / f"_data_cache_v{VENDOR_DATA_EXPORT_VERSION}" / dataset


def ensure_vendor_data(bundle: DatasetBundle, root: str | Path | None = None) -> Path:
    """Export (or reuse) the vendor-format data dir for ``bundle``."""
    out = data_cache_dir(bundle.name, root)
    expected = _fingerprint(bundle)
    manifest_path = out / "manifest.json"
    if _manifest_matches(manifest_path, expected):
        return out
    # tabddpm/tabsyn/tabdiff and multiple task ranks share this cache.  The
    # second check makes waiters reuse the export produced by the first writer.
    with exclusive_directory_lock(
        Path(f"{out}.lockdir"), timeout_seconds=900, stale_after_seconds=3600
    ):
        if _manifest_matches(manifest_path, expected):
            return out
        export_vendor_data(bundle, out)
        return out


def _manifest_matches(path: Path, expected: str) -> bool:
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
            if manifest.get("fingerprint") == expected:
                return True
        except Exception:  # noqa: BLE001 — rebuild on any corruption
            pass
    return False


def export_vendor_data(bundle: DatasetBundle, output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    schema = bundle.schema

    # The vendored diffusion repositories must see exactly the modeled schema.
    # ``bundle.train.columns`` may also contain declared identifiers; those are
    # neither model features nor targets and corrupt the grouped index mapping.
    columns = list(schema.all_columns)
    target_col = schema.target_col
    num_cols = data_utils.numerical_feature_columns(schema)
    cat_cols = data_utils.categorical_feature_columns(schema)
    task_type = _vendor_task_type(bundle)

    full_train = bundle.train.reset_index(drop=True)[columns].copy()
    test = bundle.test.reset_index(drop=True)[columns].copy()
    # Disjoint fit/val: the exported "train" is what models actually fit on.
    train, val = train_val_split(full_train)

    # classification labels -> stable int codes derived from the FIT split only.
    # Using train+test here would let a test-only class change the model's output
    # dimension, i.e. leak the test label domain into training.
    y_classes: list[Any] = []
    if task_type != "regression":
        y_classes = pd.Series(train[target_col].dropna().unique()).sort_values(kind="stable").tolist()
    class_to_code = {value: idx for idx, value in enumerate(y_classes)}
    unknown_test_classes = (
        []
        if task_type == "regression"
        else sorted(
            {str(v) for v in test[target_col].dropna().unique() if v not in class_to_code}
        )
    )

    def _write_layout(root: Path, splits: dict[str, pd.DataFrame]) -> None:
        root.mkdir(parents=True, exist_ok=True)
        for split, frame in splits.items():
            X_num = frame[num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32) \
                if num_cols else np.zeros((len(frame), 0), dtype=np.float32)
            X_cat = _cat_matrix(frame, cat_cols)
            if task_type == "regression":
                y = pd.to_numeric(frame[target_col], errors="coerce").to_numpy(dtype=np.float32)
            else:
                # -1 marks a label outside the train vocabulary (test only).
                y = frame[target_col].map(class_to_code).fillna(-1).to_numpy(dtype=np.int64)
            np.save(root / f"X_num_{split}.npy", X_num)
            np.save(root / f"X_cat_{split}.npy", X_cat)
            np.save(root / f"y_{split}.npy", y)
            frame.to_csv(root / f"{split}.csv", index=False)
        synthetic = root / "synthetic" / bundle.name
        synthetic.mkdir(parents=True, exist_ok=True)
        splits["train"].to_csv(synthetic / "real.csv", index=False)
        splits["test"].to_csv(synthetic / "test.csv", index=False)
        splits["val"].to_csv(synthetic / "val.csv", index=False)
        info = _build_info(bundle, train, columns, num_cols, cat_cols, target_col, task_type,
                           y_classes, len(splits["train"]), len(splits["val"]), len(splits["test"]))
        info["unknown_test_classes"] = unknown_test_classes
        _write_json(root / "info.json", info)

    # main layout: inference view (test = benchmark test)
    _write_layout(out, {"train": train, "val": val, "test": test})
    # training view: identical train split, but "test"/"val" point at the val
    # subset so training-time selection/metrics never touch the benchmark test.
    _write_layout(out / "train_view", {"train": train, "val": val, "test": val})

    manifest = {
        "export_version": VENDOR_DATA_EXPORT_VERSION,
        "fingerprint": _fingerprint(bundle),
        "dataset": bundle.name,
        "splits": {"train": int(len(train)), "val": int(len(val)), "test": int(len(test))},
        "train_val_disjoint": True,
        "label_vocabulary_source": "train",
        "unknown_test_classes": unknown_test_classes,
    }
    _write_json(out / "manifest.json", manifest)
    return out


def train_view_dir(data_dir: str | Path) -> Path:
    """The layout whose 'test' split is the val subset (safe for training)."""
    return Path(data_dir) / "train_view"


def load_info(data_dir: str | Path) -> dict:
    with (Path(data_dir) / "info.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def decode_y(values: np.ndarray, info: dict) -> np.ndarray:
    """Map predicted target codes/values back to original label space."""
    if info["task_type"] == "regression":
        return np.asarray(values, dtype=np.float64)
    classes = info.get("y_classes") or []
    codes = np.clip(np.rint(np.asarray(values, dtype=np.float64)).astype(int), 0, len(classes) - 1)
    return np.asarray([classes[c] for c in codes], dtype=object)


# ---------------------------------------------------------------------------
def _vendor_task_type(bundle: DatasetBundle) -> str:
    if bundle.schema.task_type == "regression":
        return "regression"
    n_classes = bundle.train[bundle.schema.target_col].nunique(dropna=True)
    return "binclass" if n_classes <= 2 else "multiclass"


def train_val_indices(n_rows: int, val_fraction: float = VAL_FRACTION) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic disjoint (fit, val) positional indices for a train split.

    Shared by the vendored exporters and by any adapter that needs an unbiased
    validation set, so "the validation split" means the same rows everywhere.
    """
    if n_rows <= 1:
        return np.arange(n_rows), np.arange(n_rows)
    rng = np.random.default_rng(0)
    size = min(max(1, int(round(n_rows * val_fraction))), n_rows - 1)
    val_idx = np.sort(rng.choice(n_rows, size=size, replace=False))
    fit_idx = np.setdiff1d(np.arange(n_rows), val_idx, assume_unique=True)
    return fit_idx, val_idx


def train_val_split(train: pd.DataFrame, val_fraction: float = VAL_FRACTION):
    """Split a train frame into disjoint (fit, val) frames."""
    fit_idx, val_idx = train_val_indices(len(train), val_fraction)
    return (
        train.iloc[fit_idx].reset_index(drop=True),
        train.iloc[val_idx].reset_index(drop=True),
    )


def _cat_matrix(frame: pd.DataFrame, cat_cols: list[str]) -> np.ndarray:
    if not cat_cols:
        # A zero-width STRING array poisons downstream np.concatenate([X_num,
        # X_cat]) (result_type(float, str) -> str), which then breaks
        # torch.from_numpy in the vendored pipelines. Use a numeric zero-width
        # array so the concat stays float/int. Only the empty case matters —
        # non-empty categorical blocks are still raw strings below.
        return np.zeros((len(frame), 0), dtype=np.int64)
    block = frame[cat_cols].copy()
    for col in cat_cols:
        block[col] = block[col].astype(object).where(block[col].notna(), "nan").astype(str)
    return block.to_numpy(dtype=object)


def _build_info(
    bundle: DatasetBundle,
    fit_train: pd.DataFrame,
    columns: list[str],
    num_cols: list[str],
    cat_cols: list[str],
    target_col: str,
    task_type: str,
    y_classes: list[Any],
    n_train: int,
    n_val: int,
    n_test: int,
) -> dict:
    col_to_idx = {col: idx for idx, col in enumerate(columns)}
    num_col_idx = [col_to_idx[c] for c in num_cols]
    cat_col_idx = [col_to_idx[c] for c in cat_cols]
    target_col_idx = [col_to_idx[target_col]]

    # original idx -> grouped idx (num..., cat..., target) as in process_dataset.py
    idx_mapping: dict[int, int] = {}
    curr_num, curr_cat, curr_target = 0, len(num_col_idx), len(num_col_idx) + len(cat_col_idx)
    for idx in range(len(columns)):
        if idx in num_col_idx:
            idx_mapping[idx] = curr_num
            curr_num += 1
        elif idx in cat_col_idx:
            idx_mapping[idx] = curr_cat
            curr_cat += 1
        else:
            idx_mapping[idx] = curr_target
            curr_target += 1
    inverse_idx_mapping = {v: k for k, v in idx_mapping.items()}
    idx_name_mapping = {idx: col for idx, col in enumerate(columns)}

    # Integer-valued numeric feature columns (for TabDiff int dequantization),
    # decided on the FIT split only: this flag changes the model's noise schedule,
    # so consulting test here would make the test split alter training.
    int_columns, int_col_idx, int_col_idx_wrt_num = [], [], []
    for i, col in enumerate(num_cols):
        values = pd.to_numeric(fit_train[col], errors="coerce").dropna()
        is_int = bool(len(values)) and bool((values % 1 == 0).all())
        if is_int:
            int_columns.append(col)
            int_col_idx.append(col_to_idx[col])
            int_col_idx_wrt_num.append(i)

    metadata = {"columns": {}}
    for i in num_col_idx:
        metadata["columns"][i] = {"sdtype": "numerical", "computer_representation": "Float"}
    for i in cat_col_idx:
        metadata["columns"][i] = {"sdtype": "categorical"}
    for i in target_col_idx:
        if task_type == "regression":
            metadata["columns"][i] = {"sdtype": "numerical", "computer_representation": "Float"}
        else:
            metadata["columns"][i] = {"sdtype": "categorical"}

    cat_categories = {
        col: pd.Series(
            fit_train[col].astype(object).where(fit_train[col].notna(), "nan").astype(str).unique()
        ).sort_values(kind="stable").tolist()
        for col in cat_cols
    }

    return {
        "name": bundle.name,
        "task_type": task_type,
        "column_names": columns,
        "num_col_idx": num_col_idx,
        "cat_col_idx": cat_col_idx,
        "target_col_idx": target_col_idx,
        "target_col": target_col,
        "idx_mapping": idx_mapping,
        "inverse_idx_mapping": inverse_idx_mapping,
        "idx_name_mapping": idx_name_mapping,
        "train_num": int(n_train),
        "val_num": int(n_val),
        "test_num": int(n_test),
        "n_num_features": len(num_col_idx),
        "n_cat_features": len(cat_col_idx),
        "n_classes": int(len(y_classes)) if task_type != "regression" else 0,
        "y_classes": y_classes,
        "cat_categories": cat_categories,
        "int_columns": int_columns,
        "int_col_idx": int_col_idx,
        "int_col_idx_wrt_num": int_col_idx_wrt_num,
        "metadata": metadata,
    }


def _fingerprint(bundle: DatasetBundle) -> str:
    payload = "|".join(
        [
            str(VENDOR_DATA_EXPORT_VERSION),
            data_utils.dataframe_hash(bundle.train),
            data_utils.dataframe_hash(bundle.test),
            data_utils.schema_hash(bundle.schema),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _write_json(path: Path, payload: dict) -> None:
    with atomic_output_path(path) as tmp:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(_jsonable(payload), f, indent=2, ensure_ascii=False)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
