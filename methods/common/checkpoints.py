"""Checkpoint directory layout, metadata contract, and reuse validation.

Layout:
    checkpoints/<method>/<dataset>/train_seed<seed>/   # trainable methods
    checkpoints/<method>/pretrained/                    # foundation models

``metadata.json`` records dataset/schema/config hashes; a checkpoint is reused
only when all of them match the current run ("has ckpt -> infer directly").
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Optional

from . import data as data_utils
from .config import config_hash
from .locking import atomic_output_path, exclusive_file_lock
from .resources import (workspace_root, methods_source_hash, runtime_identity,
                        pretrained_resources, method_resources)

CHECKPOINTS_ROOT = workspace_root() / "checkpoints"
METADATA_FILE = "metadata.json"
# The metadata version binds dataset/schema/config, training implementation,
# runtime and resource identities. A different contract cannot be reused.
METADATA_VERSION = 4


def runtime_fingerprint(packages: tuple[str, ...] = ()) -> dict[str, Any]:
    """Return the Python/package ABI needed to safely unpickle a checkpoint."""
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = "missing"
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "packages": versions,
    }


def checkpoints_root(root: str | Path | None = None) -> Path:
    return Path(root).expanduser().resolve() if root is not None else workspace_root() / "checkpoints"


def checkpoint_dir(
    method: str,
    dataset: str,
    train_seed: int,
    root: str | Path | None = None,
) -> Path:
    return checkpoints_root(root) / method / dataset / f"train_seed{train_seed}"


def checkpoint_dir_for_method(
    method_obj,
    dataset: str,
    train_seed: int,
    root: str | Path | None = None,
) -> Path:
    """Route checkpoint storage through the registry-declared sharing group."""
    group = method_obj.capabilities.checkpoint_group or method_obj.name
    return checkpoint_dir(group, dataset, train_seed, root)


def pretrained_dir(method: str, root: str | Path | None = None) -> Path:
    return checkpoints_root(root) / method / "pretrained"


def checkpoint_lock(ckpt_dir: str | Path):
    """Serialize writers for one registry-routed checkpoint directory."""
    return exclusive_file_lock(Path(f"{Path(ckpt_dir)}.lock"))


def training_protocol(method: str) -> dict[str, Any]:
    """Training inputs not represented by the method TOML configuration.

    The three vendored diffusion pipelines train from a derived on-disk view.
    Its exporter version is therefore part of their checkpoint ABI: a schema
    change in that view must invalidate the checkpoint even when dataset/config
    hashes happen to remain unchanged.
    """
    if method in {"tabddpm", "tabsyn", "tabdiff"}:
        # Lazy import avoids the vendor_data -> checkpoints module dependency
        # during package initialization while keeping one source of truth.
        from .vendor_data import VENDOR_DATA_EXPORT_VERSION

        return {"vendor_data_export_version": VENDOR_DATA_EXPORT_VERSION}
    return {}


def _hashes(bundle, config: dict, method: str | None = None,
            checkpoint_dir: str | Path | None = None) -> dict[str, str]:
    # Inference controls do not retrain. Other declared sections include native
    # training parameters such as CARD's diffusion/guidance and DiffPuter's em.
    inference_sections = {"sample", "impute", "predict", "generate"}
    train_config = {key: value for key, value in config.items() if key not in inference_sections}
    hashes = {
        "dataset_hash": data_utils.dataframe_hash(bundle.train),
        "schema_hash": data_utils.schema_hash(bundle.schema),
        "column_order_hash": data_utils.column_order_hash(bundle.train.columns),
        "config_hash": config_hash(train_config),
        "training_code_hash": methods_source_hash(),
        "training_runtime_hash": config_hash(runtime_identity()),
    }
    if pretrained_resources(method or ""):
        from types import SimpleNamespace
        # Trainable backends such as GReaT also depend on a frozen base model.
        # Recover the root from the canonical group/dataset/train_seed layout.
        root = Path(checkpoint_dir).resolve().parents[2] if checkpoint_dir is not None else None
        descriptor = SimpleNamespace(name=method, config=config, checkpoints_root=root)
        hashes["training_resources_hash"] = config_hash(method_resources(descriptor))
    protocol = training_protocol(method or "")
    if protocol:
        hashes["training_protocol_hash"] = config_hash(protocol)
    return hashes


def write_metadata(
    ckpt_dir: str | Path,
    method: str,
    bundle,
    config: dict,
    train_seed: int,
    artifacts: Optional[dict[str, Any]] = None,
    extra: Optional[dict[str, Any]] = None,
    runtime_packages: tuple[str, ...] = (),
) -> dict:
    """Mark a checkpoint directory complete and record its reuse contract."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": METADATA_VERSION,
        "status": "complete",
        "method": method,
        "dataset": bundle.name,
        "train_seed": int(train_seed),
        "created_at": datetime.now(timezone.utc).isoformat(),
        **_hashes(bundle, config, method=method, checkpoint_dir=ckpt_dir),
        **(
            {"training_protocol": training_protocol(method)}
            if training_protocol(method) else {}
        ),
        "config": config,
        "artifacts": artifacts or {},
        **(
            {"serialization_runtime": runtime_fingerprint(runtime_packages)}
            if runtime_packages else {}
        ),
        **(extra or {}),
    }
    # Readers perform an optimistic check before taking the checkpoint lock.
    # Atomic replacement means they observe either the old complete marker or
    # the new complete marker, never half-written JSON.
    with atomic_output_path(ckpt_dir / METADATA_FILE) as tmp:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    return payload


def load_metadata(ckpt_dir: str | Path) -> Optional[dict]:
    path = Path(ckpt_dir) / METADATA_FILE
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        # An interrupted writer may have left a partial completion
        # marker. Treat it as invalid so the locked writer can rebuild it.
        return None


def is_checkpoint_valid(
    ckpt_dir: str | Path,
    method: str,
    bundle,
    config: dict,
    runtime_packages: tuple[str, ...] = (),
    require_runtime_fingerprint: bool = False,
) -> bool:
    """True iff a complete checkpoint with matching hashes and artifacts exists."""
    meta = load_metadata(ckpt_dir)
    if not meta or meta.get("status") != "complete":
        return False
    if int(meta.get("version", 0)) != METADATA_VERSION:
        return False
    if meta.get("method") != method:
        return False
    expected = _hashes(bundle, config, method=method, checkpoint_dir=ckpt_dir)
    for key, value in expected.items():
        if meta.get(key) != value:
            return False
    expected_protocol = training_protocol(method)
    if expected_protocol and meta.get("training_protocol") != expected_protocol:
        return False
    if runtime_packages:
        recorded_runtime = meta.get("serialization_runtime")
        if recorded_runtime is None:
            if require_runtime_fingerprint:
                return False
        elif recorded_runtime != runtime_fingerprint(runtime_packages):
            return False
    for rel in (meta.get("artifacts") or {}).values():
        if isinstance(rel, str) and not (Path(ckpt_dir) / rel).exists():
            return False
    return True

