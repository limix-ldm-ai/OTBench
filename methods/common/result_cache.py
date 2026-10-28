# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Provenance-complete result-cache contracts for all three task runners.

A non-empty metrics JSON is not evidence that it belongs to the requested
run.  A reusable result records the effective data/schema/config/checkpoint,
task arguments, code/runtime identity, and hashes of every output artifact.
Reports without that proof cannot be reused.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import checkpoints as ckpt_utils
from . import data as data_utils
from .config import config_hash
from .locking import atomic_output_path
from .resources import methods_source_hash, method_resources, file_identity, runtime_identity

RESULT_CACHE_VERSION = 2
_RUNTIME_PACKAGES = ("numpy", "pandas", "scikit-learn", "scipy")


@dataclass(frozen=True)
class CacheValidation:
    valid: bool
    reason: str
    payload: dict[str, Any] | None = None


def build_run_fingerprint(
    task: str,
    method_obj,
    bundle,
    checkpoint_dir: str | Path,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Build the complete, deterministic identity of one requested run."""
    return {
        "result_cache_version": RESULT_CACHE_VERSION,
        "task": task,
        "method": method_obj.name,
        "dataset": bundle.name,
        "dataset_type": bundle.dataset_type,
        "train_data_hash": data_utils.dataframe_hash(bundle.train),
        "test_data_hash": data_utils.dataframe_hash(bundle.test),
        "schema_hash": data_utils.schema_hash(bundle.schema),
        "effective_config_hash": config_hash(method_obj.config),
        "checkpoint": _checkpoint_identity(method_obj, checkpoint_dir),
        "parameters": _jsonable(parameters),
        "code_hash": source_code_hash(task, method_obj.name),
        "runtime": _runtime_identity(),
    }


def fingerprint_hash(fingerprint: dict[str, Any]) -> str:
    return _sha256_bytes(_canonical_bytes(fingerprint))


def fingerprint_tag(fingerprint: dict[str, Any]) -> str:
    """Short content-addressed namespace used in result paths."""
    return fingerprint_hash(fingerprint)[:16]


def attach_result_cache(
    payload: dict[str, Any],
    fingerprint: dict[str, Any],
    artifact_paths: Iterable[str | Path],
) -> dict[str, Any]:
    """Return ``payload`` with a successful cache completion marker."""
    result = dict(payload)
    result["_result_cache"] = {
        "version": RESULT_CACHE_VERSION,
        "status": "succeeded",
        "fingerprint": fingerprint,
        "fingerprint_hash": fingerprint_hash(fingerprint),
        "artifacts": [_artifact_record(path) for path in artifact_paths],
    }
    return result


def validate_result_cache(
    metrics_path: str | Path,
    expected_fingerprint: dict[str, Any],
    *,
    expected_artifacts: Iterable[str | Path] | None = None,
) -> CacheValidation:
    """Validate a metrics/report file and every artifact it commits to."""
    path = Path(metrics_path)
    if not path.is_file() or path.stat().st_size <= 0:
        return CacheValidation(False, "metrics file is missing or empty")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return CacheValidation(False, f"metrics JSON is unreadable: {type(exc).__name__}")
    if not isinstance(payload, dict):
        return CacheValidation(False, "metrics JSON is not an object")

    cache = payload.get("_result_cache")
    statuses = [payload.get("provenance_status")]
    if isinstance(cache, dict):
        statuses.append(cache.get("provenance_status"))
    if any(status not in (None, "verified") for status in statuses):
        return CacheValidation(False, "result provenance is unverified")
    if cache is None:
        return CacheValidation(False, "result has no provenance fingerprint")
    if not isinstance(cache, dict):
        return CacheValidation(False, "result cache marker is malformed")
    if cache.get("version") != RESULT_CACHE_VERSION:
        return CacheValidation(False, "result cache version differs")
    if cache.get("status") != "succeeded":
        return CacheValidation(False, "result cache status is not succeeded")

    recorded = cache.get("fingerprint")
    if not isinstance(recorded, dict):
        return CacheValidation(False, "recorded fingerprint is missing")
    recorded_hash = cache.get("fingerprint_hash")
    if recorded_hash != fingerprint_hash(recorded):
        return CacheValidation(False, "recorded fingerprint hash is corrupt")
    if recorded_hash != fingerprint_hash(expected_fingerprint):
        return CacheValidation(False, "requested run fingerprint differs")
    if _canonical_bytes(recorded) != _canonical_bytes(expected_fingerprint):
        return CacheValidation(False, "requested run fingerprint payload differs")

    artifacts = cache.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return CacheValidation(False, "artifact manifest is missing")
    recorded_paths: list[str] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            return CacheValidation(False, "artifact manifest entry is malformed")
        artifact_path = Path(str(artifact.get("path", "")))
        if not artifact_path.is_file():
            return CacheValidation(False, f"artifact is missing: {artifact_path}")
        try:
            size = artifact_path.stat().st_size
        except OSError:
            return CacheValidation(False, f"artifact cannot be stat'ed: {artifact_path}")
        if size != artifact.get("size"):
            return CacheValidation(False, f"artifact size differs: {artifact_path}")
        if size <= 0:
            return CacheValidation(False, f"artifact is empty: {artifact_path}")
        if _sha256_file(artifact_path) != artifact.get("sha256"):
            return CacheValidation(False, f"artifact hash differs: {artifact_path}")
        recorded_paths.append(str(artifact_path.resolve()))

    if expected_artifacts is not None:
        expected_paths = sorted(str(Path(p).resolve()) for p in expected_artifacts)
        if sorted(recorded_paths) != expected_paths:
            return CacheValidation(False, "artifact path set differs")
    return CacheValidation(True, "validated provenance and artifacts", payload)


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    with atomic_output_path(path) as tmp:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)


def write_csv_atomic(frame, path: str | Path, schema=None) -> None:
    with atomic_output_path(path) as tmp:
        if schema is None:
            frame.to_csv(tmp, index=False)
        else:
            from otbench.tabdata.io import write_frame_csv
            write_frame_csv(frame, tmp, schema)


def source_code_hash(task: str, method: str) -> str:
    """Hash runner/evaluator/common/adapter Python sources used by a run."""
    package_root = Path(__file__).resolve().parents[2]
    roots = [
        package_root / task,
        package_root / "metrics",
        package_root / "tabdata",
    ]
    files: set[Path] = {package_root / "run.py", package_root / "__init__.py", package_root / "paths.py"}
    for root in roots:
        if root.is_dir():
            files.update(root.rglob("*.py"))
    digest = hashlib.sha256()
    for path in sorted((p for p in files if p.is_file()), key=lambda p: str(p)):
        digest.update(str(path.relative_to(package_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    digest.update(methods_source_hash().encode("ascii"))
    return digest.hexdigest()


def _checkpoint_identity(method_obj, checkpoint_dir: str | Path) -> dict[str, Any]:
    ckpt_dir = Path(checkpoint_dir)
    if method_obj.trainable:
        metadata = ckpt_utils.load_metadata(ckpt_dir)
        if not metadata:
            raise RuntimeError(f"cannot fingerprint missing checkpoint metadata: {ckpt_dir}")
        stable_metadata = {k: v for k, v in metadata.items() if k != "created_at"}
        return {
            "kind": "trained",
            "metadata_hash": _sha256_bytes(_canonical_bytes(stable_metadata)),
            "metadata": _jsonable(stable_metadata),
            "resources": method_resources(method_obj),
            "artifact_manifest": _checkpoint_artifact_manifest(
                ckpt_dir, metadata.get("artifacts") or {},
            ),
        }

    return {"kind": "training_free", "resources": method_resources(method_obj)}


def _checkpoint_artifact_manifest(
    checkpoint_dir: Path, artifacts: dict[str, Any],
) -> list[dict[str, Any]]:
    """Identify checkpoint contents; unchanged files share the process hash cache."""
    paths: set[Path] = set()

    def collect(value: Any) -> None:
        if isinstance(value, str):
            candidate = checkpoint_dir / value
            if candidate.is_file():
                paths.add(candidate)
            elif candidate.is_dir():
                paths.update(path for path in candidate.rglob("*") if path.is_file())
        elif isinstance(value, dict):
            for nested in value.values():
                collect(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                collect(nested)

    collect(artifacts)
    manifest = []
    for path in sorted(paths, key=lambda item: str(item)):
        try:
            label = str(path.relative_to(checkpoint_dir))
        except ValueError:
            label = str(path.resolve())
        manifest.append({**file_identity(path), "path": label})
    return manifest


def _artifact_record(path: str | Path) -> dict[str, Any]:
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"cannot commit missing result artifact: {artifact}")
    size = artifact.stat().st_size
    if size <= 0:
        raise ValueError(f"cannot commit empty result artifact: {artifact}")
    return {
        "path": str(artifact.resolve()),
        "size": size,
        "sha256": _sha256_file(artifact),
    }


def _runtime_identity() -> dict[str, Any]:
    return runtime_identity()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(item) for item in value), key=lambda item: repr(item))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
