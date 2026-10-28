# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Workspace and pretrained resources shared by adapters, readiness and caches."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from functools import lru_cache

from otbench.paths import workspace_root


@dataclass(frozen=True)
class PretrainedResource:
    key: str
    filename: str
    env_vars: tuple[str, ...] = ()
    weight_group: str = ""
    directory: bool = False


_CLASSIFIER = "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt"
_TABICL_CLASSIFIER = "tabicl-classifier-v2-20260212.ckpt"
_TABICL_REGRESSOR = "tabicl-regressor-v2-20260212.ckpt"
_PRETRAINED = {
    "tabpfn": (
        PretrainedResource("classifier_ckpt", _CLASSIFIER, ("TABPFN_CLASSIFIER_MODEL_PATH", "TABPFN_MODEL_PATH")),
        PretrainedResource("regressor_ckpt", "tabpfn-v2-regressor.ckpt", ("TABPFN_REGRESSOR_MODEL_PATH",)),
    ),
    "tabpfn_unsupervised": (
        PretrainedResource("classifier_ckpt", "tabpfn-v3-classifier-v3_default.ckpt", ("TABPFN_UNSUPERVISED_CLASSIFIER_MODEL_PATH", "TABPFN_V3_CLASSIFIER_MODEL_PATH")),
        PretrainedResource("regressor_ckpt", "tabpfn-v3-regressor-v3_default.ckpt", ("TABPFN_UNSUPERVISED_REGRESSOR_MODEL_PATH", "TABPFN_V3_REGRESSOR_MODEL_PATH")),
    ),
    "tabicl": (PretrainedResource("classifier_ckpt", _TABICL_CLASSIFIER, ("TABICL_CLASSIFIER_MODEL_PATH",)),),
    "tabiclv2": (
        PretrainedResource("classifier_ckpt", _TABICL_CLASSIFIER, ("TABICLV2_CLASSIFIER_MODEL_PATH", "TABICL_CLASSIFIER_MODEL_PATH"), "tabicl"),
        PretrainedResource("regressor_ckpt", _TABICL_REGRESSOR, ("TABICLV2_REGRESSOR_MODEL_PATH", "TABICL_REGRESSOR_MODEL_PATH"), "tabicl"),
    ),
    "limix": (PretrainedResource("model_ckpt", "LimiX-16M.ckpt", ("LIMIX_MODEL_PATH",)),),
    "limix2": (PretrainedResource("model_ckpt", "LimiX-2.ckpt", ("LIMIX2_MODEL_PATH", "LIMIX_MODEL_PATH")),),
    "tabimpute": (PretrainedResource("model_ckpt", "tabimputev2.ckpt", ("TABIMPUTE_MODEL_PATH",)),),
    "tabebm": (PretrainedResource("classifier_ckpt", _CLASSIFIER, ("TABPFN_MODEL_PATH",), "tabpfn"),),
    "great": (PretrainedResource("llm", "distilgpt2", directory=True),),
}


def pretrained_resources(name: str) -> tuple[PretrainedResource, ...]:
    return _PRETRAINED.get(name, ())


def resolve_resource(name: str, config: dict, resource: PretrainedResource,
                     checkpoints_root: str | Path | None = None, *,
                     require_exists: bool = True) -> Path:
    """Config absolute path > environment > checkpoint-root/group/pretrained/name."""
    configured = config.get("model", {}).get(resource.key)
    candidate = Path(str(configured)).expanduser() if configured else None
    if candidate is None or not candidate.is_absolute():
        external = next((os.environ[v] for v in resource.env_vars if os.environ.get(v)), None)
        if external:
            candidate = Path(external).expanduser()
        else:
            root = Path(checkpoints_root).expanduser() if checkpoints_root is not None else workspace_root() / "checkpoints"
            candidate = root / (resource.weight_group or name) / "pretrained" / str(configured or resource.filename)
    candidate = candidate.resolve()
    exists = candidate.is_dir() if resource.directory else candidate.is_file()
    if require_exists and not exists:
        kind = "directory" if resource.directory else "file"
        raise FileNotFoundError(
            f"{name} pretrained {kind} not found: {candidate}. Set model.{resource.key} "
            f"or place it under the configured checkpoints root."
        )
    return candidate


@lru_cache(maxsize=8192)
def _content_hash(path: str, size: int, mtime_ns: int, ctime_ns: int) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: str | Path) -> dict:
    """Content identity cached only while the file's stat identity is unchanged."""
    path = Path(path).resolve()
    if not path.is_file():
        return {"path": str(path), "missing": True}
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size,
            "sha256": _content_hash(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)}


def resource_identity(path: str | Path) -> dict:
    path = Path(path).resolve()
    if path.is_dir():
        return {"path": str(path), "files": [file_identity(p) for p in sorted(path.rglob("*")) if p.is_file()]}
    return file_identity(path)


def method_resources(method_obj) -> list[dict]:
    """Describe exactly the configured resource locations, including missing ones.

    All declared heads are included so an imputation schema can use either head;
    an unused missing head does not prevent a classification-only run.
    """
    resources = [
        {"key": resource.key, **resource_identity(resolve_resource(
            method_obj.name, method_obj.config, resource,
            getattr(method_obj, "checkpoints_root", None), require_exists=False,
        ))}
        for resource in pretrained_resources(method_obj.name)
    ]
    resources.extend(getattr(method_obj, "additional_resource_identities", lambda: [])())
    return resources


def source_tree_identity(root: str | Path) -> list[dict]:
    """Hash Python and static inference config files, excluding runtime artifacts."""
    root = Path(root).resolve()
    if not root.is_dir():
        return [{"path": str(root), "missing": True}]
    suffixes = {".py", ".json", ".yaml", ".yml"}
    excluded = {"__pycache__", ".git", "outputs", "checkpoints", "store", "build", "dist"}
    return [file_identity(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix in suffixes
            and not excluded.intersection(p.relative_to(root).parts)]


def methods_source_hash() -> str:
    """Conservative implementation identity including aliases and shared bases.

    The small method source tree is intentionally hashed together. An unrelated
    method edit can invalidate a checkpoint; missing an imported base cannot.
    """
    import json
    root = Path(__file__).resolve().parents[1]
    identities = source_tree_identity(root)
    relative = [{**item, "path": str(Path(item["path"]).relative_to(root))} for item in identities]
    payload = json.dumps(relative, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def runtime_identity() -> dict:
    """Conservative installed-distribution identity for optional model backends."""
    import sys
    from importlib import metadata
    packages = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name.lower().replace("_", "-")] = distribution.version
    return {"python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "packages": dict(sorted(packages.items()))}
