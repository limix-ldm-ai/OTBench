# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Probe missing optional packages and configured resources without importing models.

A present but broken backend is deliberately not classified as absent: its import
or execution error must fail a test/run. Package version/API compatibility is
validated by backend tests, rather than hidden by this availability probe.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from .config import load_method_config
from .registry import METHODS, get_spec
from .resources import pretrained_resources, resolve_resource

_SYNTH = ("torch", "torchvision", "PIL", "pydantic", "optuna", "loguru", "cloudpickle", "xgboost")
_SYNTH_METHODS = frozenset({"arf", "bn", "bn_original_units", "ctgan", "tvae", "nflow", "nrgboost", "scm", "goggle", "great"})
_MODULES = {
    "arf": ("arfpy",), "bn": ("pgmpy",), "bn_original_units": ("pgmpy",), "nflow": ("nflows",),
    "nrgboost": ("nrgboost",), "scm": ("lingam", "causallearn"),
    "goggle": ("dgl", "torch_geometric", "torchdata"), "great": ("be_great",),
    "smote": ("imblearn",), "smotenc": ("imblearn",),
    "forest_vp": ("ForestDiffusion",), "forest_vp_joint": ("ForestDiffusion",),
    "hyperimpute": ("hyperimpute",), "tabebm": ("torch", "tabpfn"),
    "tabpfn": ("torch", "tabpfn"), "tabpfn_unsupervised": ("torch", "tabpfn_extensions"),
    "tabicl": ("torch", "tabicl"), "tabiclv2": ("torch", "tabicl"),
    "tabimpute": ("torch", "tabimpute"), "limix": ("torch", "einops"),
    "limix2": ("torch", "einops", "kditransform"),
    "tabddpm": ("torch", "category_encoders", "tomli", "tomli_w"),
    "tabsyn": ("torch", "category_encoders", "prdc"),
    "tabdiff": ("torch", "category_encoders", "tomli_w"),
    "tabcsdi": ("torch",), "diffputer": ("torch",),
    "ft_transformer": ("torch", "rtdl_revisiting_models"), "tabm": ("torch", "tabm"),
    "ngboost": ("ngboost",), "xgboost": ("xgboost",), "lightgbm": ("lightgbm",),
    "lightgbm_quantile": ("lightgbm",), "catboost": ("catboost",),
}


def optional_modules(name: str) -> tuple[str, ...]:
    get_spec(name)
    return tuple(dict.fromkeys(((_SYNTH if name in _SYNTH_METHODS else ()) + _MODULES.get(name, ()))))


def method_directory(name: str) -> Path:
    module = get_spec(name).import_path.split(":", 1)[0]
    return Path(__file__).resolve().parents[2].joinpath(*module.split(".")[1:-1])


def method_config(name: str) -> dict:
    return load_method_config(method_directory(name))


def missing_requirements(name: str, *, checkpoints_root=None, include_weights=True,
                         config: dict | None = None, resource_keys=None) -> list[str]:
    """Report excluded source or absent prerequisites; installed backend errors propagate."""
    spec = get_spec(name)
    if spec.capabilities.status == "external_only":
        upstream = spec.capabilities.upstream
        return [
            f"upstream source excluded from otbench distribution: {upstream.name}; "
            f"no redistribution license supplied in reviewed source; obtain from {upstream.url}; "
            f"see methods/{name}/model/README.md (external-only placeholder)"
        ]
    missing = [f"missing optional package: {module}" for module in optional_modules(name)
               if importlib.util.find_spec(module) is None]
    if include_weights:
        config = method_config(name) if config is None else config
        for resource in pretrained_resources(name):
            if resource_keys is not None and resource.key not in resource_keys:
                continue
            path = resolve_resource(name, config, resource, checkpoints_root, require_exists=False)
            exists = path.is_dir() and any(path.iterdir()) if resource.directory else path.is_file()
            if not exists:
                missing.append(f"missing pretrained {'directory' if resource.directory else 'file'}: {path}")
    return missing


def reference_markdown() -> str:
    rows = ["# Method reference", "", "Generated from the registry, optional-package readiness, and resource descriptors.",
            "Regenerate with `python -m otbench.methods.common.readiness --format markdown`.", "",
            "| Method | Tasks | Availability | Optional import modules | Pretrained resources |", "| --- | --- | --- | --- | --- |"]
    for name, spec in sorted(METHODS.items()):
        tasks = ", ".join(sorted(task.value for task in spec.tasks))
        modules = ", ".join(f"`{module}`" for module in optional_modules(name)) or "core"
        resources = ", ".join(f"`{resource.key}`" for resource in pretrained_resources(name)) or "none"
        rows.append(f"| `{name}` | {tasks} | {spec.capabilities.status} | {modules} | {resources} |")
    rows.extend(["", "`external_only` entries are unavailable placeholders: their upstream source is not distributed.",
                 "CTGAN requires `opacus` when differential privacy is enabled."])
    return "\n".join(rows) + "\n"


def notice_markdown() -> str:
    rows = ["## Registry and bundled-notice inventory", "",
            "Generated with `python -m otbench.methods.common.readiness --format notices`.", "",
            "| Method | Registry license label | Upstream | Bundled notice paths |", "| --- | --- | --- | --- |"]
    root = Path(__file__).resolve().parents[2]
    for name, spec in sorted(METHODS.items()):
        upstream = spec.capabilities.upstream
        notices = sorted({path for pattern in ("LICENSE*", "NOTICE*", "VENDOR*")
                          for path in method_directory(name).rglob(pattern) if path.is_file()})
        paths = ", ".join(f"`{path.relative_to(root)}`" for path in notices) or "none in method directory"
        source = f"[{upstream.name}]({upstream.url})" if upstream.url else upstream.name
        rows.append(f"| `{name}` | {upstream.license or 'pending'} | {source} | {paths} |")
    return "\n".join(rows) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", nargs="?", choices=sorted(METHODS))
    parser.add_argument("--checkpoints", default=None)
    parser.add_argument("--format", choices=("text", "markdown", "notices"), default="text")
    args = parser.parse_args(argv)
    if args.format in {"markdown", "notices"}:
        print(reference_markdown() if args.format == "markdown" else notice_markdown(), end="")
        return 0
    if args.method is None:
        parser.error("provide a method name or --format markdown")
    missing = missing_requirements(args.method, checkpoints_root=args.checkpoints)
    print("; ".join(missing) if missing else "READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
