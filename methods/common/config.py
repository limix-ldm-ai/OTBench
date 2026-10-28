"""Per-method TOML config loading, override merging, and hashing.

Every method ships a ``config.toml`` next to its adapter with the unified
sections ``[model]`` / ``[train]`` / ``[sample]`` / ``[impute]`` (values sourced
from the original repositories). The adapter maps these unified sections back
to the original implementation's parameter names.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    import tomllib  # py311+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def load_toml(path: str | Path) -> dict:
    with Path(path).open("rb") as f:
        return tomllib.load(f)


def load_method_config(
    method_dir: str | Path,
    config_file: str | Path | None = None,
    overrides: dict | None = None,
) -> dict:
    """Merge a required shipped default, an optional file, and validated overrides."""
    method_dir = Path(method_dir)
    default_path = method_dir / "config.toml"
    if not default_path.is_file():
        raise FileNotFoundError(f"Method default config not found: {default_path}")
    config = load_toml(default_path)
    known = _known_keys(method_dir, config)
    if config_file is not None:
        path = Path(config_file).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Requested method config not found: {path}")
        supplied = load_toml(path)
        _validate_values(supplied, config, known)
        config = deep_update(config, supplied)
    if overrides:
        _validate_values(overrides, config, known)
        config = deep_update(config, overrides)
    return config


def _known_keys(method_dir: Path, defaults: dict) -> set[tuple[str, ...]]:
    known: set[tuple[str, ...]] = set()
    def collect(values, prefix=()):
        for key, value in values.items():
            path = prefix + (key,)
            known.add(path)
            if isinstance(value, dict):
                collect(value, path)
    collect(defaults)
    from .resources import pretrained_resources
    for resource in pretrained_resources(method_dir.name):
        known.update({("model",), ("model", resource.key)})
    # Optional adapter parameters need not occur in the shipped TOML. Recognize
    # literal cfg(section, key) calls, including inherited common hooks.
    sources = [method_dir / "adapter.py"]
    common = Path(__file__).parent
    sources.extend(common.glob("*.py"))
    for source in sources:
        if not source.is_file():
            continue
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "cfg" and len(node.args) >= 2:
                if all(isinstance(arg, ast.Constant) and isinstance(arg.value, str) for arg in node.args[:2]):
                    section, key = (arg.value for arg in node.args[:2])
                    known.update({(section,), (section, key)})
    return known


def _validate_values(values: dict, defaults: dict, known: set, prefix=()) -> None:
    for key, value in values.items():
        path = prefix + (key,)
        # Backend keyword dictionaries are an explicitly open extension point.
        if any(part in {"estimator_kwargs", "predictor_kwargs"} for part in path):
            continue
        if path not in known:
            raise ValueError(f"Unknown method config key: {'.'.join(path)}")
        expected = defaults.get(key)
        if isinstance(value, dict):
            if expected is not None and not isinstance(expected, dict):
                raise ValueError(f"Config {'.'.join(path)} must be {type(expected).__name__}")
            _validate_values(value, expected or {}, known, path)
        elif expected is not None:
            compatible = isinstance(value, type(expected))
            if isinstance(expected, float):
                compatible = isinstance(value, (int, float)) and not isinstance(value, bool)
            elif isinstance(expected, int) and not isinstance(expected, bool):
                compatible = isinstance(value, int) and not isinstance(value, bool)
            if not compatible:
                raise ValueError(f"Config {'.'.join(path)} must be {type(expected).__name__}, got {type(value).__name__}")


def deep_update(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def parse_overrides(pairs: list[str]) -> dict:
    """Parse CLI ``--override section.key=value`` pairs into a nested dict.

    Values are parsed as TOML literals (int/float/bool/string); bare words fall
    back to strings.
    """
    result: dict = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"Override {pair!r} must look like section.key=value")
        dotted, raw = pair.split("=", 1)
        value = _parse_literal(raw)
        node = result
        keys = dotted.strip().split(".")
        if not all(key and key.strip() == key for key in keys):
            raise ValueError(f"Invalid override key: {dotted!r}")
        for key in keys[:-1]:
            if key in node and not isinstance(node[key], dict):
                raise ValueError(f"Override key collides with scalar: {dotted!r}")
            node = node.setdefault(key, {})
        node[keys[-1]] = value
    return result


def _parse_literal(raw: str) -> Any:
    raw = raw.strip()
    try:
        return tomllib.loads(f"v = {raw}")["v"]
    except Exception:  # noqa: BLE001 — bare word -> string
        return raw


def config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
