# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Locating third-party method implementations.

Third-party implementations are loaded lazily from optional installed packages
or included under ``methods/<name>/model/`` with their upstream notices. CARD
and ReMasker are external-only placeholders: their reviewed source supplied no
redistribution license, so no implementation or external-code loader is shipped.

Missing-implementation errors identify the upstream, its source, and any
validated revision. A bare ``ModuleNotFoundError`` from six frames
deep inside an adapter tells whoever hits it none of that, and a silent
fallback to some approximation would quietly change what the benchmark measures.
"""
from __future__ import annotations

import importlib
from types import ModuleType


class UpstreamUnavailable(ImportError):
    """An official implementation this method requires is not installed."""


def require_package(module: str, *, method: str, url: str, version: str = "",
                    extra: str = "") -> ModuleType:
    """Import an upstream package, or explain precisely what is missing."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        pin = f"=={version}" if version else ""
        install = (f"pip install -r requirements/{extra}.txt" if extra
                   else f"pip install {module}{pin}")
        raise UpstreamUnavailable(
            f"{method} runs the official {module} implementation, which is not importable "
            f"in this environment ({exc}).\n"
            f"  upstream: {url}\n"
            f"  validated version: {version or 'see the registry entry'}\n"
            f"  install: {install}"
        ) from exc


def require_torch(method: str) -> ModuleType:
    return require_package("torch", method=method, url="https://pytorch.org",
                           extra="diffusion")
