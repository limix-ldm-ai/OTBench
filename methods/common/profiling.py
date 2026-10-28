"""Unified compute profiling for the fit / inference / eval phases.

Every task runner wraps its phases in :meth:`PhaseProfiler.phase`, so the
``COST`` metric family (wall clock, throughput, peak memory, model size) is
produced the same way for every method and never has to be reconstructed from
log scraping.

Failures are recorded rather than swallowed: a phase that raises still produces
a record with ``status="failed"`` and the exception type, which keeps "the
method does not support this" distinguishable from "the method crashed" and
from "the metric was never attempted".
"""
from __future__ import annotations

import gc
import json
import platform
import resource
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path

PHASES = ("fit", "inference", "eval")


@dataclass
class PhaseRecord:
    name: str
    wall_clock_sec: float = 0.0
    status: str = "ok"                 # "ok" | "failed"
    error: Optional[str] = None
    rows: Optional[int] = None
    cells: Optional[int] = None
    sec_per_1k_rows: Optional[float] = None
    sec_per_1k_cells: Optional[float] = None
    peak_ram_mb: Optional[float] = None
    peak_vram_mb: Optional[float] = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "wall_clock_sec": self.wall_clock_sec,
            "status": self.status,
            "error": self.error,
            "rows": self.rows,
            "cells": self.cells,
            "sec_per_1k_rows": self.sec_per_1k_rows,
            "sec_per_1k_cells": self.sec_per_1k_cells,
            "peak_ram_mb": self.peak_ram_mb,
            "peak_vram_mb": self.peak_vram_mb,
            **({"extra": self.extra} if self.extra else {}),
        }


class PhaseProfiler:
    """Collects one :class:`PhaseRecord` per timed phase of a run."""

    def __init__(self, *, track_ram: bool = True, device: str | None = None) -> None:
        self.records: list[PhaseRecord] = []
        self.model: dict[str, Any] = {}
        self._track_ram = track_ram
        self._device = device

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None, *, device: str | None = None):
        """Restore completed phases saved before a later phase failed.

        Generation commits its fit/inference profile before evaluation starts.
        A metrics-only resume can therefore preserve the original compute cost
        instead of reporting the retry as if sampling had taken zero seconds.
        """
        payload = dict(payload or {})
        profiler = cls(device=device if device is not None else payload.get("device"))
        profiler.model.update(payload.get("model") or {})
        fields = {
            "name", "wall_clock_sec", "status", "error", "rows", "cells",
            "sec_per_1k_rows", "sec_per_1k_cells", "peak_ram_mb", "peak_vram_mb",
            "extra",
        }
        for raw in payload.get("phases") or []:
            if not isinstance(raw, dict) or not raw.get("name"):
                continue
            values = {key: raw[key] for key in fields if key in raw}
            profiler.records.append(PhaseRecord(**values))
        return profiler

    @contextmanager
    def phase(
        self,
        name: str,
        *,
        rows: int | None = None,
        cells: int | None = None,
        **extra: Any,
    ):
        record = PhaseRecord(name=name, rows=rows, cells=cells, extra=dict(extra))
        ram_started = self._start_ram()
        self._reset_vram()
        start = time.perf_counter()
        try:
            yield record
        except BaseException as exc:  # noqa: BLE001 — recorded, then re-raised
            record.status = "failed"
            record.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record.wall_clock_sec = time.perf_counter() - start
            record.peak_ram_mb = self._stop_ram(ram_started)
            record.peak_vram_mb = self._peak_vram()
            if record.rows:
                record.sec_per_1k_rows = record.wall_clock_sec * 1000.0 / record.rows
            if record.cells:
                record.sec_per_1k_cells = record.wall_clock_sec * 1000.0 / record.cells
            self.records.append(record)

    def record_model(self, **fields: Any) -> None:
        """Attach model-size facts: ``n_parameters``, ``n_auxiliary_models``, ..."""
        self.model.update(fields)

    def total_sec(self, name: str) -> float:
        return sum(r.wall_clock_sec for r in self.records if r.name == name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phases": [r.to_dict() for r in self.records],
            "totals_sec": {name: self.total_sec(name) for name in PHASES if self.total_sec(name)},
            "model": dict(self.model),
            "device": self._device,
        }

    # --- memory backends --------------------------------------------------
    def _start_ram(self) -> bool:
        if not self._track_ram or tracemalloc.is_tracing():
            return False
        gc.collect()
        tracemalloc.start()
        return True

    def _stop_ram(self, started: bool) -> float | None:
        python_peak = None
        if started:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            python_peak = peak / (1024.0 * 1024.0)
        # tracemalloc sees only Python allocations.  ru_maxrss also includes
        # NumPy/sklearn/native buffers and is therefore the authoritative process
        # high-water mark for the run (Linux reports KiB, macOS bytes).
        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        rss_mb = rss / (1024.0 if platform.system() != "Darwin" else 1024.0 ** 2)
        return max(v for v in (python_peak, rss_mb) if v is not None)

    def _reset_vram(self) -> None:
        torch = _torch()
        cuda = getattr(torch, "cuda", None) if torch is not None else None
        if cuda is not None and callable(getattr(cuda, "is_available", None)) and cuda.is_available():
            cuda.reset_peak_memory_stats(_cuda_device(torch, self._device))

    def _peak_vram(self) -> float | None:
        torch = _torch()
        cuda = getattr(torch, "cuda", None) if torch is not None else None
        if cuda is None or not callable(getattr(cuda, "is_available", None)) or not cuda.is_available():
            return None
        return cuda.max_memory_allocated(_cuda_device(torch, self._device)) / (1024.0 * 1024.0)


def checkpoint_model_profile(method_obj: Any, checkpoint_dir: str | Path) -> dict[str, int]:
    """Collect model-size facts recorded by an adapter's checkpoint contract.

    Adapters already write small JSON sidecars.  Reading those declarations here
    keeps runners generic and avoids importing or deserialising the model merely
    to count it.  A logical single-model method always reports one auxiliary
    model; multi-model methods must record their actual count.
    """
    profile: dict[str, int] = {}
    capabilities = getattr(method_obj, "capabilities", None)
    if capabilities is not None and capabilities.auxiliary_models == "single":
        profile["n_auxiliary_models"] = 1

    root = Path(checkpoint_dir)
    metadata_path = root / "metadata.json"
    payloads: list[dict[str, Any]] = []
    if metadata_path.exists():
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            payloads.append(metadata)
            for relative in (metadata.get("artifacts") or {}).values():
                path = root / relative if isinstance(relative, str) else None
                if path is not None and path.suffix == ".json" and path.exists():
                    with path.open("r", encoding="utf-8") as handle:
                        payloads.append(json.load(handle))
        except (OSError, ValueError, TypeError):
            payloads = []

    for payload in payloads:
        for key in ("n_parameters", "n_auxiliary_models"):
            value = payload.get(key)
            if value is not None:
                try:
                    profile[key] = int(value)
                except (TypeError, ValueError):
                    pass
    return profile


def count_parameters(model: Any) -> int | None:
    """Number of trainable parameters of a torch module, if it is one."""
    params = getattr(model, "parameters", None)
    if not callable(params):
        return None
    try:
        return int(sum(p.numel() for p in params() if getattr(p, "requires_grad", False)))
    except Exception:  # noqa: BLE001 — non-torch objects with a `parameters` attr
        return None


def _torch():
    try:
        import torch

        return torch
    except Exception:  # noqa: BLE001 — torch-free environments
        return None


def _cuda_device(torch, device: str | None):
    """Resolve the configured CUDA device without assuming a complete torch install."""
    if not device or not str(device).startswith("cuda"):
        return None
    constructor = getattr(torch, "device", None)
    return constructor(device) if callable(constructor) else None
