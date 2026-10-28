# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""GOGGLE adapter (graph-based generator, synth plugin route).

Needs the GNN stack (dgl + torch-geometric); the device stays "cpu" as in the
original wrapper (DGL PyPI wheels in this environment are CPU-only).
"""
from __future__ import annotations

from otbench.methods.common.synth_method import SynthPluginMethod
from otbench.methods.common.upstream import require_package
from otbench.methods.goggle.dgl_compat import allow_dgl_without_graphbolt

UPSTREAM_URL = "https://github.com/tennisonliu/GOGGLE"


class GOGGLEMethod(SynthPluginMethod):
    name = "goggle"
    misc_plugin_kwargs = {"device": "cpu", "sampling_patience": 1, "strict": False}

    def plugin_class(self):
        allow_dgl_without_graphbolt()
        # The plugin module swallows this ImportError and leaves its model class
        # undefined, so without the check the GNN stack going missing surfaces as
        # a NameError inside _fit instead of as a missing dependency.
        for module in ("dgl", "torch_geometric"):
            require_package(module, method="goggle", url=UPSTREAM_URL, extra="generation")
        from .plugin import GOGGLEPlugin

        return GOGGLEPlugin
