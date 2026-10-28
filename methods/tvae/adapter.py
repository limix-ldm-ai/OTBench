# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""TVAE adapter (tabular VAE, synth plugin route)."""
from __future__ import annotations

from otbench.methods.common.synth_method import SynthPluginMethod


class TVAEMethod(SynthPluginMethod):
    name = "tvae"
    conditional = True
    regression_cond = False  # TVAE does not support conditional generation for regression

    def plugin_class(self):
        from .plugin import TVAEPlugin

        return TVAEPlugin
