# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""CTGAN adapter (conditional GAN, synth plugin route)."""
from __future__ import annotations

from otbench.methods.common.synth_method import SynthPluginMethod


class CTGANMethod(SynthPluginMethod):
    name = "ctgan"
    conditional = True
    regression_cond = False  # CTGAN does not support conditional generation for regression

    def plugin_class(self):
        from .plugin import CTGANPlugin

        return CTGANPlugin
