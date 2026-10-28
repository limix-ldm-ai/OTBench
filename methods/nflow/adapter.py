# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""NFlow adapter (normalizing flows, synth plugin route).

Original corner case preserved: NFlow is unstable beyond ~2k rows, so the fit
frame is capped at 2000 rows.
"""
from __future__ import annotations

from otbench.methods.common.synth_method import SynthPluginMethod


class NFlowMethod(SynthPluginMethod):
    name = "nflow"
    row_cap = 2000

    def plugin_class(self):
        from .plugin import NormalizingFlowsPlugin

        return NormalizingFlowsPlugin
