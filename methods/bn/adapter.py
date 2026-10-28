# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Bayesian network adapter (pgmpy, synth plugin route).

Original corner cases preserved: all numerical columns are discretized into
<=100 bins and the fit frame is capped at 10k rows (BN does not scale beyond).
Note that, as in the original engine, generated numerical columns stay in the
binned space (bin indices).
"""
from __future__ import annotations

from otbench.methods.common.synth_method import SynthPluginMethod


class BNMethod(SynthPluginMethod):
    name = "bn"
    row_cap = 10000
    discretize_numeric_bins = 100
    misc_plugin_kwargs = {"strict": False}

    def plugin_class(self):
        from .plugin import BayesianNetworkPlugin

        return BayesianNetworkPlugin
