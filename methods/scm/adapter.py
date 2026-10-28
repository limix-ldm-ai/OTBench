# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""SCM adapter (structural causal model via lingam, synth plugin route)."""
from __future__ import annotations

from otbench.methods.common.synth_method import SynthPluginMethod


class SCMMethod(SynthPluginMethod):
    name = "scm"
    fit_task_type_kwarg = True
    misc_plugin_kwargs = {"strict": False}

    def plugin_class(self):
        from .plugin import SCMPlugin

        return SCMPlugin
