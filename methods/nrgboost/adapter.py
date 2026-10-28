# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""NRGBoost adapter (energy-based gradient boosting, synth plugin route).

The upstream implementation is CPU/OpenMP-only. Its raw CFFI tree object is not
picklable, but the project exposes a stable serialize/deserialize representation;
the model wrapper uses that representation so training is checkpointed normally.
"""
from __future__ import annotations

from otbench.methods.common.synth_method import SynthPluginMethod


class NRGBoostMethod(SynthPluginMethod):
    name = "nrgboost"
    trainable = True
    misc_plugin_kwargs = {"strict": False}
    checkpoint_runtime_packages = SynthPluginMethod.checkpoint_runtime_packages + (
        "nrgboost", "joblib", "scikit-learn",
    )
    require_checkpoint_runtime_fingerprint = True

    def plugin_kwargs(self, ctx) -> dict:
        kwargs = super().plugin_kwargs(ctx)
        # The generic synth plugin accepts a device argument, but official
        # NRGBoost 0.0.3 has no CUDA backend. Make the actual backend explicit
        # instead of letting an auto-selected CUDA string imply GPU execution.
        kwargs["device"] = "cpu"
        return kwargs

    def checkpoint_profile(self, plugin) -> dict:
        model = getattr(plugin, "model", None)
        return {
            "execution_backend": "cpu_openmp",
            "num_threads": int(getattr(model, "num_threads", 1)),
        }

    def plugin_class(self):
        from .plugin import NRGBoostPlugin

        return NRGBoostPlugin
