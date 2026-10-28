# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""TabICL adapter: pretrained tabular foundation model (prediction + imputation)."""
from __future__ import annotations

from otbench.methods.common.base import Task
from otbench.methods.common.foundation import FoundationMethod



class TabICLMethod(FoundationMethod):
    name = "tabicl"
    tasks = frozenset({Task.PREDICTION})

    def make_classifier(self, ctx):
        from tabicl import TabICLClassifier

        path = self.pretrained_path("classifier_ckpt")
        return TabICLClassifier(model_path=path, allow_auto_download=False, **self.estimator_kwargs(ctx))
