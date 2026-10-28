# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""TabPFN adapter: pretrained tabular foundation model (prediction + imputation)."""
from __future__ import annotations

from otbench.methods.common.foundation import FoundationMethod

class TabPFNMethod(FoundationMethod):
    name = "tabpfn"
    max_imputation_classes = 10
    max_prediction_classes = 10

    def make_regressor(self, ctx):
        from tabpfn import TabPFNRegressor

        path = self.pretrained_path("regressor_ckpt")
        return TabPFNRegressor(model_path=path, **self.estimator_kwargs(ctx))

    def make_classifier(self, ctx):
        from tabpfn import TabPFNClassifier

        path = self.pretrained_path("classifier_ckpt")
        return TabPFNClassifier(model_path=path, **self.estimator_kwargs(ctx))
