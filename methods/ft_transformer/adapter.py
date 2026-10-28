# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""FT-Transformer (Gorishniy et al., NeurIPS 2021).

Runs the authors' ``rtdl_revisiting_models.FTTransformer``. The module's own
default hyperparameters are used via ``get_default_kwargs(n_blocks)``, which is
the configuration the paper reports as a strong default; the config file only
overrides ``n_blocks`` and the optimisation settings, so a run stays comparable
with the published numbers unless someone deliberately changes it.

The module emits a single ``(batch, d_out)`` tensor, so nothing needs collapsing
and the shared loop's default reduction applies.
"""
from __future__ import annotations

from otbench.methods.common.base import Task
from otbench.methods.common.deep_tabular import DeepTabularMethod
from otbench.methods.common.upstream import require_package

UPSTREAM_URL = "https://github.com/yandex-research/rtdl-revisiting-models"


class FTTransformerMethod(DeepTabularMethod):
    name = "ft_transformer"
    tasks = frozenset({Task.PREDICTION})

    def build_module(self, n_numerical, cardinalities, d_out):
        rtdl = require_package("rtdl_revisiting_models", method="ft_transformer",
                               url=UPSTREAM_URL, extra="prediction")
        kwargs = rtdl.FTTransformer.get_default_kwargs(
            n_blocks=int(self.cfg("model", "n_blocks", 3)))
        kwargs["d_out"] = d_out
        return rtdl.FTTransformer(
            n_cont_features=n_numerical,
            cat_cardinalities=list(cardinalities),
            **kwargs,
        )
