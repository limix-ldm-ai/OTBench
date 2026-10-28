# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
# stdlib
from pathlib import Path
from typing import Any, List, Optional

# third party
import pandas as pd
from pydantic import validate_call

from otbench.methods.common.synth.metrics.weighted_metrics import WeightedMetrics
from otbench.methods.common.synth.plugins.core.dataloader import DataLoader
from otbench.methods.common.synth.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    FloatDistribution,
    IntegerDistribution,
)
from .model.flows import NormalizingFlows
from .model.tabular_flows import TabularFlows
from otbench.methods.common.synth.plugins.core.plugin import Plugin
from otbench.methods.common.synth.plugins.core.schema import Schema
from otbench.methods.common.synth.utils.constants import DEVICE


class NormalizingFlowsPlugin(Plugin):
    """NormalizingFlowsPlugin: fit and sample encoded tabular data."""

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_iter: int = 1000,
        n_layers_hidden: int = 1,
        n_units_hidden: int = 100,
        batch_size: int = 200,
        num_transform_blocks: int = 1,
        dropout: float = 0.1,
        batch_norm: bool = False,
        num_bins: int = 8,
        tail_bound: float = 3,
        lr: float = 1e-3,
        apply_unconditional_transform: bool = True,
        base_distribution: str = "standard_normal",  # "standard_normal"
        linear_transform_type: str = "permutation",  # "lu", "permutation", "svd"
        base_transform_type: str = "rq-autoregressive",  # "affine-coupling", "quadratic-coupling", "rq-coupling", "affine-autoregressive", "quadratic-autoregressive", "rq-autoregressive"
        encoder_max_clusters: int = 10,
        tabular: bool = True,
        # early stopping
        n_iter_min: int = 100,
        n_iter_print: int = 50,
        patience: int = 5,
        patience_metric: Optional[WeightedMetrics] = None,
        # core plugin arguments
        workspace: Path = Path("logs/otbench_workspace"),
        compress_dataset: bool = False,
        sampling_patience: int = 500,
        random_state: int = 0,
        device: Any = DEVICE,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            device=device,
            random_state=random_state,
            sampling_patience=sampling_patience,
            workspace=workspace,
            compress_dataset=compress_dataset,
            **kwargs,
        )

        if patience_metric is None:
            patience_metric = WeightedMetrics(
                metrics=[("detection", "detection_mlp")],
                weights=[1],
                workspace=workspace,
            )

        self.n_iter = n_iter
        self.n_layers_hidden = n_layers_hidden
        self.n_units_hidden = n_units_hidden
        self.batch_size = batch_size
        self.num_transform_blocks = num_transform_blocks
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.apply_unconditional_transform = apply_unconditional_transform
        self.lr = lr

        self.base_distribution = base_distribution
        self.linear_transform_type = linear_transform_type
        self.base_transform_type = base_transform_type

        self.encoder_max_clusters = encoder_max_clusters
        self.tabular = tabular

        self.n_iter_min = n_iter_min
        self.n_iter_print = n_iter_print
        self.patience = patience
        self.patience_metric = patience_metric

    @staticmethod
    def name() -> str:
        return "nflow"

    @staticmethod
    def type() -> str:
        return "generic"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            IntegerDistribution(name="n_iter", low=100, high=5000, step=100),
            IntegerDistribution(name="n_layers_hidden", low=1, high=10),
            IntegerDistribution(name="n_units_hidden", low=10, high=100),
            CategoricalDistribution(name="batch_size", choices=[32, 64, 128, 256, 512]),
            FloatDistribution(name="dropout", low=0, high=0.2),
            CategoricalDistribution(name="batch_norm", choices=[True, False]),
            CategoricalDistribution(name="lr", choices=[1e-3, 1e-4, 2e-4]),
            CategoricalDistribution(
                name="linear_transform_type", choices=["lu", "permutation", "svd"]
            ),
            CategoricalDistribution(
                name="base_transform_type",
                choices=[
                    "affine-coupling",
                    "quadratic-coupling",
                    "rq-coupling",
                    "affine-autoregressive",
                    "quadratic-autoregressive",
                    "rq-autoregressive",
                ],
            ),
        ]

    def _fit(
        self, X: DataLoader, *args: Any, **kwargs: Any
    ) -> "NormalizingFlowsPlugin":
        if self.tabular:
            self.model = TabularFlows(
                X.dataframe(),
                n_iter=self.n_iter,
                n_layers_hidden=self.n_layers_hidden,
                n_units_hidden=self.n_units_hidden,
                batch_size=self.batch_size,
                num_transform_blocks=self.num_transform_blocks,
                dropout=self.dropout,
                batch_norm=self.batch_norm,
                num_bins=self.num_bins,
                tail_bound=self.tail_bound,
                lr=self.lr,
                apply_unconditional_transform=self.apply_unconditional_transform,
                base_distribution=self.base_distribution,
                linear_transform_type=self.linear_transform_type,
                base_transform_type=self.base_transform_type,
                encoder_max_clusters=self.encoder_max_clusters,
                n_iter_min=self.n_iter_min,
                n_iter_print=self.n_iter_print,
                patience=self.patience,
                patience_metric=self.patience_metric,
                device=self.device,
            )
        else:
            self.model = NormalizingFlows(
                n_iter=self.n_iter,
                n_layers_hidden=self.n_layers_hidden,
                n_units_hidden=self.n_units_hidden,
                batch_size=self.batch_size,
                num_transform_blocks=self.num_transform_blocks,
                dropout=self.dropout,
                batch_norm=self.batch_norm,
                num_bins=self.num_bins,
                tail_bound=self.tail_bound,
                lr=self.lr,
                apply_unconditional_transform=self.apply_unconditional_transform,
                base_distribution=self.base_distribution,
                linear_transform_type=self.linear_transform_type,
                base_transform_type=self.base_transform_type,
                n_iter_min=self.n_iter_min,
                n_iter_print=self.n_iter_print,
                patience=self.patience,
                patience_metric=self.patience_metric,
                device=self.device,
            )

        self.model.fit(X.dataframe())
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        def _internal_generate(count: int) -> pd.DataFrame:
            batch = min(5000, count)

            result = self.model.generate(1)
            max_retries = count / batch + 1

            count -= 1
            retries = 0

            while count > 0 and retries < max_retries:
                batch = min(batch, count)
                try:
                    result = pd.concat(
                        [result, self.model.generate(batch)], ignore_index=True
                    )
                except BaseException:
                    pass

                count -= batch
                retries += 1

            return result

        return self._safe_generate(_internal_generate, count, syn_schema)


plugin = NormalizingFlowsPlugin
