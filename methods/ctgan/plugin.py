# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""
Reference: "Modeling Tabular Data using Conditional GAN", Xu, Lei et al.
"""

# stdlib
from pathlib import Path
from typing import Any, List, Optional, Union

# third party
import numpy as np
import pandas as pd

# Necessary packages
from pydantic import validate_call
from torch.utils.data import sampler

from otbench.methods.common.synth.metrics.weighted_metrics import WeightedMetrics
from otbench.methods.common.synth.plugins.core.dataloader import DataLoader
from otbench.methods.common.synth.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    FloatDistribution,
    IntegerDistribution,
)
from .model.tabular_gan import TabularGAN
from otbench.methods.common.synth.plugins.core.plugin import Plugin
from otbench.methods.common.synth.plugins.core.schema import Schema
from otbench.methods.common.synth.utils.constants import DEVICE


class CTGANPlugin(Plugin):
    """CTGANPlugin: fit and sample encoded tabular data."""

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_iter: int = 2000,
        generator_n_layers_hidden: int = 2,
        generator_n_units_hidden: int = 500,
        generator_nonlin: str = "relu",
        generator_dropout: float = 0.1,
        generator_opt_betas: tuple = (0.5, 0.999),
        discriminator_n_layers_hidden: int = 2,
        discriminator_n_units_hidden: int = 500,
        discriminator_nonlin: str = "leaky_relu",
        discriminator_n_iter: int = 1,
        discriminator_dropout: float = 0.1,
        discriminator_opt_betas: tuple = (0.5, 0.999),
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        batch_size: int = 200,
        random_state: int = 0,
        clipping_value: int = 1,
        lambda_gradient_penalty: float = 10,
        encoder_max_clusters: int = 10,
        encoder: Any = None,
        dataloader_sampler: Optional[sampler.Sampler] = None,
        device: Any = DEVICE,
        patience: int = 5,
        patience_metric: Optional[WeightedMetrics] = None,
        n_iter_print: int = 50,
        n_iter_min: int = 100,
        adjust_inference_sampling: bool = False,
        # core plugin arguments
        workspace: Path = Path("logs/otbench_workspace"),
        compress_dataset: bool = False,
        sampling_patience: int = 500,
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
        self.generator_n_layers_hidden = generator_n_layers_hidden
        self.generator_n_units_hidden = generator_n_units_hidden
        self.generator_nonlin = generator_nonlin
        self.n_iter = n_iter
        self.generator_dropout = generator_dropout
        self.generator_opt_betas = generator_opt_betas
        self.discriminator_n_layers_hidden = discriminator_n_layers_hidden
        self.discriminator_n_units_hidden = discriminator_n_units_hidden
        self.discriminator_nonlin = discriminator_nonlin
        self.discriminator_n_iter = discriminator_n_iter
        self.discriminator_dropout = discriminator_dropout
        self.discriminator_opt_betas = discriminator_opt_betas

        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.random_state = random_state
        self.clipping_value = clipping_value
        self.lambda_gradient_penalty = lambda_gradient_penalty

        self.encoder_max_clusters = encoder_max_clusters
        self.encoder = encoder
        self.dataloader_sampler = dataloader_sampler

        self.device = device
        self.patience = patience
        self.patience_metric = patience_metric
        self.n_iter_min = n_iter_min
        self.n_iter_print = n_iter_print
        self.adjust_inference_sampling = adjust_inference_sampling

    @staticmethod
    def name() -> str:
        return "ctgan"

    @staticmethod
    def type() -> str:
        return "generic"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            IntegerDistribution(name="generator_n_layers_hidden", low=1, high=4),
            IntegerDistribution(
                name="generator_n_units_hidden", low=50, high=150, step=50
            ),
            CategoricalDistribution(
                name="generator_nonlin", choices=["relu", "leaky_relu", "tanh", "elu"]
            ),
            IntegerDistribution(name="n_iter", low=100, high=1000, step=100),
            FloatDistribution(name="generator_dropout", low=0, high=0.2),
            IntegerDistribution(name="discriminator_n_layers_hidden", low=1, high=4),
            IntegerDistribution(
                name="discriminator_n_units_hidden", low=50, high=150, step=50
            ),
            CategoricalDistribution(
                name="discriminator_nonlin",
                choices=["relu", "leaky_relu", "tanh", "elu"],
            ),
            IntegerDistribution(name="discriminator_n_iter", low=1, high=5),
            FloatDistribution(name="discriminator_dropout", low=0, high=0.2),
            CategoricalDistribution(name="lr", choices=[1e-3, 2e-4, 1e-4]),
            CategoricalDistribution(name="weight_decay", choices=[1e-3, 1e-4]),
            CategoricalDistribution(name="batch_size", choices=[100, 200, 500]),
            IntegerDistribution(name="encoder_max_clusters", low=2, high=20),
        ]

    def _prepare_cond(
        self, cond: Optional[Union[pd.DataFrame, pd.Series, np.ndarray, list]]
    ) -> Optional[np.ndarray]:
        if cond is None:
            return None

        cond = np.asarray(cond)
        if len(cond.shape) == 1:
            cond = cond.reshape(-1, 1)

        return cond

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "CTGANPlugin":
        cond: Optional[Union[pd.DataFrame, pd.Series]] = None
        if "cond" in kwargs:
            cond = self._prepare_cond(kwargs["cond"])

        self.model = TabularGAN(
            X.dataframe(),
            cond=cond,
            n_units_latent=self.generator_n_units_hidden,
            batch_size=self.batch_size,
            generator_n_layers_hidden=self.generator_n_layers_hidden,
            generator_n_units_hidden=self.generator_n_units_hidden,
            generator_nonlin=self.generator_nonlin,
            generator_nonlin_out_discrete="softmax",
            generator_nonlin_out_continuous="none",
            generator_lr=self.lr,
            generator_residual=True,
            generator_n_iter=self.n_iter,
            generator_batch_norm=False,
            generator_dropout=0,
            generator_weight_decay=self.weight_decay,
            generator_opt_betas=self.generator_opt_betas,
            generator_extra_penalties=[],
            discriminator_n_units_hidden=self.discriminator_n_units_hidden,
            discriminator_n_layers_hidden=self.discriminator_n_layers_hidden,
            discriminator_n_iter=self.discriminator_n_iter,
            discriminator_nonlin=self.discriminator_nonlin,
            discriminator_batch_norm=False,
            discriminator_dropout=self.discriminator_dropout,
            discriminator_lr=self.lr,
            discriminator_weight_decay=self.weight_decay,
            discriminator_opt_betas=self.discriminator_opt_betas,
            encoder=self.encoder,
            clipping_value=self.clipping_value,
            lambda_gradient_penalty=self.lambda_gradient_penalty,
            encoder_max_clusters=self.encoder_max_clusters,
            dataloader_sampler=self.dataloader_sampler,
            device=self.device,
            patience=self.patience,
            patience_metric=self.patience_metric,
            n_iter_min=self.n_iter_min,
            n_iter_print=self.n_iter_print,
            adjust_inference_sampling=self.adjust_inference_sampling,
        )
        self.model.fit(X.dataframe(), cond=cond)

        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> DataLoader:
        cond: Optional[Union[pd.DataFrame, pd.Series]] = None
        if "cond" in kwargs:
            cond = self._prepare_cond(kwargs["cond"])

        return self._safe_generate(self.model.generate, count, syn_schema, cond=cond)


plugin = CTGANPlugin
