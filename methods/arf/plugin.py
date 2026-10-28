# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""
Reference: "Adversarial random forests for density estimation and generative modeling" Authors: David S. Watson, Kristin Blesch, Jan Kapar, and Marvin N. Wright
"""

# stdlib
from pathlib import Path
from typing import Any, List, Union

# third party
import pandas as pd
import torch

# Necessary packages
from pydantic import validate_call

from otbench.methods.common.synth.plugins.core.dataloader import DataLoader
from otbench.methods.common.synth.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    IntegerDistribution,
)
from .model.tabular_arf import TabularARF
from otbench.methods.common.synth.plugins.core.plugin import Plugin
from otbench.methods.common.synth.plugins.core.schema import Schema
from otbench.methods.common.synth.utils.constants import DEVICE


class ARFPlugin(Plugin):
    """ARFPlugin: fit and sample encoded tabular data."""

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        num_trees: int = 30,
        delta: int = 0,
        max_iters: int = 10,
        early_stop: bool = True,
        verbose: bool = True,
        min_node_size: int = 5,
        # core plugin arguments
        device: Union[str, torch.device] = DEVICE,
        random_state: int = 0,
        sampling_patience: int = 500,
        workspace: Path = Path("logs/otbench_workspace"),
        compress_dataset: bool = False,
        **kwargs: Any,
    ) -> None:
        """Configure the ARF backend and shared plugin execution settings.

        ``num_trees`` and ``min_node_size`` set the forest size and leaf limit;
        ``delta``, ``max_iters`` and ``early_stop`` control adversarial fitting.
        ``verbose`` controls backend output. ``random_state`` seeds fitting.
        The remaining named arguments are shared plugin device, retry, cache
        and dataset-compression settings and are forwarded to :class:`Plugin`.
        """
        super().__init__(
            device=device,
            random_state=random_state,
            sampling_patience=sampling_patience,
            workspace=workspace,
            compress_dataset=compress_dataset,
            **kwargs,
        )
        self.num_trees = num_trees
        self.delta = delta
        self.max_iters = max_iters
        self.early_stop = early_stop
        self.verbose = verbose
        self.min_node_size = min_node_size

    @staticmethod
    def name() -> str:
        return "arf"

    @staticmethod
    def type() -> str:
        return "generic"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            IntegerDistribution(name="num_trees", low=10, high=100, step=10),
            IntegerDistribution(name="delta", low=0, high=50, step=2),
            IntegerDistribution(name="max_iters", low=1, high=5, step=1),
            CategoricalDistribution(name="early_stop", choices=[True, False]),
            IntegerDistribution(name="min_node_size", low=2, high=20, step=2),
        ]

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "ARFPlugin":
        """_summary_

        Args:
            X (DataLoader): _description_
            kwargs (Any): keyword arguments passed on to an SKLearn RandomForestClassifier

        Raises:
            NotImplementedError: _description_

        Returns:
            ARFPlugin: _description_
        """
        self.model = TabularARF(
            X.dataframe(),
            self.num_trees,
            self.delta,
            self.max_iters,
            self.early_stop,
            self.verbose,
            self.min_node_size,
            **kwargs,
        )
        if "cond" in kwargs:
            if kwargs["cond"] is not None:
                raise NotImplementedError(
                    "conditional generation is not currently available for the Adversarial Random Forest (ARF) plugin."
                )
            kwargs.pop("cond")

        self.model.fit(X.dataframe(), **kwargs)
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        if "cond" in kwargs and kwargs["cond"] is not None:
            raise NotImplementedError(
                "conditional generation is not currently available for the Adversarial Random Forest (ARF) plugin."
            )

        return self._safe_generate(self.model.generate, count, syn_schema)


plugin = ARFPlugin
