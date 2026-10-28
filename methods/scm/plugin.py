# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
# stdlib
from typing import Any, List

# third party
import pandas as pd

# Necessary packages
from pydantic import validate_call

from otbench.methods.common.synth.plugins.core.dataloader import DataLoader
from otbench.methods.common.synth.plugins.core.distribution import Distribution
from .model.tabular_scm import TabularSCM
from otbench.methods.common.synth.plugins.core.plugin import Plugin
from otbench.methods.common.synth.plugins.core.schema import Schema


class SCMPlugin(Plugin):
    """SCMPlugin: fit and sample encoded tabular data."""

    @validate_call(config=dict(arbitrary_types_allowed=True))
    # OTBench: default added so Serializable.load_dict can re-instantiate
    # the plugin with no arguments when loading a checkpoint.
    def __init__(self, cd_method: str = "direct-lingam", **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.cd_method = cd_method

    @staticmethod
    def name() -> str:
        return "scm"

    @staticmethod
    def type() -> str:
        return "generic"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return []

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "SCMPlugin":
        """Fit the Structural Causal Models (SCM) plugin.

        Args:
            X (DataLoader): The input data.

        Raises:
            NotImplementedError: Conditional generation is not currently available for the Structural Causal Models (SCM) plugin.

        Returns:
            SCMPlugin: The fitted plugin.
        """
        # === Initialize the model ===
        self.model = TabularSCM(
            cd_method=self.cd_method,
            # Encoding the target column for regression tasks generally leads to poorer performance
            encoder_whitelist=(
                X.dataframe().columns[:-1]
                if kwargs["task_type"] == "classification"
                else X.dataframe().columns
            ),
        )

        # === Fit the model ===
        if "cond" in kwargs and kwargs["cond"] is not None:
            raise NotImplementedError(
                "conditional generation is not currently available for the Structural Causal Models (SCM) plugin."
            )
        self.model.fit(X.dataframe(), **kwargs)

        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        if "cond" in kwargs and kwargs["cond"] is not None:
            raise NotImplementedError(
                "conditional generation is not currently available for the Structural Causal Models (SCM) plugin."
            )

        return self._safe_generate(self.model.generate, count, syn_schema)


plugin = SCMPlugin
