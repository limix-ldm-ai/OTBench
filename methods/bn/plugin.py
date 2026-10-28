# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""
Reference: Jankan, Ankur and Panda, Abinash, "pgmpy: Probabilistic graphical models using python,"
        Proceedings of the 14th Python in Science Conference (SCIPY 2015), 2015.
"""

# stdlib
from pathlib import Path
from typing import Any, List

# third party
import numpy as np
import pandas as pd
import pgmpy.estimators as estimators
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.sampling import BayesianModelSampling
from otbench.methods.common.synth.plugins.core.dataloader import DataLoader
from otbench.methods.common.synth.plugins.core.distribution import CategoricalDistribution, Distribution
from otbench.methods.common.synth.plugins.core.models.tabular_encoder import TabularEncoder
from otbench.methods.common.synth.plugins.core.plugin import Plugin
from otbench.methods.common.synth.plugins.core.schema import Schema


class BayesianNetworkPlugin(Plugin):
    """BayesianNetworkPlugin: fit and sample encoded tabular data."""

    def __init__(
        self,
        struct_learning_n_iter: int = 1000,
        struct_learning_search_method: str = "tree_search",  # hillclimb, pc, tree_search, mmhc, exhaustive
        struct_learning_score: str = "k2",  # k2, bdeu, bic, bds
        struct_max_indegree: int = 4,
        encoder_max_clusters: int = 10,
        encoder_noise_scale: float = 0.1,
        # core plugin
        workspace: Path = Path("logs/otbench_workspace"),
        compress_dataset: bool = False,
        random_state: int = 0,
        sampling_patience: int = 500,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            random_state=random_state,
            sampling_patience=sampling_patience,
            workspace=workspace,
            compress_dataset=compress_dataset,
            **kwargs,
        )

        self.struct_learning_n_iter = struct_learning_n_iter
        self.struct_learning_search_method = struct_learning_search_method
        self.struct_learning_score = struct_learning_score
        self.struct_max_indegree = struct_max_indegree

        self.encoder = TabularEncoder(max_clusters=encoder_max_clusters)
        self.encoder_noise_scale = encoder_noise_scale

    @staticmethod
    def name() -> str:
        return "bayesian_network"

    @staticmethod
    def type() -> str:
        return "generic"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            CategoricalDistribution(
                name="struct_learning_search_method",
                choices=["hillclimb", "pc", "tree_search"],
            ),
            CategoricalDistribution(name="struct_learning_score", choices=["k2", "bdeu", "bic", "bds"]),
        ]

    def _encode_decode(self, data: pd.DataFrame) -> pd.DataFrame:
        encoded = self.encoder.transform(data)

        # add noise to the mixture means, but keep the continuous cluster
        noise = np.random.normal(loc=0, scale=self.encoder_noise_scale, size=len(encoded))
        for col in encoded.columns:
            if col.endswith(".value"):
                encoded[col] += noise

        decoded = self.encoder.inverse_transform(encoded)
        decoded = decoded[data.columns]

        return decoded

    def _get_structure_scorer(self) -> Any:
        return {
            "k2": estimators.K2,
            "bdeu": estimators.BDeu,
            "bic": estimators.BIC,
            "bds": estimators.BDs,
        }[self.struct_learning_score]

    def _get_dag(self, X: pd.DataFrame) -> Any:
        scoring_method = scoring_method = self._get_structure_scorer()(data=X)
        if self.struct_learning_search_method == "hillclimb":
            return estimators.HillClimbSearch(data=X).estimate(
                scoring_method=scoring_method,
                max_indegree=self.struct_max_indegree,
                max_iter=self.struct_learning_n_iter,
                show_progress=False,
            )
        elif self.struct_learning_search_method == "pc":
            return estimators.PC(data=X).estimate(scoring_method=scoring_method, show_progress=False)
        elif self.struct_learning_search_method == "tree_search":
            return estimators.TreeSearch(data=X).estimate(show_progress=False)
        elif self.struct_learning_search_method == "mmhc":
            return estimators.MmhcEstimator(data=X).estimate(
                scoring_method=scoring_method,
            )
        elif self.struct_learning_search_method == "exhaustive":
            return estimators.ExhaustiveSearch(data=X).estimate()
        else:
            raise ValueError(f"invalid estimator {self.struct_learning_search_method}")

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "BayesianNetworkPlugin":
        df = X.dataframe()
        self.encoder.fit(df)

        dag = self._get_dag(df)

        network = DiscreteBayesianNetwork(dag)
        # Structure learning can leave variables with no edges out of the DAG.
        # Those columns would then be missing from forward_sample() output and
        # break the encoder (KeyError on the dropped column). Re-add any missing
        # columns as isolated nodes so every variable is modeled and sampled.
        missing_nodes = [col for col in df.columns if col not in set(network.nodes())]
        if missing_nodes:
            network.add_nodes_from(missing_nodes)
        network.fit(df)

        self.model = BayesianModelSampling(network)
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        def _sample(count: int) -> pd.DataFrame:
            vals = self.model.forward_sample(size=count, show_progress=False)

            return self._encode_decode(vals)

        return self._safe_generate(_sample, count, syn_schema)


plugin = BayesianNetworkPlugin
