# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
# stdlib
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Union

# third party
import pandas as pd
from pydantic import ConfigDict, validate_call

import otbench.methods.common.synth.logger as log
from otbench.methods.common.synth.constraints import Constraints
from otbench.methods.common.synth.plugins.core.dataloader import (DataLoader, GenericDataLoader, TimeSeriesDataLoader,
                                             TimeSeriesSurvivalDataLoader, create_from_info)
from otbench.methods.common.synth.plugins.core.distribution import (CategoricalDistribution, Distribution, FloatDistribution,
                                               IntegerDistribution)
from otbench.methods.common.synth.plugins.core.schema import Schema
from otbench.methods.common.synth.plugins.core.serializable import Serializable
from otbench.methods.common.synth.utils.constants import DEVICE
from otbench.methods.common.synth.utils.reproducibility import enable_reproducible_results

PLUGIN_NAME_NOT_SET: str = "plugin_name_not_set"
PLUGIN_TYPE_NOT_SET: str = "plugin_type_not_set"


class Plugin(Serializable, metaclass=ABCMeta):
    """
    .. inheritance-diagram:: otbench.methods.common.synth.plugins.core.plugin.Plugin
        :parts: 1

    Base class for all plugins.

    Each derived class must implement the following methods:
        type() - a static method that returns the type of the plugin. e.g., debug, generative, bayesian, etc.
        name() - a static method that returns the name of the plugin. e.g., ctgan, random_noise, etc.
        hyperparameter_space() - a static method that returns the hyperparameters that can be tuned during AutoML.
        _fit() - internal method, called by `fit` on each training set.
        _generate() - internal method, called by `generate`.

    If any method implementation is missing, the class constructor will fail.

    Args:
        strict: bool. Default = True
            If True, is raises an exception if the generated data is not following the requested constraints. If False, it returns only the rows that match the constraints.
        workspace: Path
            Path for caching intermediary results
        compress_dataset: bool. Default = False
            Drop redundant features before training the generator.
        device:
            PyTorch device: cpu or cuda.
        random_state: int
            Random seed
        sampling_patience: int.
            Max inference iterations to wait for the generated data to match the training schema.
        sampling_strategy: str
            Internal parameter for schema. marginal or uniform.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    def __init__(
        self,
        sampling_patience: int = 500,
        strict: bool = False,
        device: Any = DEVICE,
        random_state: int = 0,
        workspace: Path = Path("logs/otbench_workspace"),
        compress_dataset: bool = False,
        sampling_strategy: str = "marginal",  # uniform, marginal
    ) -> None:
        if self.name() == PLUGIN_NAME_NOT_SET:
            raise ValueError(
                f"Plugin {self.__class__.__name__} `name` must be implemented by the plugin class"
            )
        if self.type() == PLUGIN_TYPE_NOT_SET:
            raise ValueError(
                f"Plugin {self.__class__.__name__} `type` must be implemented by the plugin class"
            )

        super().__init__()

        enable_reproducible_results(random_state)

        self._schema: Optional[Schema] = None
        self._training_schema: Optional[Schema] = None
        self._data_encoders: Optional[Dict] = None

        self.sampling_strategy = sampling_strategy
        self.sampling_patience = sampling_patience
        self.strict = strict
        self.device = device
        self.random_state = random_state
        self.compress_dataset = compress_dataset

        workspace.mkdir(parents=True, exist_ok=True)
        self.workspace = workspace

        self.fitted = False
        self.expecting_conditional = False

    @staticmethod
    @abstractmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        """Returns the hyperparameter space for the derived plugin."""
        ...

    @classmethod
    def sample_hyperparameters(cls, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Sample value from the hyperparameter space for the current plugin."""
        param_space = cls.hyperparameter_space(*args, **kwargs)

        results = {}

        for hp in param_space:
            results[hp.name] = hp.sample()[0]

        return results

    @classmethod
    def sample_hyperparameters_optuna(cls, trial: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        param_space = cls.hyperparameter_space(*args, **kwargs)

        results = {}

        for hp in param_space:
            if isinstance(hp, IntegerDistribution):
                results[hp.name] = trial.suggest_int(hp.name, hp.low, hp.high, hp.step)
            elif isinstance(hp, FloatDistribution):
                results[hp.name] = trial.suggest_float(hp.name, hp.low, hp.high)
            elif isinstance(hp, CategoricalDistribution):
                results[hp.name] = trial.suggest_categorical(hp.name, hp.choices)
            else:
                raise RuntimeError(f"unknown distribution type {hp}")

        return results

    @staticmethod
    @abstractmethod
    def name() -> str:
        """The name of the plugin."""
        return PLUGIN_NAME_NOT_SET

    @staticmethod
    @abstractmethod
    def type() -> str:
        """The type of the plugin."""
        return PLUGIN_TYPE_NOT_SET

    @classmethod
    def fqdn(cls) -> str:
        """The Fully-Qualified name of the plugin."""
        return cls.type() + "." + cls.name()

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def fit(self, X: Union[DataLoader, pd.DataFrame], *args: Any, **kwargs: Any) -> Any:
        """Training method the synthetic data plugin.

        Args:
            X: DataLoader.
                The reference dataset.
            cond: Optional, Union[pd.DataFrame, pd.Series, np.ndarray]
                Optional Training Conditional.
                The training conditional can be used to control to output of some models, like GANs or VAEs. The content can be anything, as long as it maps to the training dataset X.
                Usage example:
                    >>> from sklearn.datasets import load_iris
                    >>> from otbench.methods.common.synth.plugins.core.dataloader import GenericDataLoader
                    >>> from otbench.methods.common.synth.constraints import Constraints
                    >>>
                    >>> # Load in `test_plugin` the generative model of choice
                    >>> # ....
                    >>>
                    >>> X, y = load_iris(as_frame=True, return_X_y=True)
                    >>> X["target"] = y
                    >>>
                    >>> X = GenericDataLoader(X)
                    >>> test_plugin.fit(X, cond=y)
                    >>>
                    >>> count = 10
                    >>> X_gen = test_plugin.generate(count, cond=np.ones(count))
                    >>>
                    >>> # The Conditional only optimizes the output generation
                    >>> # for GANs and VAEs, but does NOT guarantee the samples
                    >>> # are only from that condition.
                    >>> # If you want to guarantee that output contains only
                    >>> # "target" == 1 samples, use Constraints.
                    >>>
                    >>> constraints = Constraints(
                    >>>     rules=[
                    >>>         ("target", "==", 1),
                    >>>     ]
                    >>> )
                    >>> X_gen = test_plugin.generate(count,
                    >>>         cond=np.ones(count),
                    >>>         constraints=constraints
                    >>>        )
                    >>> assert (X_gen["target"] == 1).all()

        Returns:
            self
        """
        if isinstance(X, (pd.DataFrame)):
            X = GenericDataLoader(X)

        if "cond" in kwargs and kwargs["cond"] is not None:
            self.expecting_conditional = True

        enable_reproducible_results(self.random_state)

        self.data_info = X.info()

        self._schema = Schema(
            data=X,
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
        )

        self._training_schema = Schema(
            data=X,
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
        )

        output = self._fit(X, *args, **kwargs)
        self.fitted = True

        return output

    @abstractmethod
    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "Plugin":
        """Internal training method the synthetic data plugin.

        Args:
            X: DataLoader.
                The reference dataset.
            cond: Optional, Union[pd.DataFrame, pd.Series, np.ndarray]
                Training Conditional
        Returns:
            self
        """
        ...

    @validate_call
    def generate(
        self,
        count: Optional[int] = None,
        constraints: Optional[Constraints] = None,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> DataLoader:
        """Synthetic data generation method.

        Args:
            count: optional int.
                The number of samples to generate. If None, it generated len(reference_dataset) samples.
            cond: Optional, Union[pd.DataFrame, pd.Series, np.ndarray].
                Optional Generation Conditional. The conditional can be used only if the model was trained using a conditional too.
                If provided, it must have `count` length.
                Not all models support conditionals. The conditionals can be used in VAEs or GANs to speed-up the generation under some constraints. For model agnostic solutions, check out the `constraints` parameter.
            constraints: optional Constraints.
                Optional constraints to apply on the generated data. If none, the reference schema constraints are applied. The constraints are model agnostic, and will filter the output of the generative model.
                The constraints are a list of rules. Each rule is a tuple of the form (<feature>, <operation>, <value>).

                Valid Operations:
                    - "<", "lt" : less than <value>
                    - "<=", "le": less or equal with <value>
                    - ">", "gt" : greater than <value>
                    - ">=", "ge": greater or equal with <value>
                    - "==", "eq": equal with <value>
                    - "in": valid for categorical features, and <value> must be array. for example, ("target", "in", [0, 1])
                    - "dtype": <value> can be a data type. For example, ("target", "dtype", "int")

                Usage example:
                    >>> from otbench.methods.common.synth.constraints import Constraints
                    >>> constraints = Constraints(
                    >>>   rules=[
                    >>>             ("InterestingFeature", "==", 0),
                    >>>         ]
                    >>>     )
                    >>>
                    >>> syn_data = syn_model.generate(
                            count=count,
                            constraints=constraints
                        ).dataframe()
                    >>>
                    >>> assert (syn_data["InterestingFeature"] == 0).all()

            random_state: optional int.
                Optional random seed to use.

        Returns:
            <count> synthetic samples
        """
        if not self.fitted:
            raise RuntimeError("Fit the generator first")

        if self._schema is None:
            raise RuntimeError("Fit the model first")

        if random_state is not None:
            enable_reproducible_results(random_state)

        has_gen_cond = "cond" in kwargs and kwargs["cond"] is not None
        if has_gen_cond and not self.expecting_conditional:
            raise RuntimeError("Conditional mismatch. Got inference conditional, without any training conditional")

        if count is None:
            count = self.data_info["len"]

        # We use the training schema for the generation
        gen_constraints = self.training_schema().as_constraints()
        if constraints is not None:
            if self.strict:
                gen_constraints = gen_constraints.extend(constraints)
            else:
                gen_constraints = constraints
        else:
            if not self.strict:
                gen_constraints = Constraints()

        syn_schema = Schema.from_constraints(gen_constraints)

        X_syn = self._generate(count=count, syn_schema=syn_schema, **kwargs)

        # if X_syn.is_tabular():
        #     if self.compress_dataset:
        #         X_syn = X_syn.decompress(self.compress_context)
        #     if self._data_encoders is not None:
        #         X_syn = X_syn.decode(self._data_encoders)

        # The dataset is decompressed here, we can use the public schema
        gen_constraints = self.schema().as_constraints()
        if constraints is not None:
            # If strict, we extend the existing constraints
            if self.strict:
                gen_constraints = gen_constraints.extend(constraints)
            # If not strict (default), we only use the manually specified constraints
            else:
                gen_constraints = constraints
        else:
            # If not strict (default), we allow more flexibility
            if not self.strict:
                gen_constraints = Constraints()

        if not X_syn.satisfies(gen_constraints) and self.strict:
            raise RuntimeError(f"Plugin {self.name()} failed to meet the synthetic constraints ({gen_constraints}).")

        if self.strict:
            X_syn = X_syn.match(gen_constraints)

        return X_syn

    @abstractmethod
    def _generate(
        self,
        count: int,
        syn_schema: Schema,
        **kwargs: Any,
    ) -> DataLoader:
        """Internal synthetic data generation method.

        Args:
            count: optional int.
                The number of samples to generate. If None, it generated len(reference_dataset) samples.
            syn_schema:
                The schema/constraints that need to be satisfied by the synthetic data.
            cond: Optional, Union[pd.DataFrame, pd.Series, np.ndarray]
                Generation Conditional

        Returns:
            <count> synthetic samples
        """
        ...

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def _safe_generate(self, gen_cbk: Callable, count: int, syn_schema: Schema, **kwargs: Any) -> DataLoader:
        constraints = syn_schema.as_constraints()

        data_synth = pd.DataFrame([], columns=self.training_schema().features())
        for it in range(self.sampling_patience):
            # sample
            iter_samples = gen_cbk(count, **kwargs)
            iter_samples_df = pd.DataFrame(iter_samples, columns=self.training_schema().features())

            # Handle protected columns
            for col in syn_schema.protected_cols:
                if col not in iter_samples_df.columns:
                    # Sample the protected column using its distribution
                    iter_samples_df[col] = syn_schema.domain[col].sample(count)

            # validate schema
            iter_samples_df = self.training_schema().adapt_dtypes(iter_samples_df)

            iter_samples_df = constraints.match(iter_samples_df)

            if self.strict:
                iter_samples_df = iter_samples_df.drop_duplicates()

            data_synth = pd.concat([data_synth, iter_samples_df], ignore_index=True)

            if len(data_synth) >= count:
                break

        data_synth = self.training_schema().adapt_dtypes(data_synth).head(count)

        return create_from_info(data_synth, self.data_info)

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def _safe_generate_time_series(
        self, gen_cbk: Callable, count: int, syn_schema: Schema, **kwargs: Any
    ) -> DataLoader:
        if self.data_info["data_type"] not in ["time_series", "time_series_survival"]:
            raise ValueError(f"Invalid data type for time series = {self.data_info['data_type']}")
        constraints = syn_schema.as_constraints()

        data_synth = pd.DataFrame([], columns=self.training_schema().features())
        data_info = self.data_info
        offset = 0
        seq_offset = 0
        for it in range(self.sampling_patience):
            # sample
            if self.data_info["data_type"] == "time_series":
                static, temporal, observation_times, outcome = gen_cbk(count - offset, **kwargs)
                loader = TimeSeriesDataLoader(
                    temporal_data=temporal,
                    observation_times=observation_times,
                    static_data=static,
                    outcome=outcome,
                    seq_offset=seq_offset,
                )
            elif self.data_info["data_type"] == "time_series_survival":
                static, temporal, observation_times, T, E = gen_cbk(count - offset, **kwargs)
                loader = TimeSeriesSurvivalDataLoader(
                    temporal_data=temporal,
                    observation_times=observation_times,
                    static_data=static,
                    T=T,
                    E=E,
                    seq_offset=seq_offset,
                )

            # validate schema
            iter_samples_df = loader.dataframe()
            id_col = loader.info()["seq_id_feature"]

            iter_samples_df = self.training_schema().adapt_dtypes(iter_samples_df)

            if self.strict:
                iter_samples_df = constraints.match(iter_samples_df)

            if len(iter_samples_df) == 0:
                continue

            data_synth = pd.concat([data_synth, iter_samples_df], ignore_index=True)
            offset = len(data_synth[id_col].unique())
            seq_offset = max(data_synth[id_col].unique()) + 1

            if offset >= count:
                break

        data_synth = self.training_schema().adapt_dtypes(data_synth)
        return create_from_info(data_synth, data_info)

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def _safe_generate_images(self, gen_cbk: Callable, count: int, syn_schema: Schema, **kwargs: Any) -> DataLoader:
        data_synth = gen_cbk(count, **kwargs)

        return create_from_info(data_synth, self.data_info)

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def schema_includes(self, other: Union[DataLoader, pd.DataFrame]) -> bool:
        """Helper method to test if the reference schema includes a Dataset

        Args:
            other: DataLoader.
                The dataset to test

        Returns:
            bool, if the schema includes the dataset or not.

        """
        other_schema = Schema(data=other)
        return self.schema().includes(other_schema)

    def schema(self) -> Schema:
        """The reference schema"""
        if self._schema is None:
            raise RuntimeError("Fit the model first")

        return self._schema

    def training_schema(self) -> Schema:
        """The internal schema"""
        if self._training_schema is None:
            raise RuntimeError("Fit the model first")

        return self._training_schema

