# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
import os
import random
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy
import torch
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier
from tabpfn.config import ModelInterfaceConfig, PreprocessorConfig
from tabpfn.utils import meta_dataset_collator
from torch.utils.data import DataLoader

# Default TabPFN classifier checkpoint shared with the tabpfn method, so
# Generation uses the explicitly supplied pretrained model files.
_TABPFN_CLASSIFIER_CKPT = "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt"


def _resolve_tabpfn_model_path() -> str:
    """Resolve the TabPFN classifier checkpoint path for offline use.

    Resolution order:
      1. ``TABPFN_MODEL_PATH`` env var, if set (explicit override).
      2. ``checkpoints/tabpfn/pretrained/<ckpt>`` (OTBench layout).
    """
    override = os.environ.get("TABPFN_MODEL_PATH")
    if override:
        return override

    from otbench.methods.common.checkpoints import pretrained_dir

    path = pretrained_dir("tabpfn") / _TABPFN_CLASSIFIER_CKPT
    if path.is_file():
        return str(path)

    raise RuntimeError(
        f"TabPFN classifier checkpoint not found at {path}; "
        "set TABPFN_MODEL_PATH or place the checkpoint there."
    )


def to_numpy(X: Union[np.ndarray, torch.Tensor, pd.DataFrame, None]) -> Optional[np.ndarray]:
    """
    Convert input data to numpy array format.

    Args:
        X: Input data in various formats (numpy array, torch tensor, pandas DataFrame, or None)

    Returns:
        numpy array representation of the input data, or None if input is None

    Raises:
        ValueError: If input type is not supported
    """
    match type(X):
        case np.ndarray:
            return X
        case torch.Tensor:
            return X.detach().cpu().numpy()
        case pd.DataFrame:
            return X.to_numpy()
        case None:
            return None
        case _:
            raise ValueError("X must be either a np.ndarray or a torch.Tensor")


def seed_everything(seed: int) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ========================================================================
#                               TabEBM CLASS
# ========================================================================


class TabEBM:
    """
    TabEBM: Tabular Energy-Based Model for synthetic data generation.

    This class implements an energy-based model that uses TabPFN as the underlying
    classifier to define energy functions. It generates synthetic tabular data
    using Stochastic Gradient Langevin Dynamics (SGLD) sampling.

    The core idea is to treat each class as having its own energy landscape,
    where real data points have low energy and synthetic points are generated
    by following the energy gradient through SGLD sampling.
    """

    def __init__(
        self,
        max_data_size: int = 10000,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize TabEBM with optimized configuration.

        Args:
            max_data_size: Maximum number of data points to use for training.
                          Larger datasets will be subsampled to this size.
        """
        # Configure TabPFN to disable preprocessing for gradient computation
        # This is crucial for SGLD sampling as we need gradients w.r.t. input features
        no_preprocessing_inference_config = ModelInterfaceConfig(
            FINGERPRINT_FEATURE=False,
            FEATURE_SHIFT_METHOD=None,
            CLASS_SHIFT_METHOD=None,
            PREPROCESS_TRANSFORMS=[PreprocessorConfig(name="none")],
        )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize TabPFN with single estimator for gradient computation
        # Note: Multiple estimators are disabled because preprocessing coupling
        # in TabPFN-v2 prevents effective gradient-based sampling
        tabpfn_kwargs: Dict[str, Any] = dict(
            n_estimators=1,
            fit_mode="batched",
            inference_config=no_preprocessing_inference_config,
            device=self.device,
            # Wide one-hot datasets (e.g. speeddating ~506 features) exceed TabPFN's
            # 500-feature pretraining limit; allow them through instead of erroring.
            ignore_pretraining_limits=True,
        )
        tabpfn_kwargs["model_path"] = model_path or _resolve_tabpfn_model_path()
        self.model = TabPFNClassifier(**tabpfn_kwargs)

        self.max_data_size = max_data_size

        # TabPFNClassifier holds one active preprocessed fit.  Remembering only
        # a boolean per class is incorrect after another class overwrites that
        # state, so cache the identity of the actually active class instead.
        self._active_target_class: Optional[int] = None

    def generate(
        self,
        X: Union[np.ndarray, torch.Tensor, pd.DataFrame],
        y: Union[np.ndarray, torch.Tensor, pd.Series],
        num_samples: int,
        target_classes: Optional[list[int]] = None,
        starting_point_noise_std: float = 0.01,
        sgld_step_size: float = 0.1,
        sgld_noise_std: float = 0.01,
        sgld_steps: int = 200,
        distance_negative_class: float = 5,
        seed: int = 42,
        debug: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Generate synthetic samples using Stochastic Gradient Langevin Dynamics (SGLD).

        This method creates synthetic data by treating the TabPFN classifier as an energy
        function and using SGLD to sample from the learned energy landscape. For each class,
        it creates a binary classification problem (target class vs surrogate negatives)
        and samples new points by following energy gradients.

        Args:
            X: Input features of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,)
            num_samples: Number of synthetic samples to generate per requested class
            target_classes: Optional subset of class codes to sample.  This lets
                the adapter generate exact class counts without over-producing
                the minority classes of imbalanced datasets.
            starting_point_noise_std: Standard deviation of noise added to real data
                                    points when initializing SGLD chains
            sgld_step_size: Step size for gradient updates in SGLD sampling
            sgld_noise_std: Standard deviation of noise added at each SGLD step
            sgld_steps: Number of SGLD steps to perform
            distance_negative_class: Distance for placing surrogate negative samples
                                   from the origin (used to create binary classification)
            seed: Random seed for reproducibility
            debug: Whether to print debug information during sampling

        Returns:
            Dictionary mapping class names to generated samples:
            {
                'class_0': np.ndarray of shape (num_samples, n_features),
                'class_1': np.ndarray of shape (num_samples, n_features),
                ...
            }
        """
        # Preprocess and validate input data
        data_dict = self._preprocess(X, y)

        # Perform optimized sampling with caching and batch processing
        res = self._sampling_internal(
            X=data_dict["X"],
            y=data_dict["y"],
            num_samples=num_samples,
            target_classes=target_classes,
            starting_point_noise_std=starting_point_noise_std,
            sgld_step_size=sgld_step_size,
            sgld_noise_std=sgld_noise_std,
            sgld_steps=sgld_steps,
            distance_negative_class=distance_negative_class,
            seed=seed,
            debug=debug,
        )

        # Extract sampling results and format output
        augmented_data = {}
        requested_classes = (
            [int(value) for value in np.unique(data_dict["y"])]
            if target_classes is None else [int(value) for value in target_classes]
        )
        available_classes = {int(value) for value in np.unique(data_dict["y"])}
        unknown = set(requested_classes) - available_classes
        if unknown:
            raise ValueError(f"Unknown target_classes: {sorted(unknown)}")
        for target_class in requested_classes:
            class_key = f"class_{int(target_class)}"
            augmented_data[class_key] = res[class_key]["sampling_paths"]

        return augmented_data

    def _sampling_internal(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_samples: int,
        target_classes: Optional[list[int]] = None,
        starting_point_noise_std: float = 0.01,
        sgld_step_size: float = 0.1,
        sgld_noise_std: float = 0.01,
        sgld_steps: int = 200,
        distance_negative_class: float = 5,
        seed: int = 42,
        debug: bool = False,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Optimized internal SGLD sampling method with caching and batch processing.

        This method performs the core SGLD sampling for each class by:
        1. Creating binary classification datasets (target class vs surrogate negatives)
        2. Fitting TabPFN models (with caching to avoid redundant computation)
        3. Initializing SGLD chains from noisy real data points
        4. Running SGLD updates using energy gradients

        Key optimizations:
        - Reuse the currently fitted class while emitting its chunks
        - Allocate only the current SGLD step's noise
        - Compute input-only gradients and detach between steps
        - Reduced device transfers

        Args:
            X, y: Preprocessed input data and labels
            num_samples: Number of samples to generate per class
            starting_point_noise_std, sgld_step_size, sgld_noise_std, sgld_steps: SGLD parameters
            distance_negative_class: Distance for surrogate negative samples
            seed: Random seed
            debug: Debug flag

        Returns:
            Dictionary containing sampling results for each class
        """
        if debug:
            print("=== TabEBM Optimized Sampling ===")
            print(f"Device: {self.device}")
            print(f"SGLD parameters: step_size={sgld_step_size}, noise_std={sgld_noise_std}, steps={sgld_steps}")
            print(f"Surrogate negatives distance: {distance_negative_class}")
            print(f"Starting point noise std: {starting_point_noise_std}")

        # Process only the requested class.  The plugin keeps each class active
        # while emitting all of its chunks, so the fitted TabPFN state is valid.
        unique_classes = (
            np.unique(y) if target_classes is None else np.asarray(target_classes)
        )
        synthetic_data_per_class = {}

        # Set seed for reproducible noise generation
        seed_everything(seed)

        for target_class in unique_classes:
            if debug:
                print(f"\n--- Processing class {target_class} ---")

            # Create or retrieve cached EBM dataset and model
            ebm_dict = self._get_or_create_ebm_dataset(X, y, target_class, distance_negative_class)
            X_ebm = ebm_dict["X_ebm"]
            y_ebm = ebm_dict["y_ebm"]

            # Fit predictor with caching
            self._fit_predictor_cached(X_ebm, y_ebm, target_class)

            # Initialize SGLD starting points
            # TabPFN's validation layer requires at least two query rows.  A
            # class can legitimately request one row (or leave a one-row tail),
            # so sample two chains and slice the deterministic first result.
            internal_samples = max(2, int(num_samples))
            start_dict = self._initialize_sgld_starting_points(
                X_ebm, y_ebm, internal_samples, starting_point_noise_std, seed
            )
            X_sgld = start_dict["X_start"]
            # Inference preprocessing is configured as exactly ``none`` with no
            # feature/class shifts.  ``get_preprocessed_datasets`` therefore
            # returned this same tensor with a leading estimator dimension, but
            # re-running sklearn validation for every SGLD chunk cost minutes on
            # 14k columns and rejects legitimate one-row tails.
            X_sgld_tensor = X_sgld.unsqueeze(0).to(self.device).requires_grad_(True)

            # Generate one noise tensor per step.  The previous
            # [steps, samples, features] allocation alone was hundreds of MiB
            # on Amazon and remained live beside the TabPFN gradient graph.
            X_sgld_tensor = self._perform_sgld_sampling(
                X_sgld_tensor, sgld_step_size, sgld_noise_std, sgld_steps, debug
            )

            # Store results
            synthetic_data_per_class[f"class_{int(target_class)}"] = {
                "sampling_paths": X_sgld_tensor.detach().cpu().squeeze(0).numpy()[:num_samples]
            }

        return synthetic_data_per_class

    def _preprocess(
        self,
        X: Union[np.ndarray, torch.Tensor, pd.DataFrame],
        y: Union[np.ndarray, torch.Tensor, pd.Series],
    ) -> Dict[str, np.ndarray]:
        """
        Preprocess input data with optimized memory usage and stratified sampling.

        Converts inputs to numpy arrays and applies stratified subsampling for large datasets
        to ensure balanced representation across classes while maintaining efficiency.

        Args:
            X: Input features in various formats
            y: Target labels in various formats

        Returns:
            Dictionary containing preprocessed X and y as numpy arrays
        """
        # Convert to numpy arrays
        if not isinstance(X, np.ndarray):
            X = to_numpy(X)
        if not isinstance(y, np.ndarray):
            y = to_numpy(y).reshape(-1)

        return {"X": X, "y": y}

    def _get_or_create_ebm_dataset(
        self, X: np.ndarray, y: np.ndarray, target_class: int, distance_negative_class: float
    ) -> Dict[str, torch.Tensor]:
        """
        Create EBM dataset with caching for better performance.

        Args:
            X, y: Input data and labels
            target_class: Target class for binary classification
            distance_negative_class: Distance for surrogate negatives

        Returns:
            Dictionary containing X_ebm and y_ebm tensors
        """
        # Use the existing method but add caching potential
        X_one_class = X[y == target_class]
        # The model is fitted independently for each class, so cap each class
        # directly.  Stratifying the full dataset first starved minority classes
        # when the width-aware Amazon context was necessarily small.
        max_real_rows = max(1, int(self.max_data_size) - 4)
        if len(X_one_class) > max_real_rows:
            rng = np.random.default_rng(42 + int(target_class))
            selected = np.sort(rng.choice(len(X_one_class), size=max_real_rows, replace=False))
            X_one_class = X_one_class[selected]
        X_ebm, y_ebm = TabEBM.add_surrogate_negative_samples(
            X_one_class, distance_negative_class=distance_negative_class
        )

        # Convert to tensors with proper device placement
        X_ebm = torch.from_numpy(X_ebm).float().to(self.device)
        y_ebm = torch.from_numpy(y_ebm).long().to(self.device)

        return {"X_ebm": X_ebm, "y_ebm": y_ebm}

    def _fit_predictor_cached(self, X_ebm: torch.Tensor, y_ebm: torch.Tensor, target_class: int) -> None:
        """
        Fit predictor with caching to avoid redundant model training.

        This method caches fitted models per class to avoid retraining when
        generating multiple batches for the same class.

        Args:
            X_ebm: EBM training features
            y_ebm: EBM training labels
            target_class: Target class identifier for caching
        """
        if self._active_target_class == int(target_class):
            return

        # Prepare batch data for TabPFN
        batch_dict = self._prepare_tabpfn_batch_data(X_ebm, y_ebm)
        X_ebm_list = [X_batch.to(self.device) for X_batch in batch_dict["X_train"]]
        y_ebm_list = [y_batch.to(self.device) for y_batch in batch_dict["y_train"]]
        cat_ixs = batch_dict["cat_ixs"]
        confs = batch_dict["confs"]

        # Train the model
        self.model.fit_from_preprocessed(
            X_ebm_list,
            y_ebm_list,
            cat_ix=cat_ixs,
            configs=confs,
        )

        self._active_target_class = int(target_class)

    def _initialize_sgld_starting_points(
        self, X_ebm: torch.Tensor, y_ebm: torch.Tensor, num_samples: int, starting_point_noise_std: float, seed: int
    ) -> Dict[str, torch.Tensor]:
        """
        Optimized initialization of SGLD starting points.

        Args:
            X_ebm: EBM training features
            y_ebm: EBM training labels
            num_samples: Number of starting points needed
            starting_point_noise_std: Noise level for starting points
            seed: Random seed

        Returns:
            Dictionary with starting points
        """
        seed_everything(seed)

        # Select random samples from the target class (class 0 by convention)
        real_sample_mask = y_ebm == 0
        real_samples = X_ebm[real_sample_mask]

        # Efficient random sampling
        num_real_samples = real_samples.shape[0]
        start_indices = torch.randint(0, num_real_samples, (num_samples,), device=self.device)
        X_start = real_samples[start_indices]
        y_start = torch.zeros(num_samples, dtype=torch.long, device=self.device)

        # Add noise to starting points (vectorized operation)
        if starting_point_noise_std > 0:
            noise = torch.randn_like(X_start, device=self.device) * starting_point_noise_std
            X_start = X_start + noise

        return {"X_start": X_start, "y_start": y_start}

    def _prepare_tabpfn_batch_data(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
        """
        Optimized preparation of TabPFN batch data with reduced overhead.

        Args:
            X: Input features tensor
            y: Input labels tensor

        Returns:
            Dictionary containing prepared batch data
        """
        # Use efficient train/test split with no validation data
        splitter = partial(TabEBM.train_test_split_allow_full_train, test_size=0, random_state=42, shuffle=False)

        # Convert to CPU for preprocessing (TabPFN requirement)
        X_cpu = X.cpu()
        y_cpu = y.cpu()

        # Get preprocessed datasets
        batched_datasets = self.model.get_preprocessed_datasets(
            X_cpu, y_cpu, splitter, max_data_size=self.max_data_size
        )

        # Create dataloader with minimal overhead
        batch_dataloader = DataLoader(
            batched_datasets,
            batch_size=1,
            collate_fn=meta_dataset_collator,
            pin_memory=self.device == "cuda",  # Optimize data transfer
        )

        # Extract first (and only) batch efficiently
        batch_data = next(iter(batch_dataloader))
        X_train, X_val, y_train, y_val, cat_ixs, confs = batch_data

        return {
            "X_train": X_train,
            "X_val": X_val,
            "y_train": y_train,
            "y_val": y_val,
            "cat_ixs": cat_ixs,
            "confs": confs,
        }

    def _perform_sgld_sampling(
        self,
        X_sgld: torch.Tensor,
        sgld_step_size: float,
        sgld_noise_std: float,
        sgld_steps: int,
        debug: bool,
    ) -> torch.Tensor:
        """
        Optimized SGLD sampling loop with reduced memory allocations.

        This method performs the core SGLD updates with one-step noise and
        input-only gradients, releasing each forward graph before the next.

        Args:
            X_sgld: Starting points tensor with gradients enabled
            sgld_step_size: SGLD step size
            sgld_noise_std: SGLD noise standard deviation
            sgld_steps: Number of SGLD steps
            debug: Debug flag

        Returns:
            Final samples after SGLD sampling
        """
        current = X_sgld.detach().requires_grad_(True)
        for t in range(sgld_steps):
            # Forward pass to compute energy
            logits = self.model.forward([current], return_logits=True)

            # Compute energy and total loss
            energy = TabEBM.compute_energy(logits.reshape(logits.shape[-1], -1))
            total_energy = energy.sum() / current.shape[1]
            gradient, = torch.autograd.grad(
                total_energy, current, retain_graph=False, create_graph=False,
            )

            # Debug output (optional)
            if debug and t % 10 == 0:  # Print every 10 steps to reduce overhead
                grad_norm = gradient.norm().item()
                print(f"Step {t}: energy={total_energy.item():.3f}, grad_norm={grad_norm:.4f}")

            # Detach every step so no graph or parameter gradients accumulate.
            with torch.no_grad():
                noise = (
                    torch.randn_like(current) * sgld_noise_std
                    if sgld_noise_std else torch.zeros_like(current)
                )
                current = current - sgld_step_size * gradient + noise
            current = current.detach().requires_grad_(True)

        return current.detach()

    @staticmethod
    def compute_energy(
        logits: Union[torch.Tensor, np.ndarray],
        return_unnormalized_prob: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Compute TabEBM class-specific energy function.

        The energy function is defined as:
        E_c(x) = -log(exp(f^c(x)[0]) + exp(f^c(x)[1]))

        Where f^c(x) are the logits from the class-specific binary classifier.
        Lower energy corresponds to higher probability of belonging to the target class.

        Args:
            logits: Model's unnormalized logits for each class
                   Shape: (num_samples, num_classes)
            return_unnormalized_prob: If True, return exp(-energy) instead of energy

        Returns:
            Energy values (or unnormalized probabilities) for each sample

        Raises:
            ValueError: If logits are not unnormalized or have wrong type
        """
        if isinstance(logits, torch.Tensor):
            # Validate that logits are unnormalized (not probabilities)
            logit_sums = logits.sum(dim=1)
            if (logit_sums - 1).abs().max() <= 1e-5:
                raise ValueError("Logits must be unnormalized (not probabilities)")

            # Compute energy using log-sum-exp for numerical stability
            energy = -torch.logsumexp(logits, dim=1)

            if return_unnormalized_prob:
                return torch.exp(-energy)
            return energy

        elif isinstance(logits, np.ndarray):
            # Validate that logits are unnormalized (not probabilities)
            logit_sums = logits.sum(axis=1)
            if (logit_sums - 1).max() <= 1e-5:
                raise ValueError("Logits must be unnormalized (not probabilities)")

            # Compute energy using scipy's log-sum-exp for numerical stability
            energy = -scipy.special.logsumexp(logits, axis=1)

            if return_unnormalized_prob:
                return np.exp(-energy)
            return energy
        else:
            raise ValueError("Logits must be either a torch.Tensor or a np.ndarray")

    @staticmethod
    def add_surrogate_negative_samples(
        X: Union[np.ndarray, torch.Tensor],
        distance_negative_class: float = 5,
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """
        Create surrogate negative samples for TabEBM's binary classification approach.

        This method creates artificial "negative" samples at specified distances from the origin
        to enable energy-based modeling through binary classification. The surrogate negatives
        help define the energy landscape by providing clear decision boundaries.

        For 2D data, negatives are placed at the four corners of a square centered at origin.
        For higher dimensions, random combinations of ±distance_negative_class are used.

        Args:
            X: Real data samples (expected to be approximately standardized)
               Shape: (num_samples, num_features)
            distance_negative_class: Distance of surrogate negatives from origin
                                   Larger values create more distinct separation

        Returns:
            Tuple of (X_ebm, y_ebm) where:
            - X_ebm: Combined real and surrogate samples
            - y_ebm: Binary labels (0 for real data, 1 for surrogates)
        """
        num_features = X.shape[1]

        if num_features == 2:
            # For 2D case, use deterministic corner placement for consistency
            surrogate_negatives = [
                [-distance_negative_class, -distance_negative_class],
                [distance_negative_class, distance_negative_class],
                [-distance_negative_class, distance_negative_class],
                [distance_negative_class, -distance_negative_class],
            ]
        else:
            # For higher dimensions, generate diverse surrogate negatives
            # Use a set to ensure uniqueness and symmetry
            surrogate_set = set()

            # Generate points with random ±distance_negative_class coordinates
            while len(surrogate_set) < 4:
                point = np.random.choice([-distance_negative_class, distance_negative_class], num_features)
                point_tuple = tuple(point)

                # Add both the point and its negation for symmetry
                if point_tuple not in surrogate_set:
                    surrogate_set.add(point_tuple)
                    surrogate_set.add(tuple(-np.array(point)))

            surrogate_negatives = list(surrogate_set)

        num_surrogates = len(surrogate_negatives)

        # Surrogates are placed FIRST (not appended). TabPFN may split a large
        # context with split_large_data, which slices sequential chunks and only
        # the first is used here; trailing surrogates would be sliced off, leaving
        # a real-only chunk whose all-zero one-hot columns get dropped by TabPFN's
        # constant-feature step. That shrinks the fitted context's feature width
        # below the (noisy, non-constant) SGLD query, breaking the forward cat.
        # Keeping surrogates in the first chunk gives every column variance, so the
        # widths stay aligned (and the used chunk keeps both classes).
        if isinstance(X, np.ndarray):
            # NumPy array processing
            X_surrogates = np.array(surrogate_negatives, dtype=X.dtype)
            X_ebm = np.concatenate([X_surrogates, X], axis=0)

            # Create binary labels: 1 for surrogates, 0 for real data
            y_ebm = np.concatenate([np.ones(num_surrogates, dtype=np.int64), np.zeros(X.shape[0], dtype=np.int64)])

            return X_ebm, y_ebm

        elif isinstance(X, torch.Tensor):
            # PyTorch tensor processing
            X_surrogates = torch.tensor(surrogate_negatives, dtype=X.dtype, device=X.device)
            X_ebm = torch.cat([X_surrogates, X], dim=0)

            # Create binary labels: 1 for surrogates, 0 for real data
            y_ebm = torch.cat(
                [
                    torch.ones(num_surrogates, dtype=torch.long, device=X.device),
                    torch.zeros(X.shape[0], dtype=torch.long, device=X.device),
                ]
            )

            return X_ebm, y_ebm
        else:
            raise ValueError("X must be either a np.ndarray or a torch.Tensor")

    @staticmethod
    def train_test_split_allow_full_train(
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        test_size: Optional[float] = None,
        train_size: Optional[float] = None,
        random_state: Optional[int] = None,
        shuffle: bool = True,
        stratify: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Tuple[
        Union[np.ndarray, torch.Tensor],
        Union[np.ndarray, torch.Tensor],
        Union[np.ndarray, torch.Tensor],
        Union[np.ndarray, torch.Tensor],
    ]:
        """
        Enhanced train-test split that supports full training mode.

        This method extends sklearn's train_test_split to handle the case where
        test_size=0, which means we want to use all data for training (no validation).
        This is useful for TabEBM's energy-based training approach.

        Args:
            X, y: Input features and labels
            test_size: Fraction of data for testing (if 0, enables full train mode)
            train_size: Fraction of data for training
            random_state: Random seed for reproducibility
            shuffle: Whether to shuffle data before splitting
            stratify: Array-like for stratified splitting

        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
            In full train mode, X_train=X and y_train=y
        """
        # Detect full training mode
        full_train_mode = test_size == 0

        if full_train_mode:
            # Reset test_size to None for sklearn compatibility
            test_size = None

        # Perform standard train-test split
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
        )

        # Override with full data in full train mode
        if full_train_mode:
            X_train = X
            y_train = y

        return X_train, X_val, y_train, y_val
