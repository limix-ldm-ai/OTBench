# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Single source of truth for available methods (lazy adapter imports).

The spec table is pure data so callers (run.py, batch scripts, metric runners)
can check what a method can do without importing torch or any heavy backend.
Each entry carries a :class:`~.capabilities.MethodCapabilities` block declaring
column types, task types, mask granularity, which parts of a predictive
distribution the adapter emits natively, its checkpoint group, and the upstream
implementation it wraps. Runners consult it before a run and emit a structured
skip instead of a silent approximation when a combination is unsupported.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .base import TabularMethod, Task
from .capabilities import MethodCapabilities, UpstreamRef
from .config import load_method_config

_GIP = frozenset({Task.GENERATION, Task.IMPUTATION, Task.PREDICTION})
_GI = frozenset({Task.GENERATION, Task.IMPUTATION})
_IP = frozenset({Task.IMPUTATION, Task.PREDICTION})
_G = frozenset({Task.GENERATION})
_I = frozenset({Task.IMPUTATION})
_P = frozenset({Task.PREDICTION})

_MIXED = frozenset({"numerical", "categorical"})
_NUM = frozenset({"numerical"})
_ALL_TASK_TYPES = frozenset({"binclass", "multiclass", "regression"})
_CLASSIFICATION = frozenset({"binclass", "multiclass"})
_ANY_MASK = frozenset({"cell", "column", "target"})


@dataclass(frozen=True)
class MethodSpec:
    name: str
    import_path: str                 # "otbench.methods.<m>.adapter:<Class>"
    tasks: frozenset
    trainable: bool
    description: str = ""
    capabilities: MethodCapabilities = field(default_factory=MethodCapabilities)


def _caps(**kwargs) -> MethodCapabilities:
    kwargs.setdefault("column_types", _MIXED)
    kwargs.setdefault("task_types", _ALL_TASK_TYPES)
    return MethodCapabilities(**kwargs)


METHODS: dict[str, MethodSpec] = {
    # --- diffusion methods, original repos, one ckpt for all three tasks ---
    "tabddpm": MethodSpec(
        "tabddpm", "otbench.methods.tabddpm.adapter:TabDDPMMethod", _GIP, True,
        "Gaussian-multinomial diffusion (original tab-ddpm; trained with is_y_cond=False)",
        _caps(
            mask_granularity=_ANY_MASK,
            prediction_outputs=frozenset({"point", "proba", "samples"}),
            imputation_outputs=frozenset({"point", "draws"}),
            checkpoint_group="tabddpm",
            upstream=UpstreamRef(
                "tab-ddpm", "https://github.com/yandex-research/tab-ddpm",
                version="vendored", license="MIT",
            ),
        ),
    ),
    "tabsyn": MethodSpec(
        "tabsyn", "otbench.methods.tabsyn.adapter:TabSynMethod", _GIP, True,
        "VAE + latent EDM diffusion (original tabsyn)",
        _caps(
            mask_granularity=_ANY_MASK,
            prediction_outputs=frozenset({"point", "proba", "samples"}),
            imputation_outputs=frozenset({"point", "draws"}),
            checkpoint_group="tabsyn",
            upstream=UpstreamRef(
                "tabsyn", "https://github.com/amazon-science/tabsyn",
                version="vendored", license="Apache-2.0",
            ),
        ),
    ),
    "tabdiff": MethodSpec(
        "tabdiff", "otbench.methods.tabdiff.adapter:TabDiffMethod", _GIP, True,
        "Unified continuous-time diffusion (original TabDiff; y_only guidance for impute)",
        _caps(
            mask_granularity=_ANY_MASK,
            prediction_outputs=frozenset({"point", "samples"}),
            imputation_outputs=frozenset({"point", "draws"}),
            checkpoint_group="tabdiff",
            upstream=UpstreamRef(
                "TabDiff", "https://github.com/MinkaiXu/TabDiff",
                version="vendored", license="MIT",
            ),
        ),
    ),
    # --- foundation models: pretrained, training-free ---
    "tabpfn": MethodSpec(
        "tabpfn", "otbench.methods.tabpfn.adapter:TabPFNMethod", _IP, False,
        "TabPFN v2 (pretrained; column-wise imputation, native prediction)",
        _caps(
            mask_granularity=_ANY_MASK,
            prediction_outputs=frozenset({"point", "proba"}),
            imputation_outputs=frozenset({"point"}),
            checkpoint_group="tabpfn_pretrained",
            auxiliary_models="per_column",
            upstream=UpstreamRef(
                "tabpfn", "https://github.com/PriorLabs/TabPFN",
                version="2.1.2", license="Apache-2.0",
                notes="version pinned; see the TabPFN/TabImpute compatibility gate",
            ),
        ),
    ),
    "tabpfn_unsupervised": MethodSpec(
        "tabpfn_unsupervised",
        "otbench.methods.tabpfn_unsupervised.adapter:TabPFNUnsupervisedMethod",
        _GIP, False,
        "TabPFN-3 with unsupervised G, target-observed I, and native V3 prediction",
        _caps(
            mask_granularity=_ANY_MASK,
            prediction_outputs=frozenset({"point", "proba"}),
            imputation_outputs=frozenset({"point", "draws"}),
            checkpoint_group="tabpfn_unsupervised_pretrained",
            auxiliary_models="per_column",
            status="experimental",
            notes="G uses the upstream sequential sampler. I conditions on the "
                  "observed supervised target by default, fills only explicit mask "
                  "cells, and fits each masked-column conditional once per dataset "
                  "setting grid. Categorical columns modeled as targets are limited "
                  "to 160 train levels. Prediction uses the native V3 estimator.",
            upstream=UpstreamRef(
                "tabpfn-extensions unsupervised",
                "https://github.com/PriorLabs/tabpfn-extensions",
                version="0.6.2 + tabpfn 8.4.0 / TabPFN-3 weights",
                license="Apache-2.0 code; tabpfn-3-license-v1.0 weights",
                notes="separate method identity from the historical TabPFN v2 baseline",
            ),
        ),
    ),
    "tabiclv2": MethodSpec(
        "tabiclv2",
        "otbench.methods.tabiclv2.adapter:TabICLv2Method",
        _GIP, False,
        "TabICL v2 with native unsupervised generation/imputation and prediction",
        _caps(
            mask_granularity=_ANY_MASK,
            prediction_outputs=frozenset({"point", "proba"}),
            imputation_outputs=frozenset({"point", "draws"}),
            checkpoint_group="tabiclv2_pretrained",
            auxiliary_models="per_column",
            status="experimental",
            notes="G and I use the upstream TabICLUnsupervised chain-rule "
                  "sampler with train-only schema encoding. I fills explicit "
                  "mask cells only and conditions on the observed supervised "
                  "target by default. The classifier and regressor weights are "
                  "loaded from the historical tabicl pretrained directory.",
            upstream=UpstreamRef(
                "TabICL v2 / TabICLUnsupervised",
                "https://github.com/soda-inria/tabicl",
                version="2.2.0", license="BSD-3-Clause",
            ),
        ),
    ),
    "limix2": MethodSpec(
        "limix2", "otbench.methods.limix2.adapter:LimiX2Method", _GIP, False,
        "LimiX-2 native prediction/MVI plus experimental conditional generation",
        _caps(
            mask_granularity=_ANY_MASK,
            prediction_outputs=frozenset({"point", "proba"}),
            imputation_outputs=frozenset({"point"}),
            checkpoint_group="limix2_pretrained",
            auxiliary_models="per_column",
            status="experimental",
            notes="I uses the public Feature_imputation head and is point-only. "
                  "P uses native classification probabilities for 2--10-class "
                  "targets and native regression points for numerical targets; "
                  "classification targets above 10 classes follow the requested "
                  "ordinal-regression point route and are not skipped. "
                  "G samples sequential conditionals: categorical targets with "
                  "2--10 levels use classification; categorical targets above "
                  "10 levels are ordinally represented and sampled through the "
                  "native regression bucket distribution, then rounded/clipped "
                  "during train-fitted decoding. High cardinality is never skipped.",
            upstream=UpstreamRef(
                "LimiX-2", "https://github.com/limix-ldm-ai/LimiX",
                version="2.0.0 / architecture 2.0",
                license="Stable AI Technology License v1.0 code; "
                        "StableAI LimiX Non-Commercial License weights",
                notes="Generation uses the current v2 bucket-regression internals; "
                      "imputation and prediction use the public predictor API.",
            ),
        ),
    ),
    "tabicl": MethodSpec(
        "tabicl", "otbench.methods.tabicl.adapter:TabICLMethod", _P, False,
        "Historical TabICL classification predictor",
        _caps(
            task_types=_CLASSIFICATION,
            prediction_outputs=frozenset({"point", "proba"}),
            checkpoint_group="tabicl_pretrained",
            upstream=UpstreamRef(
                "tabicl", "https://github.com/soda-inria/tabicl",
                license="BSD-3-Clause",
                notes="classification-only historical prediction identity; "
                      "all TabICL v2 imputation uses the target-observed tabiclv2 adapter",
            ),
        ),
    ),
    "limix": MethodSpec(
        "limix", "otbench.methods.limix.adapter:LimiXMethod", _IP, False,
        "LimiX (pretrained; column-wise imputation, native prediction)",
        _caps(
            mask_granularity=_ANY_MASK,
            prediction_outputs=frozenset({"point", "proba"}),
            imputation_outputs=frozenset({"point"}),
            checkpoint_group="limix_pretrained",
            auxiliary_models="per_column",
            upstream=UpstreamRef(
                "LimiX", "https://github.com/limix-ldm-ai/LimiX", version="vendored legacy snapshot",
                license="Apache-2.0",
                notes="legacy code retains methods/limix/LICENSE.txt; current LimiX releases "
                      "and separately acquired weights may have different terms",
            ),
        ),
    ),
    # --- generation-only methods ---
    "independent_marginal": MethodSpec(
        "independent_marginal",
        "otbench.methods.independent_marginal.adapter:IndependentMarginalMethod", _G, True,
        "Independent per-column marginal sampler (diagnostic: destroys all dependence)",
        _caps(
            checkpoint_group="independent_marginal",
            status="diagnostic",
            upstream=UpstreamRef(
                "OTBench diagnostic baseline",
                license="Apache-2.0",
                notes="samples each column from its train marginal independently; "
                      "identifies metrics that only look at marginals",
            ),
        ),
    ),
    "train_bootstrap": MethodSpec(
        "train_bootstrap", "otbench.methods.train_bootstrap.adapter:TrainBootstrapMethod",
        _G, True,
        "Resampling of training rows (diagnostic: fidelity ceiling, privacy floor)",
        _caps(
            checkpoint_group="train_bootstrap",
            status="diagnostic",
            upstream=UpstreamRef(
                "OTBench diagnostic baseline",
                license="Apache-2.0",
                notes="copy-risk control, not a privacy-preserving generator",
            ),
        ),
    ),
    "smote": MethodSpec(
        "smote", "otbench.methods.smote.adapter:SMOTEMethod", _G, False,
        "SMOTE interpolation oversampling; all-numerical classification only",
        _caps(
            column_types=_NUM,
            task_types=_CLASSIFICATION,
            checkpoint_group="smote",
            status="limited",
            notes="the original SMOTE interpolation is undefined for categorical columns; "
                  "mixed data must use smotenc",
            upstream=UpstreamRef(
                "imbalanced-learn SMOTE", "https://imbalanced-learn.org",
                license="MIT", notes="Chawla, Bowyer, Hall & Kegelmeyer (2002)",
            ),
        ),
    ),
    "smotenc": MethodSpec(
        "smotenc", "otbench.methods.smotenc.adapter:SMOTENCMethod", _G, False,
        "SMOTE-NC interpolation for mixed numerical/categorical classification data",
        _caps(
            task_types=_CLASSIFICATION,
            # SMOTE-NC is specifically the mixed-data variant: it interpolates
            # numeric features and votes over at least one categorical feature.
            requires_column_types=frozenset({"numerical", "categorical"}),
            checkpoint_group="smotenc",
            upstream=UpstreamRef(
                "imbalanced-learn SMOTENC", "https://imbalanced-learn.org",
                license="MIT",
                notes="Chawla et al. (2002) section 6.1; categorical indices fixed by the schema",
            ),
        ),
    ),
    "smoter": MethodSpec(
        "smoter", "otbench.methods.smoter.adapter:SMOTERMethod", _G, False,
        "SMOTER interpolation for regression (Torgo et al. 2013)",
        _caps(
            task_types=frozenset({"regression"}),
            checkpoint_group="smoter",
            upstream=UpstreamRef(
                "SMOTER", "https://doi.org/10.1007/978-3-642-40669-0_33",
                notes="Torgo, Ribeiro, Pfahringer & Branco (2013): interpolate features and "
                      "the target by the same weight; NOT a dummy-class SMOTE",
            ),
        ),
    ),
    "ctgan": MethodSpec(
        "ctgan", "otbench.methods.ctgan.adapter:CTGANMethod", _G, True,
        "Conditional GAN (synth plugin)",
        _caps(
            checkpoint_group="ctgan",
            upstream=UpstreamRef(
                "CTGAN (synthcity implementation)", "https://github.com/vanderschaarlab/synthcity",
                version="adapted 0.2.11 lineage", license="Apache-2.0 bundled implementation",
                notes="not the separately distributed BSL-1.1 CTGAN package; see methods/ctgan/VENDOR.md",
            ),
        ),
    ),
    "tvae": MethodSpec(
        "tvae", "otbench.methods.tvae.adapter:TVAEMethod", _G, True,
        "Tabular VAE (synth plugin)",
        _caps(
            checkpoint_group="tvae",
            upstream=UpstreamRef(
                "TVAE (synthcity implementation)", "https://github.com/vanderschaarlab/synthcity",
                version="adapted 0.2.11 lineage", license="Apache-2.0 bundled implementation",
                notes="not the separately distributed BSL-1.1 CTGAN package; see methods/tvae/VENDOR.md",
            ),
        ),
    ),
    "arf": MethodSpec(
        "arf", "otbench.methods.arf.adapter:ARFMethod", _G, True,
        "Adversarial random forests (synth plugin)",
        _caps(
            checkpoint_group="arf",
            upstream=UpstreamRef(
                "arfpy / synthcity wrapper", "https://github.com/bips-hb/arfpy",
                license="Apache-2.0 wrapper; MIT external arfpy backend",
                notes="adapted synthcity 0.2.11 lineage; see methods/arf/VENDOR.md",
            ),
        ),
    ),
    "nflow": MethodSpec(
        "nflow", "otbench.methods.nflow.adapter:NFlowMethod", _G, True,
        "Normalizing flows (synth plugin; trains on <=2k rows)",
        _caps(
            checkpoint_group="nflow", status="limited",
            notes="upstream plugin subsamples training data to 2k rows",
            upstream=UpstreamRef(
                "synthcity nflow", "https://github.com/vanderschaarlab/synthcity",
                version="adapted 0.2.11 lineage", license="Apache-2.0 wrapper; separate external flow backends",
                notes="see methods/nflow/VENDOR.md",
            ),
        ),
    ),
    "bn": MethodSpec(
        "bn", "otbench.methods.bn.adapter:BNMethod", _G, True,
        "Bayesian network (synth plugin; numerics discretized to <=100 bins)",
        _caps(
            checkpoint_group="bn", status="limited",
            notes="numerical columns are discretized before fitting; bin indices must be "
                  "inverse-transformed before mixing with ordinary numerical output",
            upstream=UpstreamRef(
                "pgmpy / synthcity bayesian_network", "https://github.com/vanderschaarlab/synthcity",
                license="Apache-2.0 wrapper; MIT external pgmpy backend",
                notes="see methods/bn/VENDOR.md for implementation scope",
            ),
        ),
    ),
    "bn_original_units": MethodSpec(
        "bn_original_units", "otbench.methods.bn_original_units.adapter:BNOriginalUnitsMethod", _G, True,
        "Bayesian network with train-fitted bin-midpoint decoding in original units",
        _caps(
            checkpoint_group="bn_original_units", status="limited",
            notes="Protocol native-units-midpoint-v1: numeric features are discretized "
                  "to <=100 bins, fitted edges are checkpointed, and generated codes "
                  "decode to bin midpoints. Separate from historical binned-space bn.",
            upstream=UpstreamRef(
                "pgmpy / synthcity bayesian_network", "https://github.com/vanderschaarlab/synthcity",
                license="Apache-2.0 wrapper; MIT external pgmpy backend",
                notes="see methods/bn/VENDOR.md for implementation scope",
            ),
        ),
    ),
    "nrgboost": MethodSpec(
        "nrgboost", "otbench.methods.nrgboost.adapter:NRGBoostMethod", _G, True,
        "Energy-based gradient boosting (checkpointed CPU/OpenMP backend)",
        _caps(
            checkpoint_group="nrgboost", status="limited",
            notes="official 0.0.3 backend is CPU/OpenMP-only; fitted CFFI state is "
                  "checkpointed through upstream serialization; unrelated to the "
                  "NGBoost probabilistic regressor",
            upstream=UpstreamRef(
                "NRGBoost", "https://github.com/ajoo/nrgboost",
                version="0.0.3 external package", license="MIT backend; Apache-2.0 shared synth support",
                notes="backend installed separately; see methods/nrgboost/VENDOR.md",
            ),
        ),
    ),
    "scm": MethodSpec(
        "scm", "otbench.methods.scm.adapter:SCMMethod", _G, True,
        "Structural causal model generator (lingam)",
        _caps(
            checkpoint_group="scm", status="limited",
            notes="the generator named 'scm' is unrelated to the SCM *datasets* used by "
                  "the conditional-independence metrics",
            upstream=UpstreamRef("lingam", "https://github.com/cdt15/lingam", license="MIT"),
        ),
    ),
    "goggle": MethodSpec(
        "goggle", "otbench.methods.goggle.adapter:GOGGLEMethod", _G, True,
        "Graph-based generator (needs dgl + torch-geometric)",
        _caps(
            checkpoint_group="goggle", status="limited",
            notes="heavy optional graph dependencies",
            upstream=UpstreamRef(
                "GOGGLE / synthcity", "https://github.com/tennisonliu/GOGGLE",
                version="adapted synthcity 0.2.11 lineage; original commit unrecorded",
                license="Apache-2.0 synthcity implementation; MIT GOGGLE project",
                notes="bundled model follows synthcity; see methods/goggle/VENDOR.md",
            ),
        ),
    ),
    "tabebm": MethodSpec(
        "tabebm", "otbench.methods.tabebm.adapter:TabEBMMethod", _G, False,
        "Energy-based, training-free (TabPFN backbone)",
        _caps(
            task_types=_CLASSIFICATION,
            checkpoint_group="tabebm", status="limited",
            auxiliary_models="per_class",
            notes="one energy model per class; SGLD sampling cost scales with data size",
            upstream=UpstreamRef(
                "TabEBM", "https://github.com/andreimargeloiu/TabEBM",
                version="adapted official source; original commit unrecorded",
                license="Apache-2.0",
                notes="official source/license checked 2026-10-25; see methods/tabebm/VENDOR.md",
            ),
        ),
    ),
    # --- shared multi-task methods ----------------------------------------
    "gaussian_copula": MethodSpec(
        "gaussian_copula", "otbench.methods.gaussian_copula.adapter:GaussianCopulaMethod",
        _GIP, True,
        "Gaussian copula over marginal-transformed columns; one joint fit serves G/I/P",
        _caps(
            mask_granularity=_ANY_MASK,
            prediction_outputs=frozenset({"point", "proba", "samples"}),
            imputation_outputs=frozenset({"point", "draws"}),
            checkpoint_group="gaussian_copula",
            upstream=UpstreamRef(
                "OTBench Gaussian-copula baseline",
                version="project implementation", license="Apache-2.0",
                notes="standard construction; no Copulas backend is bundled. Conditional sampling uses the "
                      "multivariate-normal conditioning formula. Empirical sample quantiles "
                      "and a step-marginal density are deliberately not claimed as native.",
            ),
        ),
    ),
    "forest_vp": MethodSpec(
        "forest_vp", "otbench.methods.forest_vp.adapter:ForestVPMethod", _GI, True,
        "Forest-VP diffusion (ForestDiffusion, diffusion_type='vp')",
        _caps(
            mask_granularity=_ANY_MASK,
            imputation_outputs=frozenset({"point", "draws"}),
            checkpoint_group="forest_vp",
            auxiliary_models="per_class",
            upstream=UpstreamRef(
                "ForestDiffusion",
                "https://github.com/SamsungSAILMontreal/ForestDiffusion",
                version="1.0.6",
                license="MIT",
                notes="only diffusion_type='vp' supports imputation (REPAINT); "
                      "'flow' is generation-only upstream",
            ),
        ),
    ),
    "forest_vp_joint": MethodSpec(
        "forest_vp_joint",
        "otbench.methods.forest_vp_joint.adapter:ForestVPJointMethod",
        _GIP, True,
        "Forest-VP strict joint diffusion; prediction masks the modeled target",
        _caps(
            mask_granularity=_ANY_MASK,
            prediction_outputs=frozenset({"point", "proba", "samples"}),
            imputation_outputs=frozenset({"point", "draws"}),
            checkpoint_group="forest_vp_joint",
            upstream=UpstreamRef(
                "ForestDiffusion",
                "https://github.com/SamsungSAILMontreal/ForestDiffusion",
                version="1.0.6", license="MIT",
                notes="separate family member: label_y is never supplied; target-mask "
                      "prediction is sampling from the one joint VP model",
            ),
        ),
    ),
    "great": MethodSpec(
        "great", "otbench.methods.great.adapter:GReaTMethod", _GIP, True,
        "GReaT language-model generator with official arbitrary-mask imputation",
        _caps(
            mask_granularity=_ANY_MASK,
            prediction_outputs=frozenset({"point", "proba", "samples"}),
            imputation_outputs=frozenset({"point", "draws"}),
            checkpoint_group="great",
            status="limited",
            notes="no fixed feature-count cap; serialized rows are preflighted against "
                  "the configured local LLM context window; parsing failures, retries, "
                  "decode repairs and sampling time are recorded",
            upstream=UpstreamRef(
                "be-great", "https://github.com/tabularis-ai/be_great",
                version="0.0.14", license="Apache-2.0 wrapper; MIT external be-great backend",
                notes="uses public GReaT.impute; adapted synthcity wrapper; see methods/great/VENDOR.md",
            ),
        ),
    ),
    # --- imputation-only methods ------------------------------------------
    "simple_mean_mode": MethodSpec(
        "simple_mean_mode", "otbench.methods.simple_impute.adapter:MeanModeMethod", _I, True,
        "Column mean (numerical) + mode (categorical) imputation",
        _caps(
            mask_granularity=_ANY_MASK,
            imputation_outputs=frozenset({"point"}),
            checkpoint_group="simple_mean_mode",
            upstream=UpstreamRef(
                "scikit-learn SimpleImputer",
                "https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html",
                license="BSD-3-Clause",
            ),
        ),
    ),
    "simple_median_mode": MethodSpec(
        "simple_median_mode", "otbench.methods.simple_impute.adapter:MedianModeMethod", _I, True,
        "Column median (numerical) + mode (categorical) imputation",
        _caps(
            mask_granularity=_ANY_MASK,
            imputation_outputs=frozenset({"point"}),
            checkpoint_group="simple_median_mode",
            upstream=UpstreamRef(
                "scikit-learn SimpleImputer",
                "https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html",
                license="BSD-3-Clause",
            ),
        ),
    ),
    "knn_imputer": MethodSpec(
        "knn_imputer", "otbench.methods.knn_imputer.adapter:KNNImputerMethod", _I, True,
        "k-nearest-neighbour imputation over a train-fitted encoding",
        _caps(
            mask_granularity=_ANY_MASK,
            imputation_outputs=frozenset({"point"}),
            checkpoint_group="knn_imputer",
            upstream=UpstreamRef(
                "scikit-learn KNNImputer",
                "https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html",
                license="BSD-3-Clause",
                notes="neighbours are always drawn from the train reference set",
            ),
        ),
    ),
    "mice": MethodSpec(
        "mice", "otbench.methods.mice.adapter:MICEMethod", _I, True,
        "Multiple imputation by chained equations: K independent posterior chains",
        _caps(
            mask_granularity=_ANY_MASK,
            imputation_outputs=frozenset({"point", "draws"}),
            checkpoint_group="mice",
            auxiliary_models="per_column",
            upstream=UpstreamRef(
                "MICE (van Buuren & Groothuis-Oudshoorn 2011)",
                "https://www.jstatsoft.org/article/view/v045i03",
                notes="each chain uses a Bayesian-bootstrap conditional-model draw and "
                      "then samples from the posterior predictive distribution; "
                      "categorical columns use a classifier, never a rounded regression",
            ),
        ),
    ),
    "missforest": MethodSpec(
        "missforest", "otbench.methods.missforest.adapter:MissForestMethod", _I, True,
        "Iterative random-forest imputation (MissForest)",
        _caps(
            mask_granularity=_ANY_MASK,
            imputation_outputs=frozenset({"point"}),
            checkpoint_group="missforest",
            auxiliary_models="per_column",
            upstream=UpstreamRef(
                "MissForest", "https://doi.org/10.1093/bioinformatics/btr597",
                notes="Stekhoven & Buhlmann (2012): RandomForestRegressor/Classifier per "
                      "column, iterated until the change criterion stops decreasing",
            ),
        ),
    ),
    "hyperimpute": MethodSpec(
        "hyperimpute", "otbench.methods.hyperimpute.adapter:HyperImputeMethod", _I, False,
        "HyperImpute transductive AutoML iterative imputation (official package)",
        _caps(
            mask_granularity=_ANY_MASK,
            trains_with_missing=True,
            learning_mode="transductive",
            imputation_outputs=frozenset({"point"}),
            checkpoint_group="hyperimpute",
            auxiliary_models="per_column",
            upstream=UpstreamRef(
                "hyperimpute", "https://github.com/vanderschaarlab/hyperimpute",
                version="0.1.17", license="MIT",
                notes="fit() is a no-op and fit_transform() adaptively selects models on each "
                      "incomplete matrix; fitting is charged to inference and no cross-setting "
                      "checkpoint is claimed; only point metrics are declared",
            ),
        ),
    ),
    "remasker": MethodSpec(
        "remasker", "otbench.methods.remasker.adapter:ReMaskerMethod", _I, True,
        "ReMasker masked-autoencoder imputation (external-only; source not distributed)",
        _caps(
            mask_granularity=_ANY_MASK,
            imputation_outputs=frozenset({"point"}),
            checkpoint_group="remasker",
            status="external_only",
            notes="Unavailable in this distribution. Scientific capability metadata is "
                  "retained for identification; no backend or upstream-derived loop is bundled.",
            upstream=UpstreamRef(
                "ReMasker", "https://github.com/tydusky/remasker",
                version="excluded from distribution",
                license="not supplied in reviewed upstream source; not redistributed",
                notes="Obtain code directly from the authors and verify its terms. "
                      "See methods/remasker/model/README.md; no external-code loader is provided.",
            ),
        ),
    ),
    "tabcsdi": MethodSpec(
        "tabcsdi", "otbench.methods.tabcsdi.adapter:TabCSDIMethod", _I, True,
        "TabCSDI conditional score-based diffusion imputation (official implementation)",
        _caps(
            mask_granularity=_ANY_MASK,
            imputation_outputs=frozenset({"point", "draws"}),
            checkpoint_group="tabcsdi",
            upstream=UpstreamRef(
                "TabCSDI", "https://github.com/pfnet-research/TabCSDI",
                version="vendored-sha256:682c3edb0d6f62872", license="MIT",
                notes="uses the official mixed-type (analog bits / one-hot) encoding; "
                      "K samples per mask are kept joint",
            ),
        ),
    ),
    "diffputer": MethodSpec(
        "diffputer", "otbench.methods.diffputer.adapter:DiffPuterMethod", _I, True,
        "DiffPuter EM diffusion imputation (official implementation)",
        _caps(
            mask_granularity=_ANY_MASK,
            imputation_outputs=frozenset({"point", "draws"}),
            checkpoint_group="diffputer",
            learning_mode="transductive",
            trains_with_missing=True,
            status="limited",
            upstream=UpstreamRef(
                "DiffPuter", "https://github.com/hengruizhang98/DiffPuter",
                version="vendored-sha256:6bac335ce53a7a937", license="MIT",
                notes="EM over the incomplete data itself: an independent transductive track; "
                      "per-round M/E cost is recorded separately",
            ),
        ),
    ),
    "tabimpute": MethodSpec(
        "tabimpute", "otbench.methods.tabimpute.adapter:TabImputeMethod", _I, False,
        "TabImpute pretrained zero-shot imputation (official implementation)",
        _caps(
            mask_granularity=_ANY_MASK,
            imputation_outputs=frozenset({"point"}),
            checkpoint_group="tabimpute_pretrained",
            status="limited",
            upstream=UpstreamRef(
                "TabImpute", "https://github.com/jacobf18/tabular",
                version="0.0.4", license="not declared in package metadata",
                notes="package metadata pins tabpfn==6.0.0 against the 2.1.2 pinned elsewhere, "
                      "but the V2 path never imports tabpfn, so it installs with --no-deps",
            ),
        ),
    ),
    # --- prediction-only methods ------------------------------------------
    "linear_regression": MethodSpec(
        "linear_regression", "otbench.methods.linear_model.adapter:LinearRegressionMethod",
        _P, True,
        "Ridge linear regression on train-fitted preprocessing",
        _caps(
            task_types=frozenset({"regression"}),
            prediction_outputs=frozenset({"point"}),
            checkpoint_group="linear_regression",
            upstream=UpstreamRef("scikit-learn Ridge", license="BSD-3-Clause"),
        ),
    ),
    "logistic_regression": MethodSpec(
        "logistic_regression", "otbench.methods.linear_model.adapter:LogisticRegressionMethod",
        _P, True,
        "Logistic regression on train-fitted preprocessing",
        _caps(
            task_types=_CLASSIFICATION,
            prediction_outputs=frozenset({"point", "proba"}),
            checkpoint_group="logistic_regression",
            upstream=UpstreamRef("scikit-learn LogisticRegression", license="BSD-3-Clause"),
        ),
    ),
    "random_forest": MethodSpec(
        "random_forest", "otbench.methods.random_forest.adapter:RandomForestMethod", _P, True,
        "Random forest classifier / regressor",
        _caps(
            prediction_outputs=frozenset({"point", "proba"}),
            checkpoint_group="random_forest",
            upstream=UpstreamRef("scikit-learn ensemble", license="BSD-3-Clause"),
        ),
    ),
    "xgboost": MethodSpec(
        "xgboost", "otbench.methods.xgboost.adapter:XGBoostMethod", _P, True,
        "XGBoost gradient boosting",
        _caps(
            prediction_outputs=frozenset({"point", "proba"}),
            checkpoint_group="xgboost",
            upstream=UpstreamRef("xgboost", "https://github.com/dmlc/xgboost", license="Apache-2.0"),
        ),
    ),
    "lightgbm": MethodSpec(
        "lightgbm", "otbench.methods.lightgbm.adapter:LightGBMMethod", _P, True,
        "LightGBM gradient boosting",
        _caps(
            prediction_outputs=frozenset({"point", "proba"}),
            checkpoint_group="lightgbm",
            upstream=UpstreamRef("LightGBM", "https://github.com/microsoft/LightGBM", license="MIT"),
        ),
    ),
    "lightgbm_quantile": MethodSpec(
        "lightgbm_quantile",
        "otbench.methods.lightgbm_quantile.adapter:LightGBMQuantileMethod", _P, True,
        "LightGBM quantile regression over a fixed quantile grid",
        _caps(
            task_types=frozenset({"regression"}),
            prediction_outputs=frozenset({"point", "quantiles"}),
            checkpoint_group="lightgbm_quantile",
            auxiliary_models="per_quantile",
            upstream=UpstreamRef(
                "LightGBM objective='quantile'",
                "https://lightgbm.readthedocs.io/en/latest/Parameters.html#objective",
                version="4.7.0", license="MIT",
                notes="one model per quantile; crossing is measured and reported, not repaired",
            ),
        ),
    ),
    "catboost": MethodSpec(
        "catboost", "otbench.methods.catboost.adapter:CatBoostMethod", _P, True,
        "CatBoost gradient boosting with native categorical handling",
        _caps(
            prediction_outputs=frozenset({"point", "proba"}),
            checkpoint_group="catboost",
            upstream=UpstreamRef("CatBoost", "https://github.com/catboost/catboost", license="Apache-2.0"),
        ),
    ),
    "mlp": MethodSpec(
        "mlp", "otbench.methods.mlp.adapter:MLPMethod", _P, True,
        "Multi-layer perceptron with validation-based early stopping",
        _caps(
            prediction_outputs=frozenset({"point", "proba"}),
            checkpoint_group="mlp",
            upstream=UpstreamRef("scikit-learn MLP", license="BSD-3-Clause"),
        ),
    ),
    "ft_transformer": MethodSpec(
        "ft_transformer", "otbench.methods.ft_transformer.adapter:FTTransformerMethod",
        _P, True,
        "FT-Transformer (official RTDL implementation)",
        _caps(
            prediction_outputs=frozenset({"point", "proba"}),
            checkpoint_group="ft_transformer",
            upstream=UpstreamRef(
                "rtdl_revisiting_models",
                "https://github.com/yandex-research/rtdl-revisiting-models",
                version="0.0.2", license="MIT", notes="Gorishniy et al. (2021)",
            ),
        ),
    ),
    "tabm": MethodSpec(
        "tabm", "otbench.methods.tabm.adapter:TabMMethod", _P, True,
        "TabM parameter-efficient deep ensemble (official implementation)",
        _caps(
            prediction_outputs=frozenset({"point", "proba"}),
            checkpoint_group="tabm",
            auxiliary_models="ensemble",
            upstream=UpstreamRef(
                "tabm", "https://github.com/yandex-research/tabm",
                version="0.0.3", license="MIT",
                notes="each member is trained on its own loss and classification probabilities "
                      "are averaged; regression member outputs are an ensemble spread, NOT "
                      "samples from p(y|x), so no sample-based metric is declared",
            ),
        ),
    ),
    "card": MethodSpec(
        "card", "otbench.methods.card.adapter:CARDMethod", _P, True,
        "CARD classification-and-regression diffusion (external-only; source not distributed)",
        _caps(
            prediction_outputs=frozenset({"point", "proba", "samples"}),
            checkpoint_group="card",
            auxiliary_models="ensemble",
            status="external_only",
            notes="Unavailable in this distribution. Scientific capability metadata is "
                  "retained for identification; no backend or upstream-derived loop is bundled.",
            upstream=UpstreamRef(
                "CARD", "https://github.com/XzwHan/CARD",
                version="excluded from distribution",
                license="not supplied in reviewed upstream source; not redistributed",
                notes="Obtain code directly from the authors and verify its terms. "
                      "See methods/card/model/README.md; no external-code loader is provided.",
            ),
        ),
    ),
    "ngboost": MethodSpec(
        "ngboost", "otbench.methods.ngboost.adapter:NGBoostMethod", _P, True,
        "Natural-gradient boosting with an explicit predictive distribution",
        _caps(
            task_types=frozenset({"regression"}),
            prediction_outputs=frozenset({"point", "samples", "density", "quantiles"}),
            checkpoint_group="ngboost",
            upstream=UpstreamRef(
                "ngboost", "https://github.com/stanfordmlgroup/ngboost",
                version="0.5.11", license="Apache-2.0",
                notes="samples / logpdf / ppf are read off pred_dist; unrelated to NRGBoost",
            ),
        ),
    ),
}


def available_methods() -> list[str]:
    return sorted(METHODS)


def get_spec(name: str) -> MethodSpec:
    try:
        return METHODS[name]
    except KeyError:
        raise KeyError(f"Unknown method {name!r}. Known: {available_methods()}") from None


def method_tasks(name: str) -> frozenset:
    return get_spec(name).tasks


def method_capabilities(name: str) -> MethodCapabilities:
    return get_spec(name).capabilities


def methods_for_task(task: Task | str) -> list[str]:
    task = Task(task)
    return sorted(name for name, spec in METHODS.items() if task in spec.tasks)


def method_dir(name: str) -> Path:
    module_path, _ = get_spec(name).import_path.split(":")
    package = module_path.rsplit(".", 1)[0]
    module = importlib.import_module(package)
    return Path(module.__file__).resolve().parent


def get_method(
    name: str,
    config_file: str | Path | None = None,
    overrides: Optional[dict] = None,
    checkpoints_root: str | Path | None = None,
) -> TabularMethod:
    """Import the adapter lazily and instantiate it with its merged config."""
    spec = get_spec(name)
    module_path, class_name = spec.import_path.split(":")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    config = load_method_config(Path(module.__file__).resolve().parent, config_file, overrides)
    method: TabularMethod = cls(config=config)
    method.checkpoints_root = Path(checkpoints_root).expanduser().resolve() if checkpoints_root is not None else None
    if method.name != spec.name or frozenset(method.tasks) != spec.tasks or method.trainable != spec.trainable:
        raise RuntimeError(
            f"Registry/adapter mismatch for {name!r}: "
            f"adapter declares name={method.name!r} tasks={sorted(t.value for t in method.tasks)} "
            f"trainable={method.trainable}"
        )
    return method

