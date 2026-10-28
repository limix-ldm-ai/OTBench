# Method reference

Generated from the registry, optional-package readiness, and resource descriptors.
Regenerate with `python -m otbench.methods.common.readiness --format markdown`.

| Method | Tasks | Availability | Optional import modules | Pretrained resources |
| --- | --- | --- | --- | --- |
| `arf` | generation | supported | `torch`, `torchvision`, `PIL`, `pydantic`, `optuna`, `loguru`, `cloudpickle`, `xgboost`, `arfpy` | none |
| `bn` | generation | limited | `torch`, `torchvision`, `PIL`, `pydantic`, `optuna`, `loguru`, `cloudpickle`, `xgboost`, `pgmpy` | none |
| `bn_original_units` | generation | limited | `torch`, `torchvision`, `PIL`, `pydantic`, `optuna`, `loguru`, `cloudpickle`, `xgboost`, `pgmpy` | none |
| `card` | prediction | external_only | core | none |
| `catboost` | prediction | supported | `catboost` | none |
| `ctgan` | generation | supported | `torch`, `torchvision`, `PIL`, `pydantic`, `optuna`, `loguru`, `cloudpickle`, `xgboost` | none |
| `diffputer` | imputation | limited | `torch` | none |
| `forest_vp` | generation, imputation | supported | `ForestDiffusion` | none |
| `forest_vp_joint` | generation, imputation, prediction | supported | `ForestDiffusion` | none |
| `ft_transformer` | prediction | supported | `torch`, `rtdl_revisiting_models` | none |
| `gaussian_copula` | generation, imputation, prediction | supported | core | none |
| `goggle` | generation | limited | `torch`, `torchvision`, `PIL`, `pydantic`, `optuna`, `loguru`, `cloudpickle`, `xgboost`, `dgl`, `torch_geometric`, `torchdata` | none |
| `great` | generation, imputation, prediction | limited | `torch`, `torchvision`, `PIL`, `pydantic`, `optuna`, `loguru`, `cloudpickle`, `xgboost`, `be_great` | `llm` |
| `hyperimpute` | imputation | supported | `hyperimpute` | none |
| `independent_marginal` | generation | diagnostic | core | none |
| `knn_imputer` | imputation | supported | core | none |
| `lightgbm` | prediction | supported | `lightgbm` | none |
| `lightgbm_quantile` | prediction | supported | `lightgbm` | none |
| `limix` | imputation, prediction | supported | `torch`, `einops` | `model_ckpt` |
| `limix2` | generation, imputation, prediction | experimental | `torch`, `einops`, `kditransform` | `model_ckpt` |
| `linear_regression` | prediction | supported | core | none |
| `logistic_regression` | prediction | supported | core | none |
| `mice` | imputation | supported | core | none |
| `missforest` | imputation | supported | core | none |
| `mlp` | prediction | supported | core | none |
| `nflow` | generation | limited | `torch`, `torchvision`, `PIL`, `pydantic`, `optuna`, `loguru`, `cloudpickle`, `xgboost`, `nflows` | none |
| `ngboost` | prediction | supported | `ngboost` | none |
| `nrgboost` | generation | limited | `torch`, `torchvision`, `PIL`, `pydantic`, `optuna`, `loguru`, `cloudpickle`, `xgboost`, `nrgboost` | none |
| `random_forest` | prediction | supported | core | none |
| `remasker` | imputation | external_only | core | none |
| `scm` | generation | limited | `torch`, `torchvision`, `PIL`, `pydantic`, `optuna`, `loguru`, `cloudpickle`, `xgboost`, `lingam`, `causallearn` | none |
| `simple_mean_mode` | imputation | supported | core | none |
| `simple_median_mode` | imputation | supported | core | none |
| `smote` | generation | limited | `imblearn` | none |
| `smotenc` | generation | supported | `imblearn` | none |
| `smoter` | generation | supported | core | none |
| `tabcsdi` | imputation | supported | `torch` | none |
| `tabddpm` | generation, imputation, prediction | supported | `torch`, `category_encoders`, `tomli`, `tomli_w` | none |
| `tabdiff` | generation, imputation, prediction | supported | `torch`, `category_encoders`, `tomli_w` | none |
| `tabebm` | generation | limited | `torch`, `tabpfn` | `classifier_ckpt` |
| `tabicl` | prediction | supported | `torch`, `tabicl` | `classifier_ckpt` |
| `tabiclv2` | generation, imputation, prediction | experimental | `torch`, `tabicl` | `classifier_ckpt`, `regressor_ckpt` |
| `tabimpute` | imputation | limited | `torch`, `tabimpute` | `model_ckpt` |
| `tabm` | prediction | supported | `torch`, `tabm` | none |
| `tabpfn` | imputation, prediction | supported | `torch`, `tabpfn` | `classifier_ckpt`, `regressor_ckpt` |
| `tabpfn_unsupervised` | generation, imputation, prediction | experimental | `torch`, `tabpfn_extensions` | `classifier_ckpt`, `regressor_ckpt` |
| `tabsyn` | generation, imputation, prediction | supported | `torch`, `category_encoders`, `prdc` | none |
| `train_bootstrap` | generation | diagnostic | core | none |
| `tvae` | generation | supported | `torch`, `torchvision`, `PIL`, `pydantic`, `optuna`, `loguru`, `cloudpickle`, `xgboost` | none |
| `xgboost` | prediction | supported | `xgboost` | none |

`external_only` entries are unavailable placeholders: their upstream source is not distributed.
CTGAN requires `opacus` when differential privacy is enabled.
