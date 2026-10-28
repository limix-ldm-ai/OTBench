# OTBench notices

**Built with StableAI LimiX**

Copyright © 2026 Stable AI Technology Co., Ltd.
Copyright © 2026 Wen Zhun (Xiong'an) Technology Co., Ltd.
Copyright © 2026 稳准智能(雄安)科技有限公司

These names identify the same copyright owner. Original OTBench code is distributed
under the [Apache License 2.0](LICENSE.txt). See [the licensing guide](docs/licensing.md).

The root license does not replace the licenses of third-party source, separately
installed dependencies, datasets, or model weights. Retain the upstream notices
and license texts beside their implementations. No datasets or pretrained weights
are included. LimiX-2 weights have separate non-commercial terms.
The bundled LimiX-2 code retains its own license in
`methods/limix2/model/LICENSE.txt`, including its applicable attribution and model
naming conditions. The visible attribution above is retained for that component.

## Excluded upstream implementations

The inspected CARD and ReMasker source snapshots did not supply redistribution
licenses. Their implementation files are excluded; explanation files remain in
`methods/card/model/README.md` and `methods/remasker/model/README.md`. Obtain any
needed source directly from the original projects and check their current terms:

- CARD: <https://github.com/XzwHan/CARD>
- ReMasker: <https://github.com/tydusky/remasker>

OTBench does not grant redistribution permission for those implementations.
They are marked `external_only`; their placeholder adapters explain their
unavailability. Locally obtained source must not be included in public packages.

## Provenance and modifications

The 2026-10-25 release update renames the project, Python package, imports, console
command, scripts, and documentation to OTBench; adopts the root license; adds
third-party license notices; and excludes the unlicensed CARD/ReMasker snapshots.
Project integration changes do not replace the licenses of upstream source.

OTBench adapters, contracts, and task integration are maintained separately from
vendored implementations. Existing upstream copyright and license notices remain
in place. Modified upstream files carry prominent local modification notices.
The OTBench name identifies the benchmark; the attribution above describes the
licensed lineage and does not imply upstream endorsement or affiliation.

Upstream names, versions, URLs, and reported license labels are recorded in
`methods/common/registry.py`. Labels in that registry are provenance claims from
the inspected source snapshot; they are not substitutes for the upstream terms.
A `pending` label identifies incomplete provenance, not permission to redistribute
third-party code. External package dependencies are acquired separately under
their own terms. Contributor/citation identities and the public repository URL
remain unspecified rather than inferred.

## Registry and bundled-notice inventory

Generated with `python -m otbench.methods.common.readiness --format notices`.

| Method | Registry license label | Upstream | Bundled notice paths |
| --- | --- | --- | --- |
| `arf` | Apache-2.0 wrapper; MIT external arfpy backend | [arfpy / synthcity wrapper](https://github.com/bips-hb/arfpy) | `methods/arf/LICENSE.synthcity`, `methods/arf/VENDOR.md` |
| `bn` | Apache-2.0 wrapper; MIT external pgmpy backend | [pgmpy / synthcity bayesian_network](https://github.com/vanderschaarlab/synthcity) | `methods/bn/LICENSE.synthcity`, `methods/bn/VENDOR.md` |
| `bn_original_units` | Apache-2.0 wrapper; MIT external pgmpy backend | [pgmpy / synthcity bayesian_network](https://github.com/vanderschaarlab/synthcity) | none in method directory |
| `card` | not supplied in reviewed upstream source; not redistributed | [CARD](https://github.com/XzwHan/CARD) | `methods/card/VENDOR.md` |
| `catboost` | Apache-2.0 | [CatBoost](https://github.com/catboost/catboost) | none in method directory |
| `ctgan` | Apache-2.0 bundled implementation | [CTGAN (synthcity implementation)](https://github.com/vanderschaarlab/synthcity) | `methods/ctgan/LICENSE.synthcity`, `methods/ctgan/VENDOR.md` |
| `diffputer` | MIT | [DiffPuter](https://github.com/hengruizhang98/DiffPuter) | `methods/diffputer/VENDOR.md`, `methods/diffputer/model/LICENSE` |
| `forest_vp` | MIT | [ForestDiffusion](https://github.com/SamsungSAILMontreal/ForestDiffusion) | none in method directory |
| `forest_vp_joint` | MIT | [ForestDiffusion](https://github.com/SamsungSAILMontreal/ForestDiffusion) | none in method directory |
| `ft_transformer` | MIT | [rtdl_revisiting_models](https://github.com/yandex-research/rtdl-revisiting-models) | none in method directory |
| `gaussian_copula` | Apache-2.0 | OTBench Gaussian-copula baseline | `methods/gaussian_copula/VENDOR.md` |
| `goggle` | Apache-2.0 synthcity implementation; MIT GOGGLE project | [GOGGLE / synthcity](https://github.com/tennisonliu/GOGGLE) | `methods/goggle/LICENSE.GOGGLE`, `methods/goggle/LICENSE.synthcity`, `methods/goggle/VENDOR.md` |
| `great` | Apache-2.0 wrapper; MIT external be-great backend | [be-great](https://github.com/tabularis-ai/be_great) | `methods/great/LICENSE.synthcity`, `methods/great/VENDOR.md` |
| `hyperimpute` | MIT | [hyperimpute](https://github.com/vanderschaarlab/hyperimpute) | none in method directory |
| `independent_marginal` | Apache-2.0 | OTBench diagnostic baseline | none in method directory |
| `knn_imputer` | BSD-3-Clause | [scikit-learn KNNImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html) | none in method directory |
| `lightgbm` | MIT | [LightGBM](https://github.com/microsoft/LightGBM) | none in method directory |
| `lightgbm_quantile` | MIT | [LightGBM objective='quantile'](https://lightgbm.readthedocs.io/en/latest/Parameters.html#objective) | none in method directory |
| `limix` | Apache-2.0 | [LimiX](https://github.com/limix-ldm-ai/LimiX) | `methods/limix/LICENSE.txt` |
| `limix2` | Stable AI Technology License v1.0 code; StableAI LimiX Non-Commercial License weights | [LimiX-2](https://github.com/limix-ldm-ai/LimiX) | `methods/limix2/model/LICENSE.txt` |
| `linear_regression` | BSD-3-Clause | scikit-learn Ridge | none in method directory |
| `logistic_regression` | BSD-3-Clause | scikit-learn LogisticRegression | none in method directory |
| `mice` | pending | [MICE (van Buuren & Groothuis-Oudshoorn 2011)](https://www.jstatsoft.org/article/view/v045i03) | none in method directory |
| `missforest` | pending | [MissForest](https://doi.org/10.1093/bioinformatics/btr597) | none in method directory |
| `mlp` | BSD-3-Clause | scikit-learn MLP | none in method directory |
| `nflow` | Apache-2.0 wrapper; separate external flow backends | [synthcity nflow](https://github.com/vanderschaarlab/synthcity) | `methods/nflow/LICENSE.synthcity`, `methods/nflow/VENDOR.md` |
| `ngboost` | Apache-2.0 | [ngboost](https://github.com/stanfordmlgroup/ngboost) | none in method directory |
| `nrgboost` | MIT backend; Apache-2.0 shared synth support | [NRGBoost](https://github.com/ajoo/nrgboost) | `methods/nrgboost/LICENSE.NRGBoost`, `methods/nrgboost/VENDOR.md` |
| `random_forest` | BSD-3-Clause | scikit-learn ensemble | none in method directory |
| `remasker` | not supplied in reviewed upstream source; not redistributed | [ReMasker](https://github.com/tydusky/remasker) | `methods/remasker/VENDOR.md` |
| `scm` | MIT | [lingam](https://github.com/cdt15/lingam) | none in method directory |
| `simple_mean_mode` | BSD-3-Clause | [scikit-learn SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html) | none in method directory |
| `simple_median_mode` | BSD-3-Clause | [scikit-learn SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html) | none in method directory |
| `smote` | MIT | [imbalanced-learn SMOTE](https://imbalanced-learn.org) | none in method directory |
| `smotenc` | MIT | [imbalanced-learn SMOTENC](https://imbalanced-learn.org) | none in method directory |
| `smoter` | pending | [SMOTER](https://doi.org/10.1007/978-3-642-40669-0_33) | none in method directory |
| `tabcsdi` | MIT | [TabCSDI](https://github.com/pfnet-research/TabCSDI) | `methods/tabcsdi/VENDOR.md`, `methods/tabcsdi/model/LICENSE.md` |
| `tabddpm` | MIT | [tab-ddpm](https://github.com/yandex-research/tab-ddpm) | `methods/tabddpm/LICENSE.md` |
| `tabdiff` | MIT | [TabDiff](https://github.com/MinkaiXu/TabDiff) | `methods/tabdiff/LICENSE` |
| `tabebm` | Apache-2.0 | [TabEBM](https://github.com/andreimargeloiu/TabEBM) | `methods/tabebm/LICENSE`, `methods/tabebm/VENDOR.md` |
| `tabicl` | BSD-3-Clause | [tabicl](https://github.com/soda-inria/tabicl) | none in method directory |
| `tabiclv2` | BSD-3-Clause | [TabICL v2 / TabICLUnsupervised](https://github.com/soda-inria/tabicl) | none in method directory |
| `tabimpute` | not declared in package metadata | [TabImpute](https://github.com/jacobf18/tabular) | none in method directory |
| `tabm` | MIT | [tabm](https://github.com/yandex-research/tabm) | none in method directory |
| `tabpfn` | Apache-2.0 | [tabpfn](https://github.com/PriorLabs/TabPFN) | none in method directory |
| `tabpfn_unsupervised` | Apache-2.0 code; tabpfn-3-license-v1.0 weights | [tabpfn-extensions unsupervised](https://github.com/PriorLabs/tabpfn-extensions) | none in method directory |
| `tabsyn` | Apache-2.0 | [tabsyn](https://github.com/amazon-science/tabsyn) | `methods/tabsyn/LICENSE` |
| `train_bootstrap` | Apache-2.0 | OTBench diagnostic baseline | none in method directory |
| `tvae` | Apache-2.0 bundled implementation | [TVAE (synthcity implementation)](https://github.com/vanderschaarlab/synthcity) | `methods/tvae/LICENSE.synthcity`, `methods/tvae/VENDOR.md` |
| `xgboost` | Apache-2.0 | [xgboost](https://github.com/dmlc/xgboost) | none in method directory |
