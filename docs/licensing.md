# Licensing and redistribution

**Built with StableAI LimiX**

Original OTBench code is offered under the **Apache License 2.0** in
[LICENSE.txt](../LICENSE.txt). The standard license text is available from the
[Apache Software Foundation](https://www.apache.org/licenses/LICENSE-2.0).
This guide summarizes the distribution requirements; the applicable license text
governs each component.

## Original code and third-party boundaries

The root license applies to OTBench-authored runners, adapters, dataset and
metric contracts, tooling, and documentation, except where an individual file or
directory identifies a different license. Existing third-party implementations
retain their upstream copyright notices and license texts. Their inclusion does
not relicense them under the root license. [NOTICE.md](../NOTICE.md) records the
method inventory and locations of bundled notices. Registry labels are source
provenance, not substitutes for the applicable license texts.

Dependencies installed separately, datasets, pretrained weights, and external
source checkouts each have their own terms. OTBench distributes neither datasets
nor pretrained weights. In particular, the LimiX code license does not replace
the separate LimiX-2 non-commercial weight license. Obtain resources from their
original publishers and follow their access and redistribution conditions.

## Distributing OTBench or a derivative

Apache License 2.0 Section 4 requires the following, as applicable:

- Supply a copy of the license to recipients.
- Retain relevant copyright, patent, trademark, and attribution notices, and
  include the readable attribution notices from the supplied NOTICE file.
- Put prominent notices in files you modify, identifying that you changed them.
- Preserve and satisfy each applicable third-party component's own license and
  notices. Do not suggest endorsement or affiliation by the upstream licensors.

## Bundled LimiX-2 code

The bundled code under `methods/limix2/model/` retains the separate
[Stable AI Technology Co., Ltd. License v1.0](../methods/limix2/model/LICENSE.txt).
Changing OTBench's original-code license does not relicense this component or
replace its additional Section 10 conditions. The visible attribution
**Built with StableAI LimiX** is retained for this component.

For distributions incorporating that licensed work, follow its attribution
requirements. If that work or its covered weights are used to create, train,
fine-tune, distill, or otherwise improve an AI model that is subsequently made
available to third parties, its Section 10 model-naming condition also applies.
See the component's license for the exact scope and internal-use exception.
These additional conditions belong to that third-party component; they are not
additional conditions of OTBench's Apache License 2.0.

## Implementations not distributed

The inspected CARD and ReMasker source snapshots supplied no redistribution
license. Their upstream implementation files have therefore been removed from
this distribution. Explanation files remain at their former source locations:

- [CARD](../methods/card/model/README.md): [upstream project](https://github.com/XzwHan/CARD).
- [ReMasker](../methods/remasker/model/README.md): [upstream project](https://github.com/tydusky/remasker).

Obtain any needed source directly from the upstream project, check its current
terms, and keep locally acquired files out of public distributions. A public
repository, paper citation, or permission for research use alone does not establish
redistribution permission. OTBench's license does not authorize redistribution of
those excluded implementations. These methods are marked `external_only`, and
their placeholder adapters stop with an explanation. Restoring upstream files
alone does not enable them: no automatic download or external-code loader is
provided, and OTBench does not substitute another algorithm.
