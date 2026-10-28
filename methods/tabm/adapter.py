# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""TabM (Gorishniy et al., ICLR 2025) — an efficient deep ensemble of k MLPs.

Runs the authors' ``tabm.TabM``. The module returns ``(batch, k, d_out)``: one
prediction per ensemble member. Two upstream rules decide how that is handled,
and both matter for what the metrics are allowed to say:

* **Every member is trained on its own loss.** Upstream backpropagates the mean
  of the per-member losses, not the loss of the averaged prediction. Averaging
  first would make the k members collapse toward each other and stop being an
  ensemble, so training reduces the *loss*, while inference reduces the
  *prediction*.
* **The k regression outputs are an ensemble spread, not samples from p(y|x).**
  They describe disagreement between members of a point-predicting ensemble.
  Treating them as draws would produce a CRPS and an interval that look
  well-defined and mean nothing, so the registry declares ``point``/``proba``
  only and this adapter emits no samples.

For classification, upstream averages *probabilities* across members, so the
reduction is a mean over softmaxed member logits, converted back to logits for
the shared loop.
"""
from __future__ import annotations

from otbench.methods.common.base import Task
from otbench.methods.common.deep_tabular import DeepTabularMethod
from otbench.methods.common.upstream import require_package

UPSTREAM_URL = "https://github.com/yandex-research/tabm"


class TabMMethod(DeepTabularMethod):
    name = "tabm"
    tasks = frozenset({Task.PREDICTION})

    def auxiliary_model_count(self) -> int:
        return int(self.cfg("model", "k", 32))

    def build_module(self, n_numerical, cardinalities, d_out):
        tabm = require_package("tabm", method="tabm", url=UPSTREAM_URL, extra="prediction")
        self._d_out = d_out
        return tabm.TabM(
            n_num_features=n_numerical,
            cat_cardinalities=list(cardinalities),
            d_out=d_out,
            arch_type=str(self.cfg("model", "arch_type", "tabm")),
            k=int(self.cfg("model", "k", 32)),
            n_blocks=int(self.cfg("model", "n_blocks", 3)),
            d_block=int(self.cfg("model", "d_block", 512)),
            dropout=float(self.cfg("model", "dropout", 0.1)),
            # Required by upstream for the 'tabm' architecture. 'random-signs' is
            # the value for a model without numerical embeddings, which is the
            # configuration used here; 'normal' belongs with piecewise-linear
            # embeddings. The chunk layout is inferred by TabM itself.
            start_scaling_init=str(self.cfg("model", "start_scaling_init", "random-signs")),
        )

    def compute_loss(self, output, target, loss_fn, task_type):
        """Score each member separately and average, as upstream does.

        Reducing to the ensemble prediction first and scoring that once would
        let the members drift toward a single shared solution — the diversity
        that makes the ensemble work is a consequence of each member carrying
        its own gradient.
        """
        if output.dim() != 3:
            return loss_fn(output if task_type == "classification"
                           else output.squeeze(-1), target)

        batch, k, d_out = output.shape
        expanded = target.unsqueeze(1).expand(batch, k)
        if task_type == "classification":
            return loss_fn(output.reshape(batch * k, d_out), expanded.reshape(-1))
        return loss_fn(output.reshape(batch * k), expanded.reshape(-1).float())

    def reduce_output(self, output):
        """Inference only: ``(batch, k, d_out)`` -> ``(batch, d_out)``.

        Regression averages the member predictions. Classification averages the
        member *probabilities*, which is upstream's rule, and returns the log of
        that mean so the caller still sees a logit.
        """
        import torch

        if output.dim() != 3:
            return output
        if getattr(self, "_d_out", 1) == 1:
            return output.mean(dim=1)
        return torch.log(output.softmax(dim=-1).mean(dim=1).clamp_min(1e-12))
