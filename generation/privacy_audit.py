# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""G18-G19: the shadow-retraining privacy audit.

This is the only track that can produce a Membership Disclosure Score or a
shadow-model MIA number, and it is expensive: every shadow requires training the
generator again on a split that includes or excludes a target record. That cost
is the reason it is a separate entry point rather than part of ``--eval``.

What it does *not* do is equally important. MDS cannot be approximated from one
``train.csv`` / ``gen.csv`` pair -- without the paired include/exclude runs
there is nothing to difference, and any number computed from a single run is a
distance statistic wearing an MDS label. When the audit has not been run the
standard evaluation reports G18/G19 as ``skipped`` with that reason.

Protocol (Shokri et al. 2017, adapted to generative models as in the
membership-disclosure literature):

1. Sample ``n_targets`` target records from the training split.
2. For each shadow ``s`` and each target ``t``, train the generator on a random
   half of the training rows; ``t`` is a member in half the shadows and a
   non-member in the other half (recorded, never inferred).
3. Score each (shadow, target) pair by the target's Gower distance to the
   closest generated row.
4. MIA AUROC separates member from non-member pairs. MDS is the mean absolute
   difference between the member and non-member score distributions per target,
   i.e. how much a target's presence shifts the generator's output.
"""
from __future__ import annotations

import json
import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from otbench.generation.eval.metrics._distance import GowerSpace, tpr_at_fpr
from otbench.metrics.catalog import MetricResult

_MIA_AUROC = "generation.privacy.shadow_mia_auroc"
_MIA_TPR = "generation.privacy.shadow_mia_tpr_at_1pct_fpr"
_MDS = "generation.privacy.membership_disclosure_score"
SHADOW_METRIC_NAMES = (_MIA_AUROC, _MIA_TPR, _MDS)


@dataclass
class ShadowAuditConfig:
    n_shadows: int = 8
    n_targets: int = 32
    member_fraction: float = 0.5      # share of the train split each shadow is fitted on
    n_generated: Optional[int] = None  # default: the shadow's own training size
    seed: int = 0
    timeout_sec: Optional[float] = None

    def __post_init__(self) -> None:
        if self.n_shadows < 2 or self.n_shadows % 2:
            raise ValueError("n_shadows must be an even integer >= 2 for paired membership")
        if self.n_targets < 1:
            raise ValueError("n_targets must be >= 1")
        if not 0.0 < self.member_fraction < 1.0:
            raise ValueError("member_fraction must lie strictly between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_shadows": self.n_shadows, "n_targets": self.n_targets,
            "member_fraction": self.member_fraction, "n_generated": self.n_generated,
            "seed": self.seed, "timeout_sec": self.timeout_sec,
        }


@dataclass
class ShadowAuditResult:
    config: dict
    scores: list[dict] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)
    elapsed_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config, "scores": self.scores,
            "failures": self.failures, "elapsed_sec": self.elapsed_sec,
        }


def skipped_results(reason: str) -> list[MetricResult]:
    """What the standard evaluation reports when no audit artifact exists."""
    return [MetricResult.skipped(name, reason) for name in SHADOW_METRIC_NAMES]


def run_shadow_audit(
    bundle,
    method_name: str,
    config: ShadowAuditConfig | None = None,
    checkpoints_root: str | Path | None = None,
    device: str | None = None,
    overrides: dict | None = None,
) -> ShadowAuditResult:
    """Train the include/exclude shadows and score every (shadow, target) pair."""
    from otbench.methods.common import get_method
    from otbench.methods.common.base import GenerateContext, Task
    from otbench.methods.common.runtime import ensure_trained

    config = config or ShadowAuditConfig()
    started = time.time()
    result = ShadowAuditResult(config=config.to_dict())

    train = bundle.train.reset_index(drop=True)
    rng = np.random.default_rng(config.seed)
    target_idx = np.sort(rng.choice(len(train), size=min(config.n_targets, len(train)), replace=False))
    targets = train.iloc[target_idx].reset_index(drop=True)
    space = GowerSpace.fit(train, bundle.schema)
    membership_plan = _balanced_membership(config.n_shadows, len(target_idx))
    target_set = set(target_idx.tolist())
    background_rows = np.asarray([i for i in range(len(train)) if i not in target_set], dtype=int)

    root = Path(checkpoints_root) if checkpoints_root else Path("checkpoints/_shadow_audit")
    fit_size = max(1, int(round(len(train) * config.member_fraction)))

    for shadow in range(config.n_shadows):
        if config.timeout_sec is not None and time.time() - started > config.timeout_sec:
            result.failures.append({"shadow": shadow, "reason": "timeout"})
            break
        shadow_rng = np.random.default_rng([config.seed, shadow])
        membership = membership_plan[shadow]
        forced_members = target_idx[membership]
        need = max(0, fit_size - len(forced_members))
        if need > len(background_rows):
            # Never fill a shadow with a target assigned to its exclusion arm.
            # On tiny datasets this makes the shadow smaller than member_fraction,
            # which is preferable to corrupting the paired audit design.
            need = len(background_rows)
        background = (
            shadow_rng.choice(background_rows, size=need, replace=False)
            if need else np.asarray([], dtype=int)
        )
        member_rows = np.sort(np.concatenate([forced_members, background]))
        member_set = set(member_rows.tolist())
        # Membership is recorded from the split we built, never inferred later.
        recorded_membership = np.array([int(i) in member_set for i in target_idx], dtype=bool)
        if not np.array_equal(recorded_membership, membership):
            raise RuntimeError("shadow membership construction violated the paired plan")

        shadow_bundle = _replace_train(bundle, train.iloc[member_rows].reset_index(drop=True))
        ckpt_dir = root / method_name / bundle.name / f"shadow{shadow}"
        try:
            method = get_method(method_name, overrides=overrides)
            if not method.supports(Task.GENERATION):
                raise ValueError(f"{method_name} does not support generation")
            remaining = None if config.timeout_sec is None else (
                config.timeout_sec - (time.time() - started)
            )
            with _time_limit(remaining):
                ensure_trained(method, shadow_bundle, ckpt_dir, train_seed=shadow, device=device)
                generated = method.generate(GenerateContext(
                    bundle=shadow_bundle, config=method.config, checkpoint_dir=ckpt_dir,
                    train_seed=shadow, seed=shadow, device=device,
                    n_samples=config.n_generated or len(member_rows),
                ))
        except TimeoutError as exc:
            result.failures.append({"shadow": shadow, "reason": "timeout", "error": str(exc)})
            break
        except MemoryError as exc:
            result.failures.append({"shadow": shadow, "reason": "oom", "error": str(exc)})
            continue
        except Exception as exc:  # noqa: BLE001 — a failed shadow is data, not a crash
            reason = "oom" if "out of memory" in str(exc).lower() else "failed"
            result.failures.append({"shadow": shadow, "reason": reason,
                                    "error": f"{type(exc).__name__}: {exc}"})
            continue

        distances = space.min_distances(targets, generated.reset_index(drop=True))
        for position, (distance, is_member) in enumerate(zip(distances, membership, strict=True)):
            result.scores.append({
                "shadow": shadow,
                "target_row": int(target_idx[position]),
                "is_member": bool(is_member),
                "gower_dcr": None if not np.isfinite(distance) else float(distance),
            })

    result.elapsed_sec = time.time() - started
    return result


def evaluate_shadow_audit(audit: ShadowAuditResult | dict | None) -> list[MetricResult]:
    """Turn a stored audit artifact into G18/G19 results."""
    if audit is None:
        return skipped_results(
            "no shadow audit artifact; MDS and shadow MIA require include/exclude "
            "retraining runs and cannot be derived from a single train/generated pair"
        )
    payload = audit.to_dict() if isinstance(audit, ShadowAuditResult) else dict(audit)
    frame = pd.DataFrame(payload.get("scores", []))
    context = {"config": payload.get("config"), "failures": payload.get("failures", []),
               "elapsed_sec": payload.get("elapsed_sec")}
    if frame.empty or frame["gower_dcr"].isna().all():
        return skipped_results("the shadow audit produced no usable score")

    frame = frame.dropna(subset=["gower_dcr"])
    if frame["is_member"].nunique() < 2:
        return skipped_results("the audit contains only member or only non-member pairs")

    from sklearn.metrics import roc_auc_score

    # Closer to the generated data => more likely a member, so negate.
    scores = -frame["gower_dcr"].to_numpy(dtype=float)
    labels = frame["is_member"].to_numpy(dtype=int)
    results = [
        MetricResult.ok(_MIA_AUROC, float(roc_auc_score(labels, scores)),
                        n=len(labels), details=context),
    ]
    tpr = tpr_at_fpr(scores, labels, target_fpr=0.01)
    results.append(
        MetricResult.ok(_MIA_TPR, tpr, n=len(labels), details=context)
        if tpr is not None
        else MetricResult.skipped(_MIA_TPR, "the ROC curve has no point at or below 1% FPR")
    )

    # MDS: per target, how much its presence moves the generator's output.
    per_target = []
    for row, group in frame.groupby("target_row"):
        member = group.loc[group["is_member"], "gower_dcr"]
        non_member = group.loc[~group["is_member"], "gower_dcr"]
        if member.empty or non_member.empty:
            continue
        per_target.append({"target_row": int(row),
                           "shift": float(non_member.mean() - member.mean())})
    if not per_target:
        results.append(MetricResult.skipped(
            _MDS, "no target appeared both as a member and as a non-member across shadows"
        ))
        return results
    shifts = np.array([entry["shift"] for entry in per_target], dtype=float)
    results.append(MetricResult.ok(
        _MDS, float(np.mean(np.abs(shifts))), n=len(shifts),
        details={**context, "per_target": per_target},
        extra={"definition": "mean over targets of |E[DCR | excluded] - E[DCR | included]|; "
                             "0 means the generator's output is unaffected by any single "
                             "training record"},
    ))
    return results


def save_audit(audit: ShadowAuditResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(audit.to_dict(), f, indent=2, ensure_ascii=False, default=str)
    return path


def load_audit(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _replace_train(bundle, train: pd.DataFrame):
    import copy

    from otbench.tabdata.schema import enrich_schema, load_schema

    shadow = copy.copy(bundle)
    shadow.train = train
    schema_path = Path(bundle.dir) / "schema.json"
    if schema_path.exists():
        base_schema = load_schema(schema_path)
    else:
        # In-memory bundles do not preserve whether category_domains were
        # declared or train-derived.  Drop derived state rather than leaking the
        # full training set into a shadow model.
        base_schema = replace(
            bundle.schema,
            constraints=bundle.schema.constraints_of("declared"),
            category_domains={},
        )
    shadow.schema = enrich_schema(train, base_schema)
    return shadow


def _balanced_membership(n_shadows: int, n_targets: int) -> np.ndarray:
    """Each target is included in exactly half of the shadow models."""
    plan = np.zeros((n_shadows, n_targets), dtype=bool)
    half = n_shadows // 2
    for target in range(n_targets):
        plan[(target + np.arange(half)) % n_shadows, target] = True
    return plan


@contextmanager
def _time_limit(seconds: float | None):
    """Best-effort hard deadline for one shadow fit/sample on POSIX main threads."""
    if seconds is None:
        yield
        return
    if seconds <= 0:
        raise TimeoutError("shadow audit time budget exhausted")
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "SIGALRM"):
        yield
        return

    def expired(_signum, _frame):
        raise TimeoutError("shadow fit/sample exceeded the remaining audit time budget")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)
