# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
from __future__ import annotations

import bisect
import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from otbench.generation.eval.data import EvalFrames
from otbench.metrics.catalog import MetricResult

# pgmpy is only needed by the SCM datasets, so it stays a lazy import: a
# real-data evaluation must not require it to be installed.
_LOCAL = "generation.structural.local_conditional_independence"
_GLOBAL = "generation.structural.global_conditional_independence"

# Conditional-independence tests are much more expensive than constructing the
# graph workload.  In particular, evaluating every proper subset of a large
# d-separator grows exponentially.  The workload below is a deterministic,
# graph-only, stratified sample with a hard cap.  It deliberately does not use
# the method seed: every generator and every generation seed for a dataset must
# be judged on exactly the same CI statements.
CI_WORKLOAD_PROTOCOL = "stratified-ci-v1"
MAX_CI_STATEMENTS = 2_048
LOCAL_WORKLOAD_FRACTION = 0.25
INDEPENDENT_WORKLOAD_FRACTION = 0.40
MIN_SAMPLES_PER_STRATUM = 16


@dataclass(frozen=True)
class CIStatement:
    x: str
    y: str
    conditioning: tuple[str, ...]
    expected_independent: bool
    sampling_weight: float = 1.0
    scope: str = "nonlocal"
    stratum_population: int = 1
    stratum_sample_size: int = 1

    @property
    def is_local(self) -> bool:
        return self.scope == "local"


@dataclass(frozen=True)
class _StatementBlock:
    """A contiguous block in a stratum's implicit statement population."""

    x: str
    y: str
    separator: tuple[str, ...]
    conditioning_size: int
    expected_independent: bool
    scope: str
    population: int
    use_separator: bool = False


_Stratum = tuple[str, bool, int]


def evaluate_ci(
    frames: EvalFrames,
    scm_path: str | Path,
    alpha: float = 0.01,
    ci_test: str | None = None,
    n_bins: int = 5,
    max_statements: int = MAX_CI_STATEMENTS,
) -> tuple[dict[str, float], dict[str, Any]]:
    graph, target_col = load_scm_graph(scm_path)
    if target_col is None:
        target_col = frames.schema.target_col
    if target_col != frames.schema.target_col:
        raise ValueError(
            f"SCM target_col={target_col!r} does not match schema target_col={frames.schema.target_col!r}."
        )

    statements, workload = _build_ci_workload(
        graph,
        target_col=target_col,
        max_statements=max_statements,
    )
    data = frames.ci_frame("gen", n_bins=n_bins)
    test_func = resolve_ci_test(ci_test or frames.schema.ci_test)

    evaluated = [
        evaluate_statement(statement, data, test_func, alpha)
        for statement in statements
    ]
    local = [item for item in evaluated if item["scope"] == "local"]

    metrics = {
        "structural.local_ci": _score(local),
        "structural.global_ci": _score(evaluated),
    }
    details = {
        "ci": {
            "alpha": alpha,
            "ci_test": ci_test or frames.schema.ci_test,
            "n_statements": len(evaluated),
            "n_local_statements": len(local),
            **workload,
            "statements": evaluated,
        }
    }
    return metrics, details


def evaluate_conditional_independence(
    frames: EvalFrames,
    scm_path: str | Path | None,
    alpha: float = 0.01,
    n_bins: int = 5,
    mode: str = "real",
    max_statements: int = MAX_CI_STATEMENTS,
) -> list[MetricResult]:
    """G25/G26 in the shared result protocol.

    "CI" here is *conditional independence* against the dataset's ground-truth
    causal graph. It is unrelated to confidence intervals, and unrelated to the
    generator registered under the name ``scm``.
    """
    if mode != "scm":
        reason = "conditional-independence metrics need an SCM dataset with a ground-truth graph"
        return [MetricResult.skipped(name, reason) for name in (_LOCAL, _GLOBAL)]
    if not scm_path or not Path(scm_path).exists():
        reason = f"the SCM graph file is missing: {scm_path}"
        return [MetricResult.skipped(name, reason) for name in (_LOCAL, _GLOBAL)]
    try:
        values, details = evaluate_ci(
            frames,
            scm_path=scm_path,
            alpha=alpha,
            n_bins=n_bins,
            max_statements=max_statements,
        )
    except Exception as exc:  # noqa: BLE001 — pgmpy missing, bad graph, ...
        return [MetricResult.failed(name, exc) for name in (_LOCAL, _GLOBAL)]
    payload = details["ci"]
    out = []
    for name, key, count in (
        (_LOCAL, "structural.local_ci", payload["n_local_statements"]),
        (_GLOBAL, "structural.global_ci", payload["n_statements"]),
    ):
        value = _nan_to_none(values[key])
        out.append(
            MetricResult.ok(name, value, n=count, details=payload)
            if value is not None
            else MetricResult.skipped(name, "no finite conditional-independence score")
        )
    return out


def _nan_to_none(value: float) -> float | None:
    return None if value is None or not np.isfinite(value) else float(value)


def load_scm_graph(path: str | Path):
    from pgmpy.base import DAG

    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    nodes = raw.get("nodes", [])
    edges = [tuple(edge) for edge in raw.get("edges", [])]
    if not nodes:
        nodes = sorted(set(itertools.chain.from_iterable(edges)))
    graph = DAG()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph, raw.get("target_col") or raw.get("target")


def enumerate_ci_statements(
    graph,
    *,
    target_col: str | None = None,
    max_statements: int = MAX_CI_STATEMENTS,
) -> list[CIStatement]:
    """Return the canonical bounded CI workload for a graph.

    The exhaustive population is represented as combinatorial blocks.  A
    deterministic set of integer ranks is selected from each non-empty
    ``(local/nonlocal, expected value, conditioning size)`` stratum, and only
    those combinations are materialized.  Consequently a separator of size
    13 no longer creates all 8,191 proper subsets in memory.

    ``target_col`` only controls the local/nonlocal allocation.  Sampling never
    depends on generated data, method name, process hash seed, or run seed.
    """
    statements, _ = _build_ci_workload(
        graph,
        target_col=target_col,
        max_statements=max_statements,
    )
    return statements


def _build_ci_workload(
    graph,
    *,
    target_col: str | None,
    max_statements: int,
) -> tuple[list[CIStatement], dict[str, Any]]:
    if max_statements <= 0:
        raise ValueError("max_statements must be positive")

    canonical_graph, canonical_nodes = _canonical_integer_graph(graph)
    graph_fingerprint = _graph_fingerprint(graph, target_col)
    blocks: dict[_Stratum, list[_StatementBlock]] = {}

    for x_id, y_id in itertools.combinations(range(len(canonical_nodes)), 2):
        x, y = canonical_nodes[x_id], canonical_nodes[y_id]
        scope = "local" if target_col is not None and target_col in (x, y) else "nonlocal"
        try:
            separator_ids = canonical_graph.minimal_dseparator(start=x_id, end=y_id)
        except ValueError:
            separator_ids = None

        if separator_ids is None:
            block = _StatementBlock(
                x=x,
                y=y,
                separator=(),
                conditioning_size=0,
                expected_independent=False,
                scope=scope,
                population=1,
            )
            blocks.setdefault((scope, False, 0), []).append(block)
            continue

        separator = tuple(canonical_nodes[index] for index in sorted(separator_ids))
        size = len(separator)
        positive = _StatementBlock(
            x=x,
            y=y,
            separator=separator,
            conditioning_size=size,
            expected_independent=True,
            scope=scope,
            population=1,
            use_separator=True,
        )
        blocks.setdefault((scope, True, size), []).append(positive)

        # Store one combinatorial block per conditioning size.  Do not enumerate
        # the C(size, subset_size) concrete proper subsets here.
        for subset_size in range(size):
            population = math.comb(size, subset_size)
            negative = _StatementBlock(
                x=x,
                y=y,
                separator=separator,
                conditioning_size=subset_size,
                expected_independent=False,
                scope=scope,
                population=population,
            )
            blocks.setdefault((scope, False, subset_size), []).append(negative)

    stratum_sizes = {
        key: sum(block.population for block in value)
        for key, value in blocks.items()
    }
    population_size = sum(stratum_sizes.values())
    sample_budget = min(max_statements, population_size)
    quotas = _allocate_stratified_quotas(stratum_sizes, sample_budget)

    statements: list[CIStatement] = []
    strata_details: list[dict[str, Any]] = []
    for key in sorted(blocks, key=_stratum_sort_key):
        scope, expected_independent, conditioning_size = key
        population = stratum_sizes[key]
        sample_size = quotas[key]
        sample_weight = population / sample_size
        salt = (
            f"{CI_WORKLOAD_PROTOCOL}|{graph_fingerprint}|{scope}|"
            f"{int(expected_independent)}|{conditioning_size}"
        )
        ranks = _deterministic_sample_ranks(population, sample_size, salt)
        cumulative = _cumulative_populations(blocks[key])
        for rank in ranks:
            block_index = bisect.bisect_right(cumulative, rank)
            prior = cumulative[block_index - 1] if block_index else 0
            block = blocks[key][block_index]
            within_block_rank = rank - prior
            if block.use_separator:
                conditioning = block.separator
            else:
                conditioning = _unrank_combination(
                    block.separator,
                    block.conditioning_size,
                    within_block_rank,
                )
            statements.append(
                CIStatement(
                    x=block.x,
                    y=block.y,
                    conditioning=conditioning,
                    expected_independent=block.expected_independent,
                    sampling_weight=sample_weight,
                    scope=block.scope,
                    stratum_population=population,
                    stratum_sample_size=sample_size,
                )
            )
        strata_details.append(
            {
                "scope": scope,
                "expected_independent": expected_independent,
                "conditioning_size": conditioning_size,
                "population": population,
                "sampled": sample_size,
                "sampling_weight": sample_weight,
            }
        )

    local_population = sum(
        size for key, size in stratum_sizes.items() if key[0] == "local"
    )
    local_sampled = sum(
        quota for key, quota in quotas.items() if key[0] == "local"
    )
    workload_id = _workload_id(statements)
    metadata = {
        "workload_protocol": CI_WORKLOAD_PROTOCOL,
        "workload_id": workload_id,
        "graph_fingerprint": graph_fingerprint,
        "max_statements": max_statements,
        "n_population_statements": population_size,
        "n_local_population_statements": local_population,
        "n_sampled_statements": len(statements),
        "n_local_sampled_statements": local_sampled,
        "conditioning_sizes": sorted({key[2] for key in stratum_sizes}),
        "strata": strata_details,
    }
    return statements, metadata


def _canonical_integer_graph(graph):
    """Relabel a DAG to stable integers before pgmpy's set-based algorithm.

    Python salts string hashes per process.  pgmpy's ``minimal_dseparator``
    iterates a set, so using the original string labels can select a different
    valid minimal separator in different processes.  Integer labels make that
    choice stable, while sorted insertion makes the graph representation itself
    canonical.
    """
    from pgmpy.base import DAG

    nodes = tuple(sorted(graph.nodes(), key=_node_sort_key))
    node_ids = {node: index for index, node in enumerate(nodes)}
    edges = sorted(
        ((node_ids[start], node_ids[end]) for start, end in graph.edges()),
        key=lambda edge: (edge[0], edge[1]),
    )
    canonical = DAG()
    canonical.add_nodes_from(range(len(nodes)))
    canonical.add_edges_from(edges)
    if getattr(graph, "latents", None):
        canonical.latents.update(node_ids[node] for node in graph.latents)
    return canonical, nodes


def _node_sort_key(node: Any) -> str:
    try:
        value = json.dumps(node, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        value = repr(node)
    node_type = f"{type(node).__module__}.{type(node).__qualname__}"
    return f"{node_type}:{value}"


def _graph_fingerprint(graph, target_col: str | None) -> str:
    nodes = sorted(graph.nodes(), key=_node_sort_key)
    edges = sorted(
        ((_node_sort_key(start), _node_sort_key(end)) for start, end in graph.edges()),
        key=lambda edge: (edge[0], edge[1]),
    )
    payload = {
        "nodes": [_node_sort_key(node) for node in nodes],
        "edges": edges,
        "target_col": None if target_col is None else _node_sort_key(target_col),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stratum_sort_key(key: _Stratum) -> tuple[int, int, int]:
    scope, expected_independent, conditioning_size = key
    return (
        0 if scope == "local" else 1,
        0 if expected_independent else 1,
        conditioning_size,
    )


def _allocate_stratified_quotas(
    stratum_sizes: dict[_Stratum, int],
    budget: int,
) -> dict[_Stratum, int]:
    if not stratum_sizes:
        return {}
    if budget < len(stratum_sizes):
        raise ValueError(
            f"max_statements={budget} is too small to cover all "
            f"{len(stratum_sizes)} non-empty CI strata"
        )
    if budget >= sum(stratum_sizes.values()):
        return dict(stratum_sizes)

    by_scope = {
        scope: {key: size for key, size in stratum_sizes.items() if key[0] == scope}
        for scope in ("local", "nonlocal")
        if any(key[0] == scope for key in stratum_sizes)
    }
    if len(by_scope) == 1:
        only = next(iter(by_scope.values()))
        return _allocate_scope_quotas(only, budget)

    local = by_scope["local"]
    nonlocal_ = by_scope["nonlocal"]
    local_population = sum(local.values())
    nonlocal_population = sum(nonlocal_.values())
    local_minimum = _preferred_minimum(local)
    nonlocal_minimum = _preferred_minimum(nonlocal_)

    # With an unusually tiny caller-provided budget we can still guarantee one
    # item per stratum, but cannot promise the preferred 16.  The production
    # budget has ample room for all current SCM graphs.
    if budget < local_minimum + nonlocal_minimum:
        return _allocate_quotas(stratum_sizes, budget)

    local_budget = max(local_minimum, round(budget * LOCAL_WORKLOAD_FRACTION))
    local_budget = min(local_budget, local_population, budget - nonlocal_minimum)
    nonlocal_budget = budget - local_budget
    if nonlocal_budget > nonlocal_population:
        transfer = nonlocal_budget - nonlocal_population
        nonlocal_budget -= transfer
        local_budget += transfer

    quotas = _allocate_scope_quotas(local, local_budget)
    quotas.update(_allocate_scope_quotas(nonlocal_, nonlocal_budget))
    return quotas


def _allocate_scope_quotas(
    sizes: dict[_Stratum, int],
    budget: int,
) -> dict[_Stratum, int]:
    """Reserve useful coverage for full-separator independent statements."""
    independent = {key: size for key, size in sizes.items() if key[1]}
    dependent = {key: size for key, size in sizes.items() if not key[1]}
    if not independent or not dependent:
        return _allocate_quotas(sizes, budget)

    independent_minimum = _preferred_minimum(independent)
    dependent_minimum = _preferred_minimum(dependent)
    if budget < independent_minimum + dependent_minimum:
        return _allocate_quotas(sizes, budget)

    independent_population = sum(independent.values())
    dependent_population = sum(dependent.values())
    independent_budget = max(
        independent_minimum,
        round(budget * INDEPENDENT_WORKLOAD_FRACTION),
    )
    independent_budget = min(
        independent_budget,
        independent_population,
        budget - dependent_minimum,
    )
    dependent_budget = budget - independent_budget
    if dependent_budget > dependent_population:
        transfer = dependent_budget - dependent_population
        dependent_budget -= transfer
        independent_budget += transfer

    quotas = _allocate_quotas(independent, independent_budget)
    quotas.update(_allocate_quotas(dependent, dependent_budget))
    return quotas


def _preferred_minimum(sizes: dict[_Stratum, int]) -> int:
    return sum(min(size, MIN_SAMPLES_PER_STRATUM) for size in sizes.values())


def _allocate_quotas(sizes: dict[_Stratum, int], budget: int) -> dict[_Stratum, int]:
    if budget < len(sizes) or budget > sum(sizes.values()):
        raise ValueError("invalid stratum budget")
    quotas = {key: 1 for key in sizes}
    ordered = sorted(sizes, key=_stratum_sort_key)
    remaining = budget - len(quotas)

    # First strengthen every conditioning-size stratum.  If a test-only caller
    # supplies too small a budget, round-robin ordering degrades gracefully
    # while preserving at least one sample from every non-empty stratum.
    for level in range(2, MIN_SAMPLES_PER_STRATUM + 1):
        for key in ordered:
            if remaining == 0:
                return quotas
            if quotas[key] < min(level, sizes[key]):
                quotas[key] += 1
                remaining -= 1

    for _ in range(remaining):
        eligible = [key for key in ordered if quotas[key] < sizes[key]]
        selected = min(
            eligible,
            key=lambda key: (
                -(sizes[key] / (quotas[key] + 1)),
                -sizes[key],
                _stratum_sort_key(key),
            ),
        )
        quotas[selected] += 1
    return quotas


def _deterministic_sample_ranks(population: int, sample_size: int, salt: str) -> list[int]:
    if not 0 < sample_size <= population:
        raise ValueError("sample_size must be in [1, population]")
    if sample_size == population:
        return list(range(population))

    # Sampling the smaller side avoids excessive collisions near a census.  A
    # SHA-256 rejection map is independent of PYTHONHASHSEED and does not depend
    # on NumPy/Python PRNG implementation details.
    take_complement = sample_size > population // 2
    wanted = population - sample_size if take_complement else sample_size
    selected: set[int] = set()
    counter = 0
    modulus_limit = (1 << 256) - ((1 << 256) % population)
    while len(selected) < wanted:
        digest = hashlib.sha256(f"{salt}|{counter}".encode("utf-8")).digest()
        raw = int.from_bytes(digest, "big")
        if raw < modulus_limit:
            selected.add(raw % population)
        counter += 1

    if take_complement:
        return [rank for rank in range(population) if rank not in selected]
    return sorted(selected)


def _cumulative_populations(blocks: Iterable[_StatementBlock]) -> list[int]:
    total = 0
    out = []
    for block in blocks:
        total += block.population
        out.append(total)
    return out


def _unrank_combination(
    items: tuple[str, ...],
    size: int,
    rank: int,
) -> tuple[str, ...]:
    """Return one lexicographic combination without materializing its peers."""
    population = math.comb(len(items), size)
    if not 0 <= rank < population:
        raise IndexError(f"combination rank {rank} outside population {population}")
    if size == 0:
        return ()

    positions: list[int] = []
    start = 0
    remaining = size
    residual = rank
    while remaining:
        for position in range(start, len(items) - remaining + 1):
            suffix_count = math.comb(len(items) - position - 1, remaining - 1)
            if residual < suffix_count:
                positions.append(position)
                start = position + 1
                remaining -= 1
                break
            residual -= suffix_count
    return tuple(items[position] for position in positions)


def _workload_id(statements: list[CIStatement]) -> str:
    payload = [
        {
            "x": _node_sort_key(statement.x),
            "y": _node_sort_key(statement.y),
            "conditioning": [_node_sort_key(item) for item in statement.conditioning],
            "expected_independent": statement.expected_independent,
            "sampling_weight": statement.sampling_weight,
            "scope": statement.scope,
        }
        for statement in statements
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evaluate_statement(
    statement: CIStatement,
    data: pd.DataFrame,
    test_func: Callable[..., bool],
    alpha: float,
) -> dict[str, Any]:
    try:
        actual = bool(
            test_func(
                X=statement.x,
                Y=statement.y,
                Z=list(statement.conditioning),
                data=data,
                boolean=True,
                significance_level=alpha,
            )
        )
        error = None
    except Exception as exc:
        actual = None
        error = str(exc)

    passed = actual == statement.expected_independent if actual is not None else False
    return {
        "x": statement.x,
        "y": statement.y,
        "conditioning": list(statement.conditioning),
        "conditioning_size": len(statement.conditioning),
        "expected_independent": statement.expected_independent,
        "actual_independent": actual,
        "passed": bool(passed),
        "error": error,
        "sampling_weight": statement.sampling_weight,
        "scope": statement.scope,
        "stratum_population": statement.stratum_population,
        "stratum_sample_size": statement.stratum_sample_size,
    }


def resolve_ci_test(name: str) -> Callable[..., bool]:
    from pgmpy.estimators.CITests import (
        chi_square,
        g_sq,
        gcm,
        log_likelihood,
        pearsonr,
        pillai_trace,
    )

    mapping = {
        "chi_square": chi_square,
        "g_sq": g_sq,
        "log_likelihood": log_likelihood,
        "pearsonr": pearsonr,
        "pillai_trace": pillai_trace,
        "gcm": gcm,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported CI test {name!r}. Supported: {sorted(mapping)}")
    return mapping[name]


def _score(records: list[dict[str, Any]]) -> float:
    if not records:
        return float("nan")
    weights = np.asarray(
        [float(item.get("sampling_weight", 1.0)) for item in records],
        dtype=float,
    )
    passed = np.asarray([1.0 if item["passed"] else 0.0 for item in records], dtype=float)
    return float(np.average(passed, weights=weights))
