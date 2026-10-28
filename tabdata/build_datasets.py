# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Build a dataset store from a declarative source and schema manifest.

The benchmark manifest records the 50 OpenML datasets and six SCM networks
used by the evaluation. Data, caches, and pretrained weights are not bundled.
Three precomputed SCM networks require external samples and DAG definitions;
see the README for the source prerequisites.

OpenML target selection uses the recorded feature columns and requires exactly
one remaining target candidate. A build fails on ambiguous columns rather than
silently changing the prediction target. Reference split hashes can be checked
with --verify-reference; the reference split seed is unavailable.

Example:
    python -m otbench.tabdata.build_datasets --type real --store ./data
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from otbench.paths import workspace_root
from .io import csv_format_for, write_frame_csv
from .schema import schema_from_dict

HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "manifest.yaml"
STORE_ROOT = workspace_root() / "tabdata" / "store"


# ----------------------------------------------------------------------------
# Manifest
# ----------------------------------------------------------------------------
def load_manifest(path: Path = MANIFEST_PATH) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw["datasets"]


# ----------------------------------------------------------------------------
# Real datasets (tabcamel, fallback fetch_openml)
# ----------------------------------------------------------------------------
def build_real(entry: dict, seed: int, test_size: float) -> dict[str, Any]:
    name = entry["name"]
    task = entry["task_type"]
    full_df, target_col, col2type, source_used = _load_real_full(entry)

    train_df, test_df = _split(full_df, target_col, task, seed, test_size)
    schema = entry.get("schema") or _real_schema(col2type, target_col, task)
    # Drop any declared column the source no longer provides: a schema naming an
    # absent column fails _validate_optional_fields() at load time.
    group_cols = [c for c in GROUP_COLS.get(name, []) if c in col2type]
    if group_cols:
        schema["group_cols"] = group_cols
    return {
        "train": train_df,
        "test": test_df,
        "schema": schema,
        "graph": None,
        "extra_meta": {"source_resolved": source_used},
    }


def _load_real_full(entry: dict):
    """Return (full_df with target named 'target', target_col, col2type, source_str)."""
    name = entry["name"]
    task = entry["task_type"]

    # Offline path: sklearn-bundled loaders (no network), e.g. load_breast_cancer.
    if entry.get("source") == "sklearn":
        from sklearn import datasets as skdatasets

        loader = getattr(skdatasets, entry["source_id"])
        loaded = loader(as_frame=True)
        full = loaded.data.copy()
        full["target"] = loaded.target.values
        col2type = _infer_col2type(full, "target", task)
        return full, "target", col2type, f"sklearn:{entry['source_id']}"

    # Load the declared UCI source and normalize its target column name.
    if entry.get("source") == "uci":
        from ucimlrepo import fetch_ucirepo

        uci_id = int(entry["uci_id"])
        loaded = fetch_ucirepo(id=uci_id)
        x_df = loaded.data.features
        y_df = loaded.data.targets
        if y_df is None or y_df.shape[1] == 0:
            raise ValueError(f"UCI dataset id={uci_id} has no target column to use.")
        full = x_df.copy()
        full["target"] = y_df.iloc[:, 0].values
        col2type = _infer_col2type(full, "target", task)
        return full, "target", col2type, f"uci:{uci_id}"

    # URL path: download a raw CSV/whitespace file from a direct link (UCI raw,
    # author mirror, GitHub). Also OpenML-independent. Target -> 'target'.
    if entry.get("source") == "url":
        return _load_url_full(entry)

    # ARFF path: download a backup ARFF (OpenML export mirror or author repo).
    # Highest-fidelity OpenML-independent route: ARFF preserves the original
    # column names, nominal levels, and types. Target -> 'target'.
    if entry.get("source") == "arff":
        return _load_arff_full(entry)

    # Use the declared OpenML data object rather than a name-based lookup.
    if entry.get("source") == "openml-parquet":
        return _load_openml_parquet_full(entry)

    # Primary path: tabcamel by its registry key (handles target rename + types).
    loader_name = entry.get("loader_name", name)
    try:
        from tabcamel.data.dataset import TabularDataset
        from tabcamel.utils.loading import get_dataset_source

        get_dataset_source(loader_name)  # raises ValueError if unknown
        ds = TabularDataset(dataset_name=loader_name, task_type=task)
        col2type = {c: t.name for c, t in ds.col2type.items()}
        return ds.data_df.copy(), ds.target_col, col2type, f"tabcamel:{loader_name}"
    except Exception as exc:  # noqa: BLE001 — fall back to OpenML by id
        print(f"[build] tabcamel lookup failed for {loader_name!r} ({exc}); "
              f"falling back to fetch_openml(data_id={entry.get('source_id')}).")

    from sklearn.datasets import fetch_openml

    loaded = fetch_openml(data_id=int(entry["source_id"]), as_frame=True)
    x_df = loaded.data
    y_s = loaded.target
    full = x_df.copy()
    full["target"] = y_s.values
    col2type = _infer_col2type(full, "target", task)
    return full, "target", col2type, f"openml:{entry['source_id']}"


def _load_url_full(entry: dict):
    """Load a real dataset from a direct download URL (OpenML-independent).

    Manifest fields (under ``source: url``):
        url        : direct link to a CSV/whitespace-delimited file.
        sep        : column separator; ',' (default), ';', or 'whitespace'.
        header     : null/"none" if the file has no header row, else "infer".
        na_values  : optional token(s) treated as missing (e.g. '?').
        target     : how to locate the label column, one of
                       - int   : column index (supports -1 for the last column);
                       - str   : column name (when a header is present);
                       - {row_block: N} : no label column in the file — synthesize
                         target = row_index // N (the mfeat block layout, classes
                         in contiguous blocks of N rows).
    """
    import io
    import urllib.request

    task = entry["task_type"]
    url = entry["url"]

    sep = entry.get("sep", ",")
    if sep in ("whitespace", "ws"):
        sep = r"\s+"
    header = entry.get("header", "infer")
    if header in (None, "none", "None"):
        header = None
    na_values = entry.get("na_values")

    raw = urllib.request.urlopen(url, timeout=180).read()
    df = pd.read_csv(io.BytesIO(raw), sep=sep, header=header, na_values=na_values, engine="python")

    target = entry.get("target", -1)
    if isinstance(target, dict) and "row_block" in target:
        block = int(target["row_block"])
        df.columns = [f"f{i}" for i in range(df.shape[1])]
        full = df.copy()
        full["target"] = (pd.RangeIndex(len(full)) // block).astype("int64")
    else:
        if header is None:
            df.columns = [f"f{i}" for i in range(df.shape[1])]
            tcol = df.columns[target] if isinstance(target, int) else target
        else:
            tcol = target
        full = df.rename(columns={tcol: "target"})

    col2type = _infer_col2type(full, "target", task)
    return full, "target", col2type, f"url:{url.rsplit('/', 1)[-1]}"


def _load_arff_full(entry: dict):
    """Load a real dataset from a backup ARFF file (OpenML-independent).

    Manifest fields (under ``source: arff``):
        arff_url     : direct link to a ``.arff`` (or a ``.zip`` containing one).
        arff_target  : explicit label column name. Defaults to the LAST attribute,
                       which is correct for the OpenML ARFF mirror (class/Class/
                       match), but several author mirrors place the goal first
                       (e.g. ailerons/elevators 'Goal', house_16H 'Price',
                       space_ga 'ln_votes_pop') or use a decoy last column
                       (liver-disorders 'selector' vs the real target 'drinks').

    ARFF preserves the original attribute names, nominal levels, and numeric
    types, so this is the most faithful OpenML-independent source.
    """
    import io
    import urllib.request
    import zipfile

    import arff as liacarff

    task = entry["task_type"]
    url = entry["arff_url"]
    raw = urllib.request.urlopen(url, timeout=180).read()
    if raw[:2] == b"PK":  # zipped ARFF
        zf = zipfile.ZipFile(io.BytesIO(raw))
        members = [n for n in zf.namelist() if "arff" in n.lower() and not n.startswith("__MACOSX")]
        if not members:
            raise ValueError(f"no .arff member in zip at {url!r}")
        raw = zf.read(members[0])

    parsed = liacarff.loads(raw.decode("utf-8", errors="replace"))
    cols = [a[0] for a in parsed["attributes"]]
    df = pd.DataFrame(parsed["data"], columns=cols)
    # liac-arff yields strings for nominal and may leave numeric cols as object
    # when missing values are present; coerce declared-numeric attributes back.
    for aname, atype in parsed["attributes"]:
        if not isinstance(atype, list) and str(atype).upper() in ("NUMERIC", "REAL", "INTEGER"):
            df[aname] = pd.to_numeric(df[aname], errors="coerce")

    tcol = entry.get("arff_target") or cols[-1]
    if tcol not in df.columns:
        raise ValueError(f"arff_target {tcol!r} not found in {cols}")
    full = df.rename(columns={tcol: "target"})
    col2type = _infer_col2type(full, "target", task)
    fname = url.rsplit("/", 1)[-1].split("?")[0]
    return full, "target", col2type, f"arff:{fname}"


def _download_resumable(url: str, dest: str, timeout: int = 120, retries: int = 200) -> str:
    """Download ``url`` to ``dest`` with a resumable, cached, on-disk transfer.

    The data.openml.org object store is reliable, but the local proxy is very
    slow/unstable for larger files (higgs ~10MB trickles at a few KB/s and
    drops repeatedly). We therefore stream to ``dest + '.part'`` and resume
    from its current size via ``Range: bytes=<have>-`` (the CDN answers 206) on
    every reconnect, so each attempt makes forward progress and the transfer
    survives across re-runs. A completed file is reused as-is.
    """
    import os
    import time
    import urllib.request

    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    part = dest + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    total: int | None = None
    last: Exception | None = None
    for attempt in range(retries):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        try:
            headers = {"Accept-Encoding": "identity"}
            if have:
                headers["Range"] = f"bytes={have}-"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                mode = "ab"
                if "Range" in headers and resp.status == 200:
                    mode = "wb"  # server ignored Range; overwrite from scratch
                crange = resp.headers.get("Content-Range")
                if crange and "/" in crange:
                    total = int(crange.rsplit("/", 1)[-1])
                elif total is None:
                    clen = resp.headers.get("Content-Length")
                    total = int(clen) if clen else None
                with open(part, mode) as fh:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        fh.write(chunk)
            if total is None or os.path.getsize(part) >= total:
                os.replace(part, dest)
                return dest
            raise OSError(f"partial {os.path.getsize(part)}/{total}")
        except Exception as exc:  # noqa: BLE001 — resume on any transient error
            last = exc
            if attempt < retries - 1:
                time.sleep(min(1.0 + 0.5 * attempt, 6))
    raise RuntimeError(f"failed to download {url!r} after {retries} tries: {last}")


def _load_openml_parquet_full(entry: dict):
    """Load a real dataset from the OpenML parquet object store (exact fidelity).

    The OpenML data CDN serves the canonical parquet for every dataset id at a
    deterministic path and stays online even while the api.openml.org gateway
    is returning 504. Because it is the *same* file the OpenML API would hand
    out, this reproduces OpenML byte-for-byte: identical rows, column names,
    dtypes, and categorical encodings.

    Manifest fields (under ``source: openml-parquet``):
        source_id      : OpenML data id (used to build the CDN URL).
        openml_target  : explicit source target column. Otherwise, reference_columns
                         must leave exactly one target candidate after declared
                         feature columns are excluded; missing features or an
                         ambiguous target are errors. Without either declaration,
                         the final source column is used.
    """
    import os

    task = entry["task_type"]
    data_id = int(entry["source_id"])
    url = (
        f"https://data.openml.org/datasets/"
        f"{data_id // 10000:04d}/{data_id:04d}/dataset_{data_id}.pq"
    )
    cache_dir = str(STORE_ROOT / ".cache" / "openml")
    dest = os.path.join(cache_dir, f"dataset_{data_id}.pq")
    path = _download_resumable(url, dest)
    df = pd.read_parquet(path)
    # Drop OpenML "row id"/ignore columns that are not real features (e.g. the
    # splice 'Instance_name' identifier), mirroring how OpenML excludes them.
    drop_cols = entry.get("drop_cols") or []
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    # parquet categoricals are fine for features; keep them as-is.
    tcol = entry.get("openml_target")
    reference_columns = entry.get("reference_columns")
    if tcol is None and reference_columns:
        feature_columns = [c for c in reference_columns if c != entry.get("target_col", "target")]
        missing = sorted(set(feature_columns) - set(df.columns))
        candidates = [c for c in df.columns if c not in feature_columns]
        if missing or len(candidates) != 1:
            raise ValueError(
                f"Source columns do not uniquely identify the reference target: "
                f"missing_features={missing}, target_candidates={candidates}. "
                "Set openml_target and drop_cols explicitly after checking the source."
            )
        tcol = candidates[0]
    if tcol is None:
        tcol = df.columns[-1]
    if tcol not in df.columns:
        raise ValueError(f"openml_target {tcol!r} not in {list(df.columns)}")
    full = df.rename(columns={tcol: "target"})
    # Move target to the end for readability (not required downstream).
    full = full[[c for c in full.columns if c != "target"] + ["target"]]
    if reference_columns:
        if set(full.columns) != set(reference_columns):
            raise ValueError("Downloaded columns differ from the dataset contract")
        full = full[reference_columns]
    col2type = _infer_col2type(full, "target", task)
    return full, "target", col2type, f"openml-parquet:{data_id}"


def _infer_col2type(df: pd.DataFrame, target_col: str, task: str) -> dict[str, str]:
    col2type: dict[str, str] = {}
    for col in df.columns:
        if col == target_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            col2type[col] = "numerical"
        else:
            col2type[col] = "categorical"
    col2type[target_col] = "numerical" if task == "regression" else "categorical"
    return col2type


# Protected attributes for the fairness slices (imputation.fairness.group_gap,
# imputation.fairness.downstream_gap).  Declared per dataset because "which
# attribute is protected" is a normative decision the evaluator refuses to guess.
#
# A column is listed only if all three hold: it is a person-level protected
# attribute, it is categorical (the metrics group on the raw value, so a
# continuous column yields subgroups below the 20-row floor), and it has no
# missing/unknown code large enough to survive that floor as a spurious
# subgroup.  The third rule is why allbp.sex (0 = missing, 33 test rows) and
# adult-census native-country: (121 NaN rows) are absent.
GROUP_COLS: dict[str, list[str]] = {
    "adult-census": ["sex:", "race:", "marital-status:"],
    "cleveland": ["sex"],
    "credit_card_clients_default": ["SEX", "MARRIAGE"],
    "E-CommereShippingData": ["Gender"],
    "healthcare_insurance_expenses": ["sex"],
    "primary-tumor": ["sex", "age"],          # age is already banded: <30 / 30-59 / >=60
}


def _real_schema(col2type: dict[str, str], target_col: str, task: str) -> dict[str, Any]:
    numerical = [c for c, t in col2type.items() if t == "numerical"]
    categorical = [c for c, t in col2type.items() if t == "categorical"]
    return {
        "target_col": target_col,
        "task_type": task,
        "numerical_cols": numerical,
        "categorical_cols": categorical,
        "id_cols": [],
        "ci_test": "pearsonr" if task == "regression" else "chi_square",
    }


# ----------------------------------------------------------------------------
# SCM datasets (bnlearn)
# ----------------------------------------------------------------------------
def build_scm(entry: dict, seed: int, test_size: float, n_samples: int) -> dict[str, Any]:
    """Build an SCM dataset from a bnlearn network.

    Uses pgmpy.get_example_model (downloads from bnrepository) + simulate.
    Supports discrete (DiscreteBayesianNetwork) and gaussian
    (LinearGaussianBayesianNetwork). Hybrid CLG networks are not supported by
    pgmpy 1.0; for those (source=='precomputed') we ingest samples + DAG that
    require externally supplied R-simulated samples and a DAG definition.
    """
    task = entry["task_type"]

    if entry.get("source") == "precomputed":
        # Hybrid/CLG networks pgmpy can't build: read R-simulated samples + DAG.
        samples, nodes, edges = _load_scm_precomputed(entry)
    else:
        model = _load_scm_model(entry)
        nodes = list(model.nodes())
        edges = [[u, v] for u, v in model.edges()]
        samples = _simulate(model, n_samples, seed)

    target_col = _resolve_scm_target(entry.get("target_col"), nodes, edges)
    reference_columns = entry.get("reference_columns")
    if reference_columns:
        if set(nodes) != set(reference_columns) or set(samples.columns) != set(reference_columns):
            raise ValueError("SCM nodes differ from the declared dataset contract")
        nodes = list(reference_columns)
    samples = samples[nodes]

    # Infer per-node types from the simulated data (handles discrete/gaussian/mixed).
    col2type = _infer_col2type_scm(samples, target_col, task)
    numerical = [c for c in nodes if col2type[c] == "numerical"]
    categorical = [c for c in nodes if col2type[c] == "categorical"]

    split_task = "classification" if col2type[target_col] == "categorical" else "regression"
    train_df, test_df = _split(samples, target_col, split_task, seed, test_size)

    schema = {
        "target_col": target_col,
        "task_type": task,
        "numerical_cols": numerical,
        "categorical_cols": categorical,
        "id_cols": [],
        # eval discretizes numerics before CI, so chi_square is the universal choice.
        "ci_test": "chi_square",
    }
    schema = entry.get("schema") or schema
    graph = {"target_col": target_col, "nodes": list(nodes), "edges": edges}
    return {"train": train_df, "test": test_df, "schema": schema, "graph": graph, "extra_meta": {}}


def _load_scm_model(entry: dict):
    network = entry["source_id"]
    source = entry.get("source", "pgmpy")
    from pgmpy.utils import get_example_model

    if source == "bnlearn-r":
        # pgmpy cannot build hybrid CLG models; try a local .bif export if given.
        if isinstance(network, str) and os.path.exists(network):
            from pgmpy.readwrite import BIFReader

            return BIFReader(network).get_model()
        raise NotImplementedError(
            f"SCM dataset {entry['name']!r} is a hybrid/CLG bnlearn network that pgmpy "
            f"1.0 cannot construct. Export it from the R `bnlearn` package to a discretized "
            f".bif and set its `source_id` to that file path (source stays 'bnlearn-r')."
        )
    try:
        return get_example_model(network)
    except ValueError as exc:
        raise NotImplementedError(
            f"pgmpy could not load SCM network {network!r}: {exc}. "
            f"If it is a hybrid/CLG network, provide a discretized .bif (source: bnlearn-r)."
        ) from exc


def _load_scm_precomputed(entry: dict):
    """Load R-simulated samples + ground-truth DAG for a CLG/hybrid network.

    Conditional-linear-gaussian networks require externally supplied samples
    from the corresponding R `bnlearn` bn.fit object, stored as
    ``raw/<name>.csv`` plus a DAG export
    ``raw/<name>.arcs.json`` of the form ``{"nodes": [...], "edges": [[u, v]]}``.

    Returns (samples_df, nodes, edges) — mirroring the pgmpy branch in build_scm.
    """
    samples_path = HERE / entry["source_id"]
    graph_path = HERE / entry["graph_id"]
    if not samples_path.is_file() or not graph_path.is_file():
        raise FileNotFoundError(
            f"{entry['name']} requires external SCM samples {samples_path} and "
            f"DAG {graph_path}; these inputs are not included in the code release."
        )
    samples = pd.read_csv(samples_path)
    with (HERE / entry["graph_id"]).open("r", encoding="utf-8") as f:
        graph = json.load(f)
    nodes = graph.get("nodes") or list(samples.columns)
    edges = [list(e) for e in graph.get("edges", [])]
    missing = [n for n in nodes if n not in samples.columns]
    if missing:
        raise ValueError(f"precomputed SCM {entry['name']!r} missing columns {missing} in {entry['source_id']!r}")
    return samples, list(nodes), edges


def _simulate(model, n_samples: int, seed: int):
    """Simulate from a pgmpy model across the differing simulate signatures.

    DiscreteBayesianNetwork.simulate(n_samples=, seed=, show_progress=);
    LinearGaussianBayesianNetwork.simulate(n=, seed=).
    """
    for kwargs in (
        {"n_samples": n_samples, "seed": seed, "show_progress": False},
        {"n_samples": n_samples, "seed": seed},
        {"n": n_samples, "seed": seed},
        {"n": n_samples},
    ):
        try:
            return model.simulate(**kwargs)
        except TypeError:
            continue
    return model.simulate(n_samples)


def _infer_col2type_scm(df: pd.DataFrame, target_col: str, task: str) -> dict[str, str]:
    col2type: dict[str, str] = {}
    for col in df.columns:
        if col == target_col:
            continue
        is_num = pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col])
        # Treat low-cardinality integers as categorical (discrete SCM nodes).
        if is_num and df[col].nunique(dropna=False) > 20:
            col2type[col] = "numerical"
        else:
            col2type[col] = "categorical"
    col2type[target_col] = "numerical" if task == "regression" else "categorical"
    return col2type


def _resolve_scm_target(requested: Optional[str], nodes: list[str], edges: list[list[str]]) -> str:
    if requested and requested in nodes:
        return requested
    if requested:
        raise ValueError(f"Declared SCM target {requested!r} is not a network node")
    # Deterministic sink: node with no outgoing edges, max incoming, name-sorted.
    sources = {u for u, _ in edges}
    in_deg = {n: 0 for n in nodes}
    for _, v in edges:
        in_deg[v] += 1
    sinks = [n for n in nodes if n not in sources] or list(nodes)
    sinks.sort(key=lambda n: (-in_deg[n], n))
    return sinks[0]


# ----------------------------------------------------------------------------
# Shared
# ----------------------------------------------------------------------------
def _split(df: pd.DataFrame, target_col: str, task: str, seed: int, test_size: float):
    stratify = None
    if task == "classification":
        counts = df[target_col].value_counts()
        if (counts >= 2).all() and len(counts) >= 2:
            stratify = df[target_col]
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=stratify
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _constant_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if df[c].nunique(dropna=False) <= 1]


def _md5(*paths: Path) -> str:
    h = hashlib.md5()
    for p in paths:
        h.update(p.read_bytes())
    return h.hexdigest()


def _pkg_version(modname: str) -> Optional[str]:
    try:
        import importlib.metadata as im

        return im.version(modname)
    except Exception:  # noqa: BLE001
        return None


def write_dataset(entry: dict, built: dict[str, Any]) -> Path:
    name = entry["name"]
    dataset_type = entry["dataset_type"]
    out_dir = STORE_ROOT / dataset_type / name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / f"{name}_train.csv"
    test_path = out_dir / f"{name}_test.csv"
    built["schema"] = {**built["schema"], "csv_format": csv_format_for(built["train"])}
    schema = schema_from_dict(built["schema"])
    write_frame_csv(built["train"], train_path, schema)
    write_frame_csv(built["test"], test_path, schema)

    with (out_dir / "schema.json").open("w", encoding="utf-8") as f:
        json.dump(built["schema"], f, ensure_ascii=False, indent=2)

    if built.get("graph") is not None:
        with (out_dir / "graph.json").open("w", encoding="utf-8") as f:
            json.dump(built["graph"], f, ensure_ascii=False, indent=2)

    meta = {
        "name": name,
        "dataset_type": dataset_type,
        "task_type": entry["task_type"],
        "source": entry["source"],
        "source_id": entry.get("source_id"),
        "target_col": built["schema"]["target_col"],
        "columns": list(built["train"].columns),
        "n_train": int(len(built["train"])),
        "n_test": int(len(built["test"])),
        "n_features": int(len(built["train"].columns) - 1),
        "constant_columns_train": _constant_columns(built["train"]),
        "md5": _md5(train_path, test_path),
        "csv_format": built["schema"].get("csv_format", {}),
        "pkg_versions": {
            "tabcamel": _pkg_version("tabcamel"),
            "pgmpy": _pkg_version("pgmpy"),
            "pyarrow": _pkg_version("pyarrow"),
            "pandas": _pkg_version("pandas"),
        },
        **built.get("extra_meta", {}),
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return out_dir


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main() -> None:
    global STORE_ROOT
    parser = argparse.ArgumentParser("Build the OTBench dataset store")
    parser.add_argument("names", nargs="*", help="Dataset names to build (default: all in manifest).")
    parser.add_argument("--type", choices=["scm", "real"], default=None, help="Restrict to a dataset type.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--store", type=Path, default=STORE_ROOT)
    parser.add_argument("--verify-reference", action="store_true",
                        help="Fail if rebuilt split bytes differ from the reference hash.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-scm", type=int, default=20000, help="Samples to draw from each SCM network.")
    args = parser.parse_args()

    STORE_ROOT = args.store.expanduser().resolve()
    manifest = load_manifest(args.manifest)
    selected = []
    for entry in manifest:
        if args.names and entry["name"] not in args.names:
            continue
        if args.type and entry["dataset_type"] != args.type:
            continue
        selected.append(entry)

    if not selected:
        parser.error("No matching datasets in manifest.")

    ok, failed = [], []
    for entry in selected:
        name = entry["name"]
        try:
            print(f">>> building {entry['dataset_type']}/{name} ...")
            if entry["dataset_type"] == "scm":
                built = build_scm(entry, args.seed, args.test_size, args.n_scm)
            else:
                built = build_real(entry, args.seed, args.test_size)
            built.setdefault("extra_meta", {}).update({
                "split_seed": args.seed,
                "test_size": args.test_size,
                "requested_scm_samples": args.n_scm if entry["dataset_type"] == "scm" else None,
            })
            if args.verify_reference and entry.get("reference_split_md5"):
                h = hashlib.md5()
                for split in ("train", "test"):
                    h.update(built[split].to_csv(index=False).encode("utf-8"))
                if h.hexdigest() != entry["reference_split_md5"]:
                    raise ValueError(
                        "Rebuilt split differs from the reference hash; verify the split "
                        "seed, source version, preprocessing, and pandas version."
                    )
                built["extra_meta"]["reference_split_verification"] = {
                    "matched": True, "md5": entry["reference_split_md5"],
                    "representation": "pandas_csv",
                }
            out_dir = write_dataset(entry, built)
            print(f"    OK -> {out_dir}  (train={len(built['train'])}, test={len(built['test'])})")
            ok.append(name)
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"    FAILED {name}: {exc}")
            failed.append(name)

    print(f"\nDone. built={ok}")
    if failed:
        print(f"failed={failed}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
