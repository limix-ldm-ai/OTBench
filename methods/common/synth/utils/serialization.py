"""Cloudpickle persistence and dataframe identities for generator internals."""
from typing import Any

import cloudpickle
import pandas as pd


def save(model: Any) -> bytes:
    return cloudpickle.dumps(model)


def load(buff: bytes) -> Any:
    return cloudpickle.loads(buff)


def dataframe_hash(df: pd.DataFrame) -> str:
    """Dataframe hashing, used for caching/backups"""
    cols = sorted(list(df.columns))
    return str(abs(pd.util.hash_pandas_object(df[cols].fillna(0)).sum()))

