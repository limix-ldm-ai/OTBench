"""Pydantic constraint rules shared by generator plugins, schemas and loaders."""
from __future__ import annotations

import logging
from typing import Any, Dict, Generator, List, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator, validate_call

log = logging.getLogger(__name__)

Rule = Tuple[str, str, Any]

_SUPPORTED_OPS: List[str] = [
    "<", ">=", "<=", ">", "==",
    "lt", "le", "gt", "ge", "eq", "in", "dtype",
]


class Constraints(BaseModel):
    """Constraints on data features (value range / allowed set / dtype).

    rules: list of (feature, op, threshold). ``op`` in
    [<, >=, <=, >, ==, lt, le, gt, ge, eq, in, dtype]; threshold is a scalar,
    a list (for ``in``), or a dtype string (for ``dtype``).
    """

    rules: List[Rule] = []

    @field_validator("rules", mode="before")
    def _validate_rules(cls: Any, rules: List) -> List:  # noqa: N805
        for rule in rules:
            if len(rule) < 3:
                raise ValueError(f"Invalid constraint. Expecting tuple, but got {rule}")
            _feature, op, thresh = rule
            if op not in _SUPPORTED_OPS:
                raise ValueError(f"Invalid operation {op}. Supported ops: {_SUPPORTED_OPS}")
            if op == "in" and not isinstance(thresh, list):
                raise ValueError(f"Invalid type for threshold = {type(thresh)}")
            if op == "dtype" and not isinstance(thresh, str):
                raise ValueError(f"Invalid type for threshold = {type(thresh)}")
        return rules

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def _eval(self, X: pd.DataFrame, feature: str, op: str, operand: Any) -> pd.Index:
        if op in ("lt", "<"):
            return (X[feature] < operand) | X[feature].isna()
        elif op in ("le", "<="):
            return (X[feature] <= operand) | X[feature].isna()
        elif op in ("gt", ">"):
            return (X[feature] > operand) | X[feature].isna()
        elif op in ("ge", ">="):
            return (X[feature] >= operand) | X[feature].isna()
        elif op in ("eq", "=="):
            return (X[feature] == operand) | X[feature].isna()
        elif op == "in":
            return (X[feature].isin(operand)) | X[feature].isna()
        elif op == "dtype":
            return operand in str(X[feature])
        else:
            raise RuntimeError("unsupported operation", op)

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def _correct(self, X: pd.DataFrame, feature: str, op: str, operand: Any) -> pd.DataFrame:
        _filter = self._eval(X, feature, op, operand)
        if op in ("lt", "le", "gt", "ge", "eq", "<", "<=", ">", ">=", "=="):
            X.loc[~_filter, feature] = operand
        return X

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def filter(self, X: pd.DataFrame) -> pd.Series:
        X = pd.DataFrame(X)
        res = pd.Series([True] * len(X), index=X.index)
        for feature, op, thresh in self.rules:
            if feature not in X:
                res &= False
                break
            prev = res.sum()
            res &= self._eval(X, feature, op, thresh)
            if res.sum() < prev:
                log.info(
                    "[%s] quality loss for constraint %s = %s. Remaining %s (prev %s).",
                    feature, op, thresh, res.sum(), prev,
                )
        return res

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def match(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.filter(X)]

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def is_valid(self, X: pd.DataFrame) -> bool:
        return self.filter(X).sum() == len(X)

    def extend(self, other: "Constraints") -> "Constraints":
        self.rules.extend(other.rules)
        return self

    def __len__(self) -> int:
        return len(self.rules)

    def __iter__(self) -> Generator:
        for rule in self.rules:
            yield rule

    def features(self) -> List:
        return list({feature for feature, _, _ in self.rules})

    def feature_constraints(self, ref_feature: str) -> List:
        return [(op, threshold) for feature, op, threshold in self.rules if feature == ref_feature]

    def feature_params(self, feature: str) -> Tuple:
        rules = self.feature_constraints(feature)
        dist_template = "float"
        dist_args: Dict[str, Any] = {"low": np.iinfo(np.int64).min, "high": np.iinfo(np.int64).max}
        for op, value in rules:
            if op == "in":
                dist_template = "categorical"
                if "choices" not in dist_args:
                    dist_args["choices"] = value
                    continue
                dist_args["choices"] = [v for v in value if v in dist_args["choices"]]
            elif op == "dtype" and value in ["int", "int32", "int64", "integer"]:
                dist_template = "integer"
            elif (op in ("le", "<=")) and value < dist_args["high"]:
                dist_args["high"] = value
                if "choices" in dist_args:
                    dist_args["choices"] = [v for v in dist_args["choices"] if v <= value]
            elif (op in ("lt", "<")) and value < dist_args["high"]:
                dist_args["high"] = value - 1
                if "choices" in dist_args:
                    dist_args["choices"] = [v for v in dist_args["choices"] if v < value]
            elif (op in ("ge", ">=")) and dist_args["low"] < value:
                dist_args["low"] = value
                if "choices" in dist_args:
                    dist_args["choices"] = [v for v in dist_args["choices"] if v >= value]
            elif (op in ("gt", ">")) and dist_args["low"] < value:
                dist_args["low"] = value + 1
                if "choices" in dist_args:
                    dist_args["choices"] = [v for v in dist_args["choices"] if v > value]
            elif op in ("eq", "=="):
                dist_args["low"] = value
                dist_args["high"] = value
                dist_args["choices"] = [value]
        return dist_template, dist_args
