"""Polars DataFrame input to ``signapy.discover`` matches pandas input."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import signapy

pl = pytest.importorskip("polars")


def _data():
    rng = np.random.default_rng(0)
    n = 200
    return {
        "dist": rng.normal(40, 8, n),
        "roof": rng.choice(["open", "dome", "retractable"], n),
        "level": rng.choice(["low", "mid", "high"], n),
        "is_made": rng.integers(0, 2, n),
    }


_TYPES = {"dist": "continuous", "roof": "categorical", "level": "ordinal"}


def _pandas(data):
    df = pd.DataFrame(data)
    df["level"] = pd.Categorical(
        df["level"], categories=["low", "mid", "high"], ordered=True
    )
    return df


def _polars(data):
    return pl.DataFrame(data).with_columns(
        pl.col("level").cast(pl.Enum(["low", "mid", "high"]))
    )


def test_results_match_pandas():
    data = _data()
    kwargs = {"target": "is_made", "positive_class": 1, "feature_types": _TYPES}
    expected = signapy.discover(_pandas(data), **kwargs)
    actual = signapy.discover(_polars(data), **kwargs)
    assert actual == expected


def test_nulls_are_feature_specific_and_match_pandas():
    data = _data()
    pdf, pldf = _pandas(data), _polars(data)
    pdf.loc[:19, "dist"] = np.nan
    pdf.loc[10:29, "roof"] = None
    pdf.loc[:4, "is_made"] = np.nan
    pldf = pldf.with_columns(
        pl.when(pl.int_range(pl.len()) < 20)
        .then(None)
        .otherwise(pl.col("dist"))
        .alias("dist"),
        pl.when(pl.int_range(pl.len()).is_between(10, 29))
        .then(None)
        .otherwise(pl.col("roof"))
        .alias("roof"),
        pl.when(pl.int_range(pl.len()) < 5)
        .then(None)
        .otherwise(pl.col("is_made"))
        .alias("is_made"),
    )
    kwargs = {"target": "is_made", "positive_class": 1, "feature_types": _TYPES}
    expected = signapy.discover(pdf, **kwargs)
    actual = signapy.discover(pldf, **kwargs)
    assert actual == expected
    assert actual.feature("dist").n != actual.feature("roof").n


def test_does_not_mutate_input():
    df = _polars(_data())
    before = df.clone()
    signapy.discover(df, target="is_made", positive_class=1, feature_types=_TYPES)
    assert df.equals(before)
    assert df.schema == before.schema


def test_does_not_import_pyarrow():
    code = textwrap.dedent(
        """
        import importlib.abc
        import sys

        import signapy
        import polars as pl

        class BlockPyArrow(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == "pyarrow" or fullname.startswith("pyarrow."):
                    raise ImportError("PyArrow import blocked by test")
                return None

        for name in tuple(sys.modules):
            if name == "pyarrow" or name.startswith("pyarrow."):
                del sys.modules[name]
        sys.meta_path.insert(0, BlockPyArrow())

        df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "t": [1, 0, 1, 0]})
        signapy.discover(
            df,
            target="t",
            positive_class=1,
            feature_types={"a": "continuous"},
        )
        assert "pyarrow" not in sys.modules
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_nullable_integer_target_preserves_large_labels():
    low = 2**53
    high = low + 1
    df = pl.DataFrame(
        {
            "a": ["x", "x", "y", "y", "x", "y", "x", "y", "ignored"],
            "t": [low, high, low, high, low, high, low, high, None],
        },
        schema={"a": pl.String, "t": pl.Int64},
    )
    report = signapy.discover(df, target="t", positive_class=high)
    assert report.feature("a").n == 8
    assert {value.value for value in report.feature("a").values} == {"x", "y"}


def test_default_analyzes_string_and_categorical_columns():
    df = pl.DataFrame(
        {
            "a": ["x", "y", "x", "y", "x", "y"],
            "b": ["p", "p", "q", "q", "p", "q"],
            "t": [1, 0, 1, 0, 1, 1],
        }
    ).with_columns(pl.col("b").cast(pl.Categorical))
    report = signapy.discover(df, target="t", positive_class=1)
    assert [f.feature for f in report.features] == ["a", "b"]


def test_numeric_columns_still_unsupported_as_categorical():
    df = pl.DataFrame({"a": [1, 2, 1, 2], "t": [1, 0, 1, 0]})
    with pytest.raises(ValueError, match="unsupported dtype"):
        signapy.discover(df, target="t", positive_class=1)


def test_ordinal_requires_ordered_enum():
    df = pl.DataFrame({"a": ["x", "y", "x", "y"], "t": [1, 0, 1, 0]})
    with pytest.raises(ValueError, match="ordinal"):
        signapy.discover(
            df, target="t", positive_class=1, feature_types={"a": "ordinal"}
        )
    df = df.with_columns(pl.col("a").cast(pl.Categorical))
    with pytest.raises(ValueError, match="unordered"):
        signapy.discover(
            df, target="t", positive_class=1, feature_types={"a": "ordinal"}
        )


def test_boolean_feature_rejected_as_continuous_even_with_nulls():
    df = pl.DataFrame(
        {"a": [True, False, None, True, False, True], "t": [1, 0, 1, 0, 1, 1]}
    )
    with pytest.raises(ValueError, match="boolean"):
        signapy.discover(
            df, target="t", positive_class=1, feature_types={"a": "continuous"}
        )
    with pytest.raises(ValueError, match="unsupported dtype"):
        signapy.discover(
            df, target="t", positive_class=1, feature_types={"a": "categorical"}
        )


def test_integer_feature_with_nulls_and_boolean_target():
    df = pl.DataFrame(
        {
            "a": [1, 2, None, 4, 5, 6, 2, 9],
            "t": [True, False, True, False, True, True, False, False],
        }
    )
    report = signapy.discover(
        df, target="t", positive_class=True, feature_types={"a": "continuous"}
    )
    assert report.feature("a").n == 7


def test_missing_columns_report_original_columns():
    df = pl.DataFrame({"a": ["x", "y"], "t": [1, 0]})
    with pytest.raises(ValueError, match="not found"):
        signapy.discover(df, target="zzz", positive_class=1)
    with pytest.raises(ValueError, match="not a column"):
        signapy.discover(
            df, target="t", positive_class=1, feature_types={"zzz": "categorical"}
        )


def test_other_types_raise_type_error():
    with pytest.raises(TypeError, match="pandas or Polars"):
        signapy.discover({"t": [1, 0]}, target="t", positive_class=1)
