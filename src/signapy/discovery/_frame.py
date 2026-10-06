"""Dataframe adapter: lets ``discover`` accept a Polars ``DataFrame``.

SignaPy's analysis layer is pandas-based. Rather than converting through
``DataFrame.to_pandas()`` (which requires PyArrow), selected Polars columns
are rebuilt as pandas Series via ``to_numpy()`` / ``to_list()``, which need
neither PyArrow nor any other optional dependency. The Polars frame is only
read, never mutated. Polars itself is never imported unless the caller
already passed a Polars object.

Dtype mapping (Polars -> pandas), chosen so that each column lands on the
same pandas dtype a user would get from the equivalent pandas workflow, and
so SignaPy's existing dtype validation behaves identically:

- integer / float: NumPy numeric array. Nulls (and NaNs) become ``NaN``, so
  an integer column containing nulls becomes ``float64``.
- ``String``: ``object`` dtype (``None`` for nulls).
- ``Categorical``: unordered pandas ``category``.
- ``Enum``: *ordered* pandas ``category`` in the Enum's declared order, so
  an Enum column can be declared ORDINAL directly.
- ``Boolean``: ``bool``, or nullable ``boolean`` if it contains nulls
  (never ``object``, so it stays unsupported as categorical/continuous
  exactly like a pandas boolean column).
- ``Date`` / ``Datetime`` / ``Duration``: ``datetime64`` / ``timedelta64``.
- anything else: ``object`` dtype.
"""

from __future__ import annotations

from collections.abc import Container
from typing import Any

import pandas as pd


def is_polars_dataframe(obj: object) -> bool:
    """Whether ``obj`` is a Polars ``DataFrame``, without importing Polars."""
    cls = type(obj)
    return cls.__name__ == "DataFrame" and cls.__module__.split(".")[0] == "polars"


def is_polars_lazyframe(obj: object) -> bool:
    cls = type(obj)
    return cls.__name__ == "LazyFrame" and cls.__module__.split(".")[0] == "polars"


def _convert_series(series: Any) -> pd.Series:
    import polars as pl

    dtype = series.dtype
    if isinstance(dtype, pl.Enum):
        values: Any = pd.Categorical(
            series.to_list(), categories=dtype.categories.to_list(), ordered=True
        )
    elif isinstance(dtype, pl.Categorical):
        values = pd.Categorical(series.to_list())
    elif dtype == pl.Boolean:
        values = (
            pd.array(series.to_list(), dtype="boolean")
            if series.null_count()
            else series.to_numpy()
        )
    elif (dtype.is_numeric() and dtype != pl.Decimal) or (
        dtype.is_temporal() and dtype != pl.Time
    ):
        values = series.to_numpy()
    else:
        values = pd.Series(series.to_list(), dtype=object)
    return pd.Series(values, name=series.name)


def polars_to_pandas(df: Any, columns: Container[Any]) -> pd.DataFrame:
    """Convert the ``columns`` of Polars ``df`` to a pandas DataFrame.

    Column order follows ``df``. Columns not in ``columns`` are skipped
    entirely (never converted), so unused columns cost nothing.
    """
    return pd.concat(
        [_convert_series(df[name]) for name in df.columns if name in columns],
        axis=1,
    )
