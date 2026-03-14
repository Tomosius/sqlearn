"""sqlearn — Compile ML preprocessing pipelines to SQL."""

from sqlearn.core.schema import (
    ColumnSelector,
    Schema,
    boolean,
    categorical,
    dtype,
    matching,
    numeric,
    temporal,
)

__all__ = [
    "ColumnSelector",
    "Schema",
    "boolean",
    "categorical",
    "dtype",
    "matching",
    "numeric",
    "temporal",
]

__version__ = "0.0.1"
