"""sqlearn — Compile ML preprocessing pipelines to SQL."""

from sqlearn.core.errors import (
    ClassificationError,
    CompilationError,
    FitError,
    FrozenError,
    InvalidStepError,
    MissingColumnError,
    NotFittedError,
    ProFeatureError,
    SchemaError,
    SQLearnError,
    StaticViolationError,
    UnseenCategoryError,
)
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
from sqlearn.core.transformer import Transformer

__all__ = [
    "ClassificationError",
    "ColumnSelector",
    "CompilationError",
    "FitError",
    "FrozenError",
    "InvalidStepError",
    "MissingColumnError",
    "NotFittedError",
    "ProFeatureError",
    "SQLearnError",
    "Schema",
    "SchemaError",
    "StaticViolationError",
    "Transformer",
    "UnseenCategoryError",
    "boolean",
    "categorical",
    "dtype",
    "matching",
    "numeric",
    "temporal",
]

__version__ = "0.0.1"
