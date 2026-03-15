"""sqlearn — Compile ML preprocessing pipelines to SQL."""

from sqlearn.core.backend import Backend, DuckDBBackend
from sqlearn.core.compiler import compose_transform
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
from sqlearn.core.io import resolve_input
from sqlearn.core.pipeline import Pipeline
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
    "Backend",
    "ClassificationError",
    "ColumnSelector",
    "CompilationError",
    "DuckDBBackend",
    "FitError",
    "FrozenError",
    "InvalidStepError",
    "MissingColumnError",
    "NotFittedError",
    "Pipeline",
    "ProFeatureError",
    "SQLearnError",
    "Schema",
    "SchemaError",
    "StaticViolationError",
    "Transformer",
    "UnseenCategoryError",
    "boolean",
    "categorical",
    "compose_transform",
    "dtype",
    "matching",
    "numeric",
    "resolve_input",
    "temporal",
]

__version__ = "0.0.1"
