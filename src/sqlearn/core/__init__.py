"""sqlearn core — Transformer base, Pipeline, Compiler, Backend, Schema."""

from sqlearn.core.backend import Backend, DuckDBBackend
from sqlearn.core.columns import Columns
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
    all_columns,
    boolean,
    categorical,
    columns,
    dtype,
    matching,
    numeric,
    temporal,
)
from sqlearn.core.transformer import Transformer
from sqlearn.core.union import Union
from sqlearn.custom import Expression, custom

__all__ = [
    "Backend",
    "ClassificationError",
    "ColumnSelector",
    "Columns",
    "CompilationError",
    "DuckDBBackend",
    "Expression",
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
    "Union",
    "UnseenCategoryError",
    "all_columns",
    "boolean",
    "categorical",
    "columns",
    "compose_transform",
    "custom",
    "dtype",
    "matching",
    "numeric",
    "resolve_input",
    "temporal",
]
