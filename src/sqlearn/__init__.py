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
from sqlearn.custom import Expression, custom
from sqlearn.encoders.onehot import OneHotEncoder
from sqlearn.imputers.imputer import Imputer
from sqlearn.scalers.standard import StandardScaler

__all__ = [
    "Backend",
    "ClassificationError",
    "ColumnSelector",
    "CompilationError",
    "DuckDBBackend",
    "Expression",
    "FitError",
    "FrozenError",
    "Imputer",
    "InvalidStepError",
    "MissingColumnError",
    "NotFittedError",
    "OneHotEncoder",
    "Pipeline",
    "ProFeatureError",
    "SQLearnError",
    "Schema",
    "SchemaError",
    "StandardScaler",
    "StaticViolationError",
    "Transformer",
    "UnseenCategoryError",
    "boolean",
    "categorical",
    "compose_transform",
    "custom",
    "dtype",
    "matching",
    "numeric",
    "resolve_input",
    "temporal",
]

__version__ = "0.0.1"
