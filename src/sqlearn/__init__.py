"""sqlearn — Compile ML preprocessing pipelines to SQL."""

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
from sqlearn.encoders.frequency import FrequencyEncoder
from sqlearn.encoders.hash import HashEncoder
from sqlearn.encoders.onehot import OneHotEncoder
from sqlearn.encoders.ordinal import OrdinalEncoder
from sqlearn.imputers.imputer import Imputer
from sqlearn.ops.cast import Cast
from sqlearn.ops.deduplicate import Deduplicate
from sqlearn.ops.filter import Filter
from sqlearn.ops.rename import Rename
from sqlearn.ops.sample import Sample
from sqlearn.scalers.maxabs import MaxAbsScaler
from sqlearn.scalers.minmax import MinMaxScaler
from sqlearn.scalers.normalizer import Normalizer
from sqlearn.scalers.robust import RobustScaler
from sqlearn.scalers.standard import StandardScaler

__all__ = [
    "Backend",
    "Cast",
    "ClassificationError",
    "ColumnSelector",
    "Columns",
    "CompilationError",
    "Deduplicate",
    "DuckDBBackend",
    "Expression",
    "Filter",
    "FitError",
    "FrequencyEncoder",
    "FrozenError",
    "HashEncoder",
    "Imputer",
    "InvalidStepError",
    "MaxAbsScaler",
    "MinMaxScaler",
    "MissingColumnError",
    "Normalizer",
    "NotFittedError",
    "OneHotEncoder",
    "OrdinalEncoder",
    "Pipeline",
    "ProFeatureError",
    "Rename",
    "RobustScaler",
    "SQLearnError",
    "Sample",
    "Schema",
    "SchemaError",
    "StandardScaler",
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

__version__ = "0.1.0"
