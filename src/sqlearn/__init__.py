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
from sqlearn.data.concat import concat, concat_query
from sqlearn.data.lookup import Lookup
from sqlearn.data.merge import merge, merge_query
from sqlearn.encoders.frequency import FrequencyEncoder
from sqlearn.encoders.hash import HashEncoder
from sqlearn.encoders.onehot import OneHotEncoder
from sqlearn.encoders.ordinal import OrdinalEncoder
from sqlearn.feature_selection.correlated import DropCorrelated
from sqlearn.feature_selection.drop import Drop
from sqlearn.feature_selection.kbest import SelectKBest
from sqlearn.feature_selection.variance import VarianceThreshold
from sqlearn.features.arithmetic import Abs, Clip, Log, Power, Reciprocal, Round, Sqrt
from sqlearn.features.datetime import DateDiff, DateParts, IsWeekend, Quarter
from sqlearn.features.outlier import OutlierHandler
from sqlearn.features.string import Lower, Replace, StringLength, Substring, Trim, Upper
from sqlearn.features.target import TargetTransform
from sqlearn.features.window import Lag, Lead, Rank, RollingMean, RollingStd, RowNumber
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
    "Abs",
    "Backend",
    "Cast",
    "ClassificationError",
    "Clip",
    "ColumnSelector",
    "Columns",
    "CompilationError",
    "DateDiff",
    "DateParts",
    "Deduplicate",
    "Drop",
    "DropCorrelated",
    "DuckDBBackend",
    "Expression",
    "Filter",
    "FitError",
    "FrequencyEncoder",
    "FrozenError",
    "HashEncoder",
    "Imputer",
    "InvalidStepError",
    "IsWeekend",
    "Lag",
    "Lead",
    "Log",
    "Lookup",
    "Lower",
    "MaxAbsScaler",
    "MinMaxScaler",
    "MissingColumnError",
    "Normalizer",
    "NotFittedError",
    "OneHotEncoder",
    "OrdinalEncoder",
    "OutlierHandler",
    "Pipeline",
    "Power",
    "ProFeatureError",
    "Quarter",
    "Rank",
    "Reciprocal",
    "Rename",
    "Replace",
    "RobustScaler",
    "RollingMean",
    "RollingStd",
    "Round",
    "RowNumber",
    "SQLearnError",
    "Sample",
    "Schema",
    "SchemaError",
    "SelectKBest",
    "Sqrt",
    "StandardScaler",
    "StaticViolationError",
    "StringLength",
    "Substring",
    "TargetTransform",
    "Transformer",
    "Trim",
    "Union",
    "UnseenCategoryError",
    "Upper",
    "VarianceThreshold",
    "all_columns",
    "boolean",
    "categorical",
    "columns",
    "compose_transform",
    "concat",
    "concat_query",
    "custom",
    "dtype",
    "matching",
    "merge",
    "merge_query",
    "numeric",
    "resolve_input",
    "temporal",
]

__version__ = "0.2.0"
