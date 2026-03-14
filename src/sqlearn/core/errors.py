"""Error hierarchy for sqlearn.

All sqlearn exceptions inherit from SQLearnError. Error names describe
the problem, not the location. Messages are actionable with guidance.
"""

from __future__ import annotations


class SQLearnError(Exception):
    """Base exception for all sqlearn errors."""


class NotFittedError(SQLearnError):
    """Transform, query, or inspection called before fit().

    Raised when a method requiring fitted state (transform(), to_sql(),
    get_feature_names_out()) is called on an unfitted transformer.
    """


class SchemaError(SQLearnError):
    """Column-related problem: missing columns, type mismatches, conflicts.

    Parent class for MissingColumnError. Catch this to handle all
    schema-related errors uniformly.
    """


class MissingColumnError(SchemaError):
    """Column not found in schema.

    Stores the missing column name and available columns so callers
    can build fuzzy-match suggestions.

    Args:
        message: Error message.
        column: The column name that was not found.
        available: Column names that exist in the schema.
    """

    def __init__(
        self,
        message: str,
        *,
        column: str,
        available: list[str],
    ) -> None:
        super().__init__(message)
        self.column = column
        self.available = available


class FitError(SQLearnError):
    """Data problem discovered during fit.

    Raised for all-NULL columns, empty tables, zero-variance columns,
    and other data issues that prevent fitting.
    """


class CompilationError(SQLearnError):
    """SQL generation failure.

    Raised for unsupported operations in target dialect, expression
    depth overflow, and invalid AST composition.
    """


class UnseenCategoryError(SQLearnError):
    """Encoder encountered unseen category at transform time.

    Raised when a category appears in transform data that was not
    present during fit. Use handle_unknown='ignore' to suppress.

    Args:
        message: Error message.
        column: The column containing the unseen category.
        categories: The unseen category values encountered.
    """

    def __init__(
        self,
        message: str,
        *,
        column: str,
        categories: list[str],
    ) -> None:
        super().__init__(message)
        self.column = column
        self.categories = categories


class ClassificationError(SQLearnError):
    """Static/dynamic classification conflict.

    Raised when a transformer's declared classification contradicts
    its method overrides (e.g. declares static but overrides discover).
    """


class StaticViolationError(SQLearnError):
    """Static transformer accessed data-dependent state.

    Raised when a static transformer attempts to read params_ or sets_
    during compilation.
    """


class FrozenError(SQLearnError):
    """Mutation attempted on a frozen pipeline.

    Raised when fit() is called on a FrozenPipeline. Frozen pipelines
    are immutable and deployment-ready.
    """


class InvalidStepError(SQLearnError):
    """Invalid pipeline step.

    Raised when a Pipeline receives an object that is not a Transformer
    or other valid step type.
    """


class ProFeatureError(SQLearnError):
    """License-gated feature accessed without activation.

    Raised when a Studio Pro feature is used without a valid license.
    Call sq.activate("SQ-XXXX") to unlock Pro features.

    Args:
        message: Error message.
        feature: The name of the gated feature.
    """

    def __init__(
        self,
        message: str,
        *,
        feature: str,
    ) -> None:
        super().__init__(message)
        self.feature = feature
