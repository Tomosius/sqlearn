"""MaxAbsScaler — scale numeric columns by their maximum absolute value.

Compiles to inline SQL: ``col / NULLIF(max_abs, 0)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlglot.expressions as exp

from sqlearn.core.transformer import Transformer

if TYPE_CHECKING:
    from sqlearn.core.schema import ColumnSelector, Schema


class MaxAbsScaler(Transformer):
    """Scale numeric columns to [-1, 1] by dividing by the max absolute value.

    Each numeric column is transformed to ``col / NULLIF(max_abs, 0)``
    where max_abs is ``MAX(ABS(col))`` learned during ``fit()``.
    Safe division via ``NULLIF`` ensures zero-max columns produce NULL
    instead of division-by-zero errors.

    Generated SQL::

        SELECT
          price / NULLIF(99.5, 0) AS price,
          quantity / NULLIF(500.0, 0) AS quantity
        FROM __input__

    Args:
        columns: Column specification override. Defaults to all numeric
            columns via ``_default_columns = "numeric"``.

    Raises:
        NotFittedError: If ``transform()`` or ``to_sql()`` is called
            before ``fit()``.
        FitError: If the input data has issues that prevent fitting
            (e.g. empty table, all-NULL column).

    Examples:
        Basic usage — scale all numeric columns:

        >>> import sqlearn as sq
        >>> pipe = sq.Pipeline([sq.MaxAbsScaler()])
        >>> pipe.fit("train.parquet")
        >>> result = pipe.transform("test.parquet")
        >>> sql = pipe.to_sql()

        Scale specific columns only:

        >>> scaler = sq.MaxAbsScaler(columns=["price", "score"])
        >>> pipe = sq.Pipeline([scaler])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()  # only price and score are scaled

        Compose with Imputer (expressions nest automatically):

        >>> pipe = sq.Pipeline([sq.Imputer(), sq.MaxAbsScaler()])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # COALESCE(price, 3.0) / NULLIF(99.5, 0) AS price

    See Also:
        :class:`~sqlearn.scalers.standard.StandardScaler`: Zero mean, unit variance.
        :class:`~sqlearn.imputers.imputer.Imputer`: Fill NULLs before scaling.
        :class:`~sqlearn.core.pipeline.Pipeline`: Compose transformers.
    """

    _default_columns: str = "numeric"  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "dynamic"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        super().__init__(columns=columns)

    def discover(
        self,
        columns: list[str],
        schema: Schema,  # noqa: ARG002
        y_column: str | None = None,  # noqa: ARG002
    ) -> dict[str, exp.Expression]:
        """Learn the maximum absolute value per column.

        Returns sqlglot aggregate expressions that the compiler executes
        as a single batched SQL query during fit().

        Args:
            columns: Target columns to compute statistics for.
            schema: Current table schema (unused, required by protocol).
            y_column: Target column name (unused, required by protocol).

        Returns:
            Dict mapping ``'{col}__max_abs'`` to ``MAX(ABS(col))``
            for each column.
        """
        result: dict[str, exp.Expression] = {}
        for col in columns:
            result[f"{col}__max_abs"] = exp.Max(
                this=exp.Abs(this=exp.Column(this=col)),
            )
        return result

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        """Generate inline SQL expressions for max-absolute scaling.

        Composes ``exprs[col] / NULLIF(max_abs, 0)`` using learned
        params_. Uses ``exprs[col]`` (not ``exp.Column``) to compose
        correctly with prior pipeline steps.

        Args:
            columns: Target columns to transform.
            exprs: Current expression dict for ALL columns from prior steps.

        Returns:
            Dict of modified column expressions for each scaled column.
        """
        params = self.params_ or {}
        result: dict[str, exp.Expression] = {}

        for col in columns:
            max_abs = params.get(f"{col}__max_abs", 1.0)
            result[col] = exp.Div(
                this=exprs[col],
                expression=exp.Nullif(
                    this=exp.Literal.number(max_abs),  # pyright: ignore[reportUnknownMemberType]
                    expression=exp.Literal.number(0),  # pyright: ignore[reportUnknownMemberType]
                ),
            )

        return result
