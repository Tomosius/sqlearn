"""MinMaxScaler — scale numeric columns to a given range.

Compiles to inline SQL: ``(col - min) / NULLIF(max - min, 0) * (max_val - min_val) + min_val``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlglot.expressions as exp

from sqlearn.core.transformer import Transformer

if TYPE_CHECKING:
    from sqlearn.core.schema import ColumnSelector, Schema


class MinMaxScaler(Transformer):
    """Scale numeric columns to a given range via SQL.

    Each numeric column is transformed to
    ``(col - min) / NULLIF(max - min, 0) * (max_val - min_val) + min_val``
    where min and max are the column statistics learned during ``fit()``.
    Safe division via ``NULLIF`` ensures zero-range columns produce NULL
    instead of division-by-zero errors.

    With the default ``feature_range=(0, 1)``, the formula simplifies to
    ``(col - min) / NULLIF(max - min, 0)``.

    Generated SQL::

        SELECT
          (price - 1.0) / NULLIF(10.0 - 1.0, 0) AS price,
          (quantity - 5.0) / NULLIF(100.0 - 5.0, 0) AS quantity
        FROM __input__

    With ``feature_range=(-1, 1)``::

        SELECT
          (price - 1.0) / NULLIF(10.0 - 1.0, 0) * 2 + -1 AS price
        FROM __input__

    Args:
        feature_range: Desired range of transformed data as ``(min, max)``.
            Defaults to ``(0, 1)``.
        columns: Column specification override. Defaults to all numeric
            columns via ``_default_columns = "numeric"``.

    Raises:
        NotFittedError: If ``transform()`` or ``to_sql()`` is called
            before ``fit()``.
        FitError: If the input data has issues that prevent fitting
            (e.g. empty table, all-NULL column).
        ValueError: If ``feature_range[0] >= feature_range[1]``.

    Examples:
        Basic usage — scale all numeric columns to [0, 1]:

        >>> import sqlearn as sq
        >>> pipe = sq.Pipeline([sq.MinMaxScaler()])
        >>> pipe.fit("train.parquet")
        >>> result = pipe.transform("test.parquet")
        >>> sql = pipe.to_sql()

        Scale to a custom range [-1, 1]:

        >>> scaler = sq.MinMaxScaler(feature_range=(-1, 1))
        >>> pipe = sq.Pipeline([scaler])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # (col - min) / NULLIF(max - min, 0) * 2 + -1 AS col

        Scale specific columns only:

        >>> scaler = sq.MinMaxScaler(columns=["price", "score"])
        >>> pipe = sq.Pipeline([scaler])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()  # only price and score are scaled

        Compose with Imputer (expressions nest automatically):

        >>> pipe = sq.Pipeline([sq.Imputer(), sq.MinMaxScaler()])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # (COALESCE(price, 3.0) - 1.0) / NULLIF(10.0 - 1.0, 0) AS price

    See Also:
        :class:`~sqlearn.scalers.standard.StandardScaler`: Standardize to zero
            mean and unit variance.
        :class:`~sqlearn.imputers.imputer.Imputer`: Fill NULLs before scaling.
        :class:`~sqlearn.core.pipeline.Pipeline`: Compose transformers.
    """

    _default_columns: str = "numeric"  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "dynamic"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        feature_range: tuple[float, float] = (0, 1),
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        super().__init__(columns=columns)
        if feature_range[0] >= feature_range[1]:
            msg = f"feature_range[0] must be less than feature_range[1], got {feature_range}"
            raise ValueError(msg)
        self.feature_range = feature_range

    def discover(
        self,
        columns: list[str],
        schema: Schema,  # noqa: ARG002
        y_column: str | None = None,  # noqa: ARG002
    ) -> dict[str, exp.Expression]:
        """Learn min and max per column.

        Returns sqlglot aggregate expressions that the compiler executes
        as a single batched SQL query during fit().

        Args:
            columns: Target columns to compute statistics for.
            schema: Current table schema (unused, required by protocol).
            y_column: Target column name (unused, required by protocol).

        Returns:
            Dict mapping ``'{col}__min'`` to ``MIN(col)`` and
            ``'{col}__max'`` to ``MAX(col)`` for each column.
        """
        result: dict[str, exp.Expression] = {}
        for col in columns:
            result[f"{col}__min"] = exp.Min(this=exp.Column(this=col))
            result[f"{col}__max"] = exp.Max(this=exp.Column(this=col))
        return result

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        """Generate inline SQL expressions for min-max scaling.

        Composes ``(exprs[col] - min) / NULLIF(max - min, 0)`` using learned
        params_, optionally scaled to ``feature_range``. Uses ``exprs[col]``
        (not ``exp.Column``) to compose correctly with prior pipeline steps.

        Args:
            columns: Target columns to transform.
            exprs: Current expression dict for ALL columns from prior steps.

        Returns:
            Dict of modified column expressions for each transformed column.
        """
        params = self.params_ or {}
        result: dict[str, exp.Expression] = {}
        min_val, max_val = self.feature_range

        for col in columns:
            col_min = params.get(f"{col}__min", 0.0)
            col_max = params.get(f"{col}__max", 1.0)

            # (exprs[col] - min) / NULLIF(max - min, 0)
            expr: exp.Expression = exp.Div(
                this=exp.Paren(
                    this=exp.Sub(
                        this=exprs[col],
                        expression=exp.Literal.number(col_min),  # pyright: ignore[reportUnknownMemberType]
                    ),
                ),
                expression=exp.Nullif(
                    this=exp.Sub(
                        this=exp.Literal.number(col_max),  # pyright: ignore[reportUnknownMemberType]
                        expression=exp.Literal.number(col_min),  # pyright: ignore[reportUnknownMemberType]
                    ),
                    expression=exp.Literal.number(0),  # pyright: ignore[reportUnknownMemberType]
                ),
            )

            # Scale to feature_range if not default [0, 1]
            if min_val != 0 or max_val != 1:
                scale = max_val - min_val
                expr = exp.Add(
                    this=exp.Mul(
                        this=expr,
                        expression=exp.Literal.number(scale),  # pyright: ignore[reportUnknownMemberType]
                    ),
                    expression=exp.Literal.number(min_val),  # pyright: ignore[reportUnknownMemberType]
                )

            result[col] = expr

        return result
