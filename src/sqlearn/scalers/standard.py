"""StandardScaler — standardize numeric columns to zero mean and unit variance.

Compiles to inline SQL: ``(col - mean) / NULLIF(std, 0)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlglot.expressions as exp

from sqlearn.core.transformer import Transformer

if TYPE_CHECKING:
    from sqlearn.core.schema import ColumnSelector, Schema


class StandardScaler(Transformer):
    """Standardize numeric columns to zero mean and unit variance via SQL.

    Each numeric column is transformed to ``(col - mean) / NULLIF(std, 0)``
    where mean and std are the population statistics learned during ``fit()``.
    Safe division via ``NULLIF`` ensures zero-variance columns produce NULL
    instead of division-by-zero errors.

    Generated SQL::

        SELECT
          (price - 3.0) / NULLIF(1.41, 0) AS price,
          (quantity - 30.0) / NULLIF(14.14, 0) AS quantity
        FROM __input__

    Args:
        with_mean: If True (default), center data by subtracting the mean.
        with_std: If True (default), scale data by dividing by the
            population standard deviation.
        columns: Column specification override. Defaults to all numeric
            columns via ``_default_columns = "numeric"``.

    Raises:
        NotFittedError: If ``transform()`` or ``to_sql()`` is called
            before ``fit()``.
        FitError: If the input data has issues that prevent fitting
            (e.g. empty table, all-NULL column).

    Examples:
        Basic usage — standardize all numeric columns:

        >>> import sqlearn as sq
        >>> pipe = sq.Pipeline([sq.StandardScaler()])
        >>> pipe.fit("train.parquet")
        >>> result = pipe.transform("test.parquet")
        >>> sql = pipe.to_sql()

        Center only (no variance scaling):

        >>> scaler = sq.StandardScaler(with_std=False)
        >>> pipe = sq.Pipeline([scaler])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()  # (col - mean) AS col, no division

        Scale specific columns only:

        >>> scaler = sq.StandardScaler(columns=["price", "score"])
        >>> pipe = sq.Pipeline([scaler])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()  # only price and score are scaled

        Compose with Imputer (expressions nest automatically):

        >>> pipe = sq.Pipeline([sq.Imputer(), sq.StandardScaler()])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # (COALESCE(price, 3.0) - 3.0) / NULLIF(1.58, 0) AS price

    See Also:
        :class:`~sqlearn.imputers.imputer.Imputer`: Fill NULLs before scaling.
        :class:`~sqlearn.encoders.onehot.OneHotEncoder`: Encode categoricals.
        :class:`~sqlearn.core.pipeline.Pipeline`: Compose transformers.
    """

    _default_columns: str = "numeric"  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "dynamic"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        with_mean: bool = True,
        with_std: bool = True,
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        super().__init__(columns=columns)
        self.with_mean = with_mean
        self.with_std = with_std

    def discover(
        self,
        columns: list[str],
        schema: Schema,  # noqa: ARG002
        y_column: str | None = None,  # noqa: ARG002
    ) -> dict[str, exp.Expression]:
        """Learn mean and population standard deviation per column.

        Returns sqlglot aggregate expressions that the compiler executes
        as a single batched SQL query during fit().

        Args:
            columns: Target columns to compute statistics for.
            schema: Current table schema (unused, required by protocol).
            y_column: Target column name (unused, required by protocol).

        Returns:
            Dict mapping ``'{col}__mean'`` to ``AVG(col)`` and
            ``'{col}__std'`` to ``STDDEV_POP(col)`` for each column.
            Entries are omitted when ``with_mean`` or ``with_std`` is False.
        """
        result: dict[str, exp.Expression] = {}
        for col in columns:
            if self.with_mean:
                result[f"{col}__mean"] = exp.Avg(this=exp.Column(this=col))
            if self.with_std:
                result[f"{col}__std"] = exp.StddevPop(this=exp.Column(this=col))
        return result

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        """Generate inline SQL expressions for standardization.

        Composes ``(exprs[col] - mean) / NULLIF(std, 0)`` using learned
        params_. Uses ``exprs[col]`` (not ``exp.Column``) to compose
        correctly with prior pipeline steps.

        Args:
            columns: Target columns to transform.
            exprs: Current expression dict for ALL columns from prior steps.

        Returns:
            Dict of modified column expressions. Only includes columns
            that are actually transformed (omits when both flags are False).
        """
        if not self.with_mean and not self.with_std:
            return {}

        params = self.params_ or {}
        result: dict[str, exp.Expression] = {}

        for col in columns:
            expr = exprs[col]

            if self.with_mean:
                mean = params.get(f"{col}__mean", 0.0)
                expr = exp.Paren(
                    this=exp.Sub(
                        this=expr,
                        expression=exp.Literal.number(mean),  # pyright: ignore[reportUnknownMemberType]
                    ),
                )

            if self.with_std:
                std = params.get(f"{col}__std", 1.0)
                expr = exp.Div(
                    this=expr,
                    expression=exp.Nullif(
                        this=exp.Literal.number(std),  # pyright: ignore[reportUnknownMemberType]
                        expression=exp.Literal.number(0),  # pyright: ignore[reportUnknownMemberType]
                    ),
                )

            result[col] = expr

        return result
