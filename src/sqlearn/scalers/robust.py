"""RobustScaler — scale numeric columns using median and interquartile range.

Compiles to inline SQL: ``(col - median) / NULLIF(Q3 - Q1, 0)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlglot.expressions as exp

from sqlearn.core.transformer import Transformer

if TYPE_CHECKING:
    from sqlearn.core.schema import ColumnSelector, Schema


class RobustScaler(Transformer):
    """Scale numeric columns using statistics robust to outliers via SQL.

    Each numeric column is transformed to
    ``(col - median) / NULLIF(Q3 - Q1, 0)`` where median, Q1, and Q3 are
    learned during ``fit()``. The interquartile range (IQR = Q3 - Q1) is
    robust to outliers, unlike variance-based scaling. Safe division via
    ``NULLIF`` ensures zero-IQR columns produce NULL instead of
    division-by-zero errors.

    Generated SQL::

        SELECT
          (price - 5.0) / NULLIF(8.0 - 2.0, 0) AS price,
          (quantity - 30.0) / NULLIF(45.0 - 15.0, 0) AS quantity
        FROM __input__

    Args:
        with_centering: If True (default), center data by subtracting the
            median.
        with_scaling: If True (default), scale data by dividing by the
            interquartile range (Q3 - Q1).
        quantile_range: Tuple of (lower quantile, upper quantile) used to
            compute the IQR. Defaults to ``(25.0, 75.0)`` for the standard
            interquartile range. Values must be in the range [0, 100].
        columns: Column specification override. Defaults to all numeric
            columns via ``_default_columns = "numeric"``.

    Raises:
        NotFittedError: If ``transform()`` or ``to_sql()`` is called
            before ``fit()``.
        FitError: If the input data has issues that prevent fitting
            (e.g. empty table, all-NULL column).

    Examples:
        Basic usage — scale all numeric columns robustly:

        >>> import sqlearn as sq
        >>> pipe = sq.Pipeline([sq.RobustScaler()])
        >>> pipe.fit("train.parquet")
        >>> result = pipe.transform("test.parquet")
        >>> sql = pipe.to_sql()

        Center only (no IQR scaling):

        >>> scaler = sq.RobustScaler(with_scaling=False)
        >>> pipe = sq.Pipeline([scaler])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()  # (col - median) AS col, no division

        Custom quantile range (10th to 90th percentile):

        >>> scaler = sq.RobustScaler(quantile_range=(10.0, 90.0))
        >>> pipe = sq.Pipeline([scaler])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()  # divides by P90 - P10 instead of Q3 - Q1

        Scale specific columns only:

        >>> scaler = sq.RobustScaler(columns=["price", "score"])
        >>> pipe = sq.Pipeline([scaler])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()  # only price and score are scaled

        Compose with Imputer (expressions nest automatically):

        >>> pipe = sq.Pipeline([sq.Imputer(), sq.RobustScaler()])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # (COALESCE(price, 5.0) - 5.0) / NULLIF(8.0 - 2.0, 0) AS price

    See Also:
        :class:`~sqlearn.scalers.standard.StandardScaler`: Variance-based scaling.
        :class:`~sqlearn.imputers.imputer.Imputer`: Fill NULLs before scaling.
        :class:`~sqlearn.core.pipeline.Pipeline`: Compose transformers.
    """

    _default_columns: str = "numeric"  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "dynamic"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        with_centering: bool = True,
        with_scaling: bool = True,
        quantile_range: tuple[float, float] = (25.0, 75.0),
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        super().__init__(columns=columns)
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range

    def discover(
        self,
        columns: list[str],
        schema: Schema,  # noqa: ARG002
        y_column: str | None = None,  # noqa: ARG002
    ) -> dict[str, exp.Expression]:
        """Learn median and quantile boundaries per column.

        Returns sqlglot aggregate expressions that the compiler executes
        as a single batched SQL query during fit().

        Args:
            columns: Target columns to compute statistics for.
            schema: Current table schema (unused, required by protocol).
            y_column: Target column name (unused, required by protocol).

        Returns:
            Dict mapping ``'{col}__median'`` to ``MEDIAN(col)``,
            ``'{col}__q1'`` to ``QUANTILE(col, q_low/100)``, and
            ``'{col}__q3'`` to ``QUANTILE(col, q_high/100)`` for each
            column. Entries are omitted when ``with_centering`` or
            ``with_scaling`` is False.
        """
        q_low, q_high = self.quantile_range
        result: dict[str, exp.Expression] = {}
        for col in columns:
            if self.with_centering:
                result[f"{col}__median"] = exp.Median(this=exp.Column(this=col))
            if self.with_scaling:
                result[f"{col}__q1"] = exp.Quantile(
                    this=exp.Column(this=col),
                    quantile=exp.Literal.number(q_low / 100),  # pyright: ignore[reportUnknownMemberType]
                )
                result[f"{col}__q3"] = exp.Quantile(
                    this=exp.Column(this=col),
                    quantile=exp.Literal.number(q_high / 100),  # pyright: ignore[reportUnknownMemberType]
                )
        return result

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        """Generate inline SQL expressions for robust scaling.

        Composes ``(exprs[col] - median) / NULLIF(Q3 - Q1, 0)`` using
        learned params_. Uses ``exprs[col]`` (not ``exp.Column``) to
        compose correctly with prior pipeline steps.

        Args:
            columns: Target columns to transform.
            exprs: Current expression dict for ALL columns from prior steps.

        Returns:
            Dict of modified column expressions. Only includes columns
            that are actually transformed (omits when both flags are False).
        """
        if not self.with_centering and not self.with_scaling:
            return {}

        params = self.params_ or {}
        result: dict[str, exp.Expression] = {}

        for col in columns:
            expr = exprs[col]

            if self.with_centering:
                median = params.get(f"{col}__median", 0.0)
                expr = exp.Paren(
                    this=exp.Sub(
                        this=expr,
                        expression=exp.Literal.number(median),  # pyright: ignore[reportUnknownMemberType]
                    ),
                )

            if self.with_scaling:
                q1 = params.get(f"{col}__q1", 0.0)
                q3 = params.get(f"{col}__q3", 1.0)
                expr = exp.Div(
                    this=expr,
                    expression=exp.Nullif(
                        this=exp.Sub(
                            this=exp.Literal.number(q3),  # pyright: ignore[reportUnknownMemberType]
                            expression=exp.Literal.number(q1),  # pyright: ignore[reportUnknownMemberType]
                        ),
                        expression=exp.Literal.number(0),  # pyright: ignore[reportUnknownMemberType]
                    ),
                )

            result[col] = expr

        return result
