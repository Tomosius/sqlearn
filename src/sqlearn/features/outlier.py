"""OutlierHandler -- clip or remove outliers using IQR or z-score method.

Compiles to inline SQL for clip or query-level WHERE for remove.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlglot.expressions as exp

from sqlearn.core.transformer import Transformer

if TYPE_CHECKING:
    from sqlearn.core.schema import ColumnSelector, Schema


class OutlierHandler(Transformer):
    """Clip or remove outliers from numeric columns via SQL.

    Supports two detection methods:

    - **IQR** (interquartile range): outliers are values below
      ``Q1 - threshold * IQR`` or above ``Q3 + threshold * IQR``.
    - **Z-score**: outliers are values more than ``threshold``
      standard deviations from the mean.

    And two actions:

    - **clip**: cap outlier values to the fence boundaries (inline SQL
      via ``expressions()``).
    - **remove**: filter out rows containing outlier values (query-level
      WHERE via ``query()``).

    **IQR clip** generated SQL::

        SELECT
          GREATEST(LEAST(price, 12.0), -2.0) AS price
        FROM __input__

    **Z-score remove** generated SQL::

        SELECT * FROM (__input__) AS __input__
        WHERE price BETWEEN -5.5 AND 11.5

    Args:
        method: Detection method. ``"iqr"`` (default) or ``"zscore"``.
        action: What to do with outliers. ``"clip"`` (default) caps values
            to fence boundaries. ``"remove"`` filters out outlier rows.
        threshold: Sensitivity threshold. For IQR method, the IQR multiplier
            (default 1.5). For z-score method, the number of standard
            deviations (default 3.0).
        columns: Column specification override. Defaults to all numeric
            columns via ``_default_columns = "numeric"``.

    Raises:
        ValueError: If ``method`` is not ``"iqr"`` or ``"zscore"``,
            or if ``action`` is not ``"clip"`` or ``"remove"``.
        NotFittedError: If ``transform()`` or ``to_sql()`` is called
            before ``fit()``.

    Examples:
        Clip outliers using IQR (default):

        >>> import sqlearn as sq
        >>> pipe = sq.Pipeline([sq.OutlierHandler()])
        >>> pipe.fit("train.parquet")
        >>> result = pipe.transform("test.parquet")

        Remove outlier rows using z-score:

        >>> handler = sq.OutlierHandler(method="zscore", action="remove")
        >>> pipe = sq.Pipeline([handler])
        >>> pipe.fit("data.parquet")

        Custom IQR multiplier (more aggressive clipping):

        >>> handler = sq.OutlierHandler(threshold=1.0)
        >>> pipe = sq.Pipeline([handler])

        Specific columns only:

        >>> handler = sq.OutlierHandler(columns=["price", "quantity"])
        >>> pipe = sq.Pipeline([handler])

    See Also:
        :class:`~sqlearn.scalers.robust.RobustScaler`: Uses IQR for scaling.
        :class:`~sqlearn.ops.filter.Filter`: Manual row filtering.
        :class:`~sqlearn.core.pipeline.Pipeline`: Compose transformers.
    """

    _default_columns: str = "numeric"  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "dynamic"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        method: str = "iqr",
        action: str = "clip",
        threshold: float | None = None,
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        """Initialize OutlierHandler.

        Args:
            method: Detection method (``"iqr"`` or ``"zscore"``).
            action: Action for outliers (``"clip"`` or ``"remove"``).
            threshold: Sensitivity threshold. Default is 1.5 for IQR,
                3.0 for z-score. If None, uses the method's default.
            columns: Column specification override.

        Raises:
            ValueError: If method or action is not recognized.
        """
        super().__init__(columns=columns)
        if method not in ("iqr", "zscore"):
            msg = f"method must be 'iqr' or 'zscore', got {method!r}"
            raise ValueError(msg)
        if action not in ("clip", "remove"):
            msg = f"action must be 'clip' or 'remove', got {action!r}"
            raise ValueError(msg)
        self.method = method
        self.action = action
        if threshold is None:
            self.threshold = 1.5 if method == "iqr" else 3.0
        else:
            self.threshold = threshold

    def discover(
        self,
        columns: list[str],
        schema: Schema,  # noqa: ARG002
        y_column: str | None = None,  # noqa: ARG002
    ) -> dict[str, exp.Expression]:
        """Learn outlier detection statistics per column.

        For IQR method: learns Q1 (25th percentile) and Q3 (75th
        percentile) via ``QUANTILE(col, 0.25)`` and ``QUANTILE(col, 0.75)``.

        For z-score method: learns mean via ``AVG(col)`` and population
        standard deviation via ``STDDEV_POP(col)``.

        Args:
            columns: Target columns to compute statistics for.
            schema: Current table schema (unused, required by protocol).
            y_column: Target column name (unused, required by protocol).

        Returns:
            Dict mapping param names to sqlglot aggregate expressions.
            For IQR: ``'{col}__q1'`` and ``'{col}__q3'``.
            For z-score: ``'{col}__mean'`` and ``'{col}__std'``.
        """
        result: dict[str, exp.Expression] = {}
        for col in columns:
            if self.method == "iqr":
                result[f"{col}__q1"] = exp.Quantile(
                    this=exp.Column(this=col),
                    quantile=exp.Literal.number(0.25),  # pyright: ignore[reportUnknownMemberType]
                )
                result[f"{col}__q3"] = exp.Quantile(
                    this=exp.Column(this=col),
                    quantile=exp.Literal.number(0.75),  # pyright: ignore[reportUnknownMemberType]
                )
            else:  # zscore
                result[f"{col}__mean"] = exp.Avg(this=exp.Column(this=col))
                result[f"{col}__std"] = exp.StddevPop(this=exp.Column(this=col))
        return result

    def _compute_fences(
        self,
        col: str,
    ) -> tuple[float, float]:
        """Compute lower and upper fence values for a column.

        Args:
            col: Column name.

        Returns:
            Tuple of (lower_fence, upper_fence).
        """
        params = self.params_ or {}
        if self.method == "iqr":
            q1 = float(params.get(f"{col}__q1", 0.0))
            q3 = float(params.get(f"{col}__q3", 0.0))
            iqr = q3 - q1
            lower = q1 - self.threshold * iqr
            upper = q3 + self.threshold * iqr
        else:  # zscore
            mean = float(params.get(f"{col}__mean", 0.0))
            std = float(params.get(f"{col}__std", 0.0))
            lower = mean - self.threshold * std
            upper = mean + self.threshold * std
        return lower, upper

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        """Generate inline SQL expressions for clipping outliers.

        For clip action: ``GREATEST(LEAST(col, upper), lower)``
        For remove action: returns empty dict (delegates to ``query()``).

        Args:
            columns: Target columns to transform.
            exprs: Current expression dict for ALL columns from prior steps.

        Returns:
            Dict of modified column expressions. Empty if action is
            ``"remove"`` (handled by ``query()`` instead).
        """
        if self.action == "remove":
            return {}

        result: dict[str, exp.Expression] = {}
        for col in columns:
            lower, upper = self._compute_fences(col)
            # GREATEST(LEAST(col, upper), lower)
            result[col] = exp.Greatest(
                this=exp.Least(
                    this=exprs[col],
                    expressions=[
                        exp.Literal.number(upper),  # pyright: ignore[reportUnknownMemberType]
                    ],
                ),
                expressions=[
                    exp.Literal.number(lower),  # pyright: ignore[reportUnknownMemberType]
                ],
            )
        return result

    def query(
        self,
        input_query: exp.Expression,
    ) -> exp.Expression | None:
        """Generate a WHERE clause to remove outlier rows.

        Only active when ``action="remove"``. Generates
        ``SELECT * FROM (input) WHERE col BETWEEN lower AND upper``
        for each target column, combined with AND.

        Args:
            input_query: The input query to wrap.

        Returns:
            A new sqlglot SELECT with WHERE clause filtering outlier
            rows, or None if action is ``"clip"``.
        """
        if self.action != "remove":
            return None

        if self.columns_ is None:
            return None

        conditions: list[exp.Expression] = []
        for col in self.columns_:
            lower, upper = self._compute_fences(col)
            conditions.append(
                exp.Between(
                    this=exp.Column(this=col),
                    low=exp.Literal.number(lower),  # pyright: ignore[reportUnknownMemberType]
                    high=exp.Literal.number(upper),  # pyright: ignore[reportUnknownMemberType]
                )
            )

        if not conditions:
            return None

        # Combine all conditions with AND
        where_clause = conditions[0]
        for cond in conditions[1:]:
            where_clause = exp.And(this=where_clause, expression=cond)

        return (
            exp.Select(
                expressions=[exp.Star()],
            )
            .from_(  # pyright: ignore[reportUnknownMemberType]
                exp.Subquery(this=input_query, alias="__input__"),
            )
            .where(where_clause)
        )
