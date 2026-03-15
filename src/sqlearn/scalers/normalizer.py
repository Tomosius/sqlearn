"""Normalizer — normalize each row to unit norm across target columns.

Compiles to inline SQL: ``col / NULLIF(norm, 0)`` where norm is computed
across all target columns per row (L1, L2, or max norm).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

import sqlglot.expressions as exp

from sqlearn.core.transformer import Transformer

if TYPE_CHECKING:
    from sqlearn.core.schema import ColumnSelector

_VALID_NORMS = ("l1", "l2", "max")


class Normalizer(Transformer):
    """Normalize each row to unit norm across target columns via SQL.

    Unlike per-column scalers (StandardScaler, MinMaxScaler), Normalizer
    operates across columns within each row. It computes a norm from all
    target columns and divides each column by that norm. This is useful
    when the relative proportions between features matter more than their
    absolute values (e.g. TF-IDF vectors, text classification).

    This is a **static** transformer --- no ``fit()`` statistics are needed.
    The norm is computed inline from the row values themselves.

    Supported norms:

    - **L2** (default): Euclidean norm.
      ``x_col = x_col / SQRT(x1^2 + x2^2 + ... + xN^2)``
    - **L1**: Manhattan norm.
      ``x_col = x_col / (|x1| + |x2| + ... + |xN|)``
    - **max**: Maximum absolute value.
      ``x_col = x_col / MAX(|x1|, |x2|, ..., |xN|)``

    Safe division via ``NULLIF`` ensures zero-norm rows produce NULL
    instead of division-by-zero errors.

    Generated SQL (L2 norm, two columns)::

        SELECT
          price / NULLIF(SQRT(price * price + quantity * quantity), 0) AS price,
          quantity / NULLIF(SQRT(price * price + quantity * quantity), 0) AS quantity
        FROM __input__

    Args:
        norm: Norm to use. One of ``'l1'``, ``'l2'``, or ``'max'``.
            Defaults to ``'l2'``.
        columns: Column specification override. Defaults to all numeric
            columns via ``_default_columns = "numeric"``.

    Raises:
        ValueError: If ``norm`` is not one of ``'l1'``, ``'l2'``, ``'max'``.

    Examples:
        Basic usage --- L2-normalize all numeric columns:

        >>> import sqlearn as sq
        >>> pipe = sq.Pipeline([sq.Normalizer()])
        >>> pipe.fit("train.parquet")
        >>> sql = pipe.to_sql()
        ... # price / NULLIF(SQRT(price * price + quantity * quantity), 0)

        L1 normalization:

        >>> pipe = sq.Pipeline([sq.Normalizer(norm="l1")])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # price / NULLIF(ABS(price) + ABS(quantity), 0)

        Max normalization:

        >>> pipe = sq.Pipeline([sq.Normalizer(norm="max")])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # price / NULLIF(GREATEST(ABS(price), ABS(quantity)), 0)

        Normalize specific columns only:

        >>> pipe = sq.Pipeline([sq.Normalizer(columns=["x", "y", "z"])])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()  # only x, y, z are normalized

        Compose with StandardScaler (expressions nest automatically):

        >>> pipe = sq.Pipeline([sq.StandardScaler(), sq.Normalizer()])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # nested: standardized values are then row-normalized

    See Also:
        :class:`~sqlearn.scalers.standard.StandardScaler`: Per-column
            standardization to zero mean and unit variance.
        :class:`~sqlearn.core.pipeline.Pipeline`: Compose transformers.
    """

    _default_columns: str = "numeric"  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "static"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        norm: str = "l2",
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        super().__init__(columns=columns)
        if norm not in _VALID_NORMS:
            msg = f"Invalid norm {norm!r}. Must be one of {_VALID_NORMS!r}."
            raise ValueError(msg)
        self.norm = norm

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        """Generate inline SQL expressions for row-wise normalization.

        Builds a norm expression from all target columns, then divides
        each column by that norm. The norm expression is ``.copy()``-ed
        for each column to avoid sqlglot AST node sharing issues.

        Args:
            columns: Target columns to normalize.
            exprs: Current expression dict for ALL columns from prior steps.

        Returns:
            Dict of modified column expressions. Each target column is
            divided by its row norm.
        """
        if len(columns) == 0:
            return {}

        norm_expr = self._build_norm(columns, exprs)

        result: dict[str, exp.Expression] = {}
        for col in columns:
            result[col] = exp.Div(
                this=exprs[col],
                expression=exp.Nullif(
                    this=norm_expr.copy(),
                    expression=exp.Literal.number(0),  # pyright: ignore[reportUnknownMemberType]
                ),
            )
        return result

    def _build_norm(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> exp.Expression:
        """Build the norm expression across all target columns.

        Args:
            columns: Target columns contributing to the norm.
            exprs: Current expression dict for ALL columns.

        Returns:
            A sqlglot expression representing the row norm.
        """
        if self.norm == "l2":
            return self._build_l2(columns, exprs)
        if self.norm == "l1":
            return self._build_l1(columns, exprs)
        return self._build_max(columns, exprs)

    def _build_l2(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> exp.Expression:
        """Build L2 (Euclidean) norm: SQRT(x1^2 + x2^2 + ...).

        Args:
            columns: Target columns.
            exprs: Current expression dict.

        Returns:
            ``SQRT(SUM_OF_SQUARES)`` expression.
        """
        squared_terms = [exp.Mul(this=exprs[col], expression=exprs[col]) for col in columns]
        sum_of_squares = self._chain_add(squared_terms)
        return exp.Sqrt(this=sum_of_squares)

    def _build_l1(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> exp.Expression:
        """Build L1 (Manhattan) norm: |x1| + |x2| + ...

        Args:
            columns: Target columns.
            exprs: Current expression dict.

        Returns:
            Sum of absolute values expression.
        """
        abs_terms = [exp.Abs(this=exprs[col]) for col in columns]
        return self._chain_add(abs_terms)

    def _build_max(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> exp.Expression:
        """Build max norm: GREATEST(|x1|, |x2|, ...).

        Args:
            columns: Target columns.
            exprs: Current expression dict.

        Returns:
            ``GREATEST(ABS(x1), ABS(x2), ...)`` expression.
        """
        abs_terms = [exp.Abs(this=exprs[col]) for col in columns]
        return exp.Greatest(this=abs_terms[0], expressions=abs_terms[1:])

    @staticmethod
    def _chain_add(terms: Sequence[exp.Expression]) -> exp.Expression:
        """Left-fold a list of expressions into nested Add nodes.

        Args:
            terms: Non-empty sequence of sqlglot expressions to sum.

        Returns:
            Chained ``exp.Add`` expression tree.
        """
        result = terms[0]
        for term in terms[1:]:
            result = exp.Add(this=result, expression=term)
        return result
