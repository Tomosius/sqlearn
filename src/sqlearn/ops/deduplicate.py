"""Deduplicate — remove duplicate rows via SQL.

Compiles to query-level SQL: ``SELECT DISTINCT *`` for full-row dedup, or
``ROW_NUMBER() OVER (PARTITION BY ...)`` / ``COUNT(*) OVER (PARTITION BY ...)``
for subset dedup with keep="first"/"last"/"none".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlglot.expressions as exp

from sqlearn.core.transformer import Transformer

if TYPE_CHECKING:
    from sqlearn.core.schema import Schema

_VALID_KEEP = ("first", "last", "none")


class Deduplicate(Transformer):
    """Remove duplicate rows from the dataset via SQL.

    Operates at the query level using ``query()``, not per-column
    ``expressions()``. Supports three deduplication modes:

    - **first** (default): Keep the first occurrence of each duplicate group.
      Uses ``SELECT DISTINCT`` for full-row dedup, or ``ROW_NUMBER()`` for
      subset dedup.
    - **last**: Keep the last occurrence. Uses ``ROW_NUMBER()`` with
      descending order.
    - **none**: Remove all rows that have any duplicates. Uses
      ``COUNT(*) OVER (PARTITION BY ...)`` to find and exclude groups
      with count > 1.

    This is a **static** transformer --- no statistics are learned during
    ``fit()``.

    Generated SQL (full-row dedup, keep="first")::

        SELECT DISTINCT
          *
        FROM __input__ AS __input__

    Generated SQL (subset dedup, keep="first")::

        SELECT
          city, name, price
        FROM (
          SELECT
            *, ROW_NUMBER() OVER (PARTITION BY city, name ORDER BY city) AS __rn__
          FROM __input__ AS __input__
        ) AS __dedup__
        WHERE
          __rn__ = 1

    Generated SQL (subset dedup, keep="none")::

        SELECT
          city, name, price
        FROM (
          SELECT
            *, COUNT(*) OVER (PARTITION BY city, name) AS __cnt__
          FROM __input__ AS __input__
        ) AS __dedup__
        WHERE
          __cnt__ = 1

    Args:
        subset: Columns that define uniqueness. If ``None`` (default), all
            columns are used for full-row dedup. This is NOT the same as
            the base class ``columns`` parameter --- it specifies which
            columns define "duplicate", not which columns to transform.
        keep: Which duplicate to keep: ``'first'``, ``'last'``, or
            ``'none'`` (remove all duplicates). Defaults to ``'first'``.

    Raises:
        ValueError: If ``keep`` is not one of ``'first'``, ``'last'``,
            ``'none'``, or if ``subset`` is an empty list.

    Examples:
        Remove exact duplicate rows:

        >>> import sqlearn as sq
        >>> pipe = sq.Pipeline([sq.Deduplicate()])
        >>> pipe.fit("data.parquet")
        >>> result = pipe.transform("data.parquet")

        Deduplicate by specific columns, keeping first occurrence:

        >>> pipe = sq.Pipeline([sq.Deduplicate(subset=["city", "name"])])
        >>> pipe.fit("data.parquet")
        >>> sql = pipe.to_sql()
        ... # ROW_NUMBER() OVER (PARTITION BY city, name ...) AS __rn__

        Remove ALL rows that have duplicates:

        >>> pipe = sq.Pipeline([sq.Deduplicate(subset=["city"], keep="none")])
        >>> pipe.fit("data.parquet")
        >>> sql = pipe.to_sql()
        ... # COUNT(*) OVER (PARTITION BY city) AS __cnt__
        ... # WHERE __cnt__ = 1

        Compose with StandardScaler (deduplicate first, then scale):

        >>> from sqlearn.scalers.standard import StandardScaler
        >>> pipe = sq.Pipeline([sq.Deduplicate(), StandardScaler()])
        >>> pipe.fit("data.parquet")

    See Also:
        :class:`~sqlearn.core.pipeline.Pipeline`: Compose transformers.
        :class:`~sqlearn.core.transformer.Transformer`: Base class for steps.
    """

    _default_columns: None = None  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "static"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        subset: list[str] | None = None,
        keep: str = "first",
    ) -> None:
        super().__init__(columns=None)
        if keep not in _VALID_KEEP:
            msg = f"Invalid keep {keep!r}. Must be one of {_VALID_KEEP!r}."
            raise ValueError(msg)
        if subset is not None and len(subset) == 0:
            msg = "subset must be a non-empty list of column names, or None."
            raise ValueError(msg)
        self.subset = subset
        self.keep = keep

    def query(
        self,
        input_query: exp.Expression,
    ) -> exp.Expression | None:
        """Generate deduplication SQL wrapping the input query.

        For full-row dedup with keep="first", uses ``SELECT DISTINCT``.
        For subset dedup or keep="last"/"none", uses window functions.

        Args:
            input_query: The input query to wrap.

        Returns:
            A new sqlglot SELECT that deduplicates the input.
        """
        if self.subset is None and self.keep == "first":
            return self._build_distinct(input_query)

        if self.keep in ("first", "last"):
            return self._build_row_number(input_query)

        # keep="none"
        return self._build_count(input_query)

    def _build_distinct(
        self,
        input_query: exp.Expression,
    ) -> exp.Select:
        """Build SELECT DISTINCT * FROM (...) for full-row dedup.

        Args:
            input_query: The input query to wrap.

        Returns:
            A DISTINCT SELECT expression.
        """
        return (
            exp.select(exp.Star())  # pyright: ignore[reportUnknownMemberType]
            .from_(exp.Subquery(this=input_query, alias="__input__"))
            .distinct()
        )

    def _build_row_number(
        self,
        input_query: exp.Expression,
    ) -> exp.Select:
        """Build ROW_NUMBER dedup for subset or keep="last".

        Uses ROW_NUMBER() OVER (PARTITION BY ...) and filters to row 1.
        For keep="last", reverses the ORDER BY direction.

        Args:
            input_query: The input query to wrap.

        Returns:
            Outer SELECT filtering on __rn__ = 1.
        """
        partition_cols = self._partition_columns()
        order_col = partition_cols[0].copy()

        # For keep="last", reverse the ordering
        ascending = self.keep == "first"
        order_expr = exp.Ordered(this=order_col, desc=not ascending)

        window = exp.Window(
            this=exp.Anonymous(this="ROW_NUMBER", expressions=[]),
            partition_by=partition_cols,
            order=exp.Order(expressions=[order_expr]),
        )

        # Inner query: SELECT *, ROW_NUMBER() OVER (...) AS __rn__
        inner = exp.select(  # pyright: ignore[reportUnknownMemberType]
            exp.Star(),
            window.as_("__rn__"),  # pyright: ignore[reportUnknownMemberType]
        ).from_(exp.Subquery(this=input_query, alias="__input__"))

        # Outer query: SELECT <original columns> WHERE __rn__ = 1
        outer_cols = self._output_columns()
        return (
            exp.select(*outer_cols)  # pyright: ignore[reportUnknownMemberType]
            .from_(exp.Subquery(this=inner, alias="__dedup__"))
            .where(
                exp.EQ(
                    this=exp.Column(this="__rn__"),
                    expression=exp.Literal.number(1),  # pyright: ignore[reportUnknownMemberType]
                )
            )
        )

    def _build_count(
        self,
        input_query: exp.Expression,
    ) -> exp.Select:
        """Build COUNT(*) dedup for keep="none" (remove all duplicates).

        Uses COUNT(*) OVER (PARTITION BY ...) and keeps only groups with
        exactly one member.

        Args:
            input_query: The input query to wrap.

        Returns:
            Outer SELECT filtering on __cnt__ = 1.
        """
        partition_cols = self._partition_columns()

        window = exp.Window(
            this=exp.Count(this=exp.Star()),
            partition_by=partition_cols,
        )

        # Inner query: SELECT *, COUNT(*) OVER (...) AS __cnt__
        inner = exp.select(  # pyright: ignore[reportUnknownMemberType]
            exp.Star(),
            window.as_("__cnt__"),  # pyright: ignore[reportUnknownMemberType]
        ).from_(exp.Subquery(this=input_query, alias="__input__"))

        # Outer query: SELECT <original columns> WHERE __cnt__ = 1
        outer_cols = self._output_columns()
        return (
            exp.select(*outer_cols)  # pyright: ignore[reportUnknownMemberType]
            .from_(exp.Subquery(this=inner, alias="__dedup__"))
            .where(
                exp.EQ(
                    this=exp.Column(this="__cnt__"),
                    expression=exp.Literal.number(1),  # pyright: ignore[reportUnknownMemberType]
                )
            )
        )

    def _partition_columns(self) -> list[exp.Expression]:
        """Build partition column expressions from subset or input_schema_.

        Falls back to input_schema_ column names when subset is None.

        Returns:
            List of sqlglot Column expressions for PARTITION BY.
        """
        if self.subset is not None:
            return [exp.Column(this=col) for col in self.subset]
        if self.input_schema_ is not None:
            return [exp.Column(this=col) for col in self.input_schema_.columns]
        # Fallback: use columns_ if available
        if self.columns_ is not None:
            return [exp.Column(this=col) for col in self.columns_]
        return []

    def _output_columns(self) -> list[exp.Expression]:
        """Build output column expressions (excluding helper columns like __rn__).

        Returns:
            List of sqlglot Column expressions for the original columns.
        """
        if self.input_schema_ is not None:
            return [exp.Column(this=col) for col in self.input_schema_.columns]
        if self.columns_ is not None:
            return [exp.Column(this=col) for col in self.columns_]
        return [exp.Star()]

    def output_schema(self, schema: Schema) -> Schema:
        """Return the output schema (unchanged from input).

        Deduplication removes rows, not columns, so the schema is preserved.

        Args:
            schema: Input schema.

        Returns:
            The same schema (dedup does not change columns).
        """
        return schema
