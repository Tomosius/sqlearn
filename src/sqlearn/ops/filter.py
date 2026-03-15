"""Filter — filter rows based on a SQL condition.

Compiles to a query-level WHERE clause: ``SELECT * FROM (__input__) WHERE ...``.
The condition is parsed through sqlglot at init time for fail-fast validation.
"""

from __future__ import annotations

import sqlglot
import sqlglot.expressions as exp

from sqlearn.core.transformer import Transformer


class Filter(Transformer):
    """Filter rows based on a SQL WHERE condition.

    A **static** data operation that adds a WHERE clause to the pipeline query.
    Unlike column-level transformers that use ``expressions()``, Filter operates
    at the query level via ``query()`` because it selects rows, not columns.

    The condition string is parsed through sqlglot at construction time,
    ensuring invalid SQL fails immediately rather than at fit/transform time.
    All columns pass through unchanged --- only rows are filtered.

    Generated SQL::

        SELECT * FROM (__input__) AS __input__ WHERE price > 0

    Args:
        condition: SQL WHERE condition as a string. Parsed through sqlglot
            for validation. Examples: ``"price > 0"``,
            ``"city IS NOT NULL"``, ``"age BETWEEN 18 AND 65"``.

    Raises:
        ValueError: If ``condition`` is empty or not a string.
        sqlglot.errors.ParseError: If ``condition`` is not valid SQL.

    Examples:
        Filter out rows with negative prices:

        >>> import sqlearn as sq
        >>> pipe = sq.Pipeline([sq.Filter(condition="price > 0")])
        >>> pipe.fit("data.parquet")
        >>> sql = pipe.to_sql()
        ... # SELECT * FROM (__input__) AS __input__ WHERE price > 0

        Combine multiple conditions:

        >>> pipe = sq.Pipeline(
        ...     [
        ...         sq.Filter(condition="price > 0 AND quantity IS NOT NULL"),
        ...     ]
        ... )
        >>> pipe.fit("data.parquet")
        >>> sql = pipe.to_sql()

        Chain Filter with other transformers:

        >>> pipe = sq.Pipeline(
        ...     [
        ...         sq.Filter(condition="price > 0"),
        ...         sq.StandardScaler(),
        ...     ]
        ... )
        >>> pipe.fit("data.parquet")

    See Also:
        :class:`~sqlearn.core.transformer.Transformer`: Base class.
        :class:`~sqlearn.core.pipeline.Pipeline`: Compose transformers.
    """

    _default_columns: None = None  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "static"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        condition: str,
    ) -> None:
        super().__init__(columns=None)
        if not isinstance(condition, str) or not condition.strip():  # pyright: ignore[reportUnnecessaryIsInstance]
            msg = "condition must be a non-empty string"
            raise ValueError(msg)
        self.condition = condition
        # Parse at init time for fail-fast validation.
        # sqlglot.parse_one raises ParseError on invalid SQL.
        self._condition_ast: exp.Expression = sqlglot.parse_one(condition)  # pyright: ignore[reportUnknownMemberType]

    def expressions(
        self,
        columns: list[str],  # noqa: ARG002
        exprs: dict[str, exp.Expression],  # noqa: ARG002
    ) -> dict[str, exp.Expression]:
        """Return empty dict; Filter delegates to query() instead.

        Args:
            columns: Target columns (unused by Filter).
            exprs: Current expression dict (unused by Filter).

        Returns:
            Empty dict since Filter does not modify column expressions.
        """
        return {}

    def query(
        self,
        input_query: exp.Expression,
    ) -> exp.Expression:
        """Wrap the input query with a WHERE clause.

        Generates ``SELECT * FROM (input_query) AS __input__ WHERE condition``.

        Args:
            input_query: The input query to filter.

        Returns:
            A new sqlglot SELECT wrapping the input with a WHERE clause.
        """
        return (
            exp.Select(
                expressions=[exp.Star()],
            )
            .from_(  # pyright: ignore[reportUnknownMemberType]
                exp.Subquery(this=input_query, alias="__input__"),
            )
            .where(self._condition_ast.copy())
        )

    def __repr__(self) -> str:
        """Return repr showing the condition.

        Returns:
            String like ``Filter(condition='price > 0')``.
        """
        return f"Filter(condition={self.condition!r})"
