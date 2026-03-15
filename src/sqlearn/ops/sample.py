"""Sample — select a random subset of rows from the dataset.

Compiles to query-level SQL: ``ORDER BY RANDOM() LIMIT n`` for count-based
sampling, or ``WHERE RANDOM() < fraction`` for fraction-based sampling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlglot.expressions as exp

from sqlearn.core.transformer import Transformer

if TYPE_CHECKING:
    from sqlearn.core.schema import ColumnSelector, Schema


class Sample(Transformer):
    """Sample a random subset of rows from the dataset via SQL.

    A **static** transformer that operates at the query level. No statistics
    are learned during ``fit()`` --- the sampling clause is applied directly
    to the SQL query at transform time.

    Two sampling modes (mutually exclusive):

    - **Count-based** (``n``): Select exactly ``n`` rows via
      ``ORDER BY RANDOM() LIMIT n``.
    - **Fraction-based** (``fraction``): Select approximately ``fraction``
      of rows via ``WHERE RANDOM() < fraction``.

    Fraction-based sampling is approximate --- the actual number of rows
    returned will vary around ``fraction * total_rows``. For exact counts,
    use ``n`` instead.

    Generated SQL (count-based, n=100)::

        SELECT * FROM (__input__) AS __input__
        ORDER BY RANDOM()
        LIMIT 100

    Generated SQL (fraction-based, fraction=0.5)::

        SELECT * FROM (__input__) AS __input__
        WHERE RANDOM() < 0.5

    .. note::

        Seed-based reproducibility is not supported. Each execution may
        return different rows. For deterministic sampling, use DuckDB's
        ``SETSEED()`` externally before calling transform.

    Args:
        n: Number of rows to sample. Must be a positive integer.
            Mutually exclusive with ``fraction``.
        fraction: Fraction of rows to sample, in the range ``(0.0, 1.0)``.
            Mutually exclusive with ``n``.
        seed: Reserved for future use. Currently ignored.
        columns: Not used. Sample operates on rows, not columns.

    Raises:
        ValueError: If both ``n`` and ``fraction`` are provided, or neither
            is provided, or values are out of range.

    Examples:
        Sample 100 rows:

        >>> import sqlearn as sq
        >>> pipe = sq.Pipeline([sq.Sample(n=100)])
        >>> pipe.fit("train.parquet")
        >>> result = pipe.transform("train.parquet")
        >>> result.shape[0]  # exactly 100 rows
        100

        Sample approximately 50% of rows:

        >>> pipe = sq.Pipeline([sq.Sample(fraction=0.5)])
        >>> pipe.fit("data.parquet")
        >>> result = pipe.transform("data.parquet")
        >>> # result.shape[0] is approximately half of total rows

        Combine with other transformers:

        >>> pipe = sq.Pipeline(
        ...     [
        ...         sq.StandardScaler(),
        ...         sq.Sample(n=1000),
        ...     ]
        ... )
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # CTE with StandardScaler, then ORDER BY RANDOM() LIMIT 1000

    See Also:
        :class:`~sqlearn.core.transformer.Transformer`: Base class.
        :class:`~sqlearn.core.pipeline.Pipeline`: Compose transformers.
    """

    _default_columns: str | None = None
    _classification: str = "static"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        n: int | None = None,
        fraction: float | None = None,
        seed: int | None = None,
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        super().__init__(columns=columns)

        # Validate mutually exclusive
        if n is not None and fraction is not None:
            msg = (
                "Cannot specify both 'n' and 'fraction'. "
                "Use n for exact count or fraction for proportional sampling."
            )
            raise ValueError(msg)

        if n is None and fraction is None:
            msg = "Must specify either 'n' (row count) or 'fraction' (proportion)."
            raise ValueError(msg)

        # Validate n (runtime check for non-typed callers)
        if n is not None:
            if not isinstance(n, int):  # pyright: ignore[reportUnnecessaryIsInstance]
                msg = f"'n' must be an integer, got {type(n).__name__}."
                raise TypeError(msg)
            if n <= 0:
                msg = f"'n' must be a positive integer, got {n}."
                raise ValueError(msg)

        # Validate fraction (runtime check for non-typed callers)
        if fraction is not None:
            if not isinstance(fraction, int | float):  # pyright: ignore[reportUnnecessaryIsInstance]
                msg = f"'fraction' must be a float, got {type(fraction).__name__}."
                raise TypeError(msg)
            if fraction <= 0.0 or fraction >= 1.0:
                msg = f"'fraction' must be between 0.0 and 1.0 (exclusive), got {fraction}."
                raise ValueError(msg)

        self.n = n
        self.fraction = fraction
        self.seed = seed

    def query(
        self,
        input_query: exp.Expression,
    ) -> exp.Expression:
        """Generate a full query wrapping input with sampling clause.

        Builds ``SELECT * FROM (input) ORDER BY RANDOM() LIMIT n`` for
        count-based sampling, or ``SELECT * FROM (input) WHERE RANDOM() < f``
        for fraction-based sampling.

        Args:
            input_query: The input query to wrap.

        Returns:
            A new sqlglot SELECT with the sampling clause applied.
        """
        select: exp.Select = exp.Select(
            expressions=[exp.Star()],
        ).from_(  # pyright: ignore[reportUnknownMemberType]
            exp.Subquery(
                this=input_query,
                alias=exp.TableAlias(this=exp.to_identifier("__input__")),
            )
        )

        if self.n is not None:
            # ORDER BY RANDOM() LIMIT n
            random_fn = exp.Anonymous(this="RANDOM", expressions=[])
            select = select.order_by(random_fn)  # pyright: ignore[reportUnknownMemberType]
            select = select.limit(self.n)  # pyright: ignore[reportUnknownMemberType]
        elif self.fraction is not None:
            # WHERE RANDOM() < fraction
            random_fn = exp.Anonymous(this="RANDOM", expressions=[])
            select = select.where(  # pyright: ignore[reportUnknownMemberType]
                exp.LT(
                    this=random_fn,
                    expression=exp.Literal.number(self.fraction),  # pyright: ignore[reportUnknownMemberType]
                )
            )

        return select

    def output_schema(self, schema: Schema) -> Schema:
        """Return output schema unchanged --- sampling does not alter columns.

        Args:
            schema: Input schema.

        Returns:
            Same schema (row sampling does not change columns).
        """
        return schema

    def expressions(
        self,
        columns: list[str],  # noqa: ARG002
        exprs: dict[str, exp.Expression],  # noqa: ARG002
    ) -> dict[str, exp.Expression]:
        """Passthrough no-op since Sample uses query-level SQL instead.

        This method is never called because query() returns a non-None value,
        which takes precedence in the compiler. Provided to satisfy the
        base class interface.

        Args:
            columns: Target columns (unused).
            exprs: Current expression dict (unused).

        Returns:
            Empty dict.
        """
        return {}
