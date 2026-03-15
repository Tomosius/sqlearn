"""Drop — remove specified columns from the dataset.

Compiles to a SELECT that excludes the named columns:
``SELECT kept_col1, kept_col2, ... FROM __input__``

This is a **static** transformer — no ``fit()`` statistics are needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlglot.expressions as exp

from sqlearn.core.transformer import Transformer

if TYPE_CHECKING:
    from sqlearn.core.schema import Schema


class Drop(Transformer):
    """Remove specified columns from the dataset via SQL.

    Drops the named columns from the output. All other columns pass through
    unchanged. Uses ``query()`` to wrap the input in a SELECT that excludes
    the dropped columns.

    This is a **static** transformer --- no statistics are learned during
    ``fit()``. The column list is fixed at construction time.

    Generated SQL (dropping ``id`` and ``timestamp``)::

        SELECT
          price,
          quantity,
          city
        FROM (
          SELECT * FROM __input__
        ) AS __input__

    Args:
        columns: List of column names to drop. Must be a non-empty list
            of strings.

    Raises:
        TypeError: If ``columns`` is not a list, or contains non-string
            elements.
        ValueError: If ``columns`` is empty.

    Examples:
        Drop a single column:

        >>> import sqlearn as sq
        >>> pipe = sq.Pipeline([sq.Drop(columns=["id"])])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # SELECT price, quantity FROM ...

        Drop multiple columns:

        >>> pipe = sq.Pipeline([sq.Drop(columns=["id", "timestamp"])])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # SELECT price, quantity, city FROM ...

        Compose with StandardScaler (drop after scaling):

        >>> pipe = sq.Pipeline(
        ...     [
        ...         sq.StandardScaler(),
        ...         sq.Drop(columns=["irrelevant_col"]),
        ...     ]
        ... )
        >>> pipe.fit("data.parquet")
        >>> pipe.get_feature_names_out()  # irrelevant_col excluded

    See Also:
        :class:`~sqlearn.core.transformer.Transformer`: Base class.
        :class:`~sqlearn.core.pipeline.Pipeline`: Compose transformers.
        :class:`~sqlearn.feature_selection.correlated.DropCorrelated`:
            Auto-detect and drop correlated features.
    """

    _default_columns: None = None  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "static"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        columns: list[str],
    ) -> None:
        """Initialize Drop transformer.

        Args:
            columns: List of column names to drop.

        Raises:
            TypeError: If ``columns`` is not a list, or contains non-string
                elements.
            ValueError: If ``columns`` is empty.
        """
        if not isinstance(columns, list):  # pyright: ignore[reportUnnecessaryIsInstance]
            msg = f"columns must be a list, got {type(columns).__name__}"
            raise TypeError(msg)

        for item in columns:
            if not isinstance(item, str):  # pyright: ignore[reportUnnecessaryIsInstance]
                msg = f"columns must contain strings, got {type(item).__name__}: {item!r}"
                raise TypeError(msg)

        if not columns:
            msg = "columns must be non-empty"
            raise ValueError(msg)

        super().__init__(columns=None)
        self.columns = columns

    def _resolve_columns_spec(self) -> None:
        """Return None since Drop manages its own column list.

        Returns:
            None --- Drop does not use auto column routing.
        """

    def query(
        self,
        input_query: exp.Expression,
    ) -> exp.Expression:
        """Generate a SELECT wrapping the input, excluding dropped columns.

        Builds an explicit SELECT list containing all input schema columns
        except those in ``self.columns``. The input query is wrapped as a
        subquery.

        Args:
            input_query: The input query to wrap.

        Returns:
            A sqlglot SELECT expression excluding dropped columns.
        """
        if self.input_schema_ is None:
            return input_query  # pragma: no cover

        drop_set = set(self.columns)  # type: ignore[arg-type]
        selections: list[exp.Expression] = [
            exp.Column(this=exp.to_identifier(col_name))
            for col_name in self.input_schema_.columns
            if col_name not in drop_set
        ]

        return exp.Select(expressions=selections).from_(  # pyright: ignore[reportUnknownMemberType]
            exp.Subquery(
                this=input_query,
                alias=exp.to_identifier("__input__"),
            )
        )

    def output_schema(self, schema: Schema) -> Schema:
        """Declare output schema after dropping columns.

        Removes the specified columns from the input schema. Only drops
        columns that exist in the input schema (silently skips others,
        since column resolution happens at fit time).

        Args:
            schema: Input schema.

        Returns:
            Output schema with specified columns removed.
        """
        existing = [c for c in self.columns if c in schema.columns]  # type: ignore[union-attr]
        if not existing:
            return schema
        return schema.drop(existing)

    def expressions(
        self,
        columns: list[str],  # noqa: ARG002
        exprs: dict[str, exp.Expression],  # noqa: ARG002
    ) -> dict[str, exp.Expression]:
        """Not used --- Drop uses query() instead.

        Args:
            columns: Target columns (unused).
            exprs: Current expression dict (unused).

        Returns:
            Empty dict (query() handles the transformation).
        """
        return {}
