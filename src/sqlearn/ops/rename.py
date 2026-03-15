"""Rename — rename columns in the dataset via SQL.

Compiles to a SELECT with aliased columns:
``SELECT old_name AS new_name, ... FROM __input__``

This is a **static** transformer — no ``fit()`` statistics are needed.
Column names are remapped based on an explicit mapping dict.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlglot.expressions as exp

from sqlearn.core.transformer import Transformer

if TYPE_CHECKING:
    from sqlearn.core.schema import Schema


class Rename(Transformer):
    """Rename columns in the dataset via SQL.

    Applies an explicit mapping of old column names to new column names.
    Columns not in the mapping pass through unchanged. Uses ``query()``
    to wrap the input in a SELECT with aliased columns, ensuring old
    column names are fully replaced (not duplicated alongside new names).

    This is a **static** transformer --- no statistics are learned during
    ``fit()``. The mapping is fixed at construction time.

    Generated SQL (renaming ``price`` to ``cost`` and ``qty`` to ``quantity``)::

        SELECT
          cost,
          quantity,
          city
        FROM (
          SELECT
            price AS cost,
            qty AS quantity,
            city
          FROM __input__
        )

    Args:
        mapping: Dict mapping old column names to new names. Must be
            a non-empty dict with string keys and string values, and
            no duplicate target names.

    Raises:
        TypeError: If ``mapping`` is not a dict, or contains non-string
            keys or values.
        ValueError: If ``mapping`` is empty or contains duplicate target
            names.

    Examples:
        Rename a single column:

        >>> import sqlearn as sq
        >>> from sqlearn.ops.rename import Rename
        >>> pipe = sq.Pipeline([Rename(mapping={"price": "cost"})])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # SELECT price AS cost, quantity FROM ...

        Rename multiple columns:

        >>> pipe = sq.Pipeline([Rename(mapping={"price": "cost", "qty": "quantity"})])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # SELECT price AS cost, qty AS quantity, city FROM ...

        Compose with StandardScaler (rename after scaling):

        >>> pipe = sq.Pipeline(
        ...     [
        ...         sq.StandardScaler(),
        ...         Rename(mapping={"price": "scaled_price"}),
        ...     ]
        ... )
        >>> pipe.fit("data.parquet")
        >>> pipe.get_feature_names_out()
        ['scaled_price', 'quantity']

    See Also:
        :class:`~sqlearn.core.transformer.Transformer`: Base class.
        :class:`~sqlearn.core.pipeline.Pipeline`: Compose transformers.
    """

    _default_columns: None = None  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "static"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        mapping: dict[str, str],
    ) -> None:
        """Initialize Rename transformer.

        Args:
            mapping: Dict mapping old column names to new names.

        Raises:
            TypeError: If ``mapping`` is not a dict, or contains non-string
                keys or values.
            ValueError: If ``mapping`` is empty or contains duplicate target
                names.
        """
        if not isinstance(mapping, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
            msg = f"mapping must be a dict, got {type(mapping).__name__}"
            raise TypeError(msg)

        for key, value in mapping.items():
            if not isinstance(key, str):  # pyright: ignore[reportUnnecessaryIsInstance]
                msg = f"mapping keys must be strings, got {type(key).__name__}: {key!r}"
                raise TypeError(msg)
            if not isinstance(value, str):  # pyright: ignore[reportUnnecessaryIsInstance]
                msg = f"mapping values must be strings, got {type(value).__name__}: {value!r}"
                raise TypeError(msg)

        if not mapping:
            msg = "mapping must be non-empty"
            raise ValueError(msg)

        values = list(mapping.values())
        duplicates = [v for v in values if values.count(v) > 1]
        if duplicates:
            msg = f"mapping contains duplicate target names: {sorted(set(duplicates))}"
            raise ValueError(msg)

        super().__init__(columns=None)
        self.mapping = mapping

    def query(
        self,
        input_query: exp.Expression,
    ) -> exp.Expression:
        """Generate a SELECT wrapping the input with renamed columns.

        Builds an explicit SELECT list where mapped columns use
        ``old_name AS new_name`` and unmapped columns pass through.
        The input query is wrapped as a subquery.

        Args:
            input_query: The input query to wrap.

        Returns:
            A sqlglot SELECT expression with renamed columns.
        """
        if self.input_schema_ is None:
            # Fallback: cannot rename without knowing the schema
            return input_query  # pragma: no cover

        selections: list[exp.Expression] = []
        for col_name in self.input_schema_.columns:
            col_expr: exp.Expression = exp.Column(this=exp.to_identifier(col_name))
            if col_name in self.mapping:
                selections.append(
                    exp.Alias(
                        this=col_expr,
                        alias=exp.to_identifier(self.mapping[col_name]),
                    )
                )
            else:
                selections.append(col_expr)

        return exp.Select(expressions=selections).from_(  # pyright: ignore[reportUnknownMemberType]
            exp.Subquery(
                this=input_query,
                alias=exp.to_identifier("__input__"),
            )
        )

    def output_schema(self, schema: Schema) -> Schema:
        """Declare output schema after renaming columns.

        Applies the rename mapping to the input schema. Only renames
        columns that exist in the input schema (silently skips others,
        since column resolution happens at fit time).

        Args:
            schema: Input schema.

        Returns:
            Output schema with columns renamed per the mapping.
        """
        # Only rename columns that exist in the input schema
        applicable = {k: v for k, v in self.mapping.items() if k in schema.columns}
        if not applicable:
            return schema
        return schema.rename(applicable)

    def expressions(
        self,
        columns: list[str],  # noqa: ARG002
        exprs: dict[str, exp.Expression],  # noqa: ARG002
    ) -> dict[str, exp.Expression]:
        """Not used --- Rename uses query() instead.

        Args:
            columns: Target columns (unused).
            exprs: Current expression dict (unused).

        Returns:
            Empty dict (query() handles the transformation).
        """
        return {}
