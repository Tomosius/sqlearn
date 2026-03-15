"""Cast -- cast columns to specified SQL data types.

Compiles to inline SQL: ``CAST(col AS TYPE)`` for each mapped column.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlglot.expressions as exp

from sqlearn.core.transformer import Transformer

if TYPE_CHECKING:
    from sqlearn.core.schema import Schema


class Cast(Transformer):
    """Cast columns to specified SQL data types via inline CAST expressions.

    A **static** transformer -- no statistics are learned during ``fit()``.
    Each column listed in ``mapping`` is wrapped in a ``CAST(col AS type)``
    expression. Unmapped columns pass through unchanged automatically.

    Generated SQL::

        SELECT
          CAST(price AS DOUBLE) AS price,
          CAST(qty AS INTEGER) AS qty,
          name
        FROM __input__

    Args:
        mapping: Dict mapping column names to target SQL type strings.
            Example: ``{"price": "DOUBLE", "qty": "INTEGER"}``.

    Raises:
        ValueError: If ``mapping`` is empty.
        TypeError: If ``mapping`` is not a dict, or if keys/values are
            not strings.

    Examples:
        Cast price from integer to double:

        >>> import sqlearn as sq
        >>> pipe = sq.Pipeline([sq.Cast({"price": "DOUBLE"})])
        >>> pipe.fit("data.parquet")
        >>> sql = pipe.to_sql()
        ... # CAST(price AS DOUBLE) AS price

        Multiple casts in one step:

        >>> pipe = sq.Pipeline(
        ...     [sq.Cast({"price": "DOUBLE", "qty": "INTEGER", "flag": "BOOLEAN"})]
        ... )
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # CAST(price AS DOUBLE), CAST(qty AS INTEGER), CAST(flag AS BOOLEAN)

        Compose with Imputer (impute first, then cast):

        >>> pipe = sq.Pipeline([sq.Imputer(), sq.Cast({"price": "INTEGER"})])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # CAST(COALESCE(price, 42.0) AS INTEGER)

    See Also:
        :class:`~sqlearn.core.transformer.Transformer`: Base class.
        :class:`~sqlearn.core.pipeline.Pipeline`: Compose transformers.
    """

    _classification: str = "static"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(self, mapping: dict[str, str]) -> None:
        """Initialize Cast with a column-to-type mapping.

        Args:
            mapping: Dict mapping column names to target SQL type strings.
                Keys are column names, values are SQL type strings
                (e.g. ``"DOUBLE"``, ``"INTEGER"``, ``"VARCHAR"``).

        Raises:
            TypeError: If ``mapping`` is not a dict, or if any key or
                value is not a string.
            ValueError: If ``mapping`` is empty.
        """
        if not isinstance(mapping, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
            msg = f"mapping must be a dict, got {type(mapping).__name__}"
            raise TypeError(msg)
        if len(mapping) == 0:
            msg = "mapping must be non-empty"
            raise ValueError(msg)
        for key, value in mapping.items():
            if not isinstance(key, str):  # pyright: ignore[reportUnnecessaryIsInstance]
                msg = f"mapping keys must be strings, got {type(key).__name__} for key {key!r}"
                raise TypeError(msg)
            if not isinstance(value, str):  # pyright: ignore[reportUnnecessaryIsInstance]
                msg = f"mapping values must be strings, got {type(value).__name__} for key {key!r}"
                raise TypeError(msg)

        super().__init__(columns=list(mapping.keys()))
        self.mapping = mapping

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        """Generate inline CAST expressions for each mapped column.

        Wraps each target column's current expression in
        ``CAST(expr AS target_type)`` using the sqlglot AST.

        Args:
            columns: Target columns to cast (intersection of mapping
                keys and available schema columns).
            exprs: Current expression dict for ALL columns from prior steps.

        Returns:
            Dict of modified column expressions. Each target column is
            wrapped in a CAST node.
        """
        return {
            col: exp.Cast(
                this=exprs[col],
                to=exp.DataType.build(self.mapping[col]),  # pyright: ignore[reportUnknownMemberType]
            )
            for col in columns
        }

    def output_schema(self, schema: Schema) -> Schema:
        """Update schema types for cast columns.

        Returns a new schema where each cast column's type is updated
        to the target type from ``mapping``.

        Args:
            schema: Input schema.

        Returns:
            New schema with updated types for cast columns.
        """
        cast_updates = {col: dtype for col, dtype in self.mapping.items() if col in schema.columns}
        if not cast_updates:
            return schema
        return schema.cast(cast_updates)
