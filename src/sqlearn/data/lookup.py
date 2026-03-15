"""Lookup -- mid-pipeline JOIN transformer.

Joins lookup data into the pipeline during transformation.  This is a
**static** transformer --- no statistics are learned during ``fit()``.
The join is injected at the query level via ``query()``.

Generated SQL example (joining categories by ``category_id``)::

    SELECT
      __input__.product_id,
      __input__.category_id,
      __input__.price,
      __lookup__.category_name,
      __lookup__.department
    FROM (
      SELECT * FROM __input__
    ) AS __input__
    LEFT JOIN 'categories.parquet' AS __lookup__
      ON __input__.category_id = __lookup__.category_id
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlglot.expressions as exp

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.errors import SchemaError
from sqlearn.core.transformer import Transformer

if TYPE_CHECKING:
    from sqlearn.core.schema import Schema


class Lookup(Transformer):
    """Mid-pipeline JOIN transformer that enriches data with lookup columns.

    Wraps the current pipeline query with a LEFT JOIN to a lookup data
    source.  Selected columns from the lookup table are added to the
    pipeline output.

    This is a **static** transformer --- no statistics are learned
    during ``fit()``.  Uses ``query()`` to wrap the input query with
    a JOIN, and overrides ``output_schema()`` to include the selected
    lookup columns.

    Generated SQL (joining ``categories.parquet`` on ``category_id``,
    selecting ``category_name`` and ``department``)::

        SELECT
          __input__.product_id,
          __input__.category_id,
          __input__.price,
          __lookup__.category_name,
          __lookup__.department
        FROM (
          SELECT * FROM __input__
        ) AS __input__
        LEFT JOIN 'categories.parquet' AS __lookup__
          ON __input__.category_id = __lookup__.category_id

    Args:
        source: Lookup data source (file path or table name).
        on: Column name(s) for the join condition.  Must exist in both
            the pipeline data and the lookup source.
        select: Column name(s) to select from the lookup source.  If
            None, all non-key columns from the lookup source are
            selected.
        how: Join type: ``'left'`` (default) or ``'inner'``.  Left
            join preserves all pipeline rows; inner join drops rows
            without a match.
        suffix: Suffix appended to lookup columns that overlap with
            existing pipeline columns (after accounting for join keys).

    Raises:
        TypeError: If ``source`` is not a string or ``on`` has wrong type.
        ValueError: If ``how`` is not ``'left'`` or ``'inner'``, or if
            ``select`` is empty.

    Examples:
        Add category information to a pipeline:

        >>> import sqlearn as sq
        >>> pipe = sq.Pipeline(
        ...     [
        ...         sq.Lookup(
        ...             "categories.parquet",
        ...             on="category_id",
        ...             select=["category_name", "department"],
        ...         ),
        ...         sq.StandardScaler(),
        ...     ]
        ... )
        >>> pipe.fit("products.parquet")
        >>> pipe.to_sql()
        ... # SELECT ... LEFT JOIN 'categories.parquet' ...

        Join on multiple keys:

        >>> lookup = sq.Lookup(
        ...     "regions.csv", on=["country", "state"], select=["region_name"]
        ... )

        Inner join (drops non-matching rows):

        >>> lookup = sq.Lookup("valid_ids.parquet", on="id", how="inner")

    See Also:
        :func:`~sqlearn.data.merge.merge`: Standalone join function.
        :func:`~sqlearn.data.concat.concat`: Vertical concatenation.
        :class:`~sqlearn.core.transformer.Transformer`: Base class.
    """

    _default_columns: None = None  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "static"  # pyright: ignore[reportIncompatibleVariableOverride]

    _VALID_HOW = frozenset({"left", "inner"})

    def __init__(
        self,
        source: str,
        *,
        on: str | list[str],
        select: list[str] | None = None,
        how: str = "left",
        suffix: str = "_lookup",
    ) -> None:
        """Initialize Lookup transformer.

        Args:
            source: Lookup data source (file path or table name).
            on: Join key column name(s).
            select: Columns to select from lookup.  None means all
                non-key columns.
            how: Join type (``'left'`` or ``'inner'``).
            suffix: Suffix for overlapping non-key column names.

        Raises:
            TypeError: If ``source`` is not a string, or ``on`` is
                not a string or list of strings.
            ValueError: If ``how`` is not valid, or ``select`` is
                an empty list.
        """
        if not isinstance(source, str):  # pyright: ignore[reportUnnecessaryIsInstance]
            msg = f"source must be a string, got {type(source).__name__}"
            raise TypeError(msg)

        if not isinstance(on, str | list):  # pyright: ignore[reportUnnecessaryIsInstance]
            msg = f"on must be a string or list of strings, got {type(on).__name__}"
            raise TypeError(msg)

        if isinstance(on, list):
            if not on:
                msg = "on must not be an empty list"
                raise ValueError(msg)
            for key in on:
                if not isinstance(key, str):  # pyright: ignore[reportUnnecessaryIsInstance]
                    msg = f"on keys must be strings, got {type(key).__name__}"
                    raise TypeError(msg)

        if how not in self._VALID_HOW:
            msg = f"Invalid join type: {how!r}. Must be one of {sorted(self._VALID_HOW)}"
            raise ValueError(msg)

        if select is not None and not select:
            msg = "select must not be an empty list (use None for all non-key columns)"
            raise ValueError(msg)

        super().__init__(columns=None)
        self.source = source
        self.on = on
        self.select = select
        self.how = how
        self.suffix = suffix

        # Resolved at fit time
        self._lookup_schema: Schema | None = None
        self._select_columns: list[str] | None = None

    def _resolve_with_backend(self, backend: object) -> None:
        """Resolve lookup schema using the pipeline's backend.

        Called by :meth:`Pipeline._finalize_step` during fit. Describes
        the lookup source to learn its schema, then resolves join keys
        and select columns.

        Args:
            backend: DuckDB backend with ``describe()`` method.
        """
        if not isinstance(backend, DuckDBBackend):
            return  # pragma: no cover

        self._lookup_schema = backend.describe(self.source)
        _keys, select_cols = self._resolve_lookup()
        self._select_columns = select_cols

    def _resolve_lookup(self) -> tuple[list[str], list[str]]:
        """Resolve the join keys and select columns.

        Returns:
            Tuple of (join_keys, select_columns).

        Raises:
            SchemaError: If ``on`` or ``select`` columns are missing.
        """
        if self._lookup_schema is None:
            msg = "Lookup schema not resolved (not fitted)"
            raise SchemaError(msg)

        keys = [self.on] if isinstance(self.on, str) else list(self.on)

        # Validate keys exist in lookup schema
        missing_keys = set(keys) - set(self._lookup_schema.columns)
        if missing_keys:
            msg = f"Join key(s) not found in lookup source: {sorted(missing_keys)}"
            raise SchemaError(msg)

        if self.select is not None:
            select_cols = list(self.select)
            missing_select = set(select_cols) - set(self._lookup_schema.columns)
            if missing_select:
                msg = f"Select column(s) not found in lookup source: {sorted(missing_select)}"
                raise SchemaError(msg)
        else:
            # All non-key columns
            select_cols = [c for c in self._lookup_schema.columns if c not in set(keys)]

        return keys, select_cols

    def query(
        self,
        input_query: exp.Expression,
    ) -> exp.Expression:
        """Wrap the input query with a LEFT/INNER JOIN to the lookup source.

        Builds an explicit SELECT list containing all input columns plus
        the selected lookup columns (with suffix for overlaps).

        Args:
            input_query: The input query to wrap.

        Returns:
            A sqlglot SELECT expression with the JOIN.
        """
        if self.input_schema_ is None or self._select_columns is None:
            return input_query  # pragma: no cover

        keys, select_cols = self._resolve_lookup()

        input_alias = "__input__"
        lookup_alias = "__lookup__"

        # Determine overlapping columns (non-key columns that exist in both)
        input_col_names = set(self.input_schema_.columns.keys())
        overlap = (set(select_cols) & input_col_names) - set(keys)

        # Build SELECT list
        selections: list[exp.Expression] = [
            exp.Column(
                this=exp.to_identifier(col_name),
                table=exp.to_identifier(input_alias),
            )
            for col_name in self.input_schema_.columns
        ]

        # Selected lookup columns
        for col_name in select_cols:
            col_expr = exp.Column(
                this=exp.to_identifier(col_name),
                table=exp.to_identifier(lookup_alias),
            )
            if col_name in overlap:
                alias_name = f"{col_name}{self.suffix}"
                selections.append(exp.Alias(this=col_expr, alias=exp.to_identifier(alias_name)))
            else:
                selections.append(col_expr)

        # Build FROM with subqueried input
        from_expr = exp.Alias(
            this=exp.Subquery(this=input_query),
            alias=exp.to_identifier(input_alias),
        )

        # Build JOIN condition
        conditions: list[exp.Expression] = []
        for key in keys:
            eq = exp.EQ(
                this=exp.Column(
                    this=exp.to_identifier(key),
                    table=exp.to_identifier(input_alias),
                ),
                expression=exp.Column(
                    this=exp.to_identifier(key),
                    table=exp.to_identifier(lookup_alias),
                ),
            )
            conditions.append(eq)

        on_clause = conditions[0]
        for cond in conditions[1:]:
            on_clause = exp.And(this=on_clause, expression=cond)

        # Build lookup source expression
        lookup_source = _source_to_table(self.source)
        lookup_table = exp.Alias(
            this=lookup_source,
            alias=exp.to_identifier(lookup_alias),
        )

        side = "LEFT" if self.how == "left" else ""
        join = exp.Join(
            this=lookup_table,
            on=on_clause,
            side=side,
        )

        query = exp.Select(expressions=selections).from_(from_expr)  # pyright: ignore[reportUnknownMemberType]
        return query.join(join)  # pyright: ignore[reportUnknownMemberType]

    def output_schema(self, schema: Schema) -> Schema:
        """Declare output schema after adding lookup columns.

        Adds the selected lookup columns to the input schema.  Columns
        that overlap with existing input columns (excluding join keys)
        receive the configured suffix.

        Args:
            schema: Input schema.

        Returns:
            Output schema with added lookup columns.
        """
        if self._lookup_schema is None:
            return schema

        keys, select_cols = self._resolve_lookup()
        input_col_names = set(schema.columns.keys())
        overlap = (set(select_cols) & input_col_names) - set(keys)

        new_cols: dict[str, str] = {}
        for col in select_cols:
            col_type = self._lookup_schema.columns[col]
            if col in overlap:
                new_cols[f"{col}{self.suffix}"] = col_type
            else:
                new_cols[col] = col_type

        return schema.add(new_cols)

    def expressions(
        self,
        columns: list[str],  # noqa: ARG002
        exprs: dict[str, exp.Expression],  # noqa: ARG002
    ) -> dict[str, exp.Expression]:
        """Not used --- Lookup uses query() instead.

        Args:
            columns: Target columns (unused).
            exprs: Current expression dict (unused).

        Returns:
            Empty dict (query() handles the transformation).
        """
        return {}


def _source_to_table(source: str) -> exp.Expression:
    """Convert a source string to a sqlglot table or file expression.

    Args:
        source: File path or table name.

    Returns:
        sqlglot expression suitable for a FROM or JOIN clause.
    """
    lower = source.lower()
    if any(lower.endswith(ext) for ext in (".parquet", ".csv", ".json", ".tsv")):
        return exp.Literal.string(source)
    return exp.to_table(source)
