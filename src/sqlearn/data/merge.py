"""merge() -- SQL JOIN wrapper for combining two data sources.

Builds a sqlglot SELECT with an explicit JOIN clause, returning a
view name that can be used as pipeline input.  All SQL is constructed
via sqlglot ASTs, never raw strings.

Generated SQL example (inner join on ``customer_id``)::

    SELECT
      __left__.customer_id,
      __left__.name,
      __right__.order_date,
      __right__.amount
    FROM 'customers.parquet' AS __left__
    INNER JOIN 'orders.parquet' AS __right__
      ON __left__.customer_id = __right__.customer_id
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import sqlglot.expressions as exp

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.errors import SchemaError

if TYPE_CHECKING:
    from sqlearn.core.schema import Schema

_VALID_HOW = frozenset({"inner", "left", "right", "outer", "cross"})

_merge_counter: int = 0
_merge_lock = threading.Lock()


def _next_merge_name() -> str:
    """Generate the next auto-incremented merge view name.

    Thread-safe via module-level lock.

    Returns:
        Name like ``'__sqlearn_merge_0'``, ``'__sqlearn_merge_1'``, etc.
    """
    global _merge_counter  # noqa: PLW0603
    with _merge_lock:
        name = f"__sqlearn_merge_{_merge_counter}"
        _merge_counter += 1
    return name


def _source_to_table(source: str) -> exp.Expression:
    """Convert a source string to a sqlglot table or file expression.

    File paths (ending in .parquet, .csv, .json, .tsv) become quoted
    string literals readable by DuckDB.  Table names become ``exp.Table``.

    Args:
        source: File path or table name.

    Returns:
        sqlglot expression suitable for a FROM or JOIN clause.
    """
    lower = source.lower()
    if any(lower.endswith(ext) for ext in (".parquet", ".csv", ".json", ".tsv")):
        return exp.Literal.string(source)
    return exp.to_table(source)


def _build_join_condition(
    on: str | list[str] | None,
    left_on: str | list[str] | None,
    right_on: str | list[str] | None,
    left_alias: str,
    right_alias: str,
) -> exp.Expression:
    """Build the ON clause for a JOIN.

    Exactly one of ``on`` or ``left_on``/``right_on`` must be specified.

    Args:
        on: Shared join column(s).
        left_on: Left join column(s) (used with ``right_on``).
        right_on: Right join column(s) (used with ``left_on``).
        left_alias: Alias for the left table.
        right_alias: Alias for the right table.

    Returns:
        sqlglot expression for the ON clause.

    Raises:
        ValueError: If arguments are inconsistent.
    """
    if on is not None:
        cols = [on] if isinstance(on, str) else on
        left_cols = cols
        right_cols = cols
    elif left_on is not None and right_on is not None:
        left_cols = [left_on] if isinstance(left_on, str) else left_on
        right_cols = [right_on] if isinstance(right_on, str) else right_on
    else:
        msg = "Specify either 'on' or both 'left_on' and 'right_on'"
        raise ValueError(msg)

    if len(left_cols) != len(right_cols):
        msg = (
            f"left_on and right_on must have the same number of columns, "
            f"got {len(left_cols)} and {len(right_cols)}"
        )
        raise ValueError(msg)

    if not left_cols:
        msg = "Join key list must not be empty"
        raise ValueError(msg)

    conditions: list[exp.Expression] = []
    for lcol, rcol in zip(left_cols, right_cols, strict=True):
        eq = exp.EQ(
            this=exp.Column(
                this=exp.to_identifier(lcol),
                table=exp.to_identifier(left_alias),
            ),
            expression=exp.Column(
                this=exp.to_identifier(rcol),
                table=exp.to_identifier(right_alias),
            ),
        )
        conditions.append(eq)

    result = conditions[0]
    for cond in conditions[1:]:
        result = exp.And(this=result, expression=cond)
    return result


def _resolve_key_sets(
    on: str | list[str] | None,
    left_on: str | list[str] | None,
    right_on: str | list[str] | None,
) -> tuple[set[str], set[str], set[str]]:
    """Resolve join key arguments into sets.

    Args:
        on: Shared join column(s).
        left_on: Left join column(s).
        right_on: Right join column(s).

    Returns:
        Tuple of (shared_keys, left_keys, right_keys).
    """
    if on is not None:
        shared = {on} if isinstance(on, str) else set(on)
        return shared, shared, shared
    if left_on is not None and right_on is not None:
        lk = {left_on} if isinstance(left_on, str) else set(left_on)
        rk = {right_on} if isinstance(right_on, str) else set(right_on)
        return set(), lk, rk
    return set(), set(), set()


def _build_select_list(  # noqa: PLR0913
    left_cols: list[str],
    right_cols: list[str],
    left_alias: str,
    right_alias: str,
    suffix: tuple[str, str],
    shared_keys: set[str],
    right_keys: set[str],
    overlap: set[str],
) -> list[exp.Expression]:
    """Build the SELECT clause, handling overlapping column names.

    Join key columns from the left table are included once without suffix.
    Overlapping non-key columns receive the configured suffixes.

    Args:
        left_cols: Column names in the left source.
        right_cols: Column names in the right source.
        left_alias: Alias for the left table.
        right_alias: Alias for the right table.
        suffix: Tuple of (left_suffix, right_suffix) for overlapping names.
        shared_keys: Shared join key columns.
        right_keys: Right-side join key columns.
        overlap: Set of overlapping non-key column names.

    Returns:
        List of sqlglot expressions for the SELECT clause.
    """
    selections: list[exp.Expression] = []

    for col in left_cols:
        col_expr = exp.Column(
            this=exp.to_identifier(col),
            table=exp.to_identifier(left_alias),
        )
        if col in overlap:
            alias_name = f"{col}{suffix[0]}"
            selections.append(exp.Alias(this=col_expr, alias=exp.to_identifier(alias_name)))
        else:
            selections.append(col_expr)

    for col in right_cols:
        if col in shared_keys:
            continue
        col_expr = exp.Column(
            this=exp.to_identifier(col),
            table=exp.to_identifier(right_alias),
        )
        if col in overlap or (col in right_keys and col not in shared_keys and col in overlap):
            alias_name = f"{col}{suffix[1]}"
            selections.append(exp.Alias(this=col_expr, alias=exp.to_identifier(alias_name)))
        else:
            selections.append(col_expr)

    return selections


def merge(  # noqa: PLR0913
    left: str,
    right: str,
    *,
    on: str | list[str] | None = None,
    left_on: str | list[str] | None = None,
    right_on: str | list[str] | None = None,
    how: str = "inner",
    suffix: tuple[str, str] = ("_left", "_right"),
    backend: DuckDBBackend | None = None,
) -> str:
    """Join two data sources via SQL.

    Builds a sqlglot ``SELECT ... JOIN`` query and registers the result
    as a DuckDB view.  The returned view name can be used directly as
    input to :class:`~sqlearn.core.pipeline.Pipeline`.

    Exactly one of ``on`` or ``left_on``/``right_on`` must be provided
    (except for ``how='cross'`` which requires neither).

    Args:
        left: Left data source (file path or table name).
        right: Right data source (file path or table name).
        on: Column name(s) present in both sources for the join.
        left_on: Column name(s) in the left source.
        right_on: Column name(s) in the right source.
        how: Join type: ``'inner'``, ``'left'``, ``'right'``,
            ``'outer'``, or ``'cross'``.
        suffix: Tuple of ``(left_suffix, right_suffix)`` applied to
            overlapping column names that are not join keys.
        backend: DuckDB backend.  If None, a fresh in-memory backend
            is created.

    Returns:
        A view name string that can be passed as pipeline input.

    Raises:
        ValueError: If ``how`` is not a valid join type, or if join key
            arguments are inconsistent.
        SchemaError: If join key columns do not exist in the source schemas.

    Examples:
        Inner join on a shared key:

        >>> import sqlearn as sq
        >>> view = sq.merge("customers.parquet", "orders.parquet", on="customer_id")

        Left join with separate key columns:

        >>> view = sq.merge(
        ...     "a.parquet", "b.parquet", left_on="id", right_on="a_id", how="left"
        ... )

        Cross join (no keys):

        >>> view = sq.merge("a.parquet", "b.parquet", how="cross")

    See Also:
        :func:`~sqlearn.data.concat.concat`: Vertical concatenation (UNION ALL).
        :class:`~sqlearn.data.lookup.Lookup`: Mid-pipeline JOIN transformer.
    """
    if how not in _VALID_HOW:
        msg = f"Invalid join type: {how!r}. Must be one of {sorted(_VALID_HOW)}"
        raise ValueError(msg)

    be = backend if backend is not None else DuckDBBackend()

    # Validate join key arguments
    if how == "cross":
        if on is not None or left_on is not None or right_on is not None:
            msg = "Cross join does not accept 'on', 'left_on', or 'right_on'"
            raise ValueError(msg)
    else:
        _validate_join_keys(on, left_on, right_on)

    # Resolve schemas
    left_schema = be.describe(left)
    right_schema = be.describe(right)

    # Validate join columns exist in schemas
    if how != "cross":
        _validate_join_columns(on, left_on, right_on, left_schema, right_schema)

    # Build query
    left_alias = "__left__"
    right_alias = "__right__"

    query = _build_merge_query(
        left=left,
        right=right,
        on=on,
        left_on=left_on,
        right_on=right_on,
        how=how,
        suffix=suffix,
        left_alias=left_alias,
        right_alias=right_alias,
        left_cols=list(left_schema.columns.keys()),
        right_cols=list(right_schema.columns.keys()),
    )

    # Execute as a view
    view_name = _next_merge_name()
    sql = query.sql(dialect="duckdb")  # pyright: ignore[reportUnknownMemberType]
    conn = be._get_connection()  # noqa: SLF001
    conn.execute(f"CREATE OR REPLACE VIEW {view_name} AS {sql}")
    return view_name


def merge_query(  # noqa: PLR0913
    left: str,
    right: str,
    *,
    on: str | list[str] | None = None,
    left_on: str | list[str] | None = None,
    right_on: str | list[str] | None = None,
    how: str = "inner",
    suffix: tuple[str, str] = ("_left", "_right"),
    backend: DuckDBBackend | None = None,
) -> exp.Expression:
    """Build the sqlglot AST for a merge without executing it.

    Same signature as :func:`merge` but returns the sqlglot expression
    instead of creating a view.  Useful for inspection and testing.

    Args:
        left: Left data source (file path or table name).
        right: Right data source (file path or table name).
        on: Column name(s) present in both sources for the join.
        left_on: Column name(s) in the left source.
        right_on: Column name(s) in the right source.
        how: Join type.
        suffix: Suffixes for overlapping column names.
        backend: DuckDB backend for schema introspection.

    Returns:
        sqlglot SELECT expression with the JOIN.

    Raises:
        ValueError: If arguments are inconsistent.
        SchemaError: If join columns don't exist.
    """
    if how not in _VALID_HOW:
        msg = f"Invalid join type: {how!r}. Must be one of {sorted(_VALID_HOW)}"
        raise ValueError(msg)

    be = backend if backend is not None else DuckDBBackend()

    if how == "cross":
        if on is not None or left_on is not None or right_on is not None:
            msg = "Cross join does not accept 'on', 'left_on', or 'right_on'"
            raise ValueError(msg)
    else:
        _validate_join_keys(on, left_on, right_on)

    left_schema = be.describe(left)
    right_schema = be.describe(right)

    if how != "cross":
        _validate_join_columns(on, left_on, right_on, left_schema, right_schema)

    left_alias = "__left__"
    right_alias = "__right__"

    return _build_merge_query(
        left=left,
        right=right,
        on=on,
        left_on=left_on,
        right_on=right_on,
        how=how,
        suffix=suffix,
        left_alias=left_alias,
        right_alias=right_alias,
        left_cols=list(left_schema.columns.keys()),
        right_cols=list(right_schema.columns.keys()),
    )


# ── Internal helpers ─────────────────────────────────────────────


def _validate_join_keys(
    on: str | list[str] | None,
    left_on: str | list[str] | None,
    right_on: str | list[str] | None,
) -> None:
    """Validate that exactly one of on or left_on/right_on is given.

    Args:
        on: Shared join key(s).
        left_on: Left join key(s).
        right_on: Right join key(s).

    Raises:
        ValueError: If arguments are inconsistent.
    """
    has_on = on is not None
    has_left_right = left_on is not None or right_on is not None

    if has_on and has_left_right:
        msg = "Cannot specify both 'on' and 'left_on'/'right_on'"
        raise ValueError(msg)

    if not has_on and not has_left_right:
        msg = "Must specify either 'on' or both 'left_on' and 'right_on'"
        raise ValueError(msg)

    if has_left_right and (left_on is None or right_on is None):
        msg = "Both 'left_on' and 'right_on' must be specified together"
        raise ValueError(msg)


def _validate_join_columns(
    on: str | list[str] | None,
    left_on: str | list[str] | None,
    right_on: str | list[str] | None,
    left_schema: Schema,
    right_schema: Schema,
) -> None:
    """Validate that join columns exist in their respective schemas.

    Args:
        on: Shared join key(s).
        left_on: Left join key(s).
        right_on: Right join key(s).
        left_schema: Schema of the left source.
        right_schema: Schema of the right source.

    Raises:
        SchemaError: If join columns don't exist.
    """
    if on is not None:
        cols = [on] if isinstance(on, str) else on
        left_missing = set(cols) - set(left_schema.columns)
        if left_missing:
            msg = f"Join column(s) not found in left source: {sorted(left_missing)}"
            raise SchemaError(msg)
        right_missing = set(cols) - set(right_schema.columns)
        if right_missing:
            msg = f"Join column(s) not found in right source: {sorted(right_missing)}"
            raise SchemaError(msg)
    else:
        if left_on is not None:
            lcols = [left_on] if isinstance(left_on, str) else left_on
            left_missing = set(lcols) - set(left_schema.columns)
            if left_missing:
                msg = f"Join column(s) not found in left source: {sorted(left_missing)}"
                raise SchemaError(msg)
        if right_on is not None:
            rcols = [right_on] if isinstance(right_on, str) else right_on
            right_missing = set(rcols) - set(right_schema.columns)
            if right_missing:
                msg = f"Join column(s) not found in right source: {sorted(right_missing)}"
                raise SchemaError(msg)


def _build_merge_query(  # noqa: PLR0913
    *,
    left: str,
    right: str,
    on: str | list[str] | None,
    left_on: str | list[str] | None,
    right_on: str | list[str] | None,
    how: str,
    suffix: tuple[str, str],
    left_alias: str,
    right_alias: str,
    left_cols: list[str],
    right_cols: list[str],
) -> exp.Expression:
    """Build the full SELECT ... JOIN query as a sqlglot AST.

    Args:
        left: Left source.
        right: Right source.
        on: Shared join key(s).
        left_on: Left join key(s).
        right_on: Right join key(s).
        how: Join type string.
        suffix: Column name suffixes for overlaps.
        left_alias: Left table alias.
        right_alias: Right table alias.
        left_cols: Column names in left source.
        right_cols: Column names in right source.

    Returns:
        sqlglot SELECT expression.
    """
    # Resolve key sets and compute overlap
    if how == "cross":
        shared_keys: set[str] = set()
        right_keys: set[str] = set()
    else:
        shared_keys, _, right_keys = _resolve_key_sets(on, left_on, right_on)

    left_set = set(left_cols)
    right_set = set(right_cols)
    overlap = (left_set & right_set) - shared_keys

    # Build SELECT list
    selections = _build_select_list(
        left_cols=left_cols,
        right_cols=right_cols,
        left_alias=left_alias,
        right_alias=right_alias,
        suffix=suffix,
        shared_keys=shared_keys,
        right_keys=right_keys,
        overlap=overlap,
    )

    # Build FROM with left table
    left_table = exp.Alias(
        this=_source_to_table(left),
        alias=exp.to_identifier(left_alias),
    )

    # Build JOIN
    right_table = _source_to_table(right)

    query = exp.Select(expressions=selections).from_(left_table)  # pyright: ignore[reportUnknownMemberType]

    if how == "cross":
        join = exp.Join(
            this=exp.Alias(
                this=right_table,
                alias=exp.to_identifier(right_alias),
            ),
            kind="CROSS",
        )
    else:
        join_condition = _build_join_condition(
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_alias=left_alias,
            right_alias=right_alias,
        )
        # Map how -> sqlglot Join(side=, kind=) parameters
        join_side = ""
        join_kind = ""
        if how == "left":
            join_side = "LEFT"
        elif how == "right":
            join_side = "RIGHT"
        elif how == "outer":
            join_side = "FULL"
            join_kind = "OUTER"

        join = exp.Join(
            this=exp.Alias(
                this=right_table,
                alias=exp.to_identifier(right_alias),
            ),
            on=join_condition,
            side=join_side,
            kind=join_kind,
        )

    return query.join(join)  # pyright: ignore[reportUnknownMemberType]
