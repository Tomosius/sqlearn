"""concat() -- SQL UNION ALL wrapper for vertical concatenation.

Combines multiple data sources row-wise using SQL ``UNION ALL``.
All SQL is constructed via sqlglot ASTs, never raw strings.

Generated SQL example (two sources)::

    SELECT customer_id, name, city
    FROM 'train.parquet'
    UNION ALL
    SELECT customer_id, name, city
    FROM 'test.parquet'
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import sqlglot.expressions as exp

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.errors import SchemaError

if TYPE_CHECKING:
    from sqlearn.core.schema import Schema

_concat_counter: int = 0
_concat_lock = threading.Lock()


def _next_concat_name() -> str:
    """Generate the next auto-incremented concat view name.

    Thread-safe via module-level lock.

    Returns:
        Name like ``'__sqlearn_concat_0'``, ``'__sqlearn_concat_1'``, etc.
    """
    global _concat_counter  # noqa: PLW0603
    with _concat_lock:
        name = f"__sqlearn_concat_{_concat_counter}"
        _concat_counter += 1
    return name


def _source_to_from(source: str) -> exp.Expression:
    """Convert a source string to a sqlglot FROM-suitable expression.

    File paths (ending in .parquet, .csv, .json, .tsv) become quoted
    string literals readable by DuckDB.  Table names become ``exp.Table``.

    Args:
        source: File path or table name.

    Returns:
        sqlglot expression suitable for a FROM clause.
    """
    lower = source.lower()
    if any(lower.endswith(ext) for ext in (".parquet", ".csv", ".json", ".tsv")):
        return exp.Literal.string(source)
    return exp.to_table(source)


def _build_select_for_source(
    source: str,
    columns: list[str],
) -> exp.Select:
    """Build a SELECT for one source with explicit column list.

    Args:
        source: File path or table name.
        columns: Ordered list of column names to select.

    Returns:
        sqlglot SELECT expression.
    """
    col_exprs: list[exp.Expression] = [exp.Column(this=exp.to_identifier(col)) for col in columns]
    return exp.Select(expressions=col_exprs).from_(  # pyright: ignore[reportUnknownMemberType]
        _source_to_from(source)
    )


def _validate_schemas_match(
    schemas: list[Schema],
    sources: list[str],
) -> None:
    """Validate that all schemas have the same column names.

    Args:
        schemas: List of schemas for each source.
        sources: List of source names (for error messages).

    Raises:
        SchemaError: If column names differ between sources.
    """
    first_cols = set(schemas[0].columns.keys())
    for i in range(1, len(schemas)):
        other_cols = set(schemas[i].columns.keys())
        if first_cols != other_cols:
            missing = first_cols - other_cols
            extra = other_cols - first_cols
            parts: list[str] = [f"Schema mismatch between '{sources[0]}' and '{sources[i]}'."]
            if missing:
                parts.append(f"Missing in '{sources[i]}': {sorted(missing)}")
            if extra:
                parts.append(f"Extra in '{sources[i]}': {sorted(extra)}")
            msg = " ".join(parts)
            raise SchemaError(msg)


def _align_columns(
    schemas: list[Schema],
) -> list[str]:
    """Compute the union of all column names, preserving order.

    Columns from the first schema appear first, then any new columns
    from subsequent schemas are appended.

    Args:
        schemas: List of schemas.

    Returns:
        Ordered list of all unique column names.
    """
    seen: set[str] = set()
    result: list[str] = []
    for schema in schemas:
        for col in schema.columns:
            if col not in seen:
                seen.add(col)
                result.append(col)
    return result


def _build_aligned_select(
    source: str,
    schema: Schema,
    all_columns: list[str],
) -> exp.Select:
    """Build a SELECT for one source, NULLing missing columns.

    If a column does not exist in this source's schema, a
    ``NULL AS col_name`` expression is used instead.

    Args:
        source: File path or table name.
        schema: Schema for this source.
        all_columns: Full ordered list of columns in the union.

    Returns:
        sqlglot SELECT expression with aligned columns.
    """
    col_exprs: list[exp.Expression] = []
    for col in all_columns:
        if col in schema.columns:
            col_exprs.append(exp.Column(this=exp.to_identifier(col)))
        else:
            col_exprs.append(
                exp.Alias(
                    this=exp.Null(),
                    alias=exp.to_identifier(col),
                )
            )
    return exp.Select(expressions=col_exprs).from_(  # pyright: ignore[reportUnknownMemberType]
        _source_to_from(source)
    )


def concat(
    *sources: str,
    align: bool = False,
    backend: DuckDBBackend | None = None,
) -> str:
    """Concatenate multiple data sources vertically via SQL UNION ALL.

    Builds a sqlglot ``UNION ALL`` query and registers the result as a
    DuckDB view.  The returned view name can be used directly as input
    to :class:`~sqlearn.core.pipeline.Pipeline`.

    By default, all sources must have exactly the same column names.
    With ``align=True``, missing columns are filled with ``NULL``.

    Args:
        *sources: Two or more data sources (file paths or table names).
        align: If True, align schemas by filling missing columns with
            NULL.  If False (default), schemas must match exactly.
        backend: DuckDB backend.  If None, a fresh in-memory backend
            is created.

    Returns:
        A view name string that can be passed as pipeline input.

    Raises:
        ValueError: If fewer than two sources are provided.
        SchemaError: If schemas don't match and ``align=False``.

    Examples:
        Concatenate two tables:

        >>> import sqlearn as sq
        >>> view = sq.concat("train.parquet", "test.parquet")

        Concatenate with column alignment:

        >>> view = sq.concat("a.parquet", "b.parquet", align=True)

        Three or more sources:

        >>> view = sq.concat("jan.parquet", "feb.parquet", "mar.parquet")

    See Also:
        :func:`~sqlearn.data.merge.merge`: Horizontal join (SQL JOIN).
        :class:`~sqlearn.data.lookup.Lookup`: Mid-pipeline JOIN transformer.
    """
    if len(sources) < _MIN_SOURCES:
        msg = f"concat() requires at least 2 sources, got {len(sources)}"
        raise ValueError(msg)

    be = backend if backend is not None else DuckDBBackend()

    # Resolve schemas
    source_list = list(sources)
    schemas = [be.describe(s) for s in source_list]

    # Build query
    query = _build_concat_query(
        sources=source_list,
        schemas=schemas,
        align=align,
    )

    # Execute as a view
    view_name = _next_concat_name()
    sql = query.sql(dialect="duckdb")  # pyright: ignore[reportUnknownMemberType]
    conn = be._get_connection()  # noqa: SLF001
    conn.execute(f"CREATE OR REPLACE VIEW {view_name} AS {sql}")
    return view_name


def concat_query(
    *sources: str,
    align: bool = False,
    backend: DuckDBBackend | None = None,
) -> exp.Expression:
    """Build the sqlglot AST for a concat without executing it.

    Same signature as :func:`concat` but returns the sqlglot expression
    instead of creating a view.  Useful for inspection and testing.

    Args:
        *sources: Two or more data sources.
        align: If True, align schemas by filling missing columns with NULL.
        backend: DuckDB backend for schema introspection.

    Returns:
        sqlglot UNION ALL expression.

    Raises:
        ValueError: If fewer than two sources are provided.
        SchemaError: If schemas don't match and ``align=False``.
    """
    if len(sources) < _MIN_SOURCES:
        msg = f"concat_query() requires at least 2 sources, got {len(sources)}"
        raise ValueError(msg)

    be = backend if backend is not None else DuckDBBackend()

    source_list = list(sources)
    schemas = [be.describe(s) for s in source_list]

    return _build_concat_query(
        sources=source_list,
        schemas=schemas,
        align=align,
    )


_MIN_SOURCES = 2


def _build_concat_query(
    *,
    sources: list[str],
    schemas: list[Schema],
    align: bool,
) -> exp.Expression:
    """Build the UNION ALL query as a sqlglot AST.

    Args:
        sources: List of source names.
        schemas: Corresponding schemas.
        align: Whether to align mismatched schemas.

    Returns:
        sqlglot UNION ALL expression.

    Raises:
        SchemaError: If schemas don't match and align is False.
    """
    if align:
        all_cols = _align_columns(schemas)
        selects = [
            _build_aligned_select(src, schema, all_cols)
            for src, schema in zip(sources, schemas, strict=True)
        ]
    else:
        _validate_schemas_match(schemas, sources)
        # Use column order from first source
        col_order = list(schemas[0].columns.keys())
        selects = [_build_select_for_source(src, col_order) for src in sources]

    # Chain UNION ALL
    result: exp.Expression = selects[0]
    for select in selects[1:]:
        result = exp.Union(
            this=result,
            expression=select,
            distinct=False,
        )

    return result
