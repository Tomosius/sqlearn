"""Backend protocol and DuckDB implementation.

Defines the Backend protocol that all execution backends must satisfy, and
provides DuckDBBackend as the default in-memory/file-based implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import duckdb

from sqlearn.core.errors import FitError
from sqlearn.core.schema import Schema

if TYPE_CHECKING:
    from types import TracebackType

    import sqlglot.expressions as exp


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Backend(Protocol):
    """Execution backend protocol for sqlearn.

    All backends must implement this interface. The default implementation is
    :class:`DuckDBBackend`. Custom backends can target any sqlglot-supported
    database (Postgres, MySQL, Snowflake, BigQuery).
    """

    @property
    def dialect(self) -> str:
        """SQL dialect identifier (e.g. ``'duckdb'``, ``'postgres'``).

        Returns:
            Dialect string recognized by sqlglot.
        """
        ...

    def execute(self, query: exp.Expression) -> list[dict[str, Any]]:
        """Execute a sqlglot expression and return all rows.

        Args:
            query: A sqlglot AST expression to execute.

        Returns:
            List of row dicts mapping column name to value.
        """
        ...

    def fetch_one(self, query: exp.Expression) -> dict[str, Any]:
        """Execute a sqlglot expression and return the first row.

        Args:
            query: A sqlglot AST expression to execute.

        Returns:
            Single row dict mapping column name to value.

        Raises:
            FitError: If the query returns no rows.
        """
        ...

    def describe(self, source: str) -> Schema:
        """Infer schema from a table name or file path.

        Args:
            source: Table name or file path (parquet, csv).

        Returns:
            Schema with raw DuckDB type strings.
        """
        ...

    def register(self, data: object, name: str) -> str:
        """Register an in-memory DataFrame under a table name.

        Args:
            data: DataFrame to register (pandas, polars, arrow, etc.).
            name: Table name to register under.

        Returns:
            The registered table name.
        """
        ...

    def supports(self, feature: str) -> bool:
        """Check whether the backend supports a named feature.

        Args:
            feature: Feature name to check (e.g. ``'median'``, ``'parquet'``).

        Returns:
            True if the feature is supported.
        """
        ...

    def close(self) -> None:
        """Close the backend connection and release resources."""
        ...

    def __enter__(self) -> Backend:
        """Enter the context manager."""
        ...

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        """Exit the context manager, closing the connection."""
        ...


# ---------------------------------------------------------------------------
# DuckDB implementation
# ---------------------------------------------------------------------------

_FILE_EXTENSIONS = (".parquet", ".csv", ".json", ".tsv")


def _is_file_path(source: str) -> bool:
    """Return True if source looks like a file path rather than a table name."""
    lower = source.lower()
    return any(lower.endswith(ext) for ext in _FILE_EXTENSIONS)


def _describe_source(source: str) -> str:
    """Build a DESCRIBE SQL string for the given source.

    File paths are single-quoted; table names are used as-is.

    Args:
        source: Table name or file path.

    Returns:
        DESCRIBE SQL string.
    """
    if _is_file_path(source):
        escaped = source.replace("'", "''")
        return f"DESCRIBE '{escaped}'"
    return f"DESCRIBE {source}"


class DuckDBBackend:
    """DuckDB execution backend for sqlearn.

    Supports lazy in-memory connections, lazy file-based connections, and
    reuse of an existing :class:`duckdb.DuckDBPyConnection`.

    Args:
        connection: One of:

            - ``None`` (default): lazy in-memory connection, created on first use.
            - ``str``: path to a DuckDB database file, opened lazily on first use.
            - :class:`duckdb.DuckDBPyConnection`: an existing connection to reuse.

    Example::

        # In-memory (default)
        backend = DuckDBBackend()

        # File-based
        backend = DuckDBBackend("/path/to/db.duckdb")

        # Existing connection
        conn = duckdb.connect()
        backend = DuckDBBackend(conn)

        # Context manager
        with DuckDBBackend() as backend:
            schema = backend.describe("my_table")
    """

    def __init__(
        self,
        connection: str | duckdb.DuckDBPyConnection | None = None,
    ) -> None:
        self._connection: duckdb.DuckDBPyConnection | None
        self._path: str | None
        if isinstance(connection, duckdb.DuckDBPyConnection):
            self._connection = connection
            self._path = None
        elif isinstance(connection, str):
            self._connection = None
            self._path = connection
        else:
            self._connection = None
            self._path = None

    # --- Internal ---

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Return the active connection, creating it lazily if needed.

        Returns:
            An active :class:`duckdb.DuckDBPyConnection`.
        """
        if self._connection is None:
            if self._path is not None:
                self._connection = duckdb.connect(self._path)
            else:
                self._connection = duckdb.connect()
        return self._connection

    # --- Protocol implementation ---

    @property
    def dialect(self) -> str:
        """SQL dialect identifier.

        Returns:
            Always ``'duckdb'``.
        """
        return "duckdb"

    def execute(self, query: exp.Expression) -> list[dict[str, Any]]:
        """Execute a sqlglot expression and return all rows as dicts.

        The expression is transpiled to DuckDB SQL via sqlglot before execution.

        Args:
            query: A sqlglot AST expression to execute.

        Returns:
            List of row dicts. Empty list if the query returns no rows.
        """
        sql: str = query.sql(dialect="duckdb")  # pyright: ignore[reportUnknownMemberType]
        conn = self._get_connection()
        result = conn.execute(sql)
        rows = result.fetchall()
        if not rows:
            return []
        col_names = [desc[0] for desc in result.description]
        return [dict(zip(col_names, row, strict=True)) for row in rows]

    def fetch_one(self, query: exp.Expression) -> dict[str, Any]:
        """Execute a sqlglot expression and return the first row as a dict.

        Args:
            query: A sqlglot AST expression to execute.

        Returns:
            Single row dict mapping column name to value.

        Raises:
            FitError: If the query returns no rows.
        """
        sql: str = query.sql(dialect="duckdb")  # pyright: ignore[reportUnknownMemberType]
        conn = self._get_connection()
        result = conn.execute(sql)
        row = result.fetchone()
        if row is None:
            msg = "Query returned no rows"
            raise FitError(msg)
        col_names = [desc[0] for desc in result.description]
        return dict(zip(col_names, row, strict=True))

    def describe(self, source: str) -> Schema:
        """Infer schema from a table name or file path.

        File paths ending in ``.parquet`` or ``.csv`` are automatically quoted
        in the DESCRIBE statement. Raw DuckDB type strings are stored without
        normalization.

        Args:
            source: Table name or file path.

        Returns:
            :class:`Schema` mapping column names to raw DuckDB type strings.

        Raises:
            duckdb.CatalogException: If the source does not exist.
        """
        sql = _describe_source(source)
        conn = self._get_connection()
        rows = conn.execute(sql).fetchall()
        # DESCRIBE rows: (column_name, column_type, null, key, default, extra)
        columns = {row[0]: row[1] for row in rows}
        return Schema(columns)

    def register(self, data: object, name: str) -> str:
        """Register an in-memory DataFrame under a table name.

        Any object accepted by DuckDB's ``conn.register()`` is valid (pandas,
        polars, PyArrow, etc.).

        Args:
            data: DataFrame or relation to register.
            name: Table name to register under.

        Returns:
            The registered table name (same as ``name``).
        """
        conn = self._get_connection()
        conn.register(name, data)
        return name

    def supports(self, feature: str) -> bool:  # noqa: ARG002
        """Check whether DuckDB supports a named feature.

        DuckDB supports all features tracked by sqlearn, so this always
        returns ``True``.

        Args:
            feature: Feature name to check.

        Returns:
            Always ``True``.
        """
        return True

    def close(self) -> None:
        """Close the DuckDB connection and release resources.

        Safe to call even if no connection has been opened yet. After calling
        ``close()``, the backend can no longer be used.
        """
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> DuckDBBackend:
        """Enter the context manager.

        Returns:
            This backend instance.
        """
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        """Exit the context manager, closing the connection.

        Args:
            _exc_type: Exception type, if any.
            _exc_val: Exception value, if any.
            _exc_tb: Exception traceback, if any.
        """
        self.close()

    def __repr__(self) -> str:
        """Show backend type and connection state.

        Returns:
            Human-readable representation.
        """
        if self._connection is None:
            state = f"path={self._path!r}" if self._path else "lazy"
        else:
            state = "connected"
        return f"DuckDBBackend({state})"
