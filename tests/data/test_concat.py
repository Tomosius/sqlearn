"""Tests for sqlearn.data.concat -- concat() SQL UNION ALL wrapper."""

# ruff: noqa: S608

from __future__ import annotations

import duckdb
import pytest
import sqlglot.expressions as exp

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.errors import SchemaError
from sqlearn.data.concat import concat, concat_query

# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def backend() -> DuckDBBackend:
    """Create DuckDB backend with test tables for concat tests."""
    conn = duckdb.connect()
    conn.execute(
        "CREATE TABLE train AS SELECT * FROM VALUES "
        "(1, 'Alice', 100.0), (2, 'Bob', 200.0) "
        "t(id, name, score)"
    )
    conn.execute(
        "CREATE TABLE test AS SELECT * FROM VALUES "
        "(3, 'Carol', 300.0), (4, 'Dave', 400.0) "
        "t(id, name, score)"
    )
    return DuckDBBackend(connection=conn)


@pytest.fixture
def three_tables_backend() -> DuckDBBackend:
    """Backend with three tables for multi-source concat tests."""
    conn = duckdb.connect()
    conn.execute("CREATE TABLE a AS SELECT * FROM VALUES (1, 'x'), (2, 'y') t(id, val)")
    conn.execute("CREATE TABLE b AS SELECT * FROM VALUES (3, 'z') t(id, val)")
    conn.execute("CREATE TABLE c AS SELECT * FROM VALUES (4, 'w'), (5, 'v') t(id, val)")
    return DuckDBBackend(connection=conn)


@pytest.fixture
def mismatched_backend() -> DuckDBBackend:
    """Backend with tables having different column sets."""
    conn = duckdb.connect()
    conn.execute("CREATE TABLE a AS SELECT * FROM VALUES (1, 'x') t(id, name)")
    conn.execute("CREATE TABLE b AS SELECT * FROM VALUES (2, 100) t(id, score)")
    return DuckDBBackend(connection=conn)


# ── Validation tests ─────────────────────────────────────────────


class TestValidation:
    """Test concat() argument validation."""

    def test_fewer_than_two_sources_raises(self, backend: DuckDBBackend) -> None:
        """Single source raises ValueError."""
        with pytest.raises(ValueError, match="at least 2 sources"):
            concat("train", backend=backend)

    def test_zero_sources_raises(self, backend: DuckDBBackend) -> None:
        """No sources raises ValueError."""
        with pytest.raises(ValueError, match="at least 2 sources"):
            concat(backend=backend)

    def test_mismatched_schemas_raises(self, mismatched_backend: DuckDBBackend) -> None:
        """Different column names raise SchemaError when align=False."""
        with pytest.raises(SchemaError, match="Schema mismatch"):
            concat("a", "b", backend=mismatched_backend)

    def test_mismatched_schemas_with_align_ok(self, mismatched_backend: DuckDBBackend) -> None:
        """Different column names work with align=True."""
        view = concat("a", "b", align=True, backend=mismatched_backend)
        conn = mismatched_backend._get_connection()
        rows = conn.execute(f"SELECT * FROM {view}").fetchall()
        assert len(rows) == 2


# ── Two-source concat tests ──────────────────────────────────────


class TestTwoSources:
    """Test concatenation of two sources."""

    def test_basic_concat(self, backend: DuckDBBackend) -> None:
        """Two tables concatenate correctly."""
        view = concat("train", "test", backend=backend)
        conn = backend._get_connection()
        rows = conn.execute(f"SELECT * FROM {view}").fetchall()
        assert len(rows) == 4

    def test_column_count_preserved(self, backend: DuckDBBackend) -> None:
        """Column count matches source tables."""
        view = concat("train", "test", backend=backend)
        conn = backend._get_connection()
        result = conn.execute(f"SELECT * FROM {view}")
        col_names = [desc[0] for desc in result.description]
        assert len(col_names) == 3  # id, name, score

    def test_column_names_preserved(self, backend: DuckDBBackend) -> None:
        """Column names match source tables."""
        view = concat("train", "test", backend=backend)
        conn = backend._get_connection()
        result = conn.execute(f"SELECT * FROM {view}")
        col_names = [desc[0] for desc in result.description]
        assert "id" in col_names
        assert "name" in col_names
        assert "score" in col_names

    def test_values_preserved(self, backend: DuckDBBackend) -> None:
        """All values from both sources are present."""
        view = concat("train", "test", backend=backend)
        conn = backend._get_connection()
        ids = conn.execute(f"SELECT id FROM {view} ORDER BY id").fetchall()
        assert [row[0] for row in ids] == [1, 2, 3, 4]

    def test_order_is_first_then_second(self, backend: DuckDBBackend) -> None:
        """Rows from first source appear before second (UNION ALL preserves order)."""
        view = concat("train", "test", backend=backend)
        conn = backend._get_connection()
        names = conn.execute(f"SELECT name FROM {view}").fetchall()
        # Order: Alice, Bob (from train), Carol, Dave (from test)
        assert names[0][0] == "Alice"
        assert names[1][0] == "Bob"
        assert names[2][0] == "Carol"
        assert names[3][0] == "Dave"


# ── Three-or-more source concat tests ────────────────────────────


class TestMultipleSources:
    """Test concatenation of three or more sources."""

    def test_three_sources(self, three_tables_backend: DuckDBBackend) -> None:
        """Three tables concatenate correctly."""
        view = concat("a", "b", "c", backend=three_tables_backend)
        conn = three_tables_backend._get_connection()
        rows = conn.execute(f"SELECT * FROM {view}").fetchall()
        assert len(rows) == 5  # 2 + 1 + 2

    def test_three_sources_values(self, three_tables_backend: DuckDBBackend) -> None:
        """Values from all three sources are present."""
        view = concat("a", "b", "c", backend=three_tables_backend)
        conn = three_tables_backend._get_connection()
        ids = conn.execute(f"SELECT id FROM {view} ORDER BY id").fetchall()
        assert [row[0] for row in ids] == [1, 2, 3, 4, 5]

    def test_four_sources(self) -> None:
        """Four sources concatenate correctly."""
        conn = duckdb.connect()
        for name, val in [("t1", 1), ("t2", 2), ("t3", 3), ("t4", 4)]:
            conn.execute(f"CREATE TABLE {name} AS SELECT {val} AS x")
        be = DuckDBBackend(connection=conn)
        view = concat("t1", "t2", "t3", "t4", backend=be)
        rows = conn.execute(f"SELECT * FROM {view}").fetchall()
        assert len(rows) == 4


# ── Column alignment tests ───────────────────────────────────────


class TestAlignment:
    """Test column alignment with align=True."""

    def test_aligned_fills_nulls(self, mismatched_backend: DuckDBBackend) -> None:
        """Missing columns are filled with NULL when aligned."""
        view = concat("a", "b", align=True, backend=mismatched_backend)
        conn = mismatched_backend._get_connection()
        rows = conn.execute(f"SELECT * FROM {view} ORDER BY id").fetchall()
        # Table a has (id, name), table b has (id, score)
        # Result should have (id, name, score) with NULLs
        assert len(rows) == 2

    def test_aligned_column_union(self, mismatched_backend: DuckDBBackend) -> None:
        """Aligned result has union of all column names."""
        view = concat("a", "b", align=True, backend=mismatched_backend)
        conn = mismatched_backend._get_connection()
        result = conn.execute(f"SELECT * FROM {view}")
        col_names = [desc[0] for desc in result.description]
        assert "id" in col_names
        assert "name" in col_names
        assert "score" in col_names

    def test_aligned_preserves_existing_values(self) -> None:
        """Existing values are preserved during alignment."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE a AS SELECT 1 AS id, 'x' AS name")
        conn.execute("CREATE TABLE b AS SELECT 2 AS id, 100 AS score")
        be = DuckDBBackend(connection=conn)
        view = concat("a", "b", align=True, backend=be)
        rows = conn.execute(f"SELECT id, name FROM {view} WHERE id = 1").fetchall()
        assert rows[0] == (1, "x")

    def test_aligned_null_for_missing(self) -> None:
        """NULL is used for columns missing from a source."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE a AS SELECT 1 AS id, 'x' AS name")
        conn.execute("CREATE TABLE b AS SELECT 2 AS id, 100 AS score")
        be = DuckDBBackend(connection=conn)
        view = concat("a", "b", align=True, backend=be)
        rows = conn.execute(f"SELECT id, score FROM {view} WHERE id = 1").fetchall()
        assert rows[0][1] is None  # score is NULL for table a

    def test_aligned_same_schemas_identical_to_strict(self, backend: DuckDBBackend) -> None:
        """align=True with identical schemas matches strict concat."""
        view_strict = concat("train", "test", align=False, backend=backend)
        view_aligned = concat("train", "test", align=True, backend=backend)
        conn = backend._get_connection()
        rows_strict = conn.execute(f"SELECT * FROM {view_strict} ORDER BY id").fetchall()
        rows_aligned = conn.execute(f"SELECT * FROM {view_aligned} ORDER BY id").fetchall()
        assert rows_strict == rows_aligned

    def test_aligned_three_sources(self) -> None:
        """Alignment works with three sources having different schemas."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE a AS SELECT 1 AS id, 'x' AS name")
        conn.execute("CREATE TABLE b AS SELECT 2 AS id, 100 AS score")
        conn.execute("CREATE TABLE c AS SELECT 3 AS id, TRUE AS active")
        be = DuckDBBackend(connection=conn)
        view = concat("a", "b", "c", align=True, backend=be)
        result = conn.execute(f"SELECT * FROM {view}")
        col_names = [desc[0] for desc in result.description]
        assert len(col_names) == 4  # id, name, score, active
        rows = conn.execute(f"SELECT * FROM {view}").fetchall()
        assert len(rows) == 3


# ── concat_query() tests ─────────────────────────────────────────


class TestConcatQuery:
    """Test concat_query() returns sqlglot AST without executing."""

    def test_returns_expression(self, backend: DuckDBBackend) -> None:
        """concat_query returns a sqlglot expression."""
        q = concat_query("train", "test", backend=backend)
        assert isinstance(q, exp.Select | exp.Union)

    def test_contains_union(self, backend: DuckDBBackend) -> None:
        """Result contains a Union node for multi-source."""
        q = concat_query("train", "test", backend=backend)
        # The top level should be a Union
        assert isinstance(q, exp.Union)

    def test_sql_is_valid(self, backend: DuckDBBackend) -> None:
        """Generated SQL can be executed by DuckDB."""
        q = concat_query("train", "test", backend=backend)
        sql = q.sql(dialect="duckdb")
        conn = backend._get_connection()
        rows = conn.execute(sql).fetchall()
        assert len(rows) == 4

    def test_fewer_than_two_raises(self, backend: DuckDBBackend) -> None:
        """Single source raises ValueError."""
        with pytest.raises(ValueError, match="at least 2 sources"):
            concat_query("train", backend=backend)

    def test_three_source_query(self, three_tables_backend: DuckDBBackend) -> None:
        """Three-source query is valid SQL."""
        q = concat_query("a", "b", "c", backend=three_tables_backend)
        sql = q.sql(dialect="duckdb")
        conn = three_tables_backend._get_connection()
        rows = conn.execute(sql).fetchall()
        assert len(rows) == 5


# ── Return value tests ───────────────────────────────────────────


class TestReturnValue:
    """Test that concat() returns a usable view name."""

    def test_returns_string(self, backend: DuckDBBackend) -> None:
        """concat() returns a string."""
        result = concat("train", "test", backend=backend)
        assert isinstance(result, str)

    def test_view_name_prefix(self, backend: DuckDBBackend) -> None:
        """View name has expected prefix."""
        result = concat("train", "test", backend=backend)
        assert result.startswith("__sqlearn_concat_")

    def test_view_is_queryable(self, backend: DuckDBBackend) -> None:
        """Returned view can be queried."""
        view = concat("train", "test", backend=backend)
        conn = backend._get_connection()
        rows = conn.execute(f"SELECT COUNT(*) FROM {view}").fetchone()
        assert rows is not None
        assert rows[0] == 4

    def test_unique_view_names(self, backend: DuckDBBackend) -> None:
        """Multiple concats produce unique view names."""
        v1 = concat("train", "test", backend=backend)
        v2 = concat("train", "test", backend=backend)
        assert v1 != v2


# ── Edge cases ───────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases and unusual inputs."""

    def test_single_row_tables(self) -> None:
        """Concat works with single-row tables."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE a AS SELECT 1 AS x")
        conn.execute("CREATE TABLE b AS SELECT 2 AS x")
        be = DuckDBBackend(connection=conn)
        view = concat("a", "b", backend=be)
        rows = conn.execute(f"SELECT * FROM {view}").fetchall()
        assert len(rows) == 2

    def test_single_column_tables(self) -> None:
        """Concat works with single-column tables."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE a AS SELECT * FROM VALUES (1), (2) t(x)")
        conn.execute("CREATE TABLE b AS SELECT * FROM VALUES (3) t(x)")
        be = DuckDBBackend(connection=conn)
        view = concat("a", "b", backend=be)
        rows = conn.execute(f"SELECT * FROM {view}").fetchall()
        assert len(rows) == 3

    def test_empty_table_concat(self) -> None:
        """Concat with one empty table produces other table's rows."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE a AS SELECT * FROM VALUES (1, 'x'), (2, 'y') t(id, val)")
        conn.execute("CREATE TABLE b (id INTEGER, val VARCHAR)")
        be = DuckDBBackend(connection=conn)
        view = concat("a", "b", backend=be)
        rows = conn.execute(f"SELECT * FROM {view}").fetchall()
        assert len(rows) == 2

    def test_null_values_preserved(self) -> None:
        """NULL values are preserved through concat."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE a AS SELECT 1 AS id, NULL::VARCHAR AS name")
        conn.execute("CREATE TABLE b AS SELECT 2 AS id, 'Bob' AS name")
        be = DuckDBBackend(connection=conn)
        view = concat("a", "b", backend=be)
        rows = conn.execute(f"SELECT * FROM {view} ORDER BY id").fetchall()
        assert rows[0][1] is None
        assert rows[1][1] == "Bob"

    def test_many_columns(self) -> None:
        """Concat works with many columns."""
        conn = duckdb.connect()
        cols = ", ".join(f"c{i} INTEGER" for i in range(20))
        conn.execute(f"CREATE TABLE a ({cols})")
        conn.execute(f"CREATE TABLE b ({cols})")
        vals = ", ".join(["1"] * 20)
        conn.execute(f"INSERT INTO a VALUES ({vals})")
        conn.execute(f"INSERT INTO b VALUES ({vals})")
        be = DuckDBBackend(connection=conn)
        view = concat("a", "b", backend=be)
        result = conn.execute(f"SELECT * FROM {view}")
        col_names = [desc[0] for desc in result.description]
        assert len(col_names) == 20
        rows = conn.execute(f"SELECT * FROM {view}").fetchall()
        assert len(rows) == 2

    def test_duplicate_rows_preserved(self) -> None:
        """UNION ALL preserves duplicate rows."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE a AS SELECT 1 AS x, 'same' AS y")
        conn.execute("CREATE TABLE b AS SELECT 1 AS x, 'same' AS y")
        be = DuckDBBackend(connection=conn)
        view = concat("a", "b", backend=be)
        rows = conn.execute(f"SELECT * FROM {view}").fetchall()
        assert len(rows) == 2  # Both identical rows kept

    def test_schema_mismatch_error_detail(self) -> None:
        """Schema mismatch error includes missing/extra column details."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE a AS SELECT 1 AS x, 2 AS y")
        conn.execute("CREATE TABLE b AS SELECT 1 AS x, 2 AS z")
        be = DuckDBBackend(connection=conn)
        with pytest.raises(SchemaError, match="Missing"):
            concat("a", "b", backend=be)
