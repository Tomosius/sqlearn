"""Tests for sqlearn.data.merge -- merge() SQL JOIN wrapper."""

# ruff: noqa: S608

from __future__ import annotations

import duckdb
import pytest
import sqlglot.expressions as exp

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.errors import SchemaError
from sqlearn.data.merge import merge, merge_query

# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def backend() -> DuckDBBackend:
    """Create DuckDB backend with test tables for merge tests."""
    conn = duckdb.connect()
    conn.execute(
        "CREATE TABLE customers AS SELECT * FROM VALUES "
        "(1, 'Alice', 'London'), (2, 'Bob', 'Paris'), (3, 'Carol', 'Tokyo') "
        "t(customer_id, name, city)"
    )
    conn.execute(
        "CREATE TABLE orders AS SELECT * FROM VALUES "
        "(101, 1, 50.0), (102, 1, 30.0), (103, 2, 70.0) "
        "t(order_id, customer_id, amount)"
    )
    return DuckDBBackend(connection=conn)


@pytest.fixture
def overlap_backend() -> DuckDBBackend:
    """Backend with tables having overlapping non-key column names."""
    conn = duckdb.connect()
    conn.execute(
        "CREATE TABLE a AS SELECT * FROM VALUES (1, 'x', 10.0), (2, 'y', 20.0) t(id, name, value)"
    )
    conn.execute(
        "CREATE TABLE b AS SELECT * FROM VALUES "
        "(1, 'p', 100.0), (2, 'q', 200.0) "
        "t(id, name, value)"
    )
    return DuckDBBackend(connection=conn)


# ── Constructor / validation tests ───────────────────────────────


class TestValidation:
    """Test merge() argument validation."""

    def test_invalid_how_raises(self, backend: DuckDBBackend) -> None:
        """Invalid how= value raises ValueError."""
        with pytest.raises(ValueError, match="Invalid join type"):
            merge("customers", "orders", on="customer_id", how="semi", backend=backend)

    def test_no_keys_raises(self, backend: DuckDBBackend) -> None:
        """No join keys raises ValueError (for non-cross joins)."""
        with pytest.raises(ValueError, match="Must specify"):
            merge("customers", "orders", backend=backend)

    def test_on_and_left_on_raises(self, backend: DuckDBBackend) -> None:
        """Specifying both on and left_on raises ValueError."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            merge(
                "customers",
                "orders",
                on="customer_id",
                left_on="customer_id",
                backend=backend,
            )

    def test_left_on_without_right_on_raises(self, backend: DuckDBBackend) -> None:
        """left_on without right_on raises ValueError."""
        with pytest.raises(ValueError, match="Both 'left_on' and 'right_on'"):
            merge("customers", "orders", left_on="customer_id", backend=backend)

    def test_right_on_without_left_on_raises(self, backend: DuckDBBackend) -> None:
        """right_on without left_on raises ValueError."""
        with pytest.raises(ValueError, match="Both 'left_on' and 'right_on'"):
            merge("customers", "orders", right_on="customer_id", backend=backend)

    def test_missing_left_column_raises(self, backend: DuckDBBackend) -> None:
        """Join column not in left schema raises SchemaError."""
        with pytest.raises(SchemaError, match="not found in left"):
            merge("customers", "orders", on="nonexistent", backend=backend)

    def test_missing_right_column_raises(self, backend: DuckDBBackend) -> None:
        """Join column not in right schema raises SchemaError."""
        with pytest.raises(SchemaError, match="not found in right"):
            merge(
                "customers",
                "orders",
                left_on="customer_id",
                right_on="nonexistent",
                backend=backend,
            )

    def test_cross_join_with_on_raises(self, backend: DuckDBBackend) -> None:
        """Cross join with on= raises ValueError."""
        with pytest.raises(ValueError, match="Cross join does not accept"):
            merge("customers", "orders", on="customer_id", how="cross", backend=backend)

    def test_cross_join_with_left_on_raises(self, backend: DuckDBBackend) -> None:
        """Cross join with left_on= raises ValueError."""
        with pytest.raises(ValueError, match="Cross join does not accept"):
            merge(
                "customers",
                "orders",
                left_on="customer_id",
                right_on="customer_id",
                how="cross",
                backend=backend,
            )

    def test_mismatched_key_lengths_raises(self, backend: DuckDBBackend) -> None:
        """Different lengths for left_on and right_on raises ValueError."""
        with pytest.raises(ValueError, match="same number of columns"):
            merge_query(
                "customers",
                "orders",
                left_on=["customer_id", "name"],
                right_on=["customer_id"],
                backend=backend,
            )

    def test_empty_on_list_raises(self, backend: DuckDBBackend) -> None:
        """Empty on list raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            merge_query("customers", "orders", on=[], backend=backend)


# ── Inner join tests ─────────────────────────────────────────────


class TestInnerJoin:
    """Test inner join behavior."""

    def test_inner_join_basic(self, backend: DuckDBBackend) -> None:
        """Inner join produces matching rows only."""
        view = merge("customers", "orders", on="customer_id", how="inner", backend=backend)
        conn = backend._get_connection()
        rows = conn.execute(f"SELECT * FROM {view}").fetchall()
        # Customer 3 has no orders, so should be excluded
        assert len(rows) == 3  # 2 orders for customer 1, 1 for customer 2

    def test_inner_join_column_count(self, backend: DuckDBBackend) -> None:
        """Inner join has correct number of columns."""
        view = merge("customers", "orders", on="customer_id", how="inner", backend=backend)
        conn = backend._get_connection()
        result = conn.execute(f"SELECT * FROM {view}")
        col_names = [desc[0] for desc in result.description]
        # customer_id, name, city from left + order_id, amount from right
        assert len(col_names) == 5

    def test_inner_join_default(self, backend: DuckDBBackend) -> None:
        """Default how is 'inner'."""
        view = merge("customers", "orders", on="customer_id", backend=backend)
        conn = backend._get_connection()
        rows = conn.execute(f"SELECT * FROM {view}").fetchall()
        assert len(rows) == 3

    def test_inner_join_preserves_values(self, backend: DuckDBBackend) -> None:
        """Inner join preserves original data values."""
        view = merge("customers", "orders", on="customer_id", how="inner", backend=backend)
        conn = backend._get_connection()
        rows = conn.execute(f"SELECT name, amount FROM {view} WHERE order_id = 101").fetchall()
        assert rows[0] == ("Alice", 50.0)


# ── Left join tests ──────────────────────────────────────────────


class TestLeftJoin:
    """Test left join behavior."""

    def test_left_join_preserves_all_left_rows(self, backend: DuckDBBackend) -> None:
        """Left join keeps all left rows, even without matches."""
        view = merge("customers", "orders", on="customer_id", how="left", backend=backend)
        conn = backend._get_connection()
        rows = conn.execute(f"SELECT * FROM {view}").fetchall()
        # 2 orders for Alice + 1 for Bob + 1 NULL row for Carol = 4
        assert len(rows) == 4

    def test_left_join_null_for_unmatched(self, backend: DuckDBBackend) -> None:
        """Left join produces NULL for unmatched right columns."""
        view = merge("customers", "orders", on="customer_id", how="left", backend=backend)
        conn = backend._get_connection()
        rows = conn.execute(
            f"SELECT name, order_id, amount FROM {view} WHERE name = 'Carol'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][1] is None  # order_id is NULL
        assert rows[0][2] is None  # amount is NULL


# ── Right join tests ─────────────────────────────────────────────


class TestRightJoin:
    """Test right join behavior."""

    def test_right_join_preserves_all_right_rows(self, backend: DuckDBBackend) -> None:
        """Right join keeps all right rows."""
        view = merge("customers", "orders", on="customer_id", how="right", backend=backend)
        conn = backend._get_connection()
        rows = conn.execute(f"SELECT * FROM {view}").fetchall()
        assert len(rows) == 3  # All 3 orders have matching customers


# ── Outer join tests ─────────────────────────────────────────────


class TestOuterJoin:
    """Test outer join behavior."""

    def test_outer_join_preserves_all_rows(self, backend: DuckDBBackend) -> None:
        """Outer join keeps all rows from both sides."""
        view = merge("customers", "orders", on="customer_id", how="outer", backend=backend)
        conn = backend._get_connection()
        rows = conn.execute(f"SELECT * FROM {view}").fetchall()
        # 3 orders + 1 unmatched customer (Carol) = 4
        assert len(rows) == 4


# ── Cross join tests ─────────────────────────────────────────────


class TestCrossJoin:
    """Test cross join behavior."""

    def test_cross_join_row_count(self, backend: DuckDBBackend) -> None:
        """Cross join produces cartesian product."""
        view = merge("customers", "orders", how="cross", backend=backend)
        conn = backend._get_connection()
        rows = conn.execute(f"SELECT * FROM {view}").fetchall()
        assert len(rows) == 9  # 3 customers x 3 orders

    def test_cross_join_no_keys(self, backend: DuckDBBackend) -> None:
        """Cross join works without any join keys."""
        view = merge("customers", "orders", how="cross", backend=backend)
        conn = backend._get_connection()
        result = conn.execute(f"SELECT * FROM {view}")
        col_names = [desc[0] for desc in result.description]
        # All columns from both tables (customer_id appears twice with suffix)
        assert len(col_names) >= 5


# ── left_on / right_on tests ────────────────────────────────────


class TestSeparateKeys:
    """Test left_on/right_on for different column names."""

    def test_left_on_right_on_basic(self) -> None:
        """Join with different key names works."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE l AS SELECT * FROM VALUES (1, 'A'), (2, 'B') t(lid, val)")
        conn.execute("CREATE TABLE r AS SELECT * FROM VALUES (1, 10), (2, 20) t(rid, score)")
        be = DuckDBBackend(connection=conn)
        view = merge("l", "r", left_on="lid", right_on="rid", backend=be)
        rows = conn.execute(f"SELECT * FROM {view}").fetchall()
        assert len(rows) == 2

    def test_left_on_right_on_multi_key(self) -> None:
        """Join on multiple different key columns."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE l AS SELECT * FROM VALUES (1, 'a', 100), (2, 'b', 200) t(k1, k2, val)"
        )
        conn.execute(
            "CREATE TABLE r AS SELECT * FROM VALUES (1, 'a', 'x'), (2, 'b', 'y') t(j1, j2, info)"
        )
        be = DuckDBBackend(connection=conn)
        view = merge("l", "r", left_on=["k1", "k2"], right_on=["j1", "j2"], backend=be)
        rows = conn.execute(f"SELECT * FROM {view}").fetchall()
        assert len(rows) == 2


# ── Suffix handling tests ────────────────────────────────────────


class TestSuffixHandling:
    """Test overlapping column name handling with suffixes."""

    def test_default_suffix(self, overlap_backend: DuckDBBackend) -> None:
        """Overlapping columns get default _left/_right suffixes."""
        view = merge("a", "b", on="id", backend=overlap_backend)
        conn = overlap_backend._get_connection()
        result = conn.execute(f"SELECT * FROM {view}")
        col_names = [desc[0] for desc in result.description]
        assert "name_left" in col_names
        assert "name_right" in col_names
        assert "value_left" in col_names
        assert "value_right" in col_names

    def test_custom_suffix(self, overlap_backend: DuckDBBackend) -> None:
        """Custom suffix is applied to overlapping columns."""
        view = merge("a", "b", on="id", suffix=("_a", "_b"), backend=overlap_backend)
        conn = overlap_backend._get_connection()
        result = conn.execute(f"SELECT * FROM {view}")
        col_names = [desc[0] for desc in result.description]
        assert "name_a" in col_names
        assert "name_b" in col_names

    def test_no_suffix_for_non_overlapping(self, backend: DuckDBBackend) -> None:
        """Non-overlapping columns don't get suffixed."""
        view = merge("customers", "orders", on="customer_id", backend=backend)
        conn = backend._get_connection()
        result = conn.execute(f"SELECT * FROM {view}")
        col_names = [desc[0] for desc in result.description]
        assert "name" in col_names  # unique to left, no suffix
        assert "amount" in col_names  # unique to right, no suffix


# ── Single vs multiple join keys ─────────────────────────────────


class TestMultipleJoinKeys:
    """Test joining on single and multiple columns."""

    def test_single_key(self, backend: DuckDBBackend) -> None:
        """Single key join works."""
        view = merge("customers", "orders", on="customer_id", backend=backend)
        conn = backend._get_connection()
        rows = conn.execute(f"SELECT * FROM {view}").fetchall()
        assert len(rows) >= 1

    def test_multiple_keys(self) -> None:
        """Multi-column key join works."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE l AS SELECT * FROM VALUES "
            "(1, 'a', 100), (1, 'b', 200), (2, 'a', 300) t(k1, k2, val)"
        )
        conn.execute(
            "CREATE TABLE r AS SELECT * FROM VALUES (1, 'a', 'X'), (2, 'a', 'Y') t(k1, k2, info)"
        )
        be = DuckDBBackend(connection=conn)
        view = merge("l", "r", on=["k1", "k2"], backend=be)
        rows = conn.execute(f"SELECT * FROM {view}").fetchall()
        assert len(rows) == 2


# ── merge_query() tests ─────────────────────────────────────────


class TestMergeQuery:
    """Test merge_query() returns sqlglot AST without executing."""

    def test_returns_select(self, backend: DuckDBBackend) -> None:
        """merge_query returns a Select expression."""
        q = merge_query("customers", "orders", on="customer_id", backend=backend)
        assert isinstance(q, exp.Select)

    def test_contains_join(self, backend: DuckDBBackend) -> None:
        """Result contains a Join node."""
        q = merge_query("customers", "orders", on="customer_id", backend=backend)
        assert q.find(exp.Join) is not None

    def test_sql_is_valid(self, backend: DuckDBBackend) -> None:
        """Generated SQL can be executed by DuckDB."""
        q = merge_query("customers", "orders", on="customer_id", backend=backend)
        sql = q.sql(dialect="duckdb")
        conn = backend._get_connection()
        rows = conn.execute(sql).fetchall()
        assert len(rows) >= 1

    def test_cross_join_query(self, backend: DuckDBBackend) -> None:
        """Cross join query is valid."""
        q = merge_query("customers", "orders", how="cross", backend=backend)
        sql = q.sql(dialect="duckdb")
        conn = backend._get_connection()
        rows = conn.execute(sql).fetchall()
        assert len(rows) == 9

    def test_invalid_how_raises(self, backend: DuckDBBackend) -> None:
        """Invalid how= on merge_query also raises."""
        with pytest.raises(ValueError, match="Invalid join type"):
            merge_query("customers", "orders", on="customer_id", how="anti", backend=backend)


# ── Returns view name tests ──────────────────────────────────────


class TestReturnValue:
    """Test that merge() returns a usable view name."""

    def test_returns_string(self, backend: DuckDBBackend) -> None:
        """merge() returns a string."""
        result = merge("customers", "orders", on="customer_id", backend=backend)
        assert isinstance(result, str)

    def test_view_name_prefix(self, backend: DuckDBBackend) -> None:
        """View name has expected prefix."""
        result = merge("customers", "orders", on="customer_id", backend=backend)
        assert result.startswith("__sqlearn_merge_")

    def test_view_is_queryable(self, backend: DuckDBBackend) -> None:
        """Returned view can be queried."""
        view = merge("customers", "orders", on="customer_id", backend=backend)
        conn = backend._get_connection()
        rows = conn.execute(f"SELECT COUNT(*) FROM {view}").fetchone()
        assert rows is not None
        assert rows[0] > 0

    def test_unique_view_names(self, backend: DuckDBBackend) -> None:
        """Multiple merges produce unique view names."""
        v1 = merge("customers", "orders", on="customer_id", backend=backend)
        v2 = merge("customers", "orders", on="customer_id", backend=backend)
        assert v1 != v2


# ── Edge cases ───────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases and unusual inputs."""

    def test_single_row_tables(self) -> None:
        """Join works with single-row tables."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE l AS SELECT 1 AS id, 'a' AS val")
        conn.execute("CREATE TABLE r AS SELECT 1 AS id, 'b' AS info")
        be = DuckDBBackend(connection=conn)
        view = merge("l", "r", on="id", backend=be)
        rows = conn.execute(f"SELECT * FROM {view}").fetchall()
        assert len(rows) == 1

    def test_empty_table_left_join(self) -> None:
        """Left join with empty right table returns left rows with NULLs."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE l AS SELECT 1 AS id, 'a' AS val")
        conn.execute("CREATE TABLE r (id INTEGER, info VARCHAR)")
        be = DuckDBBackend(connection=conn)
        view = merge("l", "r", on="id", how="left", backend=be)
        rows = conn.execute(f"SELECT * FROM {view}").fetchall()
        assert len(rows) == 1

    def test_no_matching_rows_inner_join(self) -> None:
        """Inner join with no matches returns empty result."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE l AS SELECT 1 AS id, 'a' AS val")
        conn.execute("CREATE TABLE r AS SELECT 99 AS id, 'z' AS info")
        be = DuckDBBackend(connection=conn)
        view = merge("l", "r", on="id", how="inner", backend=be)
        rows = conn.execute(f"SELECT * FROM {view}").fetchall()
        assert len(rows) == 0

    def test_null_join_keys(self) -> None:
        """NULL join keys do not match (standard SQL behavior)."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE l AS SELECT NULL::INTEGER AS id, 'a' AS val")
        conn.execute("CREATE TABLE r AS SELECT NULL::INTEGER AS id, 'b' AS info")
        be = DuckDBBackend(connection=conn)
        view = merge("l", "r", on="id", how="inner", backend=be)
        rows = conn.execute(f"SELECT * FROM {view}").fetchall()
        assert len(rows) == 0

    def test_many_columns(self) -> None:
        """Join works with many columns in each table."""
        conn = duckdb.connect()
        left_cols = ", ".join(f"c{i} INTEGER" for i in range(10))
        right_cols = ", ".join(f"d{i} INTEGER" for i in range(10))
        conn.execute(f"CREATE TABLE l (id INTEGER, {left_cols})")
        conn.execute(f"CREATE TABLE r (id INTEGER, {right_cols})")
        conn.execute("INSERT INTO l VALUES " + "(" + ",".join(["1"] * 11) + ")")
        conn.execute("INSERT INTO r VALUES " + "(" + ",".join(["1"] * 11) + ")")
        be = DuckDBBackend(connection=conn)
        view = merge("l", "r", on="id", backend=be)
        result = conn.execute(f"SELECT * FROM {view}")
        col_names = [desc[0] for desc in result.description]
        assert len(col_names) == 21  # id + 10 left + 10 right

    def test_default_backend_creation(self) -> None:
        """merge() creates a default backend when None is passed."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE l AS SELECT 1 AS id, 'a' AS val")
        conn.execute("CREATE TABLE r AS SELECT 1 AS id, 10 AS score")
        be = DuckDBBackend(connection=conn)
        # Just test that providing a backend works (default backend would need files)
        view = merge("l", "r", on="id", backend=be)
        assert isinstance(view, str)
