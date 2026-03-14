"""Tests for sqlearn.core.backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest
import sqlglot.expressions as exp

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.errors import FitError
from sqlearn.core.schema import Schema


@pytest.fixture
def backend() -> DuckDBBackend:
    """Fresh in-memory DuckDBBackend."""
    return DuckDBBackend()


@pytest.fixture
def sample_table(backend: DuckDBBackend) -> DuckDBBackend:
    """Backend with a 'sample' table: id INTEGER, price DOUBLE, city VARCHAR."""
    conn = backend._get_connection()
    conn.execute(
        "CREATE TABLE sample AS SELECT * FROM VALUES "
        "(1, 10.5, 'London'), (2, 20.0, 'Paris'), (3, 30.5, 'Berlin') "
        "t(id, price, city)"
    )
    return backend


@pytest.fixture
def parquet_file(tmp_path: Any) -> str:
    """Small parquet file for testing describe()."""
    path = str(tmp_path / "test.parquet")
    conn = duckdb.connect()
    conn.execute(
        f"COPY (SELECT 1 AS id, 2.5 AS price, 'hello' AS name) TO '{path}' (FORMAT PARQUET)"
    )
    conn.close()
    return path


@pytest.fixture
def csv_file(tmp_path: Any) -> str:
    """Small CSV file for testing describe()."""
    path = str(tmp_path / "test.csv")
    conn = duckdb.connect()
    conn.execute(
        f"COPY (SELECT 1 AS id, 2.5 AS price, 'hello' AS name) TO '{path}' (FORMAT CSV, HEADER)"
    )
    conn.close()
    return path


class TestConstruction:
    """Test DuckDBBackend construction modes."""

    def test_default_creates_lazy_in_memory(self) -> None:
        """Default constructor creates no connection until first use."""
        b = DuckDBBackend()
        assert b._connection is None

    def test_accepts_existing_connection(self) -> None:
        """Existing duckdb connection is used directly."""
        conn = duckdb.connect()
        b = DuckDBBackend(conn)
        assert b._connection is conn
        conn.close()

    def test_accepts_string_path(self, tmp_path: Any) -> None:
        """String path is stored for lazy connection."""
        path = str(tmp_path / "test.duckdb")
        b = DuckDBBackend(path)
        assert b._connection is None
        b._get_connection()
        assert b._connection is not None
        b.close()
        Path(path).unlink()

    def test_lazy_connection_created_on_first_use(self) -> None:
        """Connection is created when first method is called."""
        b = DuckDBBackend()
        assert b._connection is None
        conn = b._get_connection()
        assert b._connection is not None
        assert conn is not None


class TestDialect:
    """Test dialect property."""

    def test_returns_duckdb(self, backend: DuckDBBackend) -> None:
        """Dialect is 'duckdb'."""
        assert backend.dialect == "duckdb"


class TestDescribe:
    """Test describe() schema inference."""

    def test_describe_table(self, sample_table: DuckDBBackend) -> None:
        """Describe a registered table."""
        schema = sample_table.describe("sample")
        assert isinstance(schema, Schema)
        assert "id" in schema
        assert "price" in schema
        assert "city" in schema

    def test_describe_parquet(self, backend: DuckDBBackend, parquet_file: str) -> None:
        """Describe a parquet file (auto-quoted)."""
        schema = backend.describe(parquet_file)
        assert isinstance(schema, Schema)
        assert "id" in schema
        assert "price" in schema

    def test_describe_csv(self, backend: DuckDBBackend, csv_file: str) -> None:
        """Describe a CSV file (auto-quoted)."""
        schema = backend.describe(csv_file)
        assert isinstance(schema, Schema)
        assert "id" in schema

    def test_raw_types_preserved(self, sample_table: DuckDBBackend) -> None:
        """DuckDB raw types stored as-is (no normalization)."""
        schema = sample_table.describe("sample")
        assert schema.columns["id"] == "INTEGER"
        assert schema.columns["price"] == "DECIMAL(3,1)"
        assert schema.columns["city"] == "VARCHAR"

    def test_describe_nonexistent_raises(self, backend: DuckDBBackend) -> None:
        """Nonexistent source raises DuckDB error (not wrapped)."""
        with pytest.raises(duckdb.CatalogException):
            backend.describe("nonexistent_table")


class TestExecute:
    """Test execute() query execution."""

    def test_returns_list_of_dicts(self, sample_table: DuckDBBackend) -> None:
        """Execute returns list[dict]."""
        query = exp.select("id", "price").from_("sample")
        result = sample_table.execute(query)
        assert isinstance(result, list)
        assert len(result) == 3
        assert isinstance(result[0], dict)
        assert "id" in result[0]
        assert "price" in result[0]

    def test_empty_result(self, sample_table: DuckDBBackend) -> None:
        """Empty result returns empty list."""
        query = exp.select("id").from_("sample").where("1 = 0")
        result = sample_table.execute(query)
        assert result == []

    def test_aggregate_query(self, sample_table: DuckDBBackend) -> None:
        """Aggregate queries return computed values."""
        query = exp.select(exp.func("AVG", exp.column("price")).as_("avg_price")).from_("sample")
        result = sample_table.execute(query)
        assert len(result) == 1
        assert abs(result[0]["avg_price"] - 20.333333) < 0.01


class TestFetchOne:
    """Test fetch_one() single-row query."""

    def test_returns_dict(self, sample_table: DuckDBBackend) -> None:
        """fetch_one returns a single dict."""
        query = exp.select(exp.func("COUNT", exp.Star()).as_("n")).from_("sample")
        result = sample_table.fetch_one(query)
        assert isinstance(result, dict)
        assert result["n"] == 3

    def test_empty_result_raises_fit_error(self, sample_table: DuckDBBackend) -> None:
        """Empty result raises FitError."""
        query = exp.select("id").from_("sample").where("1 = 0")
        with pytest.raises(FitError):
            sample_table.fetch_one(query)


class TestRegister:
    """Test register() DataFrame registration."""

    def test_register_pandas(self, backend: DuckDBBackend) -> None:
        """Register pandas DataFrame and query it."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        name = backend.register(df, "__test_df")
        assert name == "__test_df"
        result = backend.execute(exp.select("x", "y").from_("__test_df"))
        assert len(result) == 3


class TestSupports:
    """Test supports() feature checking."""

    def test_known_features(self, backend: DuckDBBackend) -> None:
        """DuckDB supports all known features."""
        assert backend.supports("filter_clause") is True
        assert backend.supports("median") is True
        assert backend.supports("hash") is True
        assert backend.supports("parquet") is True

    def test_unknown_feature(self, backend: DuckDBBackend) -> None:
        """Unknown features also return True for DuckDB."""
        assert backend.supports("anything") is True


class TestClose:
    """Test close() and context manager."""

    def test_close(self) -> None:
        """Close shuts down connection."""
        b = DuckDBBackend()
        b._get_connection()
        b.close()
        assert b._connection is None

    def test_context_manager(self) -> None:
        """with statement calls close on exit."""
        with DuckDBBackend() as b:
            b._get_connection()
            assert b._connection is not None
        assert b._connection is None

    def test_close_without_connection(self) -> None:
        """Close on never-used backend is a no-op."""
        b = DuckDBBackend()
        b.close()  # should not raise
