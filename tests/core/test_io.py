"""Tests for sqlearn.core.io."""

from __future__ import annotations

import pandas as pd
import pytest

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.io import resolve_input


@pytest.fixture
def backend() -> DuckDBBackend:
    """Fresh in-memory DuckDBBackend."""
    return DuckDBBackend()


class TestFilePathResolution:
    """String inputs with file extensions are returned as-is."""

    def test_parquet_path(self, backend: DuckDBBackend) -> None:
        """Parquet file path returned unchanged."""
        result = resolve_input("data.parquet", backend)
        assert result == "data.parquet"

    def test_csv_path(self, backend: DuckDBBackend) -> None:
        """CSV file path returned unchanged."""
        result = resolve_input("data.csv", backend)
        assert result == "data.csv"

    def test_nested_path(self, backend: DuckDBBackend) -> None:
        """Nested file path returned unchanged."""
        result = resolve_input("path/to/data.parquet", backend)
        assert result == "path/to/data.parquet"


class TestTableNameResolution:
    """String inputs without file extensions are treated as table names."""

    def test_simple_table_name(self, backend: DuckDBBackend) -> None:
        """Simple table name returned as-is."""
        result = resolve_input("my_table", backend)
        assert result == "my_table"

    def test_schema_qualified_name(self, backend: DuckDBBackend) -> None:
        """Schema-qualified name returned as-is."""
        result = resolve_input("schema.table", backend)
        assert result == "schema.table"


class TestDataFrameResolution:
    """pandas DataFrames are registered and return a table name."""

    def test_pandas_df_auto_name(self, backend: DuckDBBackend) -> None:
        """pandas DataFrame gets auto-generated name."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = resolve_input(df, backend)
        assert result.startswith("__sqlearn_input_")

    def test_pandas_df_custom_name(self, backend: DuckDBBackend) -> None:
        """Custom table_name overrides auto-naming."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = resolve_input(df, backend, table_name="my_data")
        assert result == "my_data"

    def test_pandas_df_queryable(self, backend: DuckDBBackend) -> None:
        """Registered DataFrame is queryable via backend."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        name = resolve_input(df, backend)
        import sqlglot.expressions as exp

        result = backend.execute(exp.select("x", "y").from_(name))
        assert len(result) == 3

    def test_auto_name_increments(self, backend: DuckDBBackend) -> None:
        """Each auto-named DataFrame gets a unique name."""
        df1 = pd.DataFrame({"a": [1]})
        df2 = pd.DataFrame({"b": [2]})
        name1 = resolve_input(df1, backend)
        name2 = resolve_input(df2, backend)
        assert name1 != name2
        assert name1.startswith("__sqlearn_input_")
        assert name2.startswith("__sqlearn_input_")


class TestUnsupportedType:
    """Unsupported types raise TypeError."""

    def test_int_raises(self, backend: DuckDBBackend) -> None:
        """Integer input raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported"):
            resolve_input(42, backend)

    def test_list_raises(self, backend: DuckDBBackend) -> None:
        """List input raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported"):
            resolve_input([1, 2, 3], backend)

    def test_error_message_lists_types(self, backend: DuckDBBackend) -> None:
        """Error message lists supported types."""
        with pytest.raises(TypeError, match=r"str.*DataFrame"):
            resolve_input(42, backend)
