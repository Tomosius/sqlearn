"""Tests for sqlearn.feature_selection.drop — Drop transformer."""

from __future__ import annotations

import pickle

import duckdb
import numpy as np
import pytest
import sqlglot.expressions as exp

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.pipeline import Pipeline
from sqlearn.core.schema import Schema
from sqlearn.feature_selection.drop import Drop

# -- Constructor tests -------------------------------------------------------


class TestConstructor:
    """Test Drop constructor validation and attributes."""

    def test_basic_columns(self) -> None:
        """Single-column list is accepted."""
        d = Drop(columns=["id"])
        assert d.columns == ["id"]

    def test_multi_columns(self) -> None:
        """Multi-column list is accepted."""
        d = Drop(columns=["id", "timestamp"])
        assert d.columns == ["id", "timestamp"]

    def test_empty_list_raises(self) -> None:
        """Empty columns list raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            Drop(columns=[])

    def test_non_list_raises(self) -> None:
        """Non-list columns raises TypeError."""
        with pytest.raises(TypeError, match="must be a list"):
            Drop(columns="id")  # type: ignore[arg-type]

    def test_non_string_element_raises(self) -> None:
        """Non-string element in columns raises TypeError."""
        with pytest.raises(TypeError, match="must contain strings"):
            Drop(columns=[1, 2])  # type: ignore[list-item]

    def test_classification_is_static(self) -> None:
        """Drop is classified as static."""
        assert Drop._classification == "static"

    def test_default_columns_is_none(self) -> None:
        """Drop has no default column routing."""
        assert Drop._default_columns is None

    def test_get_params(self) -> None:
        """get_params returns constructor kwargs."""
        d = Drop(columns=["a", "b"])
        params = d.get_params()
        assert params == {"columns": ["a", "b"]}

    def test_set_params(self) -> None:
        """set_params updates columns and returns self."""
        d = Drop(columns=["a"])
        result = d.set_params(columns=["x", "y"])
        assert result is d
        assert d.columns == ["x", "y"]

    def test_repr(self) -> None:
        """repr shows columns parameter."""
        d = Drop(columns=["a", "b"])
        assert "Drop" in repr(d)
        assert "columns" in repr(d)

    def test_repr_single_column(self) -> None:
        """repr shows single column."""
        d = Drop(columns=["x"])
        r = repr(d)
        assert "Drop" in r
        assert "'x'" in r

    def test_classify_method(self) -> None:
        """_classify returns static."""
        d = Drop(columns=["a"])
        assert d._classify() == "static"


# -- output_schema tests ----------------------------------------------------


class TestOutputSchema:
    """Test Drop.output_schema() reflects drops correctly."""

    def test_single_drop(self) -> None:
        """Single column drop is reflected in output schema."""
        schema = Schema({"a": "DOUBLE", "b": "VARCHAR", "c": "INTEGER"})
        d = Drop(columns=["a"])
        out = d.output_schema(schema)
        assert "a" not in out.columns
        assert "b" in out.columns
        assert "c" in out.columns
        assert len(out) == 2

    def test_multi_drop(self) -> None:
        """Multiple drops are reflected in output schema."""
        schema = Schema({"a": "DOUBLE", "b": "VARCHAR", "c": "INTEGER"})
        d = Drop(columns=["a", "c"])
        out = d.output_schema(schema)
        assert list(out.columns.keys()) == ["b"]

    def test_preserves_column_order(self) -> None:
        """Remaining columns maintain original order."""
        schema = Schema({"a": "DOUBLE", "b": "VARCHAR", "c": "INTEGER", "d": "FLOAT"})
        d = Drop(columns=["b"])
        out = d.output_schema(schema)
        assert list(out.columns.keys()) == ["a", "c", "d"]

    def test_preserves_types(self) -> None:
        """Remaining columns keep their original types."""
        schema = Schema({"price": "DECIMAL(18,3)", "city": "VARCHAR", "id": "INTEGER"})
        d = Drop(columns=["id"])
        out = d.output_schema(schema)
        assert out["price"] == "DECIMAL(18,3)"
        assert out["city"] == "VARCHAR"

    def test_nonexistent_columns_ignored(self) -> None:
        """Columns not in schema are silently ignored."""
        schema = Schema({"a": "DOUBLE", "b": "VARCHAR"})
        d = Drop(columns=["a", "nonexistent"])
        out = d.output_schema(schema)
        assert "a" not in out.columns
        assert "b" in out.columns

    def test_all_nonexistent_returns_schema(self) -> None:
        """When no columns exist in schema, return it unchanged."""
        schema = Schema({"a": "DOUBLE", "b": "VARCHAR"})
        d = Drop(columns=["x", "y"])
        out = d.output_schema(schema)
        assert out.columns == schema.columns


# -- query() tests ----------------------------------------------------------


class TestQuery:
    """Test Drop.query() generates correct sqlglot ASTs."""

    def _make_fitted_drop(self, cols: list[str], schema: Schema) -> Drop:
        """Create a Drop with input_schema_ set (simulating fitted state)."""
        d = Drop(columns=cols)
        d.input_schema_ = schema
        d.columns_ = []
        d._fitted = True
        return d

    def test_single_drop_excludes_column(self) -> None:
        """query() excludes the dropped column from SELECT."""
        schema = Schema({"a": "DOUBLE", "b": "VARCHAR", "c": "INTEGER"})
        d = self._make_fitted_drop(["a"], schema)
        input_q = exp.select("*").from_("t")
        result = d.query(input_q)
        sql = result.sql(dialect="duckdb").lower()
        assert "b" in sql
        assert "c" in sql

    def test_multi_drop_excludes_columns(self) -> None:
        """query() excludes multiple dropped columns."""
        schema = Schema({"a": "DOUBLE", "b": "VARCHAR", "c": "INTEGER"})
        d = self._make_fitted_drop(["a", "c"], schema)
        input_q = exp.select("*").from_("t")
        result = d.query(input_q)
        sql = result.sql(dialect="duckdb").lower()
        assert "b" in sql

    def test_wraps_input_as_subquery(self) -> None:
        """query() wraps the input query in a subquery."""
        schema = Schema({"a": "DOUBLE", "b": "VARCHAR"})
        d = self._make_fitted_drop(["a"], schema)
        input_q = exp.select("*").from_("t")
        result = d.query(input_q)
        assert isinstance(result, exp.Select)
        assert result.find(exp.Subquery) is not None

    def test_produces_valid_select(self) -> None:
        """query() produces a valid SELECT expression."""
        schema = Schema({"x": "DOUBLE", "y": "VARCHAR", "z": "INTEGER"})
        d = self._make_fitted_drop(["z"], schema)
        input_q = exp.select("*").from_("t")
        result = d.query(input_q)
        sql = result.sql(dialect="duckdb")
        assert "SELECT" in sql.upper()
        assert "FROM" in sql.upper()


# -- Pipeline integration tests ----------------------------------------------


class TestPipeline:
    """Test Drop integrated with Pipeline (end-to-end)."""

    @pytest.fixture
    def backend(self) -> DuckDBBackend:
        """Create DuckDB backend with test data."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0) "
            "t(a, b, c)"
        )
        return DuckDBBackend(connection=conn)

    def test_fit_transform_drops_column(self, backend: DuckDBBackend) -> None:
        """Dropping a column reduces output width."""
        pipe = Pipeline([Drop(columns=["a"])], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (3, 2)

    def test_fit_transform_drops_multiple(self, backend: DuckDBBackend) -> None:
        """Dropping multiple columns reduces output width."""
        pipe = Pipeline([Drop(columns=["a", "c"])], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (3, 1)

    def test_output_column_names(self, backend: DuckDBBackend) -> None:
        """get_feature_names_out excludes dropped column names."""
        pipe = Pipeline([Drop(columns=["a"])], backend=backend)
        pipe.fit("t")
        names = pipe.get_feature_names_out()
        assert "a" not in names
        assert "b" in names
        assert "c" in names

    def test_to_sql_valid_duckdb(self, backend: DuckDBBackend) -> None:
        """to_sql() output is valid DuckDB SQL."""
        pipe = Pipeline([Drop(columns=["a"])], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql()
        sql = sql.replace("__input__", "t")
        conn = backend._connection
        result = conn.execute(sql).fetchall()
        assert len(result) == 3

    def test_data_values_preserved(self, backend: DuckDBBackend) -> None:
        """Drop only removes columns, not data values."""
        pipe = Pipeline([Drop(columns=["a"])], backend=backend)
        result = pipe.fit_transform("t")
        # b column values: 2.0, 5.0, 8.0
        np.testing.assert_array_equal(result[:, 0], [2.0, 5.0, 8.0])

    def test_fit_then_transform_matches(self, backend: DuckDBBackend) -> None:
        """Separate fit() and transform() produce same result as fit_transform()."""
        pipe1 = Pipeline([Drop(columns=["a"])], backend=backend)
        result1 = pipe1.fit_transform("t")

        pipe2 = Pipeline([Drop(columns=["a"])], backend=backend)
        pipe2.fit("t")
        result2 = pipe2.transform("t")

        np.testing.assert_array_equal(result1, result2)


# -- Composition tests ------------------------------------------------------


class TestComposition:
    """Test Drop composing with other transformers."""

    def test_scaler_then_drop(self) -> None:
        """StandardScaler + Drop produces correct output."""
        from sqlearn.scalers.standard import StandardScaler

        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 10.0, 100.0), (2.0, 20.0, 200.0), "
            "(3.0, 30.0, 300.0) t(a, b, c)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [StandardScaler(), Drop(columns=["c"])],
            backend=backend,
        )
        pipe.fit("t")
        names = pipe.get_feature_names_out()
        assert "c" not in names
        assert "a" in names
        assert "b" in names

        result = pipe.fit_transform("t")
        assert result.shape == (3, 2)


# -- Not-fitted guard -------------------------------------------------------


class TestNotFittedGuard:
    """Test that unfitted Drop raises appropriate errors."""

    def test_get_feature_names_out_unfitted(self) -> None:
        """get_feature_names_out raises NotFittedError when not fitted."""
        from sqlearn.core.errors import NotFittedError

        d = Drop(columns=["a"])
        with pytest.raises(NotFittedError):
            d.get_feature_names_out()


# -- Clone and pickle tests --------------------------------------------------


class TestClonePickle:
    """Deep clone independence and pickle roundtrip."""

    def test_clone_roundtrip(self) -> None:
        """Cloned Drop has same columns but is independent."""
        d = Drop(columns=["a", "b"])
        cloned = d.clone()
        assert cloned.columns == ["a", "b"]
        assert cloned is not d

    def test_clone_independence(self) -> None:
        """Modifying clone does not affect original."""
        d = Drop(columns=["a"])
        cloned = d.clone()
        cloned.set_params(columns=["x", "y"])
        assert d.columns == ["a"]
        assert cloned.columns == ["x", "y"]

    def test_pickle_roundtrip(self) -> None:
        """Pickle a Drop preserves columns."""
        d = Drop(columns=["a", "b"])
        data = pickle.dumps(d)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.columns == ["a", "b"]

    def test_pickle_roundtrip_fitted(self) -> None:
        """Pickle a fitted Drop preserves fitted state."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1.0, 2.0) t(a, b)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Drop(columns=["a"])], backend=backend)
        pipe.fit("t")

        step = pipe.named_steps["step_00"]
        data = pickle.dumps(step)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.columns == ["a"]
        assert restored.is_fitted


# -- Edge cases --------------------------------------------------------------


class TestEdgeCases:
    """Extreme and unusual inputs."""

    def test_single_column_table_drop_only_column(self) -> None:
        """Dropping the only column results in empty output schema."""
        schema = Schema({"x": "DOUBLE"})
        d = Drop(columns=["x"])
        out = d.output_schema(schema)
        assert len(out) == 0

    def test_many_columns_drop_subset(self) -> None:
        """Drop works with many columns, dropping a subset."""
        conn = duckdb.connect()
        cols = [f"c{i}" for i in range(10)]
        col_defs = ", ".join(f"{c} DOUBLE" for c in cols)
        conn.execute(f"CREATE TABLE t ({col_defs})")
        values = ", ".join(["1.0"] * 10)
        conn.execute(f"INSERT INTO t VALUES ({values})")  # noqa: S608

        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Drop(columns=["c0", "c5", "c9"])], backend=backend)
        pipe.fit("t")
        names = pipe.get_feature_names_out()
        assert "c0" not in names
        assert "c5" not in names
        assert "c9" not in names
        assert len(names) == 7

    def test_null_values_preserved(self) -> None:
        """Drop preserves NULL values in remaining columns."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES (1.0, NULL, 3.0), (NULL, 2.0, 4.0) t(a, b, c)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Drop(columns=["a"])], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (2, 2)

    def test_drop_preserves_all_rows(self) -> None:
        """Drop does not filter any rows."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 10.0), (2.0, 20.0), (3.0, 30.0), "
            "(4.0, 40.0), (5.0, 50.0) t(a, b)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Drop(columns=["a"])], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (5, 1)

    def test_expressions_returns_empty(self) -> None:
        """expressions() returns empty dict (uses query instead)."""
        d = Drop(columns=["a"])
        result = d.expressions(["a"], {"a": exp.Column(this="a")})
        assert result == {}
