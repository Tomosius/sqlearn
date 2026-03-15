"""Tests for sqlearn.ops.rename -- Rename transformer."""

from __future__ import annotations

import pickle

import duckdb
import numpy as np
import pytest
import sqlglot.expressions as exp

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.pipeline import Pipeline
from sqlearn.core.schema import Schema
from sqlearn.ops.rename import Rename

# -- Constructor tests -------------------------------------------------------


class TestConstructor:
    """Test Rename constructor validation and attributes."""

    def test_basic_mapping(self) -> None:
        """Single-pair mapping is accepted."""
        r = Rename(mapping={"old": "new"})
        assert r.mapping == {"old": "new"}

    def test_multi_mapping(self) -> None:
        """Multi-pair mapping is accepted."""
        r = Rename(mapping={"a": "x", "b": "y"})
        assert r.mapping == {"a": "x", "b": "y"}

    def test_empty_mapping_raises(self) -> None:
        """Empty mapping raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            Rename(mapping={})

    def test_non_dict_mapping_raises(self) -> None:
        """Non-dict mapping raises TypeError."""
        with pytest.raises(TypeError, match="must be a dict"):
            Rename(mapping=[("a", "b")])  # type: ignore[arg-type]

    def test_non_string_key_raises(self) -> None:
        """Non-string key raises TypeError."""
        with pytest.raises(TypeError, match="keys must be strings"):
            Rename(mapping={1: "b"})  # type: ignore[dict-item]

    def test_non_string_value_raises(self) -> None:
        """Non-string value raises TypeError."""
        with pytest.raises(TypeError, match="values must be strings"):
            Rename(mapping={"a": 1})  # type: ignore[dict-item]

    def test_duplicate_values_raises(self) -> None:
        """Duplicate target names raise ValueError."""
        with pytest.raises(ValueError, match="duplicate target names"):
            Rename(mapping={"a": "x", "b": "x"})

    def test_classification_is_static(self) -> None:
        """Rename is classified as static."""
        assert Rename._classification == "static"

    def test_default_columns_is_none(self) -> None:
        """Rename has no default column routing."""
        assert Rename._default_columns is None

    def test_columns_attr_is_none(self) -> None:
        """columns attribute is None (no column routing)."""
        r = Rename(mapping={"a": "b"})
        assert r.columns is None

    def test_get_params(self) -> None:
        """get_params returns constructor kwargs."""
        r = Rename(mapping={"a": "b"})
        params = r.get_params()
        assert params == {"mapping": {"a": "b"}}

    def test_set_params(self) -> None:
        """set_params updates mapping and returns self."""
        r = Rename(mapping={"a": "b"})
        result = r.set_params(mapping={"x": "y"})
        assert result is r
        assert r.mapping == {"x": "y"}

    def test_repr(self) -> None:
        """repr shows mapping parameter."""
        r = Rename(mapping={"a": "b"})
        assert "Rename" in repr(r)
        assert "mapping" in repr(r)


# -- output_schema tests ----------------------------------------------------


class TestOutputSchema:
    """Test Rename.output_schema() reflects renames correctly."""

    def test_single_rename(self) -> None:
        """Single column rename is reflected in output schema."""
        schema = Schema({"a": "DOUBLE", "b": "VARCHAR"})
        r = Rename(mapping={"a": "x"})
        out = r.output_schema(schema)
        assert "x" in out.columns
        assert "a" not in out.columns
        assert "b" in out.columns
        assert out["x"] == "DOUBLE"

    def test_multi_rename(self) -> None:
        """Multiple renames are reflected in output schema."""
        schema = Schema({"a": "DOUBLE", "b": "VARCHAR", "c": "INTEGER"})
        r = Rename(mapping={"a": "x", "b": "y"})
        out = r.output_schema(schema)
        assert list(out.columns.keys()) == ["x", "y", "c"]
        assert out["x"] == "DOUBLE"
        assert out["y"] == "VARCHAR"

    def test_preserves_column_order(self) -> None:
        """Renamed columns appear in the same position as originals."""
        schema = Schema({"a": "DOUBLE", "b": "VARCHAR", "c": "INTEGER"})
        r = Rename(mapping={"b": "renamed_b"})
        out = r.output_schema(schema)
        assert list(out.columns.keys()) == ["a", "renamed_b", "c"]

    def test_preserves_types(self) -> None:
        """Renamed columns keep their original types."""
        schema = Schema({"price": "DECIMAL(18,3)", "city": "VARCHAR"})
        r = Rename(mapping={"price": "cost"})
        out = r.output_schema(schema)
        assert out["cost"] == "DECIMAL(18,3)"

    def test_unmapped_columns_not_in_schema_ignored(self) -> None:
        """Mapping keys not in schema are silently ignored."""
        schema = Schema({"a": "DOUBLE", "b": "VARCHAR"})
        r = Rename(mapping={"a": "x", "nonexistent": "y"})
        out = r.output_schema(schema)
        assert "x" in out.columns
        assert "nonexistent" not in out.columns
        assert "y" not in out.columns

    def test_rename_to_same_name(self) -> None:
        """Renaming a column to its own name is a no-op."""
        schema = Schema({"a": "DOUBLE", "b": "VARCHAR"})
        r = Rename(mapping={"a": "a"})
        out = r.output_schema(schema)
        assert list(out.columns.keys()) == ["a", "b"]


# -- query() tests ----------------------------------------------------------


class TestQuery:
    """Test Rename.query() generates correct sqlglot ASTs."""

    def _make_fitted_rename(self, mapping: dict[str, str], schema: Schema) -> Rename:
        """Create a Rename with input_schema_ set (simulating fitted state)."""
        r = Rename(mapping=mapping)
        r.input_schema_ = schema
        r.columns_ = list(mapping.keys())
        r._fitted = True
        return r

    def test_single_rename_produces_alias(self) -> None:
        """query() aliases the renamed column."""
        schema = Schema({"a": "DOUBLE", "b": "VARCHAR"})
        r = self._make_fitted_rename({"a": "x"}, schema)
        input_q = exp.select(exp.Column(this="a"), exp.Column(this="b")).from_("t")
        result = r.query(input_q)
        sql = result.sql(dialect="duckdb")
        assert "AS" in sql.upper()

    def test_unrenamed_columns_pass_through(self) -> None:
        """query() includes unmapped columns without alias."""
        schema = Schema({"a": "DOUBLE", "b": "VARCHAR"})
        r = self._make_fitted_rename({"a": "x"}, schema)
        input_q = exp.select(exp.Column(this="a"), exp.Column(this="b")).from_("t")
        result = r.query(input_q)
        sql = result.sql(dialect="duckdb")
        assert "b" in sql.lower()

    def test_wraps_input_as_subquery(self) -> None:
        """query() wraps the input query in a subquery."""
        schema = Schema({"a": "DOUBLE"})
        r = self._make_fitted_rename({"a": "x"}, schema)
        input_q = exp.select(exp.Column(this="a")).from_("t")
        result = r.query(input_q)
        # The result should be a SELECT FROM (subquery)
        assert isinstance(result, exp.Select)
        from_clause = result.find(exp.From)
        assert from_clause is not None
        assert result.find(exp.Subquery) is not None

    def test_multi_rename_sql(self) -> None:
        """query() handles multiple renames correctly."""
        schema = Schema({"a": "DOUBLE", "b": "VARCHAR", "c": "INTEGER"})
        r = self._make_fitted_rename({"a": "x", "b": "y"}, schema)
        input_q = exp.select(
            exp.Column(this="a"),
            exp.Column(this="b"),
            exp.Column(this="c"),
        ).from_("t")
        result = r.query(input_q)
        sql = result.sql(dialect="duckdb").lower()
        # Both renames should appear as aliases
        assert "x" in sql
        assert "y" in sql
        # Unchanged column should pass through
        assert "c" in sql

    def test_rename_to_same_name_no_alias(self) -> None:
        """Renaming to the same name produces an alias (identity rename)."""
        schema = Schema({"a": "DOUBLE", "b": "VARCHAR"})
        r = self._make_fitted_rename({"a": "a"}, schema)
        input_q = exp.select(exp.Column(this="a"), exp.Column(this="b")).from_("t")
        result = r.query(input_q)
        sql = result.sql(dialect="duckdb").lower()
        # Should still produce valid SQL
        assert "a" in sql
        assert "b" in sql


# -- Pipeline integration tests ----------------------------------------------


class TestPipeline:
    """Test Rename integrated with Pipeline (end-to-end)."""

    @pytest.fixture
    def backend(self) -> DuckDBBackend:
        """Create DuckDB backend with test data."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 'London'), (2.0, 'Paris'), (3.0, 'Tokyo') "
            "t(price, city)"
        )
        return DuckDBBackend(connection=conn)

    def test_fit_transform_single_rename(self, backend: DuckDBBackend) -> None:
        """Single rename preserves data values."""
        pipe = Pipeline([Rename(mapping={"price": "cost"})], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (3, 2)
        # Values should be unchanged
        np.testing.assert_array_equal(result[:, 0], [1.0, 2.0, 3.0])

    def test_fit_transform_multi_rename(self, backend: DuckDBBackend) -> None:
        """Multiple renames preserve data values and shape."""
        pipe = Pipeline(
            [Rename(mapping={"price": "cost", "city": "location"})],
            backend=backend,
        )
        result = pipe.fit_transform("t")
        assert result.shape == (3, 2)

    def test_output_column_names(self, backend: DuckDBBackend) -> None:
        """get_feature_names_out returns renamed column names."""
        pipe = Pipeline([Rename(mapping={"price": "cost"})], backend=backend)
        pipe.fit("t")
        names = pipe.get_feature_names_out()
        assert "cost" in names
        assert "price" not in names
        assert "city" in names

    def test_to_sql_contains_alias(self, backend: DuckDBBackend) -> None:
        """to_sql() output shows the renamed column."""
        pipe = Pipeline([Rename(mapping={"price": "cost"})], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql()
        assert isinstance(sql, str)
        upper = sql.upper()
        assert "SELECT" in upper
        assert "FROM" in upper

    def test_to_sql_valid_duckdb(self, backend: DuckDBBackend) -> None:
        """to_sql() output is valid DuckDB SQL that can be executed."""
        pipe = Pipeline([Rename(mapping={"price": "cost"})], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql()
        # Replace __input__ with real table name
        sql = sql.replace("__input__", "t")
        conn = backend._connection
        result = conn.execute(sql).fetchall()
        assert len(result) == 3

    def test_fit_then_transform_matches(self, backend: DuckDBBackend) -> None:
        """Separate fit() and transform() produce same result as fit_transform()."""
        pipe1 = Pipeline([Rename(mapping={"price": "cost"})], backend=backend)
        result1 = pipe1.fit_transform("t")

        pipe2 = Pipeline([Rename(mapping={"price": "cost"})], backend=backend)
        pipe2.fit("t")
        result2 = pipe2.transform("t")

        np.testing.assert_array_equal(result1, result2)

    def test_data_values_preserved(self, backend: DuckDBBackend) -> None:
        """Rename only changes column names, not data values."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t2 AS SELECT * FROM VALUES (10.0, 20.0), (30.0, 40.0) t2(a, b)")
        be = DuckDBBackend(connection=conn)
        pipe = Pipeline([Rename(mapping={"a": "x", "b": "y"})], backend=be)
        result = pipe.fit_transform("t2")
        np.testing.assert_array_equal(result, [[10.0, 20.0], [30.0, 40.0]])


# -- Not-fitted guard -------------------------------------------------------


class TestNotFittedGuard:
    """Test that unfitted Rename raises appropriate errors."""

    def test_get_feature_names_out_unfitted(self) -> None:
        """get_feature_names_out raises NotFittedError when not fitted."""
        from sqlearn.core.errors import NotFittedError

        r = Rename(mapping={"a": "b"})
        with pytest.raises(NotFittedError):
            r.get_feature_names_out()


# -- Composition tests ------------------------------------------------------


class TestComposition:
    """Test Rename composing with other transformers."""

    def test_rename_after_standard_scaler(self) -> None:
        """StandardScaler + Rename produces correct renamed output."""
        from sqlearn.scalers.standard import StandardScaler

        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 10.0), (2.0, 20.0), (3.0, 30.0), "
            "(4.0, 40.0), (5.0, 50.0) t(a, b)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [StandardScaler(), Rename(mapping={"a": "scaled_a"})],
            backend=backend,
        )
        pipe.fit("t")
        names = pipe.get_feature_names_out()
        assert "scaled_a" in names
        assert "a" not in names
        assert "b" in names

        result = pipe.fit_transform("t")
        assert result.shape == (5, 2)

    def test_rename_after_standard_scaler_sql(self) -> None:
        """SQL from StandardScaler + Rename shows composed expressions."""
        from sqlearn.scalers.standard import StandardScaler

        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1.0, 2.0), (3.0, 4.0) t(a, b)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [StandardScaler(), Rename(mapping={"a": "scaled_a"})],
            backend=backend,
        )
        pipe.fit("t")
        sql = pipe.to_sql()
        assert isinstance(sql, str)
        upper = sql.upper()
        assert "SELECT" in upper


# -- Clone and pickle tests --------------------------------------------------


class TestClonePickle:
    """Deep clone independence and pickle roundtrip."""

    def test_clone_roundtrip(self) -> None:
        """Cloned Rename has same mapping but is independent."""
        r = Rename(mapping={"a": "b", "c": "d"})
        cloned = r.clone()
        assert cloned.mapping == {"a": "b", "c": "d"}
        assert cloned is not r

    def test_clone_independence(self) -> None:
        """Modifying clone does not affect original."""
        r = Rename(mapping={"a": "b"})
        cloned = r.clone()
        cloned.set_params(mapping={"x": "y"})
        assert r.mapping == {"a": "b"}
        assert cloned.mapping == {"x": "y"}

    def test_pickle_roundtrip(self) -> None:
        """Pickle a Rename preserves mapping."""
        r = Rename(mapping={"a": "b", "c": "d"})
        data = pickle.dumps(r)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.mapping == {"a": "b", "c": "d"}

    def test_pickle_roundtrip_fitted(self) -> None:
        """Pickle a fitted Rename preserves fitted state."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1.0, 2.0) t(a, b)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Rename(mapping={"a": "x"})], backend=backend)
        pipe.fit("t")

        step = pipe.named_steps["step_00"]
        data = pickle.dumps(step)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.mapping == {"a": "x"}
        assert restored.is_fitted


# -- Edge cases --------------------------------------------------------------


class TestEdgeCases:
    """Extreme and unusual inputs."""

    def test_rename_to_same_name_end_to_end(self) -> None:
        """Renaming a column to its own name is a no-op."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1.0, 2.0) t(a, b)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Rename(mapping={"a": "a"})], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (1, 2)
        np.testing.assert_array_equal(result, [[1.0, 2.0]])

    def test_rename_preserves_all_rows(self) -> None:
        """Rename does not filter any rows."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES (1.0), (2.0), (3.0), (4.0), (5.0) t(val)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Rename(mapping={"val": "value"})], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (5, 1)

    def test_single_column_table(self) -> None:
        """Rename works on single-column tables."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (42.0) t(x)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Rename(mapping={"x": "answer"})], backend=backend)
        pipe.fit("t")
        names = pipe.get_feature_names_out()
        assert names == ["answer"]
        result = pipe.transform("t")
        np.testing.assert_array_equal(result, [[42.0]])

    def test_many_columns(self) -> None:
        """Rename works with many columns, renaming a subset."""
        conn = duckdb.connect()
        cols = [f"c{i}" for i in range(10)]
        col_defs = ", ".join(f"{c} DOUBLE" for c in cols)
        conn.execute(f"CREATE TABLE t ({col_defs})")
        values = ", ".join(["1.0"] * 10)
        conn.execute(f"INSERT INTO t VALUES ({values})")  # noqa: S608

        backend = DuckDBBackend(connection=conn)
        mapping = {"c0": "first", "c9": "last"}
        pipe = Pipeline([Rename(mapping=mapping)], backend=backend)
        pipe.fit("t")
        names = pipe.get_feature_names_out()
        assert "first" in names
        assert "last" in names
        assert "c0" not in names
        assert "c9" not in names
        # Middle columns unchanged
        for i in range(1, 9):
            assert f"c{i}" in names

    def test_null_values_preserved(self) -> None:
        """Rename preserves NULL values in data."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1.0, NULL), (NULL, 2.0) t(a, b)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Rename(mapping={"a": "x", "b": "y"})], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (2, 2)
        # First row: x=1.0, y=NULL
        np.testing.assert_equal(result[0, 0], 1.0)
        assert result[0, 1] is None or np.isnan(float(result[0, 1]))
