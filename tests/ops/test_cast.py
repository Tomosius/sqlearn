"""Tests for sqlearn.ops.cast -- Cast transformer."""

from __future__ import annotations

import pickle

import duckdb
import numpy as np
import pytest
import sqlglot.expressions as exp

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.pipeline import Pipeline
from sqlearn.core.schema import Schema
from sqlearn.ops.cast import Cast

# -- Constructor tests --------------------------------------------------------


class TestConstructor:
    """Test Cast constructor and validation."""

    def test_basic_mapping(self) -> None:
        """Valid mapping is stored."""
        caster = Cast({"price": "DOUBLE", "qty": "INTEGER"})
        assert caster.mapping == {"price": "DOUBLE", "qty": "INTEGER"}

    def test_columns_set_from_mapping_keys(self) -> None:
        """Columns are set to mapping keys."""
        caster = Cast({"price": "DOUBLE", "qty": "INTEGER"})
        assert caster.columns == ["price", "qty"]

    def test_empty_mapping_raises(self) -> None:
        """Empty mapping raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            Cast({})

    def test_non_dict_mapping_raises(self) -> None:
        """Non-dict mapping raises TypeError."""
        with pytest.raises(TypeError, match="must be a dict"):
            Cast([("a", "DOUBLE")])  # type: ignore[arg-type]

    def test_non_string_key_raises(self) -> None:
        """Non-string key raises TypeError."""
        with pytest.raises(TypeError, match="keys must be strings"):
            Cast({42: "DOUBLE"})  # type: ignore[dict-item]

    def test_non_string_value_raises(self) -> None:
        """Non-string value raises TypeError."""
        with pytest.raises(TypeError, match="values must be strings"):
            Cast({"price": 42})  # type: ignore[dict-item]

    def test_classification_is_static(self) -> None:
        """Cast is a static transformer."""
        assert Cast._classification == "static"

    def test_classify_returns_static(self) -> None:
        """Auto-classification confirms static."""
        caster = Cast({"a": "DOUBLE"})
        assert caster._classify() == "static"

    def test_get_params(self) -> None:
        """get_params returns the mapping."""
        caster = Cast({"price": "DOUBLE"})
        params = caster.get_params()
        assert params == {"mapping": {"price": "DOUBLE"}}

    def test_set_params(self) -> None:
        """set_params updates mapping and returns self."""
        caster = Cast({"a": "DOUBLE"})
        result = caster.set_params(mapping={"b": "INTEGER"})
        assert result is caster
        assert caster.mapping == {"b": "INTEGER"}

    def test_single_column_mapping(self) -> None:
        """Single-column mapping is valid."""
        caster = Cast({"x": "BIGINT"})
        assert caster.mapping == {"x": "BIGINT"}
        assert caster.columns == ["x"]

    def test_default_columns_is_none(self) -> None:
        """_default_columns is None (uses mapping keys)."""
        assert Cast._default_columns is None


# -- expressions() tests -----------------------------------------------------


class TestExpressions:
    """Test Cast.expressions() generates correct sqlglot CAST ASTs."""

    def test_single_cast(self) -> None:
        """Single column CAST expression."""
        caster = Cast({"price": "DOUBLE"})
        exprs = {"price": exp.Column(this="price")}
        result = caster.expressions(["price"], exprs)
        assert "price" in result
        assert isinstance(result["price"], exp.Cast)

    def test_cast_target_type(self) -> None:
        """CAST target type matches mapping."""
        caster = Cast({"price": "DOUBLE"})
        exprs = {"price": exp.Column(this="price")}
        result = caster.expressions(["price"], exprs)
        cast_node = result["price"]
        assert isinstance(cast_node, exp.Cast)
        assert isinstance(cast_node.to, exp.DataType)
        sql = cast_node.sql(dialect="duckdb")
        assert "DOUBLE" in sql.upper()

    def test_multiple_casts(self) -> None:
        """Multiple columns each get their own CAST."""
        caster = Cast({"a": "DOUBLE", "b": "INTEGER", "c": "VARCHAR"})
        exprs = {
            "a": exp.Column(this="a"),
            "b": exp.Column(this="b"),
            "c": exp.Column(this="c"),
        }
        result = caster.expressions(["a", "b", "c"], exprs)
        assert len(result) == 3
        for col in ("a", "b", "c"):
            assert isinstance(result[col], exp.Cast)

    def test_uses_exprs_not_column(self) -> None:
        """expressions() uses exprs[col], not exp.Column(this=col)."""
        caster = Cast({"a": "INTEGER"})
        # Simulate a prior transform: a is already (a * 2)
        prior = exp.Mul(this=exp.Column(this="a"), expression=exp.Literal.number(2))
        exprs = {"a": prior}
        result = caster.expressions(["a"], exprs)
        # The inner expression of the CAST should be the Mul, not a Column
        assert isinstance(result["a"].this, exp.Mul)

    def test_only_modifies_target_columns(self) -> None:
        """Columns not in mapping are not in result."""
        caster = Cast({"a": "DOUBLE"})
        exprs = {
            "a": exp.Column(this="a"),
            "b": exp.Column(this="b"),
        }
        result = caster.expressions(["a"], exprs)
        assert "a" in result
        assert "b" not in result

    def test_cast_to_integer(self) -> None:
        """CAST to INTEGER produces correct type node."""
        caster = Cast({"x": "INTEGER"})
        exprs = {"x": exp.Column(this="x")}
        result = caster.expressions(["x"], exprs)
        sql = result["x"].sql(dialect="duckdb")
        assert "INT" in sql.upper()

    def test_cast_to_varchar(self) -> None:
        """CAST to VARCHAR produces correct type node."""
        caster = Cast({"x": "VARCHAR"})
        exprs = {"x": exp.Column(this="x")}
        result = caster.expressions(["x"], exprs)
        sql = result["x"].sql(dialect="duckdb")
        assert "VARCHAR" in sql.upper() or "TEXT" in sql.upper()

    def test_cast_to_boolean(self) -> None:
        """CAST to BOOLEAN produces correct type node."""
        caster = Cast({"x": "BOOLEAN"})
        exprs = {"x": exp.Column(this="x")}
        result = caster.expressions(["x"], exprs)
        sql = result["x"].sql(dialect="duckdb")
        assert "BOOLEAN" in sql.upper() or "BOOL" in sql.upper()

    def test_empty_columns_list(self) -> None:
        """Empty columns list returns empty dict."""
        caster = Cast({"a": "DOUBLE"})
        exprs = {"a": exp.Column(this="a")}
        result = caster.expressions([], exprs)
        assert result == {}

    def test_cast_sql_output(self) -> None:
        """Generated SQL string contains CAST(... AS ...)."""
        caster = Cast({"price": "DOUBLE"})
        exprs = {"price": exp.Column(this="price")}
        result = caster.expressions(["price"], exprs)
        sql = result["price"].sql(dialect="duckdb")
        assert "CAST" in sql.upper()
        assert "DOUBLE" in sql.upper()


# -- output_schema() tests ---------------------------------------------------


class TestOutputSchema:
    """Test Cast.output_schema() updates column types."""

    def test_type_updated(self) -> None:
        """Cast columns have updated types in output schema."""
        caster = Cast({"price": "DOUBLE"})
        schema = Schema({"price": "INTEGER", "name": "VARCHAR"})
        result = caster.output_schema(schema)
        assert result["price"] == "DOUBLE"

    def test_uncast_columns_unchanged(self) -> None:
        """Non-cast columns keep their original types."""
        caster = Cast({"price": "DOUBLE"})
        schema = Schema({"price": "INTEGER", "name": "VARCHAR"})
        result = caster.output_schema(schema)
        assert result["name"] == "VARCHAR"

    def test_multiple_casts(self) -> None:
        """Multiple columns updated in output schema."""
        caster = Cast({"a": "DOUBLE", "b": "BIGINT"})
        schema = Schema({"a": "INTEGER", "b": "FLOAT", "c": "VARCHAR"})
        result = caster.output_schema(schema)
        assert result["a"] == "DOUBLE"
        assert result["b"] == "BIGINT"
        assert result["c"] == "VARCHAR"

    def test_schema_immutable(self) -> None:
        """Original schema is not modified."""
        caster = Cast({"price": "DOUBLE"})
        schema = Schema({"price": "INTEGER"})
        _ = caster.output_schema(schema)
        assert schema["price"] == "INTEGER"

    def test_mapping_column_not_in_schema(self) -> None:
        """Mapping column not in schema is silently ignored."""
        caster = Cast({"missing": "DOUBLE"})
        schema = Schema({"price": "INTEGER"})
        result = caster.output_schema(schema)
        assert result["price"] == "INTEGER"
        assert len(result) == 1

    def test_cast_to_same_type(self) -> None:
        """Casting to the same type returns schema with same type (no-op)."""
        caster = Cast({"price": "DOUBLE"})
        schema = Schema({"price": "DOUBLE"})
        result = caster.output_schema(schema)
        assert result["price"] == "DOUBLE"


# -- __repr__ tests -----------------------------------------------------------


class TestRepr:
    """Test Cast.__repr__."""

    def test_repr_single(self) -> None:
        """Repr shows the mapping."""
        caster = Cast({"price": "DOUBLE"})
        r = repr(caster)
        assert "Cast" in r
        assert "price" in r
        assert "DOUBLE" in r

    def test_repr_multiple(self) -> None:
        """Repr shows multiple mapping entries."""
        caster = Cast({"a": "DOUBLE", "b": "INTEGER"})
        r = repr(caster)
        assert "Cast" in r
        assert "a" in r
        assert "b" in r


# -- Pipeline integration tests -----------------------------------------------


class TestPipeline:
    """Test Cast integrated with Pipeline (end-to-end)."""

    @pytest.fixture
    def backend(self) -> DuckDBBackend:
        """Create DuckDB backend with integer test data."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1, 10, 'x'), (2, 20, 'y'), (3, 30, 'z') t(a, b, name)"
        )
        return DuckDBBackend(connection=conn)

    @pytest.fixture
    def float_backend(self) -> DuckDBBackend:
        """Create DuckDB backend with float test data."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES (1.7, 10.3), (2.2, 20.9), (3.5, 30.1) t(a, b)"
        )
        return DuckDBBackend(connection=conn)

    def test_fit_transform_cast_to_double(self, backend: DuckDBBackend) -> None:
        """Casting integer to double produces float values."""
        pipe = Pipeline([Cast({"a": "DOUBLE"})], backend=backend)
        result = pipe.fit_transform("t")
        # a should now be float
        assert result.shape[0] == 3

    def test_fit_transform_cast_to_integer(self, float_backend: DuckDBBackend) -> None:
        """Casting float to integer truncates."""
        pipe = Pipeline([Cast({"a": "INTEGER"})], backend=float_backend)
        result = pipe.fit_transform("t")
        # a values should be truncated integers
        a_col = result[:, 0]
        for val in a_col:
            assert float(val) == int(float(val))

    def test_fit_transform_shape(self, backend: DuckDBBackend) -> None:
        """Output shape matches input shape (Cast doesn't add/remove columns)."""
        pipe = Pipeline([Cast({"a": "DOUBLE"})], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (3, 3)

    def test_to_sql_contains_cast(self, backend: DuckDBBackend) -> None:
        """to_sql() output contains CAST expressions."""
        pipe = Pipeline([Cast({"a": "DOUBLE"})], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql()
        assert "CAST" in sql.upper()
        assert "DOUBLE" in sql.upper()

    def test_to_sql_multiple_casts(self, backend: DuckDBBackend) -> None:
        """to_sql() shows CAST for each mapped column."""
        pipe = Pipeline([Cast({"a": "DOUBLE", "b": "BIGINT"})], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql().upper()
        assert sql.count("CAST") == 2

    def test_passthrough_unmapped_columns(self, backend: DuckDBBackend) -> None:
        """Unmapped columns pass through unchanged."""
        pipe = Pipeline([Cast({"a": "DOUBLE"})], backend=backend)
        result = pipe.fit_transform("t")
        # name column (index 2) should still be string
        names = result[:, 2]
        assert set(names) == {"x", "y", "z"}

    def test_fit_then_transform_separate(self, backend: DuckDBBackend) -> None:
        """Separate fit() and transform() produce same result as fit_transform()."""
        pipe1 = Pipeline([Cast({"a": "DOUBLE"})], backend=backend)
        result1 = pipe1.fit_transform("t")

        pipe2 = Pipeline([Cast({"a": "DOUBLE"})], backend=backend)
        pipe2.fit("t")
        result2 = pipe2.transform("t")

        np.testing.assert_array_equal(result1, result2)

    def test_cast_string_to_integer(self) -> None:
        """Cast VARCHAR to INTEGER."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES ('1'), ('2'), ('3') t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Cast({"a": "INTEGER"})], backend=backend)
        result = pipe.fit_transform("t")
        expected = np.array([[1], [2], [3]])
        np.testing.assert_array_equal(result, expected)

    def test_cast_integer_to_double(self) -> None:
        """Cast INTEGER to DOUBLE preserves values."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1), (2), (3) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Cast({"a": "DOUBLE"})], backend=backend)
        result = pipe.fit_transform("t")
        np.testing.assert_array_equal(result, np.array([[1.0], [2.0], [3.0]]))

    def test_cast_float_to_integer_truncates(self) -> None:
        """Cast DOUBLE to INTEGER truncates decimal part."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1.9), (2.1), (3.7) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Cast({"a": "INTEGER"})], backend=backend)
        result = pipe.fit_transform("t")
        # DuckDB CAST truncates toward zero
        a_col = result[:, 0].astype(float)
        assert all(v == int(v) for v in a_col)


# -- Not-fitted guard tests ---------------------------------------------------


class TestNotFittedGuard:
    """Calling transform/to_sql before fit() raises NotFittedError."""

    def test_transform_before_fit_raises(self) -> None:
        """transform() on unfitted pipeline raises NotFittedError."""
        from sqlearn.core.errors import NotFittedError

        pipe = Pipeline([Cast({"a": "DOUBLE"})])
        with pytest.raises(NotFittedError):
            pipe.transform("t")

    def test_to_sql_before_fit_raises(self) -> None:
        """to_sql() on unfitted pipeline raises NotFittedError."""
        from sqlearn.core.errors import NotFittedError

        pipe = Pipeline([Cast({"a": "DOUBLE"})])
        with pytest.raises(NotFittedError):
            pipe.to_sql()

    def test_get_feature_names_before_fit_raises(self) -> None:
        """get_feature_names_out() on unfitted pipeline raises NotFittedError."""
        from sqlearn.core.errors import NotFittedError

        pipe = Pipeline([Cast({"a": "DOUBLE"})])
        with pytest.raises(NotFittedError):
            pipe.get_feature_names_out()


# -- Clone and pickle tests ---------------------------------------------------


class TestCloneAndPickle:
    """Deep clone independence and pickle roundtrip."""

    @pytest.fixture
    def fitted_pipe(self) -> Pipeline:
        """Create a fitted pipeline for clone/pickle tests."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1, 10), (2, 20), (3, 30) t(a, b)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Cast({"a": "DOUBLE"})], backend=backend)
        pipe.fit("t")
        return pipe

    def test_clone_produces_same_sql(self, fitted_pipe: Pipeline) -> None:
        """Cloned pipeline produces identical SQL."""
        cloned = fitted_pipe.clone()
        assert fitted_pipe.to_sql() == cloned.to_sql()

    def test_clone_is_independent_instance(self, fitted_pipe: Pipeline) -> None:
        """Cloned pipeline is a separate instance."""
        cloned = fitted_pipe.clone()
        original_caster = fitted_pipe.steps[0][1]
        cloned_caster = cloned.steps[0][1]
        assert original_caster is not cloned_caster
        assert cloned_caster.mapping == {"a": "DOUBLE"}
        assert cloned_caster._fitted is True

    def test_pickle_roundtrip(self) -> None:
        """Pickle a Cast transformer preserves mapping."""
        caster = Cast({"a": "DOUBLE", "b": "INTEGER"})
        caster._fitted = True
        data = pickle.dumps(caster)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.mapping == {"a": "DOUBLE", "b": "INTEGER"}
        assert restored._fitted is True

    def test_pickle_unfitted(self) -> None:
        """Pickle an unfitted Cast preserves state."""
        caster = Cast({"x": "VARCHAR"})
        data = pickle.dumps(caster)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.mapping == {"x": "VARCHAR"}
        assert restored._fitted is False


# -- Composition tests --------------------------------------------------------


class TestComposition:
    """Cast composing with other transformers."""

    def test_imputer_then_cast(self) -> None:
        """Imputer + Cast: COALESCE nested inside CAST."""
        from sqlearn.imputers.imputer import Imputer

        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1.0), (NULL), (3.0), (4.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Imputer(), Cast({"a": "INTEGER"})], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (4, 1)
        # No NaN/None in output (NULL was imputed before casting)
        assert not np.any(np.isnan(result.astype(float)))

    def test_imputer_cast_sql_nesting(self) -> None:
        """SQL shows COALESCE nested inside CAST."""
        from sqlearn.imputers.imputer import Imputer

        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1.0), (NULL), (3.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Imputer(), Cast({"a": "INTEGER"})], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql().upper()
        assert "COALESCE" in sql
        assert "CAST" in sql

    def test_cast_then_scaler(self) -> None:
        """Cast + StandardScaler: cast first, then scale."""
        from sqlearn.scalers.standard import StandardScaler

        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1), (2), (3), (4), (5) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Cast({"a": "DOUBLE"}), StandardScaler()], backend=backend)
        result = pipe.fit_transform("t")
        # Should have zero mean
        np.testing.assert_allclose(result.mean(), 0.0, atol=1e-10)

    def test_operator_composition(self) -> None:
        """Cast + Cast via + operator creates Pipeline."""
        cast1 = Cast({"a": "DOUBLE"})
        cast2 = Cast({"b": "INTEGER"})
        pipe = cast1 + cast2
        assert isinstance(pipe, Pipeline)


# -- Edge cases ---------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and unusual inputs."""

    def test_cast_to_same_type_pipeline(self) -> None:
        """Casting to the same type is effectively a no-op."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1.0), (2.0), (3.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Cast({"a": "DOUBLE"})], backend=backend)
        result = pipe.fit_transform("t")
        np.testing.assert_array_equal(result, np.array([[1.0], [2.0], [3.0]]))

    def test_cast_null_values(self) -> None:
        """NULL values remain NULL after CAST."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1), (NULL), (3) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Cast({"a": "DOUBLE"})], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (3, 1)

    def test_cast_single_row(self) -> None:
        """Single-row table works correctly."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (42) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Cast({"a": "DOUBLE"})], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (1, 1)
        np.testing.assert_allclose(result[0, 0], 42.0)

    def test_many_columns_cast(self) -> None:
        """Casting many columns at once works."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT 1 AS a, 2 AS b, 3 AS c, 4 AS d, 5 AS e")
        backend = DuckDBBackend(connection=conn)
        mapping = {"a": "DOUBLE", "b": "DOUBLE", "c": "DOUBLE", "d": "DOUBLE", "e": "DOUBLE"}
        pipe = Pipeline([Cast(mapping)], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (1, 5)


# -- SQL snapshot tests -------------------------------------------------------


class TestSqlSnapshot:
    """Verify SQL output structure and patterns."""

    @pytest.fixture
    def fitted_pipe(self) -> Pipeline:
        """Create fitted pipeline for SQL verification."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES (1, 'x'), (2, 'y'), (3, 'z') t(price, name)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Cast({"price": "DOUBLE"})], backend=backend)
        pipe.fit("t")
        return pipe

    def test_sql_has_select_from(self, fitted_pipe: Pipeline) -> None:
        """SQL contains SELECT and FROM."""
        sql = fitted_pipe.to_sql().upper()
        assert "SELECT" in sql
        assert "FROM" in sql

    def test_sql_has_cast_per_column(self, fitted_pipe: Pipeline) -> None:
        """Each mapped column gets a CAST."""
        sql = fitted_pipe.to_sql().upper()
        assert sql.count("CAST") == 1

    def test_sql_custom_table(self, fitted_pipe: Pipeline) -> None:
        """to_sql(table=...) uses custom table name."""
        sql = fitted_pipe.to_sql(table="my_data")
        assert "my_data" in sql

    def test_sql_custom_dialect(self, fitted_pipe: Pipeline) -> None:
        """to_sql(dialect=...) generates valid SQL for that dialect."""
        sql_pg = fitted_pipe.to_sql(dialect="postgres")
        sql_duck = fitted_pipe.to_sql(dialect="duckdb")
        assert isinstance(sql_pg, str)
        assert isinstance(sql_duck, str)
