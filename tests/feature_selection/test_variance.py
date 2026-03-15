"""Tests for sqlearn.feature_selection.variance — VarianceThreshold transformer."""

from __future__ import annotations

import pickle

import duckdb
import numpy as np
import pytest
import sqlglot.expressions as exp

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.pipeline import Pipeline
from sqlearn.core.schema import Schema
from sqlearn.feature_selection.variance import VarianceThreshold

# -- Constructor tests -------------------------------------------------------


class TestConstructor:
    """Test VarianceThreshold constructor validation and attributes."""

    def test_default_threshold(self) -> None:
        """Default threshold is 0.0."""
        vt = VarianceThreshold()
        assert vt.threshold == 0.0

    def test_custom_threshold(self) -> None:
        """Custom threshold is accepted."""
        vt = VarianceThreshold(threshold=0.5)
        assert vt.threshold == 0.5

    def test_negative_threshold_raises(self) -> None:
        """Negative threshold raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            VarianceThreshold(threshold=-0.1)

    def test_zero_threshold_accepted(self) -> None:
        """Zero threshold is valid (only removes constants)."""
        vt = VarianceThreshold(threshold=0.0)
        assert vt.threshold == 0.0

    def test_classification_is_dynamic(self) -> None:
        """VarianceThreshold is classified as dynamic."""
        assert VarianceThreshold._classification == "dynamic"

    def test_default_columns_is_numeric(self) -> None:
        """Default columns are numeric."""
        assert VarianceThreshold._default_columns == "numeric"

    def test_custom_columns(self) -> None:
        """Explicit column list overrides default."""
        vt = VarianceThreshold(columns=["a", "b"])
        assert vt.columns == ["a", "b"]

    def test_get_params(self) -> None:
        """get_params returns constructor kwargs."""
        vt = VarianceThreshold(threshold=0.1, columns=["x"])
        params = vt.get_params()
        assert params == {"threshold": 0.1, "columns": ["x"]}

    def test_set_params(self) -> None:
        """set_params updates threshold and returns self."""
        vt = VarianceThreshold()
        result = vt.set_params(threshold=0.5)
        assert result is vt
        assert vt.threshold == 0.5

    def test_repr_default(self) -> None:
        """Default repr shows no args."""
        vt = VarianceThreshold()
        assert repr(vt) == "VarianceThreshold()"

    def test_repr_custom(self) -> None:
        """Non-default threshold shows in repr."""
        vt = VarianceThreshold(threshold=0.5)
        assert "VarianceThreshold" in repr(vt)
        assert "0.5" in repr(vt)

    def test_classify_method(self) -> None:
        """_classify returns dynamic."""
        vt = VarianceThreshold()
        assert vt._classify() == "dynamic"


# -- discover() tests -------------------------------------------------------


class TestDiscover:
    """Test VarianceThreshold.discover() returns correct sqlglot aggregates."""

    @pytest.fixture
    def schema(self) -> Schema:
        """Simple numeric schema."""
        return Schema({"a": "DOUBLE", "b": "DOUBLE", "c": "DOUBLE"})

    def test_var_pop_per_column(self, schema: Schema) -> None:
        """Each column gets a VAR_POP aggregate."""
        vt = VarianceThreshold()
        result = vt.discover(["a", "b", "c"], schema)
        assert "a__var" in result
        assert "b__var" in result
        assert "c__var" in result
        assert len(result) == 3

    def test_var_ast_type(self, schema: Schema) -> None:
        """Variance expression is exp.VariancePop wrapping exp.Column."""
        vt = VarianceThreshold()
        result = vt.discover(["a"], schema)
        var_expr = result["a__var"]
        assert isinstance(var_expr, exp.VariancePop)
        inner = var_expr.this
        assert isinstance(inner, exp.Column)

    def test_empty_columns_returns_empty(self, schema: Schema) -> None:
        """Empty columns list returns empty dict."""
        vt = VarianceThreshold()
        result = vt.discover([], schema)
        assert result == {}

    def test_single_column(self, schema: Schema) -> None:
        """Single column produces exactly one aggregate."""
        vt = VarianceThreshold()
        result = vt.discover(["a"], schema)
        assert len(result) == 1
        assert "a__var" in result


# -- _determine_drops tests -------------------------------------------------


class TestDetermineDrops:
    """Test VarianceThreshold._determine_drops() logic."""

    def test_zero_variance_dropped(self) -> None:
        """Column with zero variance is dropped at default threshold."""
        vt = VarianceThreshold(threshold=0.0)
        vt.params_ = {"a__var": 0.0, "b__var": 1.0, "c__var": 5.0}
        result = vt._determine_drops(["a", "b", "c"])
        assert result == ["a"]

    def test_low_variance_dropped(self) -> None:
        """Column with variance below threshold is dropped."""
        vt = VarianceThreshold(threshold=1.0)
        vt.params_ = {"a__var": 0.5, "b__var": 1.0, "c__var": 5.0}
        result = vt._determine_drops(["a", "b", "c"])
        assert "a" in result
        assert "b" in result
        assert "c" not in result

    def test_all_above_threshold(self) -> None:
        """No drops when all columns exceed threshold."""
        vt = VarianceThreshold(threshold=0.0)
        vt.params_ = {"a__var": 1.0, "b__var": 2.0}
        result = vt._determine_drops(["a", "b"])
        assert result == []

    def test_all_below_threshold(self) -> None:
        """All columns dropped when all below threshold."""
        vt = VarianceThreshold(threshold=10.0)
        vt.params_ = {"a__var": 1.0, "b__var": 2.0}
        result = vt._determine_drops(["a", "b"])
        assert sorted(result) == ["a", "b"]

    def test_none_params(self) -> None:
        """None params produces no drops."""
        vt = VarianceThreshold()
        vt.params_ = None
        result = vt._determine_drops(["a"])
        assert result == []

    def test_exact_threshold_dropped(self) -> None:
        """Column with variance exactly at threshold is dropped (<=)."""
        vt = VarianceThreshold(threshold=1.0)
        vt.params_ = {"a__var": 1.0}
        result = vt._determine_drops(["a"])
        assert result == ["a"]


# -- output_schema tests ----------------------------------------------------


class TestOutputSchema:
    """Test VarianceThreshold.output_schema() reflects drops correctly."""

    def test_drops_low_variance(self) -> None:
        """Output schema excludes low-variance columns."""
        schema = Schema({"a": "DOUBLE", "b": "DOUBLE", "c": "DOUBLE"})
        vt = VarianceThreshold(threshold=0.5)
        vt.columns_ = ["a", "b", "c"]
        vt.params_ = {"a__var": 0.0, "b__var": 0.3, "c__var": 5.0}
        out = vt.output_schema(schema)
        assert "a" not in out.columns
        assert "b" not in out.columns
        assert "c" in out.columns

    def test_no_drops_returns_schema(self) -> None:
        """No low-variance columns returns schema unchanged."""
        schema = Schema({"a": "DOUBLE", "b": "DOUBLE"})
        vt = VarianceThreshold(threshold=0.0)
        vt.columns_ = ["a", "b"]
        vt.params_ = {"a__var": 1.0, "b__var": 2.0}
        out = vt.output_schema(schema)
        assert len(out) == 2


# -- Pipeline integration tests ----------------------------------------------


class TestPipeline:
    """Test VarianceThreshold integrated with Pipeline (end-to-end)."""

    @pytest.fixture
    def constant_backend(self) -> DuckDBBackend:
        """Create DuckDB backend with a constant column."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 5.0, 10.0), (2.0, 5.0, 20.0), "
            "(3.0, 5.0, 30.0), (4.0, 5.0, 40.0), "
            "(5.0, 5.0, 50.0) t(a, b, c)"
        )
        return DuckDBBackend(connection=conn)

    @pytest.fixture
    def varied_backend(self) -> DuckDBBackend:
        """Create DuckDB backend with varied columns."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 10.0, 100.0), (2.0, 20.0, 200.0), "
            "(3.0, 30.0, 300.0), (4.0, 40.0, 400.0), "
            "(5.0, 50.0, 500.0) t(a, b, c)"
        )
        return DuckDBBackend(connection=conn)

    def test_drops_constant_column(self, constant_backend: DuckDBBackend) -> None:
        """Constant column is dropped from output."""
        pipe = Pipeline([VarianceThreshold()], backend=constant_backend)
        pipe.fit("t")
        names = pipe.get_feature_names_out()
        assert "b" not in names
        assert "a" in names
        assert "c" in names

    def test_shape_reduced(self, constant_backend: DuckDBBackend) -> None:
        """Output shape is reduced when constant column is dropped."""
        pipe = Pipeline([VarianceThreshold()], backend=constant_backend)
        result = pipe.fit_transform("t")
        assert result.shape == (5, 2)

    def test_no_drops_varied(self, varied_backend: DuckDBBackend) -> None:
        """No columns dropped when all have variance."""
        pipe = Pipeline([VarianceThreshold()], backend=varied_backend)
        pipe.fit("t")
        names = pipe.get_feature_names_out()
        assert len(names) == 3

    def test_to_sql_valid_duckdb(self, constant_backend: DuckDBBackend) -> None:
        """to_sql() output is valid DuckDB SQL."""
        pipe = Pipeline([VarianceThreshold()], backend=constant_backend)
        pipe.fit("t")
        sql = pipe.to_sql()
        sql = sql.replace("__input__", "t")
        conn = constant_backend._connection
        result = conn.execute(sql).fetchall()
        assert len(result) == 5

    def test_fit_then_transform_matches(self, constant_backend: DuckDBBackend) -> None:
        """Separate fit() and transform() produce same result."""
        pipe1 = Pipeline([VarianceThreshold()], backend=constant_backend)
        result1 = pipe1.fit_transform("t")

        pipe2 = Pipeline([VarianceThreshold()], backend=constant_backend)
        pipe2.fit("t")
        result2 = pipe2.transform("t")

        np.testing.assert_array_equal(result1, result2)

    def test_data_values_preserved(self, constant_backend: DuckDBBackend) -> None:
        """Values in remaining columns are unchanged."""
        pipe = Pipeline([VarianceThreshold()], backend=constant_backend)
        result = pipe.fit_transform("t")
        # a column: 1, 2, 3, 4, 5
        np.testing.assert_array_equal(result[:, 0], [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_custom_threshold(self) -> None:
        """Custom threshold drops columns with low variance."""
        conn = duckdb.connect()
        # a has var ~2.5, b has var ~0.0004, c has var ~250
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 5.01, 10.0), (2.0, 5.02, 20.0), "
            "(3.0, 5.00, 30.0), (4.0, 5.01, 40.0), "
            "(5.0, 5.02, 50.0) t(a, b, c)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([VarianceThreshold(threshold=1.0)], backend=backend)
        pipe.fit("t")
        names = pipe.get_feature_names_out()
        # b has very low variance, should be dropped
        assert "b" not in names
        assert "a" in names
        assert "c" in names


# -- Composition tests ------------------------------------------------------


class TestComposition:
    """Test VarianceThreshold composing with other transformers."""

    def test_imputer_then_variance(self) -> None:
        """Imputer + VarianceThreshold works end-to-end."""
        from sqlearn.imputers.imputer import Imputer

        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 5.0, NULL), (2.0, 5.0, 20.0), "
            "(3.0, 5.0, 30.0), (4.0, 5.0, 40.0) t(a, b, c)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [Imputer(), VarianceThreshold()],
            backend=backend,
        )
        result = pipe.fit_transform("t")
        # b is constant → dropped
        assert result.shape[1] == 2


# -- Clone and pickle tests --------------------------------------------------


class TestClonePickle:
    """Deep clone independence and pickle roundtrip."""

    def test_clone_roundtrip(self) -> None:
        """Cloned VarianceThreshold has same threshold."""
        vt = VarianceThreshold(threshold=0.5)
        cloned = vt.clone()
        assert cloned.threshold == 0.5
        assert cloned is not vt

    def test_clone_independence(self) -> None:
        """Modifying clone does not affect original."""
        vt = VarianceThreshold(threshold=0.5)
        cloned = vt.clone()
        cloned.set_params(threshold=1.0)
        assert vt.threshold == 0.5
        assert cloned.threshold == 1.0

    def test_clone_preserves_dropped(self) -> None:
        """Clone preserves the _dropped list."""
        vt = VarianceThreshold()
        vt._dropped = ["b"]
        vt._fitted = True
        cloned = vt.clone()
        assert cloned._dropped == ["b"]
        cloned._dropped.append("c")
        assert vt._dropped == ["b"]

    def test_pickle_roundtrip(self) -> None:
        """Pickle a VarianceThreshold preserves threshold."""
        vt = VarianceThreshold(threshold=0.5)
        data = pickle.dumps(vt)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.threshold == 0.5

    def test_pickle_roundtrip_fitted(self) -> None:
        """Pickle a fitted VarianceThreshold preserves state."""
        vt = VarianceThreshold(threshold=0.5)
        vt._fitted = True
        vt.params_ = {"a__var": 0.0, "b__var": 5.0}
        vt._dropped = ["a"]
        data = pickle.dumps(vt)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.params_ == {"a__var": 0.0, "b__var": 5.0}
        assert restored._dropped == ["a"]
        assert restored.is_fitted


# -- Not-fitted guard -------------------------------------------------------


class TestNotFittedGuard:
    """Test that unfitted VarianceThreshold raises appropriate errors."""

    def test_get_feature_names_out_unfitted(self) -> None:
        """get_feature_names_out raises NotFittedError when not fitted."""
        from sqlearn.core.errors import NotFittedError

        vt = VarianceThreshold()
        with pytest.raises(NotFittedError):
            vt.get_feature_names_out()


# -- Edge cases --------------------------------------------------------------


class TestEdgeCases:
    """Extreme and unusual inputs."""

    def test_all_constant_columns(self) -> None:
        """All constant columns are all dropped."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES (5.0, 3.0), (5.0, 3.0), (5.0, 3.0) t(a, b)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([VarianceThreshold()], backend=backend)
        pipe.fit("t")
        names = pipe.get_feature_names_out()
        assert len(names) == 0

    def test_single_row(self) -> None:
        """Single row has zero variance — all numeric columns dropped."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (42.0, 7.0) t(a, b)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([VarianceThreshold()], backend=backend)
        pipe.fit("t")
        names = pipe.get_feature_names_out()
        assert len(names) == 0

    def test_expressions_returns_empty(self) -> None:
        """expressions() returns empty dict (uses query instead)."""
        vt = VarianceThreshold()
        result = vt.expressions(["a"], {"a": exp.Column(this="a")})
        assert result == {}

    def test_high_threshold_drops_all(self) -> None:
        """Very high threshold drops all columns."""
        vt = VarianceThreshold(threshold=1e10)
        vt.columns_ = ["a", "b"]
        vt.params_ = {"a__var": 100.0, "b__var": 200.0}
        dropped = vt._determine_drops(["a", "b"])
        assert sorted(dropped) == ["a", "b"]

    def test_mixed_types_only_numeric_dropped(self) -> None:
        """Only numeric columns are considered; categorical passes through."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES (5.0, 'A'), (5.0, 'B'), (5.0, 'C') t(val, cat)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([VarianceThreshold()], backend=backend)
        pipe.fit("t")
        names = pipe.get_feature_names_out()
        # val is constant (dropped), cat is categorical (passed through)
        assert "cat" in names
        assert "val" not in names
