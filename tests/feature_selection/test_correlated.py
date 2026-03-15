"""Tests for sqlearn.feature_selection.correlated — DropCorrelated transformer."""

from __future__ import annotations

import pickle

import duckdb
import numpy as np
import pytest
import sqlglot.expressions as exp

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.pipeline import Pipeline
from sqlearn.core.schema import Schema
from sqlearn.feature_selection.correlated import DropCorrelated

# -- Constructor tests -------------------------------------------------------


class TestConstructor:
    """Test DropCorrelated constructor validation and attributes."""

    def test_default_threshold(self) -> None:
        """Default threshold is 0.95."""
        dc = DropCorrelated()
        assert dc.threshold == 0.95

    def test_custom_threshold(self) -> None:
        """Custom threshold is accepted."""
        dc = DropCorrelated(threshold=0.8)
        assert dc.threshold == 0.8

    def test_threshold_zero_raises(self) -> None:
        """Threshold 0 raises ValueError."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            DropCorrelated(threshold=0.0)

    def test_threshold_one_raises(self) -> None:
        """Threshold 1 raises ValueError."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            DropCorrelated(threshold=1.0)

    def test_threshold_negative_raises(self) -> None:
        """Negative threshold raises ValueError."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            DropCorrelated(threshold=-0.5)

    def test_threshold_above_one_raises(self) -> None:
        """Threshold above 1 raises ValueError."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            DropCorrelated(threshold=1.5)

    def test_classification_is_dynamic(self) -> None:
        """DropCorrelated is classified as dynamic."""
        assert DropCorrelated._classification == "dynamic"

    def test_default_columns_is_numeric(self) -> None:
        """Default columns are numeric."""
        assert DropCorrelated._default_columns == "numeric"

    def test_custom_columns(self) -> None:
        """Explicit column list overrides default."""
        dc = DropCorrelated(columns=["a", "b"])
        assert dc.columns == ["a", "b"]

    def test_get_params(self) -> None:
        """get_params returns constructor kwargs."""
        dc = DropCorrelated(threshold=0.9, columns=["x"])
        params = dc.get_params()
        assert params == {"threshold": 0.9, "columns": ["x"]}

    def test_set_params(self) -> None:
        """set_params updates threshold and returns self."""
        dc = DropCorrelated()
        result = dc.set_params(threshold=0.8)
        assert result is dc
        assert dc.threshold == 0.8

    def test_repr(self) -> None:
        """repr shows non-default parameters."""
        dc = DropCorrelated(threshold=0.8)
        assert "DropCorrelated" in repr(dc)
        assert "0.8" in repr(dc)

    def test_repr_default(self) -> None:
        """Default repr shows no args."""
        dc = DropCorrelated()
        assert repr(dc) == "DropCorrelated()"


# -- discover() tests -------------------------------------------------------


class TestDiscover:
    """Test DropCorrelated.discover() returns correct sqlglot aggregates."""

    @pytest.fixture
    def schema(self) -> Schema:
        """Simple numeric schema."""
        return Schema({"a": "DOUBLE", "b": "DOUBLE", "c": "DOUBLE"})

    def test_pairwise_correlations(self, schema: Schema) -> None:
        """Each unique pair gets a CORR aggregate."""
        dc = DropCorrelated()
        result = dc.discover(["a", "b", "c"], schema)
        # 3 columns → 3 pairs: a-b, a-c, b-c
        assert len(result) == 3
        assert "a__b__corr" in result
        assert "a__c__corr" in result
        assert "b__c__corr" in result

    def test_corr_ast_type(self, schema: Schema) -> None:
        """Each entry is a CORR expression."""
        dc = DropCorrelated()
        result = dc.discover(["a", "b"], schema)
        corr_expr = result["a__b__corr"]
        assert isinstance(corr_expr, exp.Corr)

    def test_single_column_no_pairs(self, schema: Schema) -> None:
        """Single column produces no correlations."""
        dc = DropCorrelated()
        result = dc.discover(["a"], schema)
        assert result == {}

    def test_two_columns_one_pair(self, schema: Schema) -> None:
        """Two columns produce exactly one pair."""
        dc = DropCorrelated()
        result = dc.discover(["a", "b"], schema)
        assert len(result) == 1
        assert "a__b__corr" in result

    def test_empty_columns_no_pairs(self, schema: Schema) -> None:
        """Empty columns list produces no pairs."""
        dc = DropCorrelated()
        result = dc.discover([], schema)
        assert result == {}

    def test_four_columns_six_pairs(self, schema: Schema) -> None:
        """Four columns produce 6 pairs (4 choose 2)."""
        schema4 = Schema({"a": "DOUBLE", "b": "DOUBLE", "c": "DOUBLE", "d": "DOUBLE"})
        dc = DropCorrelated()
        result = dc.discover(["a", "b", "c", "d"], schema4)
        assert len(result) == 6


# -- _determine_drops tests -------------------------------------------------


class TestDetermineDrops:
    """Test DropCorrelated._determine_drops() greedy strategy."""

    def test_no_correlated_pairs(self) -> None:
        """No drops when no correlations exceed threshold."""
        dc = DropCorrelated(threshold=0.9)
        dc.params_ = {"a__b__corr": 0.5, "a__c__corr": 0.3, "b__c__corr": 0.1}
        result = dc._determine_drops(["a", "b", "c"])
        assert result == []

    def test_one_correlated_pair(self) -> None:
        """One correlated pair drops one column."""
        dc = DropCorrelated(threshold=0.9)
        dc.params_ = {"a__b__corr": 0.95, "a__c__corr": 0.1, "b__c__corr": 0.2}
        result = dc._determine_drops(["a", "b", "c"])
        assert len(result) == 1
        assert result[0] in ("a", "b")

    def test_greedy_drops_most_connected(self) -> None:
        """Greedy strategy drops the column in most correlated pairs."""
        dc = DropCorrelated(threshold=0.9)
        # b is correlated with both a and c
        dc.params_ = {"a__b__corr": 0.95, "a__c__corr": 0.1, "b__c__corr": 0.95}
        result = dc._determine_drops(["a", "b", "c"])
        # b appears in 2 pairs, a and c in 1 each → b should be dropped
        assert result == ["b"]

    def test_perfectly_correlated(self) -> None:
        """Perfectly correlated (r=1.0) columns trigger drop."""
        dc = DropCorrelated(threshold=0.95)
        dc.params_ = {"a__b__corr": 1.0}
        result = dc._determine_drops(["a", "b"])
        assert len(result) == 1

    def test_negative_correlation(self) -> None:
        """Negative correlation above threshold triggers drop."""
        dc = DropCorrelated(threshold=0.9)
        dc.params_ = {"a__b__corr": -0.95}
        result = dc._determine_drops(["a", "b"])
        assert len(result) == 1

    def test_none_params(self) -> None:
        """None params produces no drops."""
        dc = DropCorrelated(threshold=0.9)
        dc.params_ = None
        result = dc._determine_drops(["a", "b"])
        assert result == []


# -- output_schema tests ----------------------------------------------------


class TestOutputSchema:
    """Test DropCorrelated.output_schema() reflects drops correctly."""

    def test_drops_correlated_column(self) -> None:
        """Output schema excludes dropped columns."""
        schema = Schema({"a": "DOUBLE", "b": "DOUBLE", "c": "DOUBLE"})
        dc = DropCorrelated(threshold=0.9)
        dc.columns_ = ["a", "b", "c"]
        dc.params_ = {"a__b__corr": 0.99, "a__c__corr": 0.1, "b__c__corr": 0.1}
        out = dc.output_schema(schema)
        assert len(out) == 2

    def test_no_drops_returns_schema(self) -> None:
        """No correlated columns returns schema unchanged."""
        schema = Schema({"a": "DOUBLE", "b": "DOUBLE"})
        dc = DropCorrelated(threshold=0.9)
        dc.columns_ = ["a", "b"]
        dc.params_ = {"a__b__corr": 0.5}
        out = dc.output_schema(schema)
        assert len(out) == 2


# -- Pipeline integration tests ----------------------------------------------


class TestPipeline:
    """Test DropCorrelated integrated with Pipeline (end-to-end)."""

    @pytest.fixture
    def correlated_backend(self) -> DuckDBBackend:
        """Create DuckDB backend with correlated columns."""
        conn = duckdb.connect()
        # a and b are perfectly correlated (b = 2*a), c is independent
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 2.0, 10.0), (2.0, 4.0, 20.0), "
            "(3.0, 6.0, 30.0), (4.0, 8.0, 5.0), "
            "(5.0, 10.0, 15.0) t(a, b, c)"
        )
        return DuckDBBackend(connection=conn)

    @pytest.fixture
    def uncorrelated_backend(self) -> DuckDBBackend:
        """Create DuckDB backend with uncorrelated columns."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 50.0, 3.0), (2.0, 10.0, 1.0), "
            "(3.0, 40.0, 4.0), (4.0, 20.0, 2.0), "
            "(5.0, 30.0, 5.0) t(a, b, c)"
        )
        return DuckDBBackend(connection=conn)

    def test_drops_correlated_column(self, correlated_backend: DuckDBBackend) -> None:
        """Correlated column is dropped from output."""
        pipe = Pipeline([DropCorrelated(threshold=0.95)], backend=correlated_backend)
        pipe.fit("t")
        names = pipe.get_feature_names_out()
        # a and b are perfectly correlated, one should be dropped
        assert len(names) == 2

    def test_shape_reduced(self, correlated_backend: DuckDBBackend) -> None:
        """Output shape is reduced when columns are dropped."""
        pipe = Pipeline([DropCorrelated(threshold=0.95)], backend=correlated_backend)
        result = pipe.fit_transform("t")
        assert result.shape[1] == 2

    def test_no_drops_uncorrelated(self, uncorrelated_backend: DuckDBBackend) -> None:
        """Uncorrelated columns are all kept."""
        pipe = Pipeline([DropCorrelated(threshold=0.95)], backend=uncorrelated_backend)
        pipe.fit("t")
        names = pipe.get_feature_names_out()
        assert len(names) == 3

    def test_to_sql_valid_duckdb(self, correlated_backend: DuckDBBackend) -> None:
        """to_sql() output is valid DuckDB SQL."""
        pipe = Pipeline([DropCorrelated(threshold=0.95)], backend=correlated_backend)
        pipe.fit("t")
        sql = pipe.to_sql()
        sql = sql.replace("__input__", "t")
        conn = correlated_backend._connection
        result = conn.execute(sql).fetchall()
        assert len(result) == 5

    def test_fit_then_transform_matches(self, correlated_backend: DuckDBBackend) -> None:
        """Separate fit() and transform() produce same result."""
        pipe1 = Pipeline([DropCorrelated(threshold=0.95)], backend=correlated_backend)
        result1 = pipe1.fit_transform("t")

        pipe2 = Pipeline([DropCorrelated(threshold=0.95)], backend=correlated_backend)
        pipe2.fit("t")
        result2 = pipe2.transform("t")

        np.testing.assert_array_equal(result1, result2)


# -- Clone and pickle tests --------------------------------------------------


class TestClonePickle:
    """Deep clone independence and pickle roundtrip."""

    def test_clone_roundtrip(self) -> None:
        """Cloned DropCorrelated has same threshold."""
        dc = DropCorrelated(threshold=0.8)
        cloned = dc.clone()
        assert cloned.threshold == 0.8
        assert cloned is not dc

    def test_clone_independence(self) -> None:
        """Modifying clone does not affect original."""
        dc = DropCorrelated(threshold=0.8)
        cloned = dc.clone()
        cloned.set_params(threshold=0.5)
        assert dc.threshold == 0.8
        assert cloned.threshold == 0.5

    def test_clone_preserves_dropped(self) -> None:
        """Clone preserves the _dropped list."""
        dc = DropCorrelated(threshold=0.9)
        dc._dropped = ["b"]
        dc._fitted = True
        cloned = dc.clone()
        assert cloned._dropped == ["b"]
        # Independence
        cloned._dropped.append("c")
        assert dc._dropped == ["b"]

    def test_pickle_roundtrip(self) -> None:
        """Pickle a DropCorrelated preserves threshold."""
        dc = DropCorrelated(threshold=0.85)
        data = pickle.dumps(dc)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.threshold == 0.85

    def test_pickle_roundtrip_fitted(self) -> None:
        """Pickle a fitted DropCorrelated preserves state."""
        dc = DropCorrelated(threshold=0.9)
        dc._fitted = True
        dc.params_ = {"a__b__corr": 0.99}
        dc._dropped = ["b"]
        data = pickle.dumps(dc)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.params_ == {"a__b__corr": 0.99}
        assert restored._dropped == ["b"]
        assert restored.is_fitted


# -- Not-fitted guard -------------------------------------------------------


class TestNotFittedGuard:
    """Test that unfitted DropCorrelated raises appropriate errors."""

    def test_get_feature_names_out_unfitted(self) -> None:
        """get_feature_names_out raises NotFittedError when not fitted."""
        from sqlearn.core.errors import NotFittedError

        dc = DropCorrelated()
        with pytest.raises(NotFittedError):
            dc.get_feature_names_out()


# -- Edge cases --------------------------------------------------------------


class TestEdgeCases:
    """Extreme and unusual inputs."""

    def test_all_columns_correlated(self) -> None:
        """When all columns are pairwise correlated, only one survives."""
        dc = DropCorrelated(threshold=0.9)
        dc.columns_ = ["a", "b", "c"]
        dc.params_ = {"a__b__corr": 0.99, "a__c__corr": 0.99, "b__c__corr": 0.99}
        dropped = dc._determine_drops(["a", "b", "c"])
        # At least 2 of 3 should be dropped
        assert len(dropped) >= 2

    def test_expressions_returns_empty(self) -> None:
        """expressions() returns empty dict (uses query instead)."""
        dc = DropCorrelated()
        result = dc.expressions(["a"], {"a": exp.Column(this="a")})
        assert result == {}

    def test_threshold_near_boundary(self) -> None:
        """Threshold just below 1.0 works correctly."""
        dc = DropCorrelated(threshold=0.999)
        dc.params_ = {"a__b__corr": 0.998}
        result = dc._determine_drops(["a", "b"])
        assert result == []

        dc.params_ = {"a__b__corr": 0.9999}
        result = dc._determine_drops(["a", "b"])
        assert len(result) == 1

    def test_null_correlation_handled(self) -> None:
        """None correlation value is safely handled."""
        dc = DropCorrelated(threshold=0.9)
        dc.params_ = {"a__b__corr": None}
        result = dc._determine_drops(["a", "b"])
        assert result == []
