"""Tests for sqlearn.feature_selection.kbest — SelectKBest transformer."""

from __future__ import annotations

import pickle

import duckdb
import numpy as np
import pytest
import sqlglot.expressions as exp

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.pipeline import Pipeline
from sqlearn.core.schema import Schema
from sqlearn.feature_selection.kbest import SelectKBest

# -- Constructor tests -------------------------------------------------------


class TestConstructor:
    """Test SelectKBest constructor validation and attributes."""

    def test_basic_k(self) -> None:
        """k=2 is accepted."""
        skb = SelectKBest(k=2)
        assert skb.k == 2

    def test_default_score_func(self) -> None:
        """Default score_func is f_regression."""
        skb = SelectKBest(k=1)
        assert skb.score_func == "f_regression"

    def test_custom_score_func(self) -> None:
        """Custom score_func is accepted."""
        skb = SelectKBest(k=1, score_func="mutual_info")
        assert skb.score_func == "mutual_info"

    def test_f_classif_score_func(self) -> None:
        """f_classif score_func is accepted."""
        skb = SelectKBest(k=1, score_func="f_classif")
        assert skb.score_func == "f_classif"

    def test_k_zero_raises(self) -> None:
        """k=0 raises ValueError."""
        with pytest.raises(ValueError, match="k must be >= 1"):
            SelectKBest(k=0)

    def test_k_negative_raises(self) -> None:
        """Negative k raises ValueError."""
        with pytest.raises(ValueError, match="k must be >= 1"):
            SelectKBest(k=-1)

    def test_invalid_score_func_raises(self) -> None:
        """Invalid score_func raises ValueError."""
        with pytest.raises(ValueError, match="score_func must be one of"):
            SelectKBest(k=1, score_func="invalid")

    def test_classification_is_dynamic(self) -> None:
        """SelectKBest is classified as dynamic."""
        assert SelectKBest._classification == "dynamic"

    def test_default_columns_is_numeric(self) -> None:
        """Default columns are numeric."""
        assert SelectKBest._default_columns == "numeric"

    def test_custom_columns(self) -> None:
        """Explicit column list overrides default."""
        skb = SelectKBest(k=1, columns=["a", "b"])
        assert skb.columns == ["a", "b"]

    def test_get_params(self) -> None:
        """get_params returns constructor kwargs."""
        skb = SelectKBest(k=3, score_func="mutual_info", columns=["x"])
        params = skb.get_params()
        assert params == {"k": 3, "score_func": "mutual_info", "columns": ["x"]}

    def test_set_params(self) -> None:
        """set_params updates k and returns self."""
        skb = SelectKBest(k=2)
        result = skb.set_params(k=5)
        assert result is skb
        assert skb.k == 5

    def test_repr(self) -> None:
        """repr shows k parameter."""
        skb = SelectKBest(k=3)
        r = repr(skb)
        assert "SelectKBest" in r
        assert "k=3" in r

    def test_repr_default_score_func(self) -> None:
        """Default score_func not shown in repr."""
        skb = SelectKBest(k=2)
        r = repr(skb)
        assert "f_regression" not in r

    def test_repr_custom_score_func(self) -> None:
        """Non-default score_func shows in repr."""
        skb = SelectKBest(k=2, score_func="mutual_info")
        r = repr(skb)
        assert "mutual_info" in r

    def test_classify_method(self) -> None:
        """_classify returns dynamic."""
        skb = SelectKBest(k=1)
        assert skb._classify() == "dynamic"


# -- discover() tests -------------------------------------------------------


class TestDiscover:
    """Test SelectKBest.discover() returns correct sqlglot aggregates."""

    @pytest.fixture
    def schema(self) -> Schema:
        """Schema with numeric columns."""
        return Schema({"a": "DOUBLE", "b": "DOUBLE", "c": "DOUBLE", "target": "DOUBLE"})

    def test_corr_per_column(self, schema: Schema) -> None:
        """Each column gets a CORR expression against target."""
        skb = SelectKBest(k=2)
        result = skb.discover(["a", "b", "c"], schema, y_column="target")
        assert "a__score" in result
        assert "b__score" in result
        assert "c__score" in result
        assert len(result) == 3

    def test_corr_ast_type(self, schema: Schema) -> None:
        """Each entry is a CORR expression."""
        skb = SelectKBest(k=1)
        result = skb.discover(["a"], schema, y_column="target")
        corr_expr = result["a__score"]
        assert isinstance(corr_expr, exp.Corr)

    def test_no_y_column_raises(self, schema: Schema) -> None:
        """discover() raises ValueError when y_column is None."""
        skb = SelectKBest(k=1)
        with pytest.raises(ValueError, match="requires y"):
            skb.discover(["a"], schema, y_column=None)

    def test_excludes_target_column(self, schema: Schema) -> None:
        """Target column is excluded from scoring."""
        skb = SelectKBest(k=1)
        result = skb.discover(["a", "target"], schema, y_column="target")
        assert "target__score" not in result
        assert "a__score" in result

    def test_empty_columns(self, schema: Schema) -> None:
        """Empty columns list returns empty dict."""
        skb = SelectKBest(k=1)
        result = skb.discover([], schema, y_column="target")
        assert result == {}


# -- _determine_selected tests -----------------------------------------------


class TestDetermineSelected:
    """Test SelectKBest._determine_selected() logic."""

    def test_selects_top_k(self) -> None:
        """Selects the top k columns by absolute score."""
        skb = SelectKBest(k=2)
        skb.params_ = {"a__score": 0.9, "b__score": 0.3, "c__score": 0.7}
        result = skb._determine_selected(["a", "b", "c"])
        assert len(result) == 2
        assert "a" in result
        assert "c" in result
        assert "b" not in result

    def test_uses_absolute_score(self) -> None:
        """Absolute value of score is used for ranking."""
        skb = SelectKBest(k=1)
        skb.params_ = {"a__score": -0.9, "b__score": 0.3}
        result = skb._determine_selected(["a", "b"])
        assert result == ["a"]

    def test_k_exceeds_columns(self) -> None:
        """k > number of scored columns returns all columns."""
        skb = SelectKBest(k=10)
        skb.params_ = {"a__score": 0.9, "b__score": 0.3}
        result = skb._determine_selected(["a", "b"])
        assert len(result) == 2
        assert "a" in result
        assert "b" in result

    def test_preserves_original_order(self) -> None:
        """Selected columns are returned in original column order."""
        skb = SelectKBest(k=2)
        skb.params_ = {"a__score": 0.3, "b__score": 0.9, "c__score": 0.7}
        result = skb._determine_selected(["a", "b", "c"])
        # b and c are top 2, but should come in original order
        assert result == ["b", "c"]

    def test_tie_breaking(self) -> None:
        """Tied scores break alphabetically."""
        skb = SelectKBest(k=1)
        skb.params_ = {"a__score": 0.5, "b__score": 0.5}
        result = skb._determine_selected(["a", "b"])
        assert len(result) == 1
        assert result[0] == "a"

    def test_none_params(self) -> None:
        """None params returns no columns."""
        skb = SelectKBest(k=2)
        skb.params_ = None
        result = skb._determine_selected(["a", "b"])
        assert result == []

    def test_k_equals_columns(self) -> None:
        """k equal to number of columns keeps all."""
        skb = SelectKBest(k=3)
        skb.params_ = {"a__score": 0.9, "b__score": 0.5, "c__score": 0.7}
        result = skb._determine_selected(["a", "b", "c"])
        assert len(result) == 3


# -- output_schema tests ----------------------------------------------------


class TestOutputSchema:
    """Test SelectKBest.output_schema() reflects selection correctly."""

    def test_keeps_selected_columns(self) -> None:
        """Output schema includes only selected columns."""
        schema = Schema({"a": "DOUBLE", "b": "DOUBLE", "c": "DOUBLE"})
        skb = SelectKBest(k=2)
        skb.columns_ = ["a", "b", "c"]
        skb.params_ = {"a__score": 0.9, "b__score": 0.1, "c__score": 0.7}
        out = skb.output_schema(schema)
        assert "a" in out.columns
        assert "c" in out.columns
        assert "b" not in out.columns

    def test_preserves_non_target_columns(self) -> None:
        """Non-targeted columns (e.g., categoricals) pass through."""
        schema = Schema({"a": "DOUBLE", "b": "DOUBLE", "cat": "VARCHAR"})
        skb = SelectKBest(k=1)
        skb.columns_ = ["a", "b"]
        skb.params_ = {"a__score": 0.9, "b__score": 0.1}
        out = skb.output_schema(schema)
        assert "a" in out.columns
        assert "cat" in out.columns
        assert "b" not in out.columns


# -- Pipeline integration tests ----------------------------------------------


class TestPipeline:
    """Test SelectKBest integrated with Pipeline (end-to-end)."""

    @pytest.fixture
    def backend(self) -> DuckDBBackend:
        """Create DuckDB backend with test data and a target column."""
        conn = duckdb.connect()
        # a is perfectly correlated with target, b is weakly correlated,
        # c is moderately correlated
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 50.0, 2.0, 10.0), (2.0, 10.0, 4.0, 20.0), "
            "(3.0, 40.0, 6.0, 30.0), (4.0, 20.0, 8.0, 40.0), "
            "(5.0, 30.0, 10.0, 50.0) t(a, b, c, target)"
        )
        return DuckDBBackend(connection=conn)

    def test_selects_k_features(self, backend: DuckDBBackend) -> None:
        """Selects exactly k features from output."""
        pipe = Pipeline([SelectKBest(k=2)], backend=backend)
        pipe.fit("t", y="target")
        names = pipe.get_feature_names_out()
        # Should have k=2 numeric features (excluding target)
        numeric_names = [n for n in names if n != "target"]
        assert len(numeric_names) == 2

    def test_shape_correct(self, backend: DuckDBBackend) -> None:
        """Output shape reflects selected features."""
        pipe = Pipeline([SelectKBest(k=1)], backend=backend)
        result = pipe.fit_transform("t", y="target")
        # 1 selected feature (target is excluded by Pipeline)
        assert result.shape == (5, 1)

    def test_to_sql_valid_duckdb(self, backend: DuckDBBackend) -> None:
        """to_sql() output is valid DuckDB SQL."""
        pipe = Pipeline([SelectKBest(k=2)], backend=backend)
        pipe.fit("t", y="target")
        sql = pipe.to_sql()
        sql = sql.replace("__input__", "t")
        conn = backend._connection
        result = conn.execute(sql).fetchall()
        assert len(result) == 5

    def test_fit_then_transform_matches(self, backend: DuckDBBackend) -> None:
        """Separate fit() and transform() produce same result."""
        pipe1 = Pipeline([SelectKBest(k=2)], backend=backend)
        result1 = pipe1.fit_transform("t", y="target")

        pipe2 = Pipeline([SelectKBest(k=2)], backend=backend)
        pipe2.fit("t", y="target")
        result2 = pipe2.transform("t")

        np.testing.assert_array_equal(result1, result2)

    def test_k_greater_than_columns(self, backend: DuckDBBackend) -> None:
        """k > available features keeps all features."""
        pipe = Pipeline([SelectKBest(k=100)], backend=backend)
        pipe.fit("t", y="target")
        names = pipe.get_feature_names_out()
        numeric_names = [n for n in names if n != "target"]
        assert len(numeric_names) == 3  # a, b, c


# -- Composition tests ------------------------------------------------------


class TestComposition:
    """Test SelectKBest composing with other transformers."""

    def test_scaler_then_kbest(self) -> None:
        """StandardScaler + SelectKBest works end-to-end."""
        from sqlearn.scalers.standard import StandardScaler

        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 50.0, 10.0), (2.0, 10.0, 20.0), "
            "(3.0, 40.0, 30.0), (4.0, 20.0, 40.0), "
            "(5.0, 30.0, 50.0) t(a, b, target)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [StandardScaler(), SelectKBest(k=1)],
            backend=backend,
        )
        result = pipe.fit_transform("t", y="target")
        assert result.shape[1] == 1


# -- Clone and pickle tests --------------------------------------------------


class TestClonePickle:
    """Deep clone independence and pickle roundtrip."""

    def test_clone_roundtrip(self) -> None:
        """Cloned SelectKBest has same k and score_func."""
        skb = SelectKBest(k=3, score_func="mutual_info")
        cloned = skb.clone()
        assert cloned.k == 3
        assert cloned.score_func == "mutual_info"
        assert cloned is not skb

    def test_clone_independence(self) -> None:
        """Modifying clone does not affect original."""
        skb = SelectKBest(k=3)
        cloned = skb.clone()
        cloned.set_params(k=5)
        assert skb.k == 3
        assert cloned.k == 5

    def test_clone_preserves_selected(self) -> None:
        """Clone preserves the _selected list."""
        skb = SelectKBest(k=2)
        skb._selected = ["a", "b"]
        skb._fitted = True
        cloned = skb.clone()
        assert cloned._selected == ["a", "b"]
        cloned._selected.append("c")
        assert skb._selected == ["a", "b"]

    def test_pickle_roundtrip(self) -> None:
        """Pickle a SelectKBest preserves k."""
        skb = SelectKBest(k=3)
        data = pickle.dumps(skb)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.k == 3

    def test_pickle_roundtrip_fitted(self) -> None:
        """Pickle a fitted SelectKBest preserves state."""
        skb = SelectKBest(k=2)
        skb._fitted = True
        skb.params_ = {"a__score": 0.9, "b__score": 0.3}
        skb._selected = ["a"]
        data = pickle.dumps(skb)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.params_ == {"a__score": 0.9, "b__score": 0.3}
        assert restored._selected == ["a"]
        assert restored.is_fitted


# -- Not-fitted guard -------------------------------------------------------


class TestNotFittedGuard:
    """Test that unfitted SelectKBest raises appropriate errors."""

    def test_get_feature_names_out_unfitted(self) -> None:
        """get_feature_names_out raises NotFittedError when not fitted."""
        from sqlearn.core.errors import NotFittedError

        skb = SelectKBest(k=2)
        with pytest.raises(NotFittedError):
            skb.get_feature_names_out()


# -- Edge cases --------------------------------------------------------------


class TestEdgeCases:
    """Extreme and unusual inputs."""

    def test_k_equals_one(self) -> None:
        """k=1 selects exactly one feature."""
        skb = SelectKBest(k=1)
        skb.params_ = {"a__score": 0.9, "b__score": 0.5, "c__score": 0.3}
        result = skb._determine_selected(["a", "b", "c"])
        assert result == ["a"]

    def test_all_zero_scores(self) -> None:
        """All zero scores still selects k features."""
        skb = SelectKBest(k=2)
        skb.params_ = {"a__score": 0.0, "b__score": 0.0, "c__score": 0.0}
        result = skb._determine_selected(["a", "b", "c"])
        assert len(result) == 2

    def test_expressions_returns_empty(self) -> None:
        """expressions() returns empty dict (uses query instead)."""
        skb = SelectKBest(k=1)
        result = skb.expressions(["a"], {"a": exp.Column(this="a")})
        assert result == {}

    def test_negative_scores(self) -> None:
        """Negative correlations are ranked by absolute value."""
        skb = SelectKBest(k=1)
        skb.params_ = {"a__score": -0.99, "b__score": 0.5}
        result = skb._determine_selected(["a", "b"])
        assert result == ["a"]

    def test_nan_in_scores(self) -> None:
        """NaN scores are handled without crashing."""
        skb = SelectKBest(k=1)
        skb.params_ = {"a__score": float("nan"), "b__score": 0.5}
        result = skb._determine_selected(["a", "b"])
        # Should select exactly 1 without crashing
        assert len(result) == 1
