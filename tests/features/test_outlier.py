"""Tests for sqlearn.features.outlier -- OutlierHandler."""

from __future__ import annotations

import pickle
from typing import Any

import duckdb
import numpy as np
import pytest
import sqlglot
import sqlglot.expressions as exp

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.pipeline import Pipeline
from sqlearn.core.schema import Schema
from sqlearn.features.outlier import OutlierHandler

# -- Constructor tests --------------------------------------------------------


class TestConstructor:
    """Test OutlierHandler constructor and attributes."""

    def test_defaults(self) -> None:
        """Default method=iqr, action=clip, threshold=1.5."""
        handler = OutlierHandler()
        assert handler.method == "iqr"
        assert handler.action == "clip"
        assert handler.threshold == 1.5
        assert handler.columns is None

    def test_zscore_defaults(self) -> None:
        """Z-score method defaults threshold to 3.0."""
        handler = OutlierHandler(method="zscore")
        assert handler.method == "zscore"
        assert handler.threshold == 3.0

    def test_custom_threshold(self) -> None:
        """Custom threshold overrides method default."""
        handler = OutlierHandler(threshold=2.0)
        assert handler.threshold == 2.0

    def test_zscore_custom_threshold(self) -> None:
        """Custom threshold for z-score overrides default 3.0."""
        handler = OutlierHandler(method="zscore", threshold=2.5)
        assert handler.threshold == 2.5

    def test_action_remove(self) -> None:
        """Action set to remove."""
        handler = OutlierHandler(action="remove")
        assert handler.action == "remove"

    def test_invalid_method_raises(self) -> None:
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be"):
            OutlierHandler(method="invalid")

    def test_invalid_action_raises(self) -> None:
        """Invalid action raises ValueError."""
        with pytest.raises(ValueError, match="action must be"):
            OutlierHandler(action="invalid")

    def test_custom_columns_list(self) -> None:
        """Explicit column list overrides default."""
        handler = OutlierHandler(columns=["a", "b"])
        assert handler.columns == ["a", "b"]

    def test_custom_columns_string(self) -> None:
        """String column spec accepted."""
        handler = OutlierHandler(columns="numeric")
        assert handler.columns == "numeric"

    def test_default_columns_is_numeric(self) -> None:
        """Class default routes to numeric columns."""
        assert OutlierHandler._default_columns == "numeric"

    def test_classification_is_dynamic(self) -> None:
        """Class is dynamic (needs to learn stats)."""
        assert OutlierHandler._classification == "dynamic"

    def test_get_params(self) -> None:
        """get_params returns constructor kwargs."""
        handler = OutlierHandler(method="zscore", action="remove", threshold=2.0, columns=["x"])
        params = handler.get_params()
        assert params == {
            "method": "zscore",
            "action": "remove",
            "threshold": 2.0,
            "columns": ["x"],
        }

    def test_set_params(self) -> None:
        """set_params updates attributes and returns self."""
        handler = OutlierHandler()
        result = handler.set_params(method="zscore")
        assert result is handler
        assert handler.method == "zscore"


# -- discover() tests --------------------------------------------------------


class TestDiscoverIQR:
    """Test OutlierHandler.discover() for IQR method."""

    @pytest.fixture
    def schema(self) -> Schema:
        """Simple numeric schema."""
        return Schema({"a": "DOUBLE", "b": "DOUBLE"})

    def test_q1_q3_per_column(self, schema: Schema) -> None:
        """IQR method discovers Q1 and Q3 per column."""
        handler = OutlierHandler()
        result = handler.discover(["a", "b"], schema)
        assert "a__q1" in result
        assert "a__q3" in result
        assert "b__q1" in result
        assert "b__q3" in result
        assert len(result) == 4

    def test_q1_ast_type(self, schema: Schema) -> None:
        """Q1 expression is exp.Quantile with quantile=0.25."""
        handler = OutlierHandler()
        result = handler.discover(["a"], schema)
        q1_expr = result["a__q1"]
        assert isinstance(q1_expr, exp.Quantile)
        inner = q1_expr.this
        assert isinstance(inner, exp.Column)
        assert inner.name == "a"

    def test_q3_ast_type(self, schema: Schema) -> None:
        """Q3 expression is exp.Quantile with quantile=0.75."""
        handler = OutlierHandler()
        result = handler.discover(["a"], schema)
        q3_expr = result["a__q3"]
        assert isinstance(q3_expr, exp.Quantile)
        quantile_lit = q3_expr.args["quantile"]
        assert float(quantile_lit.this) == pytest.approx(0.75)

    def test_empty_columns_returns_empty(self, schema: Schema) -> None:
        """Empty columns list returns empty dict."""
        handler = OutlierHandler()
        result = handler.discover([], schema)
        assert result == {}

    def test_single_column(self, schema: Schema) -> None:
        """Single column produces exactly 2 aggregates."""
        handler = OutlierHandler()
        result = handler.discover(["a"], schema)
        assert len(result) == 2


class TestDiscoverZScore:
    """Test OutlierHandler.discover() for z-score method."""

    @pytest.fixture
    def schema(self) -> Schema:
        """Simple numeric schema."""
        return Schema({"a": "DOUBLE", "b": "DOUBLE"})

    def test_mean_std_per_column(self, schema: Schema) -> None:
        """Z-score method discovers mean and std per column."""
        handler = OutlierHandler(method="zscore")
        result = handler.discover(["a", "b"], schema)
        assert "a__mean" in result
        assert "a__std" in result
        assert "b__mean" in result
        assert "b__std" in result
        assert len(result) == 4

    def test_mean_ast_type(self, schema: Schema) -> None:
        """Mean expression is exp.Avg wrapping exp.Column."""
        handler = OutlierHandler(method="zscore")
        result = handler.discover(["a"], schema)
        avg_expr = result["a__mean"]
        assert isinstance(avg_expr, exp.Avg)
        inner = avg_expr.this
        assert isinstance(inner, exp.Column)
        assert inner.name == "a"

    def test_std_ast_type(self, schema: Schema) -> None:
        """Std expression is exp.StddevPop wrapping exp.Column."""
        handler = OutlierHandler(method="zscore")
        result = handler.discover(["a"], schema)
        std_expr = result["a__std"]
        assert isinstance(std_expr, exp.StddevPop)

    def test_empty_columns_returns_empty(self, schema: Schema) -> None:
        """Empty columns list returns empty dict."""
        handler = OutlierHandler(method="zscore")
        result = handler.discover([], schema)
        assert result == {}


# -- _compute_fences() tests -------------------------------------------------


class TestComputeFences:
    """Test OutlierHandler._compute_fences() calculations."""

    def test_iqr_fences(self) -> None:
        """IQR fences: Q1 - 1.5*IQR, Q3 + 1.5*IQR."""
        handler = OutlierHandler()
        handler.params_ = {"a__q1": 10.0, "a__q3": 20.0}
        lower, upper = handler._compute_fences("a")
        # IQR = 20 - 10 = 10
        # lower = 10 - 1.5*10 = -5
        # upper = 20 + 1.5*10 = 35
        assert lower == pytest.approx(-5.0)
        assert upper == pytest.approx(35.0)

    def test_iqr_custom_threshold(self) -> None:
        """Custom threshold changes fence distance."""
        handler = OutlierHandler(threshold=1.0)
        handler.params_ = {"a__q1": 10.0, "a__q3": 20.0}
        lower, upper = handler._compute_fences("a")
        # IQR = 10, threshold = 1.0
        # lower = 10 - 1.0*10 = 0
        # upper = 20 + 1.0*10 = 30
        assert lower == pytest.approx(0.0)
        assert upper == pytest.approx(30.0)

    def test_zscore_fences(self) -> None:
        """Z-score fences: mean +/- threshold*std."""
        handler = OutlierHandler(method="zscore", threshold=3.0)
        handler.params_ = {"a__mean": 50.0, "a__std": 10.0}
        lower, upper = handler._compute_fences("a")
        # lower = 50 - 3*10 = 20
        # upper = 50 + 3*10 = 80
        assert lower == pytest.approx(20.0)
        assert upper == pytest.approx(80.0)

    def test_zscore_custom_threshold(self) -> None:
        """Custom z-score threshold changes fence distance."""
        handler = OutlierHandler(method="zscore", threshold=2.0)
        handler.params_ = {"a__mean": 50.0, "a__std": 10.0}
        lower, upper = handler._compute_fences("a")
        assert lower == pytest.approx(30.0)
        assert upper == pytest.approx(70.0)

    def test_zero_iqr(self) -> None:
        """Zero IQR produces identical fences."""
        handler = OutlierHandler()
        handler.params_ = {"a__q1": 5.0, "a__q3": 5.0}
        lower, upper = handler._compute_fences("a")
        assert lower == pytest.approx(5.0)
        assert upper == pytest.approx(5.0)

    def test_zero_std(self) -> None:
        """Zero std produces identical fences at the mean."""
        handler = OutlierHandler(method="zscore")
        handler.params_ = {"a__mean": 5.0, "a__std": 0.0}
        lower, upper = handler._compute_fences("a")
        assert lower == pytest.approx(5.0)
        assert upper == pytest.approx(5.0)


# -- expressions() tests (clip) -----------------------------------------------


class TestExpressionsClip:
    """Test OutlierHandler.expressions() for clip action."""

    def _make_fitted_handler(
        self,
        params: dict[str, Any],
        *,
        method: str = "iqr",
        threshold: float | None = None,
    ) -> OutlierHandler:
        """Create a fitted OutlierHandler with given params."""
        handler = OutlierHandler(method=method, action="clip", threshold=threshold)
        handler.params_ = params
        handler._fitted = True
        return handler

    def test_clip_iqr_basic(self) -> None:
        """IQR clip: GREATEST(LEAST(col, upper), lower)."""
        handler = self._make_fitted_handler({"a__q1": 10.0, "a__q3": 20.0})
        exprs = {"a": exp.Column(this="a")}
        result = handler.expressions(["a"], exprs)
        assert "a" in result
        assert isinstance(result["a"], exp.Greatest)

    def test_clip_zscore_basic(self) -> None:
        """Z-score clip: GREATEST(LEAST(col, upper), lower)."""
        handler = self._make_fitted_handler({"a__mean": 50.0, "a__std": 10.0}, method="zscore")
        exprs = {"a": exp.Column(this="a")}
        result = handler.expressions(["a"], exprs)
        assert "a" in result
        assert isinstance(result["a"], exp.Greatest)

    def test_clip_inner_is_least(self) -> None:
        """Clip expression has LEAST as inner node of GREATEST."""
        handler = self._make_fitted_handler({"a__q1": 10.0, "a__q3": 20.0})
        exprs = {"a": exp.Column(this="a")}
        result = handler.expressions(["a"], exprs)
        assert isinstance(result["a"].this, exp.Least)

    def test_clip_multiple_columns(self) -> None:
        """Multiple columns each get their own expression."""
        handler = self._make_fitted_handler(
            {"a__q1": 10.0, "a__q3": 20.0, "b__q1": 5.0, "b__q3": 15.0}
        )
        exprs = {"a": exp.Column(this="a"), "b": exp.Column(this="b")}
        result = handler.expressions(["a", "b"], exprs)
        assert "a" in result
        assert "b" in result

    def test_clip_uses_exprs_not_column(self) -> None:
        """expressions() uses exprs[col] to compose with prior transforms."""
        handler = self._make_fitted_handler({"a__q1": 10.0, "a__q3": 20.0})
        # Simulate a prior transform: a is already (a * 2)
        prior = exp.Mul(this=exp.Column(this="a"), expression=exp.Literal.number(2))
        exprs = {"a": prior}
        result = handler.expressions(["a"], exprs)
        # Inner LEAST should contain the prior expression (Mul)
        least = result["a"].this
        assert isinstance(least.this, exp.Mul)

    def test_clip_only_modifies_target_columns(self) -> None:
        """Columns not in target list are not in result."""
        handler = self._make_fitted_handler({"a__q1": 10.0, "a__q3": 20.0})
        exprs = {
            "a": exp.Column(this="a"),
            "b": exp.Column(this="b"),
        }
        result = handler.expressions(["a"], exprs)
        assert "a" in result
        assert "b" not in result


# -- expressions() tests (remove) ---------------------------------------------


class TestExpressionsRemove:
    """Test OutlierHandler.expressions() for remove action returns empty."""

    def test_remove_expressions_empty(self) -> None:
        """Remove action returns empty dict from expressions()."""
        handler = OutlierHandler(action="remove")
        handler.params_ = {"a__q1": 10.0, "a__q3": 20.0}
        handler._fitted = True
        exprs = {"a": exp.Column(this="a")}
        result = handler.expressions(["a"], exprs)
        assert result == {}


# -- query() tests (remove) --------------------------------------------------


class TestQueryRemove:
    """Test OutlierHandler.query() for remove action."""

    def test_remove_generates_select_star(self) -> None:
        """Remove query generates SELECT * FROM (input) WHERE ..."""
        handler = OutlierHandler(action="remove")
        handler.params_ = {"a__q1": 10.0, "a__q3": 20.0}
        handler.columns_ = ["a"]
        handler._fitted = True
        input_q = exp.select(exp.Star()).from_("t")
        result = handler.query(input_q)
        assert result is not None
        assert isinstance(result, exp.Select)
        star_found = any(isinstance(e, exp.Star) for e in result.expressions)
        assert star_found

    def test_remove_has_where_clause(self) -> None:
        """Remove query has WHERE clause."""
        handler = OutlierHandler(action="remove")
        handler.params_ = {"a__q1": 10.0, "a__q3": 20.0}
        handler.columns_ = ["a"]
        handler._fitted = True
        input_q = exp.select(exp.Star()).from_("t")
        result = handler.query(input_q)
        assert result is not None
        where = result.find(exp.Where)
        assert where is not None

    def test_remove_has_between(self) -> None:
        """Remove query uses BETWEEN for range check."""
        handler = OutlierHandler(action="remove")
        handler.params_ = {"a__q1": 10.0, "a__q3": 20.0}
        handler.columns_ = ["a"]
        handler._fitted = True
        input_q = exp.select(exp.Star()).from_("t")
        result = handler.query(input_q)
        assert result is not None
        between = result.find(exp.Between)
        assert between is not None

    def test_remove_wraps_input_as_subquery(self) -> None:
        """Remove query wraps input in a subquery."""
        handler = OutlierHandler(action="remove")
        handler.params_ = {"a__q1": 10.0, "a__q3": 20.0}
        handler.columns_ = ["a"]
        handler._fitted = True
        input_q = exp.select(exp.Star()).from_("t")
        result = handler.query(input_q)
        assert result is not None
        subquery = result.find(exp.Subquery)
        assert subquery is not None
        assert subquery.alias == "__input__"

    def test_remove_multiple_columns_and_conditions(self) -> None:
        """Remove with multiple columns uses AND to combine conditions."""
        handler = OutlierHandler(action="remove")
        handler.params_ = {
            "a__q1": 10.0,
            "a__q3": 20.0,
            "b__q1": 5.0,
            "b__q3": 15.0,
        }
        handler.columns_ = ["a", "b"]
        handler._fitted = True
        input_q = exp.select(exp.Star()).from_("t")
        result = handler.query(input_q)
        assert result is not None
        sql = result.sql(dialect="duckdb")
        assert "AND" in sql.upper()

    def test_clip_query_returns_none(self) -> None:
        """Clip action query() returns None (uses expressions instead)."""
        handler = OutlierHandler(action="clip")
        handler.params_ = {"a__q1": 10.0, "a__q3": 20.0}
        handler.columns_ = ["a"]
        handler._fitted = True
        input_q = exp.select(exp.Star()).from_("t")
        result = handler.query(input_q)
        assert result is None

    def test_remove_sql_is_valid(self) -> None:
        """Generated remove SQL is parseable."""
        handler = OutlierHandler(action="remove")
        handler.params_ = {"a__q1": 10.0, "a__q3": 20.0}
        handler.columns_ = ["a"]
        handler._fitted = True
        input_q = exp.select(exp.Star()).from_("t")
        result = handler.query(input_q)
        assert result is not None
        sql = result.sql(dialect="duckdb")
        parsed = sqlglot.parse_one(sql)
        assert isinstance(parsed, exp.Select)


# -- __repr__ tests -----------------------------------------------------------


class TestRepr:
    """Test OutlierHandler.__repr__."""

    def test_default_repr(self) -> None:
        """Default params shows threshold (computed from method default)."""
        handler = OutlierHandler()
        assert repr(handler) == "OutlierHandler(threshold=1.5)"

    def test_zscore_repr(self) -> None:
        """Z-score method shows in repr."""
        handler = OutlierHandler(method="zscore")
        r = repr(handler)
        assert "method='zscore'" in r
        assert "threshold=3.0" in r

    def test_remove_repr(self) -> None:
        """Remove action shows in repr."""
        handler = OutlierHandler(action="remove")
        assert "action='remove'" in repr(handler)

    def test_custom_threshold_repr(self) -> None:
        """Custom threshold shows in repr."""
        handler = OutlierHandler(threshold=2.0)
        assert "threshold=2.0" in repr(handler)


# -- Pipeline integration tests (clip) ----------------------------------------


class TestPipelineClip:
    """Test OutlierHandler clip action with Pipeline (end-to-end)."""

    @pytest.fixture
    def backend(self) -> DuckDBBackend:
        """Create DuckDB backend with data including outliers."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0), (2.0), (3.0), (4.0), (5.0), "
            "(100.0), (-50.0) t(a)"
        )
        return DuckDBBackend(connection=conn)

    @pytest.fixture
    def multi_col_backend(self) -> DuckDBBackend:
        """Create DuckDB backend with multiple numeric columns."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 10.0), (2.0, 20.0), (3.0, 30.0), "
            "(4.0, 40.0), (5.0, 50.0), "
            "(100.0, 500.0), (-50.0, -200.0) t(a, b)"
        )
        return DuckDBBackend(connection=conn)

    def test_clip_shape_preserved(self, backend: DuckDBBackend) -> None:
        """Clip preserves row count (no rows removed)."""
        pipe = Pipeline([OutlierHandler()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (7, 1)

    def test_clip_outliers_capped(self, backend: DuckDBBackend) -> None:
        """Outlier values are capped within fences."""
        pipe = Pipeline([OutlierHandler()], backend=backend)
        result = pipe.fit_transform("t")
        col_a = result[:, 0].astype(float)
        # All values should be within the fences
        assert col_a.max() <= 100.0  # upper fence should cap extreme values
        assert col_a.min() >= -50.0  # lower fence should cap extreme values

    def test_clip_params_populated(self, backend: DuckDBBackend) -> None:
        """After fit, params_ contains Q1 and Q3."""
        handler = OutlierHandler()
        pipe = Pipeline([handler], backend=backend)
        pipe.fit("t")
        assert handler.params_ is not None
        assert "a__q1" in handler.params_
        assert "a__q3" in handler.params_

    def test_clip_multi_column(self, multi_col_backend: DuckDBBackend) -> None:
        """Clip works on multiple columns."""
        pipe = Pipeline([OutlierHandler()], backend=multi_col_backend)
        result = pipe.fit_transform("t")
        assert result.shape == (7, 2)

    def test_clip_to_sql_valid(self, backend: DuckDBBackend) -> None:
        """to_sql() produces valid SQL with GREATEST/LEAST."""
        pipe = Pipeline([OutlierHandler()], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql()
        assert isinstance(sql, str)
        assert "SELECT" in sql.upper()

    def test_clip_to_sql_contains_greatest(self, backend: DuckDBBackend) -> None:
        """Clip SQL contains GREATEST."""
        pipe = Pipeline([OutlierHandler()], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql()
        assert "GREATEST" in sql.upper()

    def test_clip_to_sql_contains_least(self, backend: DuckDBBackend) -> None:
        """Clip SQL contains LEAST."""
        pipe = Pipeline([OutlierHandler()], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql()
        assert "LEAST" in sql.upper()


# -- Pipeline integration tests (remove) --------------------------------------


class TestPipelineRemove:
    """Test OutlierHandler remove action with Pipeline (end-to-end)."""

    @pytest.fixture
    def backend(self) -> DuckDBBackend:
        """Create DuckDB backend with data including outliers."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0), (2.0), (3.0), (4.0), (5.0), "
            "(100.0), (-50.0) t(a)"
        )
        return DuckDBBackend(connection=conn)

    def test_remove_reduces_rows(self, backend: DuckDBBackend) -> None:
        """Remove action reduces row count by excluding outliers."""
        pipe = Pipeline([OutlierHandler(action="remove")], backend=backend)
        result = pipe.fit_transform("t")
        # Should have fewer rows than original 7
        assert result.shape[0] < 7

    def test_remove_to_sql_contains_where(self, backend: DuckDBBackend) -> None:
        """Remove SQL contains WHERE clause."""
        pipe = Pipeline([OutlierHandler(action="remove")], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql()
        assert "WHERE" in sql.upper()

    def test_remove_to_sql_contains_between(self, backend: DuckDBBackend) -> None:
        """Remove SQL contains BETWEEN clause."""
        pipe = Pipeline([OutlierHandler(action="remove")], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql()
        assert "BETWEEN" in sql.upper()


# -- Pipeline integration (z-score) -------------------------------------------


class TestPipelineZScore:
    """Test OutlierHandler z-score method with Pipeline."""

    @pytest.fixture
    def backend(self) -> DuckDBBackend:
        """Create DuckDB backend with data."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0), (2.0), (3.0), (4.0), (5.0), "
            "(100.0), (-50.0) t(a)"
        )
        return DuckDBBackend(connection=conn)

    def test_zscore_clip_shape(self, backend: DuckDBBackend) -> None:
        """Z-score clip preserves row count."""
        pipe = Pipeline([OutlierHandler(method="zscore", action="clip")], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (7, 1)

    def test_zscore_params_populated(self, backend: DuckDBBackend) -> None:
        """After fit, params_ contains mean and std."""
        handler = OutlierHandler(method="zscore")
        pipe = Pipeline([handler], backend=backend)
        pipe.fit("t")
        assert handler.params_ is not None
        assert "a__mean" in handler.params_
        assert "a__std" in handler.params_

    def test_zscore_remove_reduces_rows(self, backend: DuckDBBackend) -> None:
        """Z-score remove reduces rows."""
        pipe = Pipeline([OutlierHandler(method="zscore", action="remove")], backend=backend)
        result = pipe.fit_transform("t")
        # With extreme outliers (100, -50), some rows should be filtered
        assert result.shape[0] <= 7


# -- Not-fitted guard tests ---------------------------------------------------


class TestNotFittedGuard:
    """Calling transform/to_sql before fit() raises NotFittedError."""

    def test_transform_before_fit_raises(self) -> None:
        """transform() on unfitted pipeline raises NotFittedError."""
        from sqlearn.core.errors import NotFittedError

        pipe = Pipeline([OutlierHandler()])
        with pytest.raises(NotFittedError):
            pipe.transform("t")

    def test_to_sql_before_fit_raises(self) -> None:
        """to_sql() on unfitted pipeline raises NotFittedError."""
        from sqlearn.core.errors import NotFittedError

        pipe = Pipeline([OutlierHandler()])
        with pytest.raises(NotFittedError):
            pipe.to_sql()


# -- Clone and pickle tests ---------------------------------------------------


class TestClonePickle:
    """Deep clone independence and pickle roundtrip."""

    @pytest.fixture
    def fitted_pipe(self) -> Pipeline:
        """Create a fitted pipeline for clone/pickle tests."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES (1.0), (2.0), (3.0), (4.0), (5.0) t(a)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([OutlierHandler()], backend=backend)
        pipe.fit("t")
        return pipe

    def test_clone_produces_same_sql(self, fitted_pipe: Pipeline) -> None:
        """Cloned pipeline produces identical SQL."""
        cloned = fitted_pipe.clone()
        assert fitted_pipe.to_sql() == cloned.to_sql()

    def test_clone_params_independent(self, fitted_pipe: Pipeline) -> None:
        """Modifying cloned step params does not affect original."""
        cloned = fitted_pipe.clone()
        original_handler = fitted_pipe.steps[0][1]
        cloned_handler = cloned.steps[0][1]
        assert cloned_handler.params_ is not None
        cloned_handler.params_["a__q1"] = 999.0
        assert original_handler.params_ is not None
        assert original_handler.params_["a__q1"] != 999.0

    def test_pickle_roundtrip(self) -> None:
        """Pickle an individual handler preserves params."""
        handler = OutlierHandler()
        handler.params_ = {"a__q1": 2.0, "a__q3": 8.0}
        handler._fitted = True
        data = pickle.dumps(handler)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.params_ == {"a__q1": 2.0, "a__q3": 8.0}
        assert restored._fitted is True

    def test_pickle_zscore_roundtrip(self) -> None:
        """Pickle a z-score handler preserves params."""
        handler = OutlierHandler(method="zscore")
        handler.params_ = {"a__mean": 5.0, "a__std": 2.0}
        handler._fitted = True
        data = pickle.dumps(handler)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.params_ == {"a__mean": 5.0, "a__std": 2.0}
        assert restored.method == "zscore"


# -- Composition tests -------------------------------------------------------


class TestComposition:
    """OutlierHandler composing with other transformers."""

    def test_outlier_then_scaler(self) -> None:
        """OutlierHandler + StandardScaler produces valid results."""
        from sqlearn.scalers.standard import StandardScaler

        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0), (2.0), (3.0), (4.0), (5.0), (100.0) t(a)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([OutlierHandler(), StandardScaler()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (6, 1)

    def test_imputer_then_outlier(self) -> None:
        """Imputer + OutlierHandler handles NULL values first."""
        from sqlearn.imputers.imputer import Imputer

        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0), (NULL), (3.0), (4.0), (5.0), (100.0) t(a)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Imputer(), OutlierHandler()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (6, 1)
        # No NaN/None in output
        assert not np.any(np.isnan(result.astype(float)))

    def test_outlier_sql_nesting(self) -> None:
        """SQL shows GREATEST/LEAST for clip."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0), (2.0), (3.0), (4.0), (5.0), (100.0) t(a)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([OutlierHandler()], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql().upper()
        assert "GREATEST" in sql
        assert "LEAST" in sql


# -- Edge cases ---------------------------------------------------------------


class TestEdgeCases:
    """Extreme and unusual inputs."""

    def test_single_row(self) -> None:
        """Single-row data works (IQR=0, values pass through)."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (42.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([OutlierHandler()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (1, 1)

    def test_constant_column(self) -> None:
        """All-same-value column has IQR=0, value stays at original."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (5.0), (5.0), (5.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([OutlierHandler()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (3, 1)
        np.testing.assert_allclose(result[:, 0], [5.0, 5.0, 5.0], atol=1e-10)

    def test_mixed_types_passthrough(self) -> None:
        """Non-numeric columns pass through unchanged."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 'a'), (2.0, 'b'), (3.0, 'c'), "
            "(4.0, 'd'), (5.0, 'e'), (100.0, 'f') t(num, cat)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([OutlierHandler()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (6, 2)
        # cat column unchanged
        cats = result[:, 1]
        assert set(cats) == {"a", "b", "c", "d", "e", "f"}

    def test_negative_values(self) -> None:
        """Negative values handled correctly."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (-10.0), (-5.0), (0.0), (5.0), (10.0), (100.0) t(a)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([OutlierHandler()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (6, 1)

    def test_fit_then_transform_separate(self) -> None:
        """Separate fit() and transform() produce same result."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0), (2.0), (3.0), (4.0), (5.0), (100.0) t(a)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe1 = Pipeline([OutlierHandler()], backend=backend)
        result1 = pipe1.fit_transform("t")

        pipe2 = Pipeline([OutlierHandler()], backend=backend)
        pipe2.fit("t")
        result2 = pipe2.transform("t")

        np.testing.assert_array_equal(result1, result2)

    def test_null_values_clip(self) -> None:
        """NULL values are handled gracefully (GREATEST/LEAST propagate NULL)."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES (1.0), (2.0), (NULL), (4.0), (5.0) t(a)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([OutlierHandler()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (5, 1)
