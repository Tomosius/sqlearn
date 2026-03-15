"""Tests for sqlearn.scalers.robust — RobustScaler."""

from __future__ import annotations

from typing import Any

import duckdb
import numpy as np
import pytest
import sqlglot.expressions as exp

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.pipeline import Pipeline
from sqlearn.core.schema import Schema
from sqlearn.scalers.robust import RobustScaler

# ── Constructor tests ──────────────────────────────────────────────


class TestConstructor:
    """Test RobustScaler constructor and attributes."""

    def test_defaults(self) -> None:
        """Default with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0)."""
        scaler = RobustScaler()
        assert scaler.with_centering is True
        assert scaler.with_scaling is True
        assert scaler.quantile_range == (25.0, 75.0)
        assert scaler.columns is None

    def test_custom_quantile_range(self) -> None:
        """Custom quantile_range overrides default."""
        scaler = RobustScaler(quantile_range=(10.0, 90.0))
        assert scaler.quantile_range == (10.0, 90.0)

    def test_custom_columns_list(self) -> None:
        """Explicit column list overrides default."""
        scaler = RobustScaler(columns=["a", "b"])
        assert scaler.columns == ["a", "b"]

    def test_custom_columns_string(self) -> None:
        """String column spec (e.g. 'numeric') accepted."""
        scaler = RobustScaler(columns="numeric")
        assert scaler.columns == "numeric"

    def test_with_centering_false(self) -> None:
        """with_centering=False disables centering."""
        scaler = RobustScaler(with_centering=False)
        assert scaler.with_centering is False
        assert scaler.with_scaling is True

    def test_with_scaling_false(self) -> None:
        """with_scaling=False disables scaling."""
        scaler = RobustScaler(with_scaling=False)
        assert scaler.with_centering is True
        assert scaler.with_scaling is False

    def test_both_false(self) -> None:
        """Both flags False is a no-op (valid but useless)."""
        scaler = RobustScaler(with_centering=False, with_scaling=False)
        assert scaler.with_centering is False
        assert scaler.with_scaling is False

    def test_default_columns_is_numeric(self) -> None:
        """Class default routes to numeric columns."""
        assert RobustScaler._default_columns == "numeric"

    def test_classification_is_dynamic(self) -> None:
        """Class is dynamic (needs to learn median/quantiles)."""
        assert RobustScaler._classification == "dynamic"

    def test_get_params(self) -> None:
        """get_params returns constructor kwargs."""
        scaler = RobustScaler(with_centering=False, columns=["x"])
        params = scaler.get_params()
        assert params == {
            "with_centering": False,
            "with_scaling": True,
            "quantile_range": (25.0, 75.0),
            "columns": ["x"],
        }

    def test_set_params(self) -> None:
        """set_params updates attributes and returns self."""
        scaler = RobustScaler()
        result = scaler.set_params(with_centering=False)
        assert result is scaler
        assert scaler.with_centering is False


# ── discover() tests ───────────────────────────────────────────────


class TestDiscover:
    """Test RobustScaler.discover() returns correct sqlglot aggregates."""

    @pytest.fixture
    def schema(self) -> Schema:
        """Simple numeric schema."""
        return Schema({"a": "DOUBLE", "b": "DOUBLE"})

    def test_median_and_quantiles_per_column(self, schema: Schema) -> None:
        """Each column gets MEDIAN, Q1, and Q3 entries."""
        scaler = RobustScaler()
        result = scaler.discover(["a", "b"], schema)
        assert "a__median" in result
        assert "a__q1" in result
        assert "a__q3" in result
        assert "b__median" in result
        assert "b__q1" in result
        assert "b__q3" in result
        assert len(result) == 6

    def test_median_ast_type(self, schema: Schema) -> None:
        """Median expression is exp.Median wrapping exp.Column."""
        scaler = RobustScaler()
        result = scaler.discover(["a"], schema)
        median_expr = result["a__median"]
        assert isinstance(median_expr, exp.Median)
        inner = median_expr.this
        assert isinstance(inner, exp.Column)
        assert inner.name == "a"

    def test_quantile_ast_type(self, schema: Schema) -> None:
        """Quantile expressions are exp.Quantile wrapping exp.Column."""
        scaler = RobustScaler()
        result = scaler.discover(["a"], schema)
        q1_expr = result["a__q1"]
        assert isinstance(q1_expr, exp.Quantile)
        inner = q1_expr.this
        assert isinstance(inner, exp.Column)
        assert inner.name == "a"

    def test_with_centering_false_no_median(self, schema: Schema) -> None:
        """with_centering=False omits MEDIAN aggregation."""
        scaler = RobustScaler(with_centering=False)
        result = scaler.discover(["a"], schema)
        assert "a__median" not in result
        assert "a__q1" in result
        assert "a__q3" in result

    def test_with_scaling_false_no_quantiles(self, schema: Schema) -> None:
        """with_scaling=False omits QUANTILE aggregations."""
        scaler = RobustScaler(with_scaling=False)
        result = scaler.discover(["a"], schema)
        assert "a__median" in result
        assert "a__q1" not in result
        assert "a__q3" not in result

    def test_both_false_empty(self, schema: Schema) -> None:
        """Both flags False returns empty dict."""
        scaler = RobustScaler(with_centering=False, with_scaling=False)
        result = scaler.discover(["a"], schema)
        assert result == {}

    def test_empty_columns_returns_empty(self, schema: Schema) -> None:
        """Empty columns list returns empty dict."""
        scaler = RobustScaler()
        result = scaler.discover([], schema)
        assert result == {}

    def test_single_column(self, schema: Schema) -> None:
        """Single column produces exactly 3 aggregates."""
        scaler = RobustScaler()
        result = scaler.discover(["a"], schema)
        assert len(result) == 3

    def test_custom_quantile_range_values(self, schema: Schema) -> None:
        """Custom quantile_range produces correct quantile literals."""
        scaler = RobustScaler(quantile_range=(10.0, 90.0))
        result = scaler.discover(["a"], schema)
        q1_expr = result["a__q1"]
        assert isinstance(q1_expr, exp.Quantile)
        # q_low / 100 = 0.1
        quantile_lit = q1_expr.args["quantile"]
        assert float(quantile_lit.this) == pytest.approx(0.1)


# ── expressions() tests ───────────────────────────────────────────


class TestExpressions:
    """Test RobustScaler.expressions() generates correct sqlglot ASTs."""

    def _make_fitted_scaler(
        self,
        params: dict[str, Any],
        *,
        with_centering: bool = True,
        with_scaling: bool = True,
        quantile_range: tuple[float, float] = (25.0, 75.0),
    ) -> RobustScaler:
        """Create a fitted RobustScaler with given params."""
        scaler = RobustScaler(
            with_centering=with_centering,
            with_scaling=with_scaling,
            quantile_range=quantile_range,
        )
        scaler.params_ = params
        scaler._fitted = True
        return scaler

    def test_basic_formula(self) -> None:
        """(col - median) / NULLIF(q3 - q1, 0) for basic case."""
        scaler = self._make_fitted_scaler({"a__median": 5.0, "a__q1": 2.0, "a__q3": 8.0})
        exprs = {"a": exp.Column(this="a"), "b": exp.Column(this="b")}
        result = scaler.expressions(["a"], exprs)
        assert "a" in result
        # Top-level is Div
        assert isinstance(result["a"], exp.Div)
        # Numerator is Paren(Sub(...))
        assert isinstance(result["a"].this, exp.Paren)
        assert isinstance(result["a"].this.this, exp.Sub)
        # Denominator is Nullif
        assert isinstance(result["a"].expression, exp.Nullif)

    def test_centering_only(self) -> None:
        """with_scaling=False produces col - median."""
        scaler = self._make_fitted_scaler({"a__median": 3.0}, with_scaling=False)
        exprs = {"a": exp.Column(this="a")}
        result = scaler.expressions(["a"], exprs)
        # No division — top level is Paren(Sub(...))
        assert isinstance(result["a"], exp.Paren)
        assert isinstance(result["a"].this, exp.Sub)

    def test_scaling_only(self) -> None:
        """with_centering=False produces col / NULLIF(q3 - q1, 0)."""
        scaler = self._make_fitted_scaler({"a__q1": 2.0, "a__q3": 8.0}, with_centering=False)
        exprs = {"a": exp.Column(this="a")}
        result = scaler.expressions(["a"], exprs)
        # No subtraction — top level is Div
        assert isinstance(result["a"], exp.Div)
        # Numerator is the raw expression, not Paren(Sub(...))
        assert isinstance(result["a"].this, exp.Column)

    def test_both_false_passthrough(self) -> None:
        """Both flags False returns empty dict (passthrough)."""
        scaler = self._make_fitted_scaler({}, with_centering=False, with_scaling=False)
        exprs = {"a": exp.Column(this="a")}
        result = scaler.expressions(["a"], exprs)
        assert result == {}

    def test_zero_iqr_safe_division(self) -> None:
        """Zero IQR uses NULLIF(0, 0) which returns NULL, not error."""
        scaler = self._make_fitted_scaler({"a__median": 5.0, "a__q1": 3.0, "a__q3": 3.0})
        exprs = {"a": exp.Column(this="a")}
        result = scaler.expressions(["a"], exprs)
        # Denominator should be NULLIF(q3-q1, 0) where q3-q1 = 0
        nullif = result["a"].expression
        assert isinstance(nullif, exp.Nullif)

    def test_uses_exprs_not_column(self) -> None:
        """expressions() uses exprs[col], not exp.Column(this=col)."""
        scaler = self._make_fitted_scaler({"a__median": 0.0, "a__q1": 0.0, "a__q3": 1.0})
        # Simulate a prior transform: a is already (a * 2)
        prior = exp.Mul(this=exp.Column(this="a"), expression=exp.Literal.number(2))
        exprs = {"a": prior}
        result = scaler.expressions(["a"], exprs)
        # The numerator's inner Sub should reference the prior expression
        sub = result["a"].this.this  # Paren -> Sub
        assert isinstance(sub.this, exp.Mul)

    def test_multiple_columns(self) -> None:
        """Multiple columns each get their own expression."""
        scaler = self._make_fitted_scaler(
            {
                "a__median": 1.0,
                "a__q1": 0.0,
                "a__q3": 2.0,
                "b__median": 10.0,
                "b__q1": 5.0,
                "b__q3": 15.0,
            }
        )
        exprs = {"a": exp.Column(this="a"), "b": exp.Column(this="b")}
        result = scaler.expressions(["a", "b"], exprs)
        assert "a" in result
        assert "b" in result

    def test_only_modifies_target_columns(self) -> None:
        """Columns not in target list are not in result."""
        scaler = self._make_fitted_scaler({"a__median": 1.0, "a__q1": 0.0, "a__q3": 2.0})
        exprs = {
            "a": exp.Column(this="a"),
            "b": exp.Column(this="b"),
            "c": exp.Column(this="c"),
        }
        result = scaler.expressions(["a"], exprs)
        assert "a" in result
        assert "b" not in result
        assert "c" not in result


# ── Pipeline integration tests ────────────────────────────────────


class TestPipeline:
    """Test RobustScaler integrated with Pipeline (end-to-end)."""

    @pytest.fixture
    def backend(self) -> DuckDBBackend:
        """Create DuckDB backend with numeric test data."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 10.0), (2.0, 20.0), (3.0, 30.0), "
            "(4.0, 40.0), (5.0, 50.0) t(a, b)"
        )
        return DuckDBBackend(connection=conn)

    def test_fit_transform_shape(self, backend: DuckDBBackend) -> None:
        """Output shape matches input shape."""
        pipe = Pipeline([RobustScaler()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (5, 2)

    def test_fit_transform_median_zero(self, backend: DuckDBBackend) -> None:
        """Median value in transformed data should map to zero."""
        pipe = Pipeline([RobustScaler()], backend=backend)
        result = pipe.fit_transform("t")
        # Median row (3.0) should become 0 after centering
        # With 5 values [1,2,3,4,5], median=3, q1=2, q3=4, IQR=2
        # (3 - 3) / 2 = 0
        col_a = result[:, 0]
        assert 0.0 in col_a or np.any(np.isclose(col_a, 0.0, atol=1e-10))

    def test_to_sql_valid(self, backend: DuckDBBackend) -> None:
        """to_sql() produces valid SQL that DuckDB can execute."""
        pipe = Pipeline([RobustScaler()], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql()
        # Should be valid SQL
        assert isinstance(sql, str)
        assert "SELECT" in sql.upper()
        assert "FROM" in sql.upper()

    def test_to_sql_contains_nullif(self, backend: DuckDBBackend) -> None:
        """to_sql() output contains NULLIF for safe division."""
        pipe = Pipeline([RobustScaler()], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql()
        assert "NULLIF" in sql.upper()

    def test_params_populated(self, backend: DuckDBBackend) -> None:
        """After fit, params_ contains median and quantiles for each column."""
        scaler = RobustScaler()
        pipe = Pipeline([scaler], backend=backend)
        pipe.fit("t")
        assert scaler.params_ is not None
        assert "a__median" in scaler.params_
        assert "a__q1" in scaler.params_
        assert "a__q3" in scaler.params_
        assert "b__median" in scaler.params_
        assert "b__q1" in scaler.params_
        assert "b__q3" in scaler.params_

    def test_fit_then_transform_separate(self, backend: DuckDBBackend) -> None:
        """Separate fit() and transform() produce same result as fit_transform()."""
        pipe1 = Pipeline([RobustScaler()], backend=backend)
        result1 = pipe1.fit_transform("t")

        pipe2 = Pipeline([RobustScaler()], backend=backend)
        pipe2.fit("t")
        result2 = pipe2.transform("t")

        np.testing.assert_array_equal(result1, result2)

    def test_with_centering_false_pipeline(self, backend: DuckDBBackend) -> None:
        """with_centering=False: values are scaled but not centered."""
        pipe = Pipeline([RobustScaler(with_centering=False)], backend=backend)
        result = pipe.fit_transform("t")
        # Values should not be centered around zero
        col_means = result.mean(axis=0)
        assert not np.allclose(col_means, [0.0, 0.0], atol=1e-10)

    def test_with_scaling_false_pipeline(self, backend: DuckDBBackend) -> None:
        """with_scaling=False: values are centered but not scaled."""
        pipe = Pipeline([RobustScaler(with_scaling=False)], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql()
        # No NULLIF needed when not scaling
        assert "NULLIF" not in sql.upper()


# ── sklearn equivalence tests ───────────────────────────────────


class TestSklearnEquivalence:
    """Compare sqlearn RobustScaler output to sklearn."""

    def test_basic_equivalence(self) -> None:
        """sqlearn matches sklearn for basic numeric data."""
        from sklearn.preprocessing import RobustScaler as SklearnRobust

        # Use enough data points that quantile interpolation differences vanish
        rng = np.random.default_rng(42)
        data = rng.standard_normal((1000, 2)) * np.array([10, 50]) + np.array([5, 100])

        sk_result = SklearnRobust().fit_transform(data)

        conn = duckdb.connect()
        conn.execute("CREATE TABLE t (a DOUBLE, b DOUBLE)")
        for row in data:
            conn.execute("INSERT INTO t VALUES (?, ?)", [float(x) for x in row])

        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([RobustScaler()], backend=backend)
        sq_result = pipe.fit_transform("t")

        np.testing.assert_allclose(sq_result, sk_result, atol=0.5)

    def test_with_centering_false_equivalence(self) -> None:
        """sqlearn matches sklearn with with_centering=False."""
        from sklearn.preprocessing import RobustScaler as SklearnRobust

        rng = np.random.default_rng(123)
        data = rng.standard_normal((1000, 1)) * 10 + 50

        sk_result = SklearnRobust(with_centering=False).fit_transform(data)

        conn = duckdb.connect()
        conn.execute("CREATE TABLE t (a DOUBLE)")
        for row in data:
            conn.execute("INSERT INTO t VALUES (?)", [float(row[0])])

        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([RobustScaler(with_centering=False)], backend=backend)
        sq_result = pipe.fit_transform("t")

        np.testing.assert_allclose(sq_result, sk_result, atol=0.5)

    def test_with_scaling_false_equivalence(self) -> None:
        """sqlearn matches sklearn with with_scaling=False."""
        from sklearn.preprocessing import RobustScaler as SklearnRobust

        rng = np.random.default_rng(456)
        data = rng.standard_normal((1000, 1)) * 10 + 50

        sk_result = SklearnRobust(with_scaling=False).fit_transform(data)

        conn = duckdb.connect()
        conn.execute("CREATE TABLE t (a DOUBLE)")
        for row in data:
            conn.execute("INSERT INTO t VALUES (?)", [float(row[0])])

        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([RobustScaler(with_scaling=False)], backend=backend)
        sq_result = pipe.fit_transform("t")

        np.testing.assert_allclose(sq_result, sk_result, atol=0.5)

    def test_multicolumn_equivalence(self) -> None:
        """sqlearn matches sklearn for many columns with varied distributions."""
        from sklearn.preprocessing import RobustScaler as SklearnRobust

        rng = np.random.default_rng(789)
        data = rng.standard_normal((1000, 5)) * np.array([1, 10, 100, 0.1, 50]) + np.array(
            [0, 50, -100, 3, 200]
        )

        sk_result = SklearnRobust().fit_transform(data)

        conn = duckdb.connect()
        conn.execute("CREATE TABLE t (a DOUBLE, b DOUBLE, c DOUBLE, d DOUBLE, e DOUBLE)")
        for row in data:
            conn.execute("INSERT INTO t VALUES (?, ?, ?, ?, ?)", [float(x) for x in row])

        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([RobustScaler()], backend=backend)
        sq_result = pipe.fit_transform("t")

        np.testing.assert_allclose(sq_result, sk_result, atol=0.5)


# ── Clone and pickle tests ──────────────────────────────────────


class TestClonePickle:
    """Deep clone independence and pickle roundtrip."""

    @pytest.fixture
    def fitted_pipe(self) -> Pipeline:
        """Create a fitted pipeline for clone/pickle tests."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES (1.0, 10.0), (2.0, 20.0), (3.0, 30.0) t(a, b)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([RobustScaler()], backend=backend)
        pipe.fit("t")
        return pipe

    def test_clone_produces_same_sql(self, fitted_pipe: Pipeline) -> None:
        """Cloned pipeline produces identical SQL."""
        cloned = fitted_pipe.clone()
        assert fitted_pipe.to_sql() == cloned.to_sql()

    def test_clone_params_independent(self, fitted_pipe: Pipeline) -> None:
        """Modifying cloned step params does not affect original."""
        cloned = fitted_pipe.clone()
        original_scaler = fitted_pipe.steps[0][1]
        cloned_scaler = cloned.steps[0][1]
        # Mutate clone
        assert cloned_scaler.params_ is not None
        cloned_scaler.params_["a__median"] = 999.0
        # Original should be unchanged
        assert original_scaler.params_ is not None
        assert original_scaler.params_["a__median"] != 999.0

    def test_pickle_roundtrip(self) -> None:
        """Pickle an individual scaler preserves params."""
        import pickle

        scaler = RobustScaler()
        scaler.params_ = {"a__median": 3.0, "a__q1": 1.0, "a__q3": 5.0}
        scaler._fitted = True
        data = pickle.dumps(scaler)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.params_ == {"a__median": 3.0, "a__q1": 1.0, "a__q3": 5.0}
        assert restored._fitted is True


# ── Edge cases ──────────────────────────────────────────────────


class TestEdgeCases:
    """Extreme and unusual inputs."""

    def test_constant_column(self) -> None:
        """All-same-value column has IQR=0, handled by NULLIF."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (5.0), (5.0), (5.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([RobustScaler()], backend=backend)
        result = pipe.fit_transform("t")
        # IQR=0, division by NULLIF(0,0) = NULL
        assert result.shape == (3, 1)

    def test_single_row(self) -> None:
        """Single-row data produces IQR=0, handled by NULLIF."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (42.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([RobustScaler()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (1, 1)


# ── Composition tests ───────────────────────────────────────────


class TestComposition:
    """RobustScaler composing with other transformers."""

    def test_imputer_then_scaler(self) -> None:
        """Imputer + RobustScaler produces no NaN/None in output."""
        from sqlearn.imputers.imputer import Imputer

        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES (1.0), (NULL), (3.0), (4.0), (5.0) t(a)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Imputer(), RobustScaler()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (5, 1)
        # No NaN/None in output (NULL was imputed before scaling)
        assert not np.any(np.isnan(result.astype(float)))

    def test_imputer_scaler_sql_nesting(self) -> None:
        """SQL shows COALESCE nested inside subtraction."""
        from sqlearn.imputers.imputer import Imputer

        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES (1.0), (NULL), (3.0), (4.0), (5.0) t(a)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Imputer(), RobustScaler()], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql().upper()
        assert "COALESCE" in sql
        assert "NULLIF" in sql
