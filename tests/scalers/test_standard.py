"""Tests for sqlearn.scalers.standard — StandardScaler."""

from __future__ import annotations

from typing import Any

import duckdb
import numpy as np
import pytest
import sqlglot.expressions as exp

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.pipeline import Pipeline
from sqlearn.core.schema import Schema
from sqlearn.scalers.standard import StandardScaler

# ── Constructor tests ──────────────────────────────────────────────


class TestConstructor:
    """Test StandardScaler constructor and attributes."""

    def test_defaults(self) -> None:
        """Default with_mean=True, with_std=True, columns=None."""
        scaler = StandardScaler()
        assert scaler.with_mean is True
        assert scaler.with_std is True
        assert scaler.columns is None

    def test_custom_columns_list(self) -> None:
        """Explicit column list overrides default."""
        scaler = StandardScaler(columns=["a", "b"])
        assert scaler.columns == ["a", "b"]

    def test_custom_columns_string(self) -> None:
        """String column spec (e.g. 'numeric') accepted."""
        scaler = StandardScaler(columns="numeric")
        assert scaler.columns == "numeric"

    def test_with_mean_false(self) -> None:
        """with_mean=False disables centering."""
        scaler = StandardScaler(with_mean=False)
        assert scaler.with_mean is False
        assert scaler.with_std is True

    def test_with_std_false(self) -> None:
        """with_std=False disables scaling."""
        scaler = StandardScaler(with_std=False)
        assert scaler.with_mean is True
        assert scaler.with_std is False

    def test_both_false(self) -> None:
        """Both flags False is a no-op (valid but useless)."""
        scaler = StandardScaler(with_mean=False, with_std=False)
        assert scaler.with_mean is False
        assert scaler.with_std is False

    def test_default_columns_is_numeric(self) -> None:
        """Class default routes to numeric columns."""
        assert StandardScaler._default_columns == "numeric"

    def test_classification_is_dynamic(self) -> None:
        """Class is dynamic (needs to learn mean/std)."""
        assert StandardScaler._classification == "dynamic"

    def test_get_params(self) -> None:
        """get_params returns constructor kwargs."""
        scaler = StandardScaler(with_mean=False, columns=["x"])
        params = scaler.get_params()
        assert params == {"with_mean": False, "with_std": True, "columns": ["x"]}

    def test_set_params(self) -> None:
        """set_params updates attributes and returns self."""
        scaler = StandardScaler()
        result = scaler.set_params(with_mean=False)
        assert result is scaler
        assert scaler.with_mean is False


# ── discover() tests ───────────────────────────────────────────────


class TestDiscover:
    """Test StandardScaler.discover() returns correct sqlglot aggregates."""

    @pytest.fixture
    def schema(self) -> Schema:
        """Simple numeric schema."""
        return Schema({"a": "DOUBLE", "b": "DOUBLE"})

    def test_avg_and_stddev_per_column(self, schema: Schema) -> None:
        """Each column gets AVG and STDDEV_POP entries."""
        scaler = StandardScaler()
        result = scaler.discover(["a", "b"], schema)
        assert "a__mean" in result
        assert "a__std" in result
        assert "b__mean" in result
        assert "b__std" in result
        assert len(result) == 4

    def test_avg_ast_type(self, schema: Schema) -> None:
        """Mean expression is exp.Avg wrapping exp.Column."""
        scaler = StandardScaler()
        result = scaler.discover(["a"], schema)
        avg_expr = result["a__mean"]
        assert isinstance(avg_expr, exp.Avg)
        inner = avg_expr.this
        assert isinstance(inner, exp.Column)
        assert inner.name == "a"

    def test_stddev_ast_type(self, schema: Schema) -> None:
        """Std expression is exp.StddevPop wrapping exp.Column."""
        scaler = StandardScaler()
        result = scaler.discover(["a"], schema)
        std_expr = result["a__std"]
        assert isinstance(std_expr, exp.StddevPop)
        inner = std_expr.this
        assert isinstance(inner, exp.Column)
        assert inner.name == "a"

    def test_with_mean_false_no_avg(self, schema: Schema) -> None:
        """with_mean=False omits AVG aggregation."""
        scaler = StandardScaler(with_mean=False)
        result = scaler.discover(["a"], schema)
        assert "a__mean" not in result
        assert "a__std" in result

    def test_with_std_false_no_stddev(self, schema: Schema) -> None:
        """with_std=False omits STDDEV_POP aggregation."""
        scaler = StandardScaler(with_std=False)
        result = scaler.discover(["a"], schema)
        assert "a__mean" in result
        assert "a__std" not in result

    def test_both_false_empty(self, schema: Schema) -> None:
        """Both flags False returns empty dict."""
        scaler = StandardScaler(with_mean=False, with_std=False)
        result = scaler.discover(["a"], schema)
        assert result == {}

    def test_empty_columns_returns_empty(self, schema: Schema) -> None:
        """Empty columns list returns empty dict."""
        scaler = StandardScaler()
        result = scaler.discover([], schema)
        assert result == {}

    def test_single_column(self, schema: Schema) -> None:
        """Single column produces exactly 2 aggregates."""
        scaler = StandardScaler()
        result = scaler.discover(["a"], schema)
        assert len(result) == 2


# ── expressions() tests ───────────────────────────────────────────


class TestExpressions:
    """Test StandardScaler.expressions() generates correct sqlglot ASTs."""

    def _make_fitted_scaler(
        self,
        params: dict[str, Any],
        *,
        with_mean: bool = True,
        with_std: bool = True,
    ) -> StandardScaler:
        """Create a fitted StandardScaler with given params."""
        scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
        scaler.params_ = params
        scaler._fitted = True
        return scaler

    def test_basic_formula(self) -> None:
        """(col - mean) / NULLIF(std, 0) for basic case."""
        scaler = self._make_fitted_scaler({"a__mean": 2.0, "a__std": 1.0})
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

    def test_std_zero_safe_division(self) -> None:
        """std=0 uses NULLIF(0, 0) which returns NULL, not error."""
        scaler = self._make_fitted_scaler({"a__mean": 5.0, "a__std": 0.0})
        exprs = {"a": exp.Column(this="a")}
        result = scaler.expressions(["a"], exprs)
        # Denominator should be NULLIF(0.0, 0)
        nullif = result["a"].expression
        assert isinstance(nullif, exp.Nullif)

    def test_with_mean_false(self) -> None:
        """with_mean=False produces col / NULLIF(std, 0)."""
        scaler = self._make_fitted_scaler({"a__std": 2.0}, with_mean=False)
        exprs = {"a": exp.Column(this="a")}
        result = scaler.expressions(["a"], exprs)
        # No subtraction — top level is Div
        assert isinstance(result["a"], exp.Div)
        # Numerator is the raw expression, not Paren(Sub(...))
        assert isinstance(result["a"].this, exp.Column)

    def test_with_std_false(self) -> None:
        """with_std=False produces col - mean."""
        scaler = self._make_fitted_scaler({"a__mean": 3.0}, with_std=False)
        exprs = {"a": exp.Column(this="a")}
        result = scaler.expressions(["a"], exprs)
        # No division — top level is Paren(Sub(...))
        assert isinstance(result["a"], exp.Paren)
        assert isinstance(result["a"].this, exp.Sub)

    def test_both_false_passthrough(self) -> None:
        """Both flags False returns empty dict (passthrough)."""
        scaler = self._make_fitted_scaler({}, with_mean=False, with_std=False)
        exprs = {"a": exp.Column(this="a")}
        result = scaler.expressions(["a"], exprs)
        assert result == {}

    def test_multiple_columns(self) -> None:
        """Multiple columns each get their own expression."""
        scaler = self._make_fitted_scaler(
            {
                "a__mean": 1.0,
                "a__std": 2.0,
                "b__mean": 10.0,
                "b__std": 5.0,
            }
        )
        exprs = {"a": exp.Column(this="a"), "b": exp.Column(this="b")}
        result = scaler.expressions(["a", "b"], exprs)
        assert "a" in result
        assert "b" in result

    def test_only_modifies_target_columns(self) -> None:
        """Columns not in target list are not in result."""
        scaler = self._make_fitted_scaler({"a__mean": 1.0, "a__std": 2.0})
        exprs = {
            "a": exp.Column(this="a"),
            "b": exp.Column(this="b"),
            "c": exp.Column(this="c"),
        }
        result = scaler.expressions(["a"], exprs)
        assert "a" in result
        assert "b" not in result
        assert "c" not in result

    def test_uses_exprs_not_column(self) -> None:
        """expressions() uses exprs[col], not exp.Column(this=col)."""
        scaler = self._make_fitted_scaler({"a__mean": 0.0, "a__std": 1.0})
        # Simulate a prior transform: a is already (a * 2)
        prior = exp.Mul(this=exp.Column(this="a"), expression=exp.Literal.number(2))
        exprs = {"a": prior}
        result = scaler.expressions(["a"], exprs)
        # The numerator's inner Sub should reference the prior expression
        sub = result["a"].this.this  # Paren -> Sub
        assert isinstance(sub.this, exp.Mul)


# ── __repr__ tests ────────────────────────────────────────────────


class TestRepr:
    """Test StandardScaler.__repr__."""

    def test_default_repr(self) -> None:
        """Default params shows no args."""
        scaler = StandardScaler()
        assert repr(scaler) == "StandardScaler()"

    def test_with_mean_false_repr(self) -> None:
        """Non-default with_mean shows in repr."""
        scaler = StandardScaler(with_mean=False)
        assert repr(scaler) == "StandardScaler(with_mean=False)"

    def test_with_std_false_repr(self) -> None:
        """Non-default with_std shows in repr."""
        scaler = StandardScaler(with_std=False)
        assert repr(scaler) == "StandardScaler(with_std=False)"

    def test_custom_columns_repr(self) -> None:
        """Custom columns shows in repr."""
        scaler = StandardScaler(columns=["x", "y"])
        assert repr(scaler) == "StandardScaler(columns=['x', 'y'])"

    def test_all_custom_repr(self) -> None:
        """All non-default params show in repr."""
        scaler = StandardScaler(with_mean=False, with_std=False, columns=["x"])
        assert repr(scaler) == "StandardScaler(with_mean=False, with_std=False, columns=['x'])"


# ── Pipeline integration tests ────────────────────────────────────


class TestPipeline:
    """Test StandardScaler integrated with Pipeline (end-to-end)."""

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

    @pytest.fixture
    def mixed_backend(self) -> DuckDBBackend:
        """Create DuckDB backend with numeric + varchar columns."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 10.0, 'x'), (2.0, 20.0, 'y'), (3.0, 30.0, 'z'), "
            "(4.0, 40.0, 'w'), (5.0, 50.0, 'v') t(a, b, name)"
        )
        return DuckDBBackend(connection=conn)

    def test_fit_transform_zero_mean(self, backend: DuckDBBackend) -> None:
        """Transformed data should have mean ~0."""
        pipe = Pipeline([StandardScaler()], backend=backend)
        result = pipe.fit_transform("t")
        col_means = result.mean(axis=0)
        np.testing.assert_allclose(col_means, [0.0, 0.0], atol=1e-10)

    def test_fit_transform_unit_std(self, backend: DuckDBBackend) -> None:
        """Transformed data should have population std ~1."""
        pipe = Pipeline([StandardScaler()], backend=backend)
        result = pipe.fit_transform("t")
        # Population std (ddof=0) should be ~1.0
        col_stds = result.std(axis=0, ddof=0)
        np.testing.assert_allclose(col_stds, [1.0, 1.0], atol=1e-10)

    def test_fit_transform_shape(self, backend: DuckDBBackend) -> None:
        """Output shape matches input shape."""
        pipe = Pipeline([StandardScaler()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (5, 2)

    def test_to_sql_contains_operators(self, backend: DuckDBBackend) -> None:
        """to_sql() output contains subtraction and division."""
        pipe = Pipeline([StandardScaler()], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql()
        assert "-" in sql
        assert "/" in sql

    def test_to_sql_contains_nullif(self, backend: DuckDBBackend) -> None:
        """to_sql() output contains NULLIF for safe division."""
        pipe = Pipeline([StandardScaler()], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql()
        assert "NULLIF" in sql.upper()

    def test_passthrough_non_numeric(self, mixed_backend: DuckDBBackend) -> None:
        """Non-numeric columns pass through unmodified."""
        pipe = Pipeline([StandardScaler()], backend=mixed_backend)
        result = pipe.fit_transform("t")
        # Result has 3 columns (a, b, name)
        assert result.shape == (5, 3)
        # name column should be unchanged string values
        names = result[:, 2]
        assert set(names) == {"x", "y", "z", "w", "v"}

    def test_params_populated(self, backend: DuckDBBackend) -> None:
        """After fit, params_ contains mean and std for each column."""
        scaler = StandardScaler()
        pipe = Pipeline([scaler], backend=backend)
        pipe.fit("t")
        assert scaler.params_ is not None
        assert "a__mean" in scaler.params_
        assert "a__std" in scaler.params_
        assert "b__mean" in scaler.params_
        assert "b__std" in scaler.params_
        # a: mean=3.0, b: mean=30.0
        np.testing.assert_allclose(scaler.params_["a__mean"], 3.0, atol=1e-10)
        np.testing.assert_allclose(scaler.params_["b__mean"], 30.0, atol=1e-10)

    def test_with_mean_false_pipeline(self, backend: DuckDBBackend) -> None:
        """with_mean=False: mean is NOT zero, but std is 1."""
        pipe = Pipeline([StandardScaler(with_mean=False)], backend=backend)
        result = pipe.fit_transform("t")
        # Mean should NOT be zero (just scaled)
        col_means = result.mean(axis=0)
        assert not np.allclose(col_means, [0.0, 0.0], atol=1e-10)
        # Population std should still be 1
        col_stds = result.std(axis=0, ddof=0)
        np.testing.assert_allclose(col_stds, [1.0, 1.0], atol=1e-10)

    def test_with_std_false_pipeline(self, backend: DuckDBBackend) -> None:
        """with_std=False: mean is zero, but std is NOT 1."""
        pipe = Pipeline([StandardScaler(with_std=False)], backend=backend)
        result = pipe.fit_transform("t")
        # Mean should be zero (centered)
        col_means = result.mean(axis=0)
        np.testing.assert_allclose(col_means, [0.0, 0.0], atol=1e-10)

    def test_fit_then_transform_separate(self, backend: DuckDBBackend) -> None:
        """Separate fit() and transform() produce same result as fit_transform()."""
        pipe1 = Pipeline([StandardScaler()], backend=backend)
        result1 = pipe1.fit_transform("t")

        pipe2 = Pipeline([StandardScaler()], backend=backend)
        pipe2.fit("t")
        result2 = pipe2.transform("t")

        np.testing.assert_array_equal(result1, result2)


# ── Not-fitted guard tests ───────────────────────────────────────


class TestNotFittedGuard:
    """Calling transform/to_sql before fit() raises NotFittedError."""

    def test_transform_before_fit_raises(self) -> None:
        """transform() on unfitted pipeline raises NotFittedError."""
        from sqlearn.core.errors import NotFittedError

        pipe = Pipeline([StandardScaler()])
        with pytest.raises(NotFittedError):
            pipe.transform("t")

    def test_to_sql_before_fit_raises(self) -> None:
        """to_sql() on unfitted pipeline raises NotFittedError."""
        from sqlearn.core.errors import NotFittedError

        pipe = Pipeline([StandardScaler()])
        with pytest.raises(NotFittedError):
            pipe.to_sql()

    def test_get_feature_names_before_fit_raises(self) -> None:
        """get_feature_names_out() on unfitted pipeline raises NotFittedError."""
        from sqlearn.core.errors import NotFittedError

        pipe = Pipeline([StandardScaler()])
        with pytest.raises(NotFittedError):
            pipe.get_feature_names_out()


# ── Clone and pickle tests ──────────────────────────────────────


class TestCloneAndPickle:
    """Deep clone independence and pickle roundtrip."""

    @pytest.fixture
    def fitted_pipe(self) -> Pipeline:
        """Create a fitted pipeline for clone/pickle tests."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES (1.0, 10.0), (2.0, 20.0), (3.0, 30.0) t(a, b)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([StandardScaler()], backend=backend)
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
        cloned_scaler.params_["a__mean"] = 999.0
        # Original should be unchanged
        assert original_scaler.params_ is not None
        assert original_scaler.params_["a__mean"] != 999.0

    def test_pickle_roundtrip(self) -> None:
        """Pickle an individual scaler preserves params."""
        import pickle

        scaler = StandardScaler()
        scaler.params_ = {"a__mean": 3.0, "a__std": 1.5}
        scaler._fitted = True
        data = pickle.dumps(scaler)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.params_ == {"a__mean": 3.0, "a__std": 1.5}
        assert restored._fitted is True


# ── Edge cases ──────────────────────────────────────────────────


class TestEdgeCases:
    """Extreme and unusual inputs."""

    def test_single_row(self) -> None:
        """Single-row data produces std=0, handled by NULLIF."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (42.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([StandardScaler()], backend=backend)
        result = pipe.fit_transform("t")
        # std=0 → NULLIF returns NULL → result is NULL (None)
        assert result.shape == (1, 1)

    def test_constant_column(self) -> None:
        """All-same-value column has std=0, handled by NULLIF."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (5.0), (5.0), (5.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([StandardScaler()], backend=backend)
        result = pipe.fit_transform("t")
        # std=0, division by NULLIF(0,0) = NULL
        assert result.shape == (3, 1)

    def test_large_values(self) -> None:
        """Very large values don't crash."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1e100), (2e100), (3e100) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([StandardScaler()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (3, 1)
        # Mean should be ~0
        np.testing.assert_allclose(result.mean(), 0.0, atol=1e-10)

    def test_negative_values(self) -> None:
        """Negative values handled correctly."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (-10.0), (-20.0), (-30.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([StandardScaler()], backend=backend)
        result = pipe.fit_transform("t")
        np.testing.assert_allclose(result.mean(), 0.0, atol=1e-10)

    def test_mixed_positive_negative(self) -> None:
        """Mixed positive/negative values centered correctly."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES (-2.0), (-1.0), (0.0), (1.0), (2.0) t(a)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([StandardScaler()], backend=backend)
        result = pipe.fit_transform("t")
        np.testing.assert_allclose(result.mean(), 0.0, atol=1e-10)
        np.testing.assert_allclose(result.std(ddof=0), 1.0, atol=1e-10)


# ── sklearn equivalence tests ───────────────────────────────────


class TestSklearnEquivalence:
    """Compare sqlearn StandardScaler output to sklearn."""

    def test_basic_equivalence(self) -> None:
        """sqlearn matches sklearn for basic numeric data."""
        from sklearn.preprocessing import StandardScaler as SklearnScaler

        data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]])

        # sklearn
        sk_result = SklearnScaler().fit_transform(data)

        # sqlearn — use population std (ddof=0) like sklearn
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 10.0), (2.0, 20.0), (3.0, 30.0), "
            "(4.0, 40.0), (5.0, 50.0) t(a, b)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([StandardScaler()], backend=backend)
        sq_result = pipe.fit_transform("t")

        np.testing.assert_allclose(sq_result, sk_result, atol=1e-10)

    def test_with_mean_false_equivalence(self) -> None:
        """sqlearn matches sklearn with with_mean=False."""
        from sklearn.preprocessing import StandardScaler as SklearnScaler

        data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])

        sk_result = SklearnScaler(with_mean=False).fit_transform(data)

        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES (1.0), (2.0), (3.0), (4.0), (5.0) t(a)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([StandardScaler(with_mean=False)], backend=backend)
        sq_result = pipe.fit_transform("t")

        np.testing.assert_allclose(sq_result, sk_result, atol=1e-10)

    def test_with_std_false_equivalence(self) -> None:
        """sqlearn matches sklearn with with_std=False."""
        from sklearn.preprocessing import StandardScaler as SklearnScaler

        data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])

        sk_result = SklearnScaler(with_std=False).fit_transform(data)

        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES (1.0), (2.0), (3.0), (4.0), (5.0) t(a)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([StandardScaler(with_std=False)], backend=backend)
        sq_result = pipe.fit_transform("t")

        np.testing.assert_allclose(sq_result, sk_result, atol=1e-10)

    def test_multicolumn_equivalence(self) -> None:
        """sqlearn matches sklearn for many columns with varied distributions."""
        from sklearn.preprocessing import StandardScaler as SklearnScaler

        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 5)) * np.array([1, 10, 100, 0.1, 50]) + np.array(
            [0, 50, -100, 3, 200]
        )

        sk_result = SklearnScaler().fit_transform(data)

        conn = duckdb.connect()
        conn.execute("CREATE TABLE t (a DOUBLE, b DOUBLE, c DOUBLE, d DOUBLE, e DOUBLE)")
        for row in data:
            conn.execute("INSERT INTO t VALUES (?, ?, ?, ?, ?)", [float(x) for x in row])

        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([StandardScaler()], backend=backend)
        sq_result = pipe.fit_transform("t")

        np.testing.assert_allclose(sq_result, sk_result, atol=1e-10)


# ── SQL snapshot tests ──────────────────────────────────────────


class TestSqlSnapshot:
    """Verify SQL output structure and patterns."""

    @pytest.fixture
    def fitted_pipe(self) -> Pipeline:
        """Create fitted pipeline for SQL verification."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 10.0), (2.0, 20.0), (3.0, 30.0) t(price, quantity)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([StandardScaler()], backend=backend)
        pipe.fit("t")
        return pipe

    def test_sql_has_select_from(self, fitted_pipe: Pipeline) -> None:
        """SQL contains SELECT and FROM."""
        sql = fitted_pipe.to_sql().upper()
        assert "SELECT" in sql
        assert "FROM" in sql

    def test_sql_has_nullif_per_column(self, fitted_pipe: Pipeline) -> None:
        """Each column gets a NULLIF for safe division."""
        sql = fitted_pipe.to_sql().upper()
        assert sql.count("NULLIF") == 2  # one per column

    def test_sql_contains_learned_values(self, fitted_pipe: Pipeline) -> None:
        """SQL contains the learned mean and std values."""
        sql = fitted_pipe.to_sql()
        scaler = fitted_pipe.steps[0][1]
        assert scaler.params_ is not None
        # The mean of [1,2,3] = 2.0 should appear
        assert "2.0" in sql

    def test_sql_custom_table(self, fitted_pipe: Pipeline) -> None:
        """to_sql(table=...) uses custom table name."""
        sql = fitted_pipe.to_sql(table="my_data")
        assert "my_data" in sql

    def test_sql_custom_dialect(self, fitted_pipe: Pipeline) -> None:
        """to_sql(dialect=...) generates valid SQL for that dialect."""
        sql_pg = fitted_pipe.to_sql(dialect="postgres")
        sql_duck = fitted_pipe.to_sql(dialect="duckdb")
        # Both should be valid SQL strings
        assert isinstance(sql_pg, str)
        assert isinstance(sql_duck, str)


# ── Composition tests ───────────────────────────────────────────


class TestComposition:
    """StandardScaler composing with other transformers."""

    def test_imputer_then_scaler(self) -> None:
        """Imputer + StandardScaler produces nested COALESCE-in-subtract."""
        from sqlearn.imputers.imputer import Imputer

        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1.0), (NULL), (3.0), (4.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Imputer(), StandardScaler()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (4, 1)
        # No NaN/None in output (NULL was imputed before scaling)
        assert not np.any(np.isnan(result.astype(float)))

    def test_imputer_scaler_sql_nesting(self) -> None:
        """SQL shows COALESCE nested inside subtraction."""
        from sqlearn.imputers.imputer import Imputer

        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1.0), (NULL), (3.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Imputer(), StandardScaler()], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql().upper()
        assert "COALESCE" in sql
        assert "NULLIF" in sql

    def test_scaler_then_encoder(self) -> None:
        """StandardScaler + OneHotEncoder processes mixed columns."""
        from sqlearn.encoders.onehot import OneHotEncoder

        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 'A'), (2.0, 'B'), (3.0, 'A') t(price, city)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([StandardScaler(), OneHotEncoder()], backend=backend)
        result = pipe.fit_transform("t")
        # price (scaled) + city_a + city_b = 3 columns
        assert result.shape == (3, 3)
