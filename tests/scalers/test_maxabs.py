"""Tests for sqlearn.scalers.maxabs — MaxAbsScaler."""

from __future__ import annotations

from typing import Any

import duckdb
import numpy as np
import pytest
import sqlglot.expressions as exp

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.pipeline import Pipeline
from sqlearn.core.schema import Schema
from sqlearn.scalers.maxabs import MaxAbsScaler

# ── Constructor tests ──────────────────────────────────────────────


class TestConstructor:
    """Test MaxAbsScaler constructor and attributes."""

    def test_defaults(self) -> None:
        """Default columns=None."""
        scaler = MaxAbsScaler()
        assert scaler.columns is None

    def test_custom_columns_list(self) -> None:
        """Explicit column list overrides default."""
        scaler = MaxAbsScaler(columns=["a", "b"])
        assert scaler.columns == ["a", "b"]

    def test_custom_columns_string(self) -> None:
        """String column spec (e.g. 'numeric') accepted."""
        scaler = MaxAbsScaler(columns="numeric")
        assert scaler.columns == "numeric"

    def test_default_columns_is_numeric(self) -> None:
        """Class default routes to numeric columns."""
        assert MaxAbsScaler._default_columns == "numeric"

    def test_classification_is_dynamic(self) -> None:
        """Class is dynamic (needs to learn max_abs)."""
        assert MaxAbsScaler._classification == "dynamic"

    def test_get_params(self) -> None:
        """get_params returns constructor kwargs."""
        scaler = MaxAbsScaler(columns=["x"])
        params = scaler.get_params()
        assert params == {"columns": ["x"]}

    def test_get_params_defaults(self) -> None:
        """get_params with defaults returns columns=None."""
        scaler = MaxAbsScaler()
        params = scaler.get_params()
        assert params == {"columns": None}

    def test_set_params(self) -> None:
        """set_params updates attributes and returns self."""
        scaler = MaxAbsScaler()
        result = scaler.set_params(columns=["a"])
        assert result is scaler
        assert scaler.columns == ["a"]


# ── discover() tests ───────────────────────────────────────────────


class TestDiscover:
    """Test MaxAbsScaler.discover() returns correct sqlglot aggregates."""

    @pytest.fixture
    def schema(self) -> Schema:
        """Simple numeric schema."""
        return Schema({"a": "DOUBLE", "b": "DOUBLE"})

    def test_max_abs_per_column(self, schema: Schema) -> None:
        """Each column gets a max_abs entry."""
        scaler = MaxAbsScaler()
        result = scaler.discover(["a", "b"], schema)
        assert "a__max_abs" in result
        assert "b__max_abs" in result
        assert len(result) == 2

    def test_ast_type_is_max_wrapping_abs(self, schema: Schema) -> None:
        """max_abs expression is exp.Max wrapping exp.Abs wrapping exp.Column."""
        scaler = MaxAbsScaler()
        result = scaler.discover(["a"], schema)
        max_expr = result["a__max_abs"]
        assert isinstance(max_expr, exp.Max)
        abs_expr = max_expr.this
        assert isinstance(abs_expr, exp.Abs)
        inner = abs_expr.this
        assert isinstance(inner, exp.Column)
        assert inner.name == "a"

    def test_empty_columns_returns_empty(self, schema: Schema) -> None:
        """Empty columns list returns empty dict."""
        scaler = MaxAbsScaler()
        result = scaler.discover([], schema)
        assert result == {}

    def test_single_column(self, schema: Schema) -> None:
        """Single column produces exactly 1 aggregate."""
        scaler = MaxAbsScaler()
        result = scaler.discover(["a"], schema)
        assert len(result) == 1


# ── expressions() tests ───────────────────────────────────────────


class TestExpressions:
    """Test MaxAbsScaler.expressions() generates correct sqlglot ASTs."""

    def _make_fitted_scaler(
        self,
        params: dict[str, Any],
    ) -> MaxAbsScaler:
        """Create a fitted MaxAbsScaler with given params."""
        scaler = MaxAbsScaler()
        scaler.params_ = params
        scaler._fitted = True
        return scaler

    def test_basic_formula(self) -> None:
        """col / NULLIF(max_abs, 0) for basic case."""
        scaler = self._make_fitted_scaler({"a__max_abs": 10.0})
        exprs = {"a": exp.Column(this="a"), "b": exp.Column(this="b")}
        result = scaler.expressions(["a"], exprs)
        assert "a" in result
        # Top-level is Div
        assert isinstance(result["a"], exp.Div)
        # Denominator is Nullif
        assert isinstance(result["a"].expression, exp.Nullif)

    def test_zero_max_safe_division(self) -> None:
        """max_abs=0 uses NULLIF(0, 0) which returns NULL, not error."""
        scaler = self._make_fitted_scaler({"a__max_abs": 0.0})
        exprs = {"a": exp.Column(this="a")}
        result = scaler.expressions(["a"], exprs)
        # Denominator should be NULLIF(0.0, 0)
        nullif = result["a"].expression
        assert isinstance(nullif, exp.Nullif)

    def test_uses_exprs_not_column(self) -> None:
        """expressions() uses exprs[col], not exp.Column(this=col)."""
        scaler = self._make_fitted_scaler({"a__max_abs": 5.0})
        # Simulate a prior transform: a is already (a * 2)
        prior = exp.Mul(this=exp.Column(this="a"), expression=exp.Literal.number(2))
        exprs = {"a": prior}
        result = scaler.expressions(["a"], exprs)
        # The numerator should reference the prior expression
        assert isinstance(result["a"].this, exp.Mul)

    def test_multiple_columns(self) -> None:
        """Multiple columns each get their own expression."""
        scaler = self._make_fitted_scaler(
            {
                "a__max_abs": 10.0,
                "b__max_abs": 50.0,
            }
        )
        exprs = {"a": exp.Column(this="a"), "b": exp.Column(this="b")}
        result = scaler.expressions(["a", "b"], exprs)
        assert "a" in result
        assert "b" in result

    def test_only_modifies_target_columns(self) -> None:
        """Columns not in target list are not in result."""
        scaler = self._make_fitted_scaler({"a__max_abs": 10.0})
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
    """Test MaxAbsScaler integrated with Pipeline (end-to-end)."""

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

    def test_fit_transform_in_range(self, backend: DuckDBBackend) -> None:
        """Transformed data should be in [-1, 1]."""
        pipe = Pipeline([MaxAbsScaler()], backend=backend)
        result = pipe.fit_transform("t")
        assert np.all(result >= -1.0 - 1e-10)
        assert np.all(result <= 1.0 + 1e-10)

    def test_fit_transform_max_is_one(self, backend: DuckDBBackend) -> None:
        """Max absolute value of each column should be ~1."""
        pipe = Pipeline([MaxAbsScaler()], backend=backend)
        result = pipe.fit_transform("t")
        col_max_abs = np.abs(result).max(axis=0)
        np.testing.assert_allclose(col_max_abs, [1.0, 1.0], atol=1e-10)

    def test_fit_transform_shape(self, backend: DuckDBBackend) -> None:
        """Output shape matches input shape."""
        pipe = Pipeline([MaxAbsScaler()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (5, 2)

    def test_to_sql_contains_division(self, backend: DuckDBBackend) -> None:
        """to_sql() output contains division operator."""
        pipe = Pipeline([MaxAbsScaler()], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql()
        assert "/" in sql

    def test_to_sql_contains_nullif(self, backend: DuckDBBackend) -> None:
        """to_sql() output contains NULLIF for safe division."""
        pipe = Pipeline([MaxAbsScaler()], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql()
        assert "NULLIF" in sql.upper()

    def test_to_sql_is_valid(self, backend: DuckDBBackend) -> None:
        """to_sql() generates a string that can be parsed back."""
        pipe = Pipeline([MaxAbsScaler()], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql()
        assert isinstance(sql, str)
        assert len(sql) > 0

    def test_params_populated(self, backend: DuckDBBackend) -> None:
        """After fit, params_ contains max_abs for each column."""
        scaler = MaxAbsScaler()
        pipe = Pipeline([scaler], backend=backend)
        pipe.fit("t")
        assert scaler.params_ is not None
        assert "a__max_abs" in scaler.params_
        assert "b__max_abs" in scaler.params_
        # a: max(|1|,|2|,|3|,|4|,|5|) = 5.0
        np.testing.assert_allclose(float(scaler.params_["a__max_abs"]), 5.0, atol=1e-10)
        # b: max(|10|,|20|,|30|,|40|,|50|) = 50.0
        np.testing.assert_allclose(float(scaler.params_["b__max_abs"]), 50.0, atol=1e-10)

    def test_fit_then_transform_separate(self, backend: DuckDBBackend) -> None:
        """Separate fit() and transform() produce same result as fit_transform()."""
        pipe1 = Pipeline([MaxAbsScaler()], backend=backend)
        result1 = pipe1.fit_transform("t")

        pipe2 = Pipeline([MaxAbsScaler()], backend=backend)
        pipe2.fit("t")
        result2 = pipe2.transform("t")

        np.testing.assert_array_equal(result1, result2)


# ── sklearn equivalence tests ───────────────────────────────────


class TestSklearnEquivalence:
    """Compare sqlearn MaxAbsScaler output to sklearn."""

    def test_basic_equivalence(self) -> None:
        """sqlearn matches sklearn for basic numeric data."""
        from sklearn.preprocessing import MaxAbsScaler as SklearnMaxAbs

        data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]])

        # sklearn
        sk_result = SklearnMaxAbs().fit_transform(data)

        # sqlearn
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 10.0), (2.0, 20.0), (3.0, 30.0), "
            "(4.0, 40.0), (5.0, 50.0) t(a, b)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([MaxAbsScaler()], backend=backend)
        sq_result = pipe.fit_transform("t")

        np.testing.assert_allclose(sq_result, sk_result, atol=1e-10)

    def test_negative_values_equivalence(self) -> None:
        """sqlearn matches sklearn with negative values."""
        from sklearn.preprocessing import MaxAbsScaler as SklearnMaxAbs

        data = np.array([[-10.0], [5.0], [-3.0], [8.0], [-1.0]])

        sk_result = SklearnMaxAbs().fit_transform(data)

        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES (-10.0), (5.0), (-3.0), (8.0), (-1.0) t(a)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([MaxAbsScaler()], backend=backend)
        sq_result = pipe.fit_transform("t")

        np.testing.assert_allclose(sq_result, sk_result, atol=1e-10)

    def test_multicolumn_equivalence(self) -> None:
        """sqlearn matches sklearn for many columns with varied distributions."""
        from sklearn.preprocessing import MaxAbsScaler as SklearnMaxAbs

        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 5)) * np.array([1, 10, 100, 0.1, 50]) + np.array(
            [0, 50, -100, 3, 200]
        )

        sk_result = SklearnMaxAbs().fit_transform(data)

        conn = duckdb.connect()
        conn.execute("CREATE TABLE t (a DOUBLE, b DOUBLE, c DOUBLE, d DOUBLE, e DOUBLE)")
        for row in data:
            conn.execute("INSERT INTO t VALUES (?, ?, ?, ?, ?)", [float(x) for x in row])

        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([MaxAbsScaler()], backend=backend)
        sq_result = pipe.fit_transform("t")

        np.testing.assert_allclose(sq_result, sk_result, atol=1e-10)


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
        pipe = Pipeline([MaxAbsScaler()], backend=backend)
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
        cloned_scaler.params_["a__max_abs"] = 999.0
        # Original should be unchanged
        assert original_scaler.params_ is not None
        assert original_scaler.params_["a__max_abs"] != 999.0

    def test_pickle_roundtrip(self) -> None:
        """Pickle an individual scaler preserves params."""
        import pickle

        scaler = MaxAbsScaler()
        scaler.params_ = {"a__max_abs": 10.0}
        scaler._fitted = True
        data = pickle.dumps(scaler)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.params_ == {"a__max_abs": 10.0}
        assert restored._fitted is True


# ── Edge cases ──────────────────────────────────────────────────


class TestEdgeCases:
    """Extreme and unusual inputs."""

    def test_all_zero_column(self) -> None:
        """All-zero column has max_abs=0, handled by NULLIF."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (0.0), (0.0), (0.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([MaxAbsScaler()], backend=backend)
        result = pipe.fit_transform("t")
        # max_abs=0, division by NULLIF(0,0) = NULL
        assert result.shape == (3, 1)

    def test_single_row(self) -> None:
        """Single-row data scales to 1.0 (or NULL if zero)."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (42.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([MaxAbsScaler()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (1, 1)
        # 42.0 / 42.0 = 1.0
        np.testing.assert_allclose(result[0, 0], 1.0, atol=1e-10)

    def test_negative_values_in_range(self) -> None:
        """Negative values should scale to [-1, 1]."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (-10.0), (5.0), (-3.0), (8.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([MaxAbsScaler()], backend=backend)
        result = pipe.fit_transform("t")
        assert np.all(result >= -1.0 - 1e-10)
        assert np.all(result <= 1.0 + 1e-10)
        # max_abs = 10.0, so -10.0/10.0 = -1.0
        np.testing.assert_allclose(result[0, 0], -1.0, atol=1e-10)

    def test_large_values(self) -> None:
        """Very large values don't crash."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1e100), (2e100), (3e100) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([MaxAbsScaler()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (3, 1)
        # All should be in [0, 1]
        assert np.all(result >= -1e-10)
        assert np.all(result <= 1.0 + 1e-10)

    def test_mixed_positive_negative(self) -> None:
        """Mixed positive/negative values scaled correctly."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES (-2.0), (-1.0), (0.0), (1.0), (2.0) t(a)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([MaxAbsScaler()], backend=backend)
        result = pipe.fit_transform("t")
        # max_abs = 2.0, so range is [-1, 1]
        np.testing.assert_allclose(result[0, 0], -1.0, atol=1e-10)
        np.testing.assert_allclose(result[4, 0], 1.0, atol=1e-10)
        np.testing.assert_allclose(result[2, 0], 0.0, atol=1e-10)


# ── Composition tests ───────────────────────────────────────────


class TestComposition:
    """MaxAbsScaler composing with other transformers."""

    def test_imputer_then_scaler(self) -> None:
        """Imputer + MaxAbsScaler produces nested COALESCE-in-divide."""
        from sqlearn.imputers.imputer import Imputer

        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1.0), (NULL), (3.0), (4.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Imputer(), MaxAbsScaler()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (4, 1)
        # No NaN/None in output (NULL was imputed before scaling)
        assert not np.any(np.isnan(result.astype(float)))

    def test_imputer_scaler_sql_nesting(self) -> None:
        """SQL shows COALESCE nested inside division."""
        from sqlearn.imputers.imputer import Imputer

        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1.0), (NULL), (3.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Imputer(), MaxAbsScaler()], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql().upper()
        assert "COALESCE" in sql
        assert "NULLIF" in sql
