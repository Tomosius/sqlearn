"""Tests for sqlearn.scalers.minmax — MinMaxScaler."""

from __future__ import annotations

from typing import Any

import duckdb
import numpy as np
import pytest
import sqlglot.expressions as exp

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.pipeline import Pipeline
from sqlearn.core.schema import Schema
from sqlearn.scalers.minmax import MinMaxScaler

# ── Constructor tests ──────────────────────────────────────────────


class TestConstructor:
    """Test MinMaxScaler constructor and attributes."""

    def test_defaults(self) -> None:
        """Default feature_range=(0, 1), columns=None."""
        scaler = MinMaxScaler()
        assert scaler.feature_range == (0, 1)
        assert scaler.columns is None

    def test_custom_feature_range(self) -> None:
        """Custom feature_range is stored."""
        scaler = MinMaxScaler(feature_range=(-1, 1))
        assert scaler.feature_range == (-1, 1)

    def test_custom_columns_list(self) -> None:
        """Explicit column list overrides default."""
        scaler = MinMaxScaler(columns=["a", "b"])
        assert scaler.columns == ["a", "b"]

    def test_custom_columns_string(self) -> None:
        """String column spec (e.g. 'numeric') accepted."""
        scaler = MinMaxScaler(columns="numeric")
        assert scaler.columns == "numeric"

    def test_invalid_feature_range_equal(self) -> None:
        """feature_range with min == max raises ValueError."""
        with pytest.raises(ValueError, match="feature_range"):
            MinMaxScaler(feature_range=(1, 1))

    def test_invalid_feature_range_reversed(self) -> None:
        """feature_range with min > max raises ValueError."""
        with pytest.raises(ValueError, match="feature_range"):
            MinMaxScaler(feature_range=(2, 0))

    def test_default_columns_is_numeric(self) -> None:
        """Class default routes to numeric columns."""
        assert MinMaxScaler._default_columns == "numeric"

    def test_classification_is_dynamic(self) -> None:
        """Class is dynamic (needs to learn min/max)."""
        assert MinMaxScaler._classification == "dynamic"

    def test_get_params(self) -> None:
        """get_params returns constructor kwargs."""
        scaler = MinMaxScaler(feature_range=(-1, 1), columns=["x"])
        params = scaler.get_params()
        assert params == {"feature_range": (-1, 1), "columns": ["x"]}

    def test_set_params(self) -> None:
        """set_params updates attributes and returns self."""
        scaler = MinMaxScaler()
        result = scaler.set_params(feature_range=(0, 10))
        assert result is scaler
        assert scaler.feature_range == (0, 10)


# ── discover() tests ───────────────────────────────────────────────


class TestDiscover:
    """Test MinMaxScaler.discover() returns correct sqlglot aggregates."""

    @pytest.fixture
    def schema(self) -> Schema:
        """Simple numeric schema."""
        return Schema({"a": "DOUBLE", "b": "DOUBLE"})

    def test_min_and_max_per_column(self, schema: Schema) -> None:
        """Each column gets MIN and MAX entries."""
        scaler = MinMaxScaler()
        result = scaler.discover(["a", "b"], schema)
        assert "a__min" in result
        assert "a__max" in result
        assert "b__min" in result
        assert "b__max" in result
        assert len(result) == 4

    def test_min_ast_type(self, schema: Schema) -> None:
        """Min expression is exp.Min wrapping exp.Column."""
        scaler = MinMaxScaler()
        result = scaler.discover(["a"], schema)
        min_expr = result["a__min"]
        assert isinstance(min_expr, exp.Min)
        inner = min_expr.this
        assert isinstance(inner, exp.Column)
        assert inner.name == "a"

    def test_max_ast_type(self, schema: Schema) -> None:
        """Max expression is exp.Max wrapping exp.Column."""
        scaler = MinMaxScaler()
        result = scaler.discover(["a"], schema)
        max_expr = result["a__max"]
        assert isinstance(max_expr, exp.Max)
        inner = max_expr.this
        assert isinstance(inner, exp.Column)
        assert inner.name == "a"

    def test_empty_columns_returns_empty(self, schema: Schema) -> None:
        """Empty columns list returns empty dict."""
        scaler = MinMaxScaler()
        result = scaler.discover([], schema)
        assert result == {}

    def test_single_column(self, schema: Schema) -> None:
        """Single column produces exactly 2 aggregates."""
        scaler = MinMaxScaler()
        result = scaler.discover(["a"], schema)
        assert len(result) == 2


# ── expressions() tests ───────────────────────────────────────────


class TestExpressions:
    """Test MinMaxScaler.expressions() generates correct sqlglot ASTs."""

    def _make_fitted_scaler(
        self,
        params: dict[str, Any],
        *,
        feature_range: tuple[float, float] = (0, 1),
    ) -> MinMaxScaler:
        """Create a fitted MinMaxScaler with given params."""
        scaler = MinMaxScaler(feature_range=feature_range)
        scaler.params_ = params
        scaler._fitted = True
        return scaler

    def test_basic_formula(self) -> None:
        """(col - min) / NULLIF(max - min, 0) for default range."""
        scaler = self._make_fitted_scaler({"a__min": 1.0, "a__max": 5.0})
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

    def test_custom_feature_range_formula(self) -> None:
        """Custom range wraps base formula in Add(Mul(...), min_val)."""
        scaler = self._make_fitted_scaler({"a__min": 0.0, "a__max": 10.0}, feature_range=(-1, 1))
        exprs = {"a": exp.Column(this="a")}
        result = scaler.expressions(["a"], exprs)
        # Top-level is Add (... * scale + min_val)
        assert isinstance(result["a"], exp.Add)
        # Inner is Mul (base_expr * scale)
        assert isinstance(result["a"].this, exp.Mul)

    def test_zero_range_safe_division(self) -> None:
        """min==max uses NULLIF(0, 0) which returns NULL, not error."""
        scaler = self._make_fitted_scaler({"a__min": 5.0, "a__max": 5.0})
        exprs = {"a": exp.Column(this="a")}
        result = scaler.expressions(["a"], exprs)
        # Denominator should be NULLIF(5.0 - 5.0, 0) => NULLIF(0, 0)
        nullif = result["a"].expression
        assert isinstance(nullif, exp.Nullif)

    def test_uses_exprs_not_column(self) -> None:
        """expressions() uses exprs[col], not exp.Column(this=col)."""
        scaler = self._make_fitted_scaler({"a__min": 0.0, "a__max": 10.0})
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
                "a__min": 1.0,
                "a__max": 5.0,
                "b__min": 10.0,
                "b__max": 50.0,
            }
        )
        exprs = {"a": exp.Column(this="a"), "b": exp.Column(this="b")}
        result = scaler.expressions(["a", "b"], exprs)
        assert "a" in result
        assert "b" in result

    def test_only_modifies_target_columns(self) -> None:
        """Columns not in target list are not in result."""
        scaler = self._make_fitted_scaler({"a__min": 1.0, "a__max": 5.0})
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
    """Test MinMaxScaler integrated with Pipeline (end-to-end)."""

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
        """Transformed data should be in [0, 1] range."""
        pipe = Pipeline([MinMaxScaler()], backend=backend)
        result = pipe.fit_transform("t")
        assert np.all(result >= 0.0 - 1e-10)
        assert np.all(result <= 1.0 + 1e-10)

    def test_fit_transform_min_zero(self, backend: DuckDBBackend) -> None:
        """Transformed data minimum should be ~0."""
        pipe = Pipeline([MinMaxScaler()], backend=backend)
        result = pipe.fit_transform("t")
        col_mins = result.min(axis=0)
        np.testing.assert_allclose(col_mins, [0.0, 0.0], atol=1e-10)

    def test_fit_transform_max_one(self, backend: DuckDBBackend) -> None:
        """Transformed data maximum should be ~1."""
        pipe = Pipeline([MinMaxScaler()], backend=backend)
        result = pipe.fit_transform("t")
        col_maxs = result.max(axis=0)
        np.testing.assert_allclose(col_maxs, [1.0, 1.0], atol=1e-10)

    def test_fit_transform_shape(self, backend: DuckDBBackend) -> None:
        """Output shape matches input shape."""
        pipe = Pipeline([MinMaxScaler()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (5, 2)

    def test_to_sql_contains_operators(self, backend: DuckDBBackend) -> None:
        """to_sql() output contains subtraction and division."""
        pipe = Pipeline([MinMaxScaler()], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql()
        assert "-" in sql
        assert "/" in sql

    def test_to_sql_contains_nullif(self, backend: DuckDBBackend) -> None:
        """to_sql() output contains NULLIF for safe division."""
        pipe = Pipeline([MinMaxScaler()], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql()
        assert "NULLIF" in sql.upper()

    def test_to_sql_produces_valid_sql(self, backend: DuckDBBackend) -> None:
        """to_sql() produces SQL that DuckDB can execute."""
        pipe = Pipeline([MinMaxScaler()], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql(table="t")
        conn = backend._get_connection()
        result = conn.execute(sql).fetchall()
        assert len(result) == 5

    def test_params_populated(self, backend: DuckDBBackend) -> None:
        """After fit, params_ contains min and max for each column."""
        scaler = MinMaxScaler()
        pipe = Pipeline([scaler], backend=backend)
        pipe.fit("t")
        assert scaler.params_ is not None
        assert "a__min" in scaler.params_
        assert "a__max" in scaler.params_
        assert "b__min" in scaler.params_
        assert "b__max" in scaler.params_
        # a: min=1.0, max=5.0, b: min=10.0, max=50.0
        np.testing.assert_allclose(float(scaler.params_["a__min"]), 1.0, atol=1e-10)
        np.testing.assert_allclose(float(scaler.params_["a__max"]), 5.0, atol=1e-10)
        np.testing.assert_allclose(float(scaler.params_["b__min"]), 10.0, atol=1e-10)
        np.testing.assert_allclose(float(scaler.params_["b__max"]), 50.0, atol=1e-10)

    def test_fit_then_transform_separate(self, backend: DuckDBBackend) -> None:
        """Separate fit() and transform() produce same result as fit_transform()."""
        pipe1 = Pipeline([MinMaxScaler()], backend=backend)
        result1 = pipe1.fit_transform("t")

        pipe2 = Pipeline([MinMaxScaler()], backend=backend)
        pipe2.fit("t")
        result2 = pipe2.transform("t")

        np.testing.assert_array_equal(result1, result2)


# ── sklearn equivalence tests ───────────────────────────────────


class TestSklearnEquivalence:
    """Compare sqlearn MinMaxScaler output to sklearn."""

    def test_basic_equivalence(self) -> None:
        """sqlearn matches sklearn for basic numeric data."""
        from sklearn.preprocessing import MinMaxScaler as SklearnMinMax

        data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]])

        # sklearn
        sk_result = SklearnMinMax().fit_transform(data)

        # sqlearn
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 10.0), (2.0, 20.0), (3.0, 30.0), "
            "(4.0, 40.0), (5.0, 50.0) t(a, b)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([MinMaxScaler()], backend=backend)
        sq_result = pipe.fit_transform("t")

        np.testing.assert_allclose(sq_result, sk_result, atol=1e-10)

    def test_custom_range_equivalence(self) -> None:
        """sqlearn matches sklearn with feature_range=(-1, 1)."""
        from sklearn.preprocessing import MinMaxScaler as SklearnMinMax

        data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])

        sk_result = SklearnMinMax(feature_range=(-1, 1)).fit_transform(data)

        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES (1.0), (2.0), (3.0), (4.0), (5.0) t(a)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([MinMaxScaler(feature_range=(-1, 1))], backend=backend)
        sq_result = pipe.fit_transform("t")

        np.testing.assert_allclose(sq_result, sk_result, atol=1e-10)

    def test_multicolumn_equivalence(self) -> None:
        """sqlearn matches sklearn for many columns with varied distributions."""
        from sklearn.preprocessing import MinMaxScaler as SklearnMinMax

        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 5)) * np.array([1, 10, 100, 0.1, 50]) + np.array(
            [0, 50, -100, 3, 200]
        )

        sk_result = SklearnMinMax().fit_transform(data)

        conn = duckdb.connect()
        conn.execute("CREATE TABLE t (a DOUBLE, b DOUBLE, c DOUBLE, d DOUBLE, e DOUBLE)")
        for row in data:
            conn.execute("INSERT INTO t VALUES (?, ?, ?, ?, ?)", [float(x) for x in row])

        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([MinMaxScaler()], backend=backend)
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
        pipe = Pipeline([MinMaxScaler()], backend=backend)
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
        cloned_scaler.params_["a__min"] = 999.0
        # Original should be unchanged
        assert original_scaler.params_ is not None
        assert original_scaler.params_["a__min"] != 999.0

    def test_pickle_roundtrip(self) -> None:
        """Pickle an individual scaler preserves params."""
        import pickle

        scaler = MinMaxScaler()
        scaler.params_ = {"a__min": 1.0, "a__max": 5.0}
        scaler._fitted = True
        data = pickle.dumps(scaler)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.params_ == {"a__min": 1.0, "a__max": 5.0}
        assert restored._fitted is True


# ── Edge cases ──────────────────────────────────────────────────


class TestEdgeCases:
    """Extreme and unusual inputs."""

    def test_constant_column(self) -> None:
        """All-same-value column has max==min, handled by NULLIF."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (5.0), (5.0), (5.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([MinMaxScaler()], backend=backend)
        result = pipe.fit_transform("t")
        # max-min=0, division by NULLIF(0,0) = NULL
        assert result.shape == (3, 1)

    def test_single_row(self) -> None:
        """Single-row data produces max==min, handled by NULLIF."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (42.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([MinMaxScaler()], backend=backend)
        result = pipe.fit_transform("t")
        # max==min → NULLIF returns NULL → result is NULL (None)
        assert result.shape == (1, 1)

    def test_all_null_column(self) -> None:
        """All-NULL column produces None params (MIN/MAX return NULL)."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT CAST(NULL AS DOUBLE) AS a "
            "UNION ALL SELECT CAST(NULL AS DOUBLE) "
            "UNION ALL SELECT CAST(NULL AS DOUBLE)"
        )
        backend = DuckDBBackend(connection=conn)
        scaler = MinMaxScaler()
        pipe = Pipeline([scaler], backend=backend)
        pipe.fit("t")
        # MIN/MAX of all-NULL column returns None
        assert scaler.params_ is not None
        assert scaler.params_["a__min"] is None
        assert scaler.params_["a__max"] is None


# ── Composition tests ───────────────────────────────────────────


class TestComposition:
    """MinMaxScaler composing with other transformers."""

    def test_imputer_then_scaler(self) -> None:
        """Imputer + MinMaxScaler produces nested COALESCE-in-subtract."""
        from sqlearn.imputers.imputer import Imputer

        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1.0), (NULL), (3.0), (5.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Imputer(), MinMaxScaler()], backend=backend)
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
        pipe = Pipeline([Imputer(), MinMaxScaler()], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql().upper()
        assert "COALESCE" in sql
        assert "NULLIF" in sql
