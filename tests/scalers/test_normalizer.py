"""Tests for sqlearn.scalers.normalizer — Normalizer."""

from __future__ import annotations

import pickle

import duckdb
import numpy as np
import pytest
import sqlglot.expressions as exp

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.pipeline import Pipeline
from sqlearn.scalers.normalizer import Normalizer

# ── Constructor tests ──────────────────────────────────────────────


class TestConstructor:
    """Test Normalizer constructor and attributes."""

    def test_defaults(self) -> None:
        """Default norm='l2', columns=None."""
        norm = Normalizer()
        assert norm.norm == "l2"
        assert norm.columns is None

    def test_custom_norm_l1(self) -> None:
        """Explicit norm='l1' accepted."""
        norm = Normalizer(norm="l1")
        assert norm.norm == "l1"

    def test_custom_norm_max(self) -> None:
        """Explicit norm='max' accepted."""
        norm = Normalizer(norm="max")
        assert norm.norm == "max"

    def test_invalid_norm_raises(self) -> None:
        """Invalid norm raises ValueError."""
        with pytest.raises(ValueError, match="Invalid norm"):
            Normalizer(norm="l3")

    def test_custom_columns_list(self) -> None:
        """Explicit column list overrides default."""
        norm = Normalizer(columns=["a", "b"])
        assert norm.columns == ["a", "b"]

    def test_custom_columns_string(self) -> None:
        """String column spec (e.g. 'numeric') accepted."""
        norm = Normalizer(columns="numeric")
        assert norm.columns == "numeric"

    def test_default_columns_is_numeric(self) -> None:
        """Class default routes to numeric columns."""
        assert Normalizer._default_columns == "numeric"

    def test_classification_is_static(self) -> None:
        """Class is static (no discover() needed)."""
        assert Normalizer._classification == "static"

    def test_get_params(self) -> None:
        """get_params returns constructor kwargs."""
        norm = Normalizer(norm="l1", columns=["x"])
        params = norm.get_params()
        assert params == {"norm": "l1", "columns": ["x"]}

    def test_get_params_default(self) -> None:
        """get_params for default instance returns default values."""
        norm = Normalizer()
        params = norm.get_params()
        assert params == {"norm": "l2", "columns": None}

    def test_set_params(self) -> None:
        """set_params updates attributes and returns self."""
        norm = Normalizer()
        result = norm.set_params(norm="l1")
        assert result is norm
        assert norm.norm == "l1"


# ── expressions() tests ───────────────────────────────────────────


class TestExpressions:
    """Test Normalizer.expressions() generates correct sqlglot ASTs."""

    def _make_normalizer(self, norm: str = "l2") -> Normalizer:
        """Create a Normalizer with specified norm."""
        return Normalizer(norm=norm)

    def test_l2_formula_structure(self) -> None:
        """L2: col / NULLIF(SQRT(x1*x1 + x2*x2), 0)."""
        norm = self._make_normalizer("l2")
        exprs = {"a": exp.Column(this="a"), "b": exp.Column(this="b")}
        result = norm.expressions(["a", "b"], exprs)
        # Top-level is Div for each column
        assert isinstance(result["a"], exp.Div)
        assert isinstance(result["b"], exp.Div)
        # Denominator is Nullif
        assert isinstance(result["a"].expression, exp.Nullif)
        # Inside Nullif is Sqrt
        assert isinstance(result["a"].expression.this, exp.Sqrt)

    def test_l1_formula_structure(self) -> None:
        """L1: col / NULLIF(ABS(x1) + ABS(x2), 0)."""
        norm = self._make_normalizer("l1")
        exprs = {"a": exp.Column(this="a"), "b": exp.Column(this="b")}
        result = norm.expressions(["a", "b"], exprs)
        assert isinstance(result["a"], exp.Div)
        nullif = result["a"].expression
        assert isinstance(nullif, exp.Nullif)
        # Inside Nullif is Add of Abs terms
        inner = nullif.this
        assert isinstance(inner, exp.Add)

    def test_max_formula_structure(self) -> None:
        """Max: col / NULLIF(GREATEST(ABS(x1), ABS(x2)), 0)."""
        norm = self._make_normalizer("max")
        exprs = {"a": exp.Column(this="a"), "b": exp.Column(this="b")}
        result = norm.expressions(["a", "b"], exprs)
        assert isinstance(result["a"], exp.Div)
        nullif = result["a"].expression
        assert isinstance(nullif, exp.Nullif)
        # Inside Nullif is Greatest
        assert isinstance(nullif.this, exp.Greatest)

    def test_zero_norm_safe_division(self) -> None:
        """All norms use NULLIF(norm, 0) for safe division."""
        for n in ("l1", "l2", "max"):
            norm = self._make_normalizer(n)
            exprs = {"a": exp.Column(this="a"), "b": exp.Column(this="b")}
            result = norm.expressions(["a", "b"], exprs)
            nullif = result["a"].expression
            assert isinstance(nullif, exp.Nullif), f"norm={n} missing NULLIF"

    def test_empty_columns_returns_empty(self) -> None:
        """Empty columns list returns empty dict."""
        norm = self._make_normalizer()
        exprs = {"a": exp.Column(this="a")}
        result = norm.expressions([], exprs)
        assert result == {}

    def test_uses_exprs_not_column(self) -> None:
        """expressions() uses exprs[col], not exp.Column(this=col)."""
        norm = self._make_normalizer("l2")
        # Simulate a prior transform: a is already (a * 2)
        prior = exp.Mul(this=exp.Column(this="a"), expression=exp.Literal.number(2))
        exprs = {"a": prior, "b": exp.Column(this="b")}
        result = norm.expressions(["a", "b"], exprs)
        # Numerator for 'a' should be the prior Mul expression
        assert isinstance(result["a"].this, exp.Mul)

    def test_multiple_columns_all_present(self) -> None:
        """All target columns appear in result."""
        norm = self._make_normalizer()
        exprs = {
            "a": exp.Column(this="a"),
            "b": exp.Column(this="b"),
            "c": exp.Column(this="c"),
        }
        result = norm.expressions(["a", "b", "c"], exprs)
        assert "a" in result
        assert "b" in result
        assert "c" in result

    def test_only_modifies_target_columns(self) -> None:
        """Columns not in target list are not in result."""
        norm = self._make_normalizer()
        exprs = {
            "a": exp.Column(this="a"),
            "b": exp.Column(this="b"),
            "c": exp.Column(this="c"),
        }
        result = norm.expressions(["a"], exprs)
        assert "a" in result
        assert "b" not in result
        assert "c" not in result

    def test_l2_sqrt_wraps_sum_of_squares(self) -> None:
        """L2 norm has Sqrt wrapping Add of Mul terms."""
        norm = self._make_normalizer("l2")
        exprs = {"a": exp.Column(this="a"), "b": exp.Column(this="b")}
        result = norm.expressions(["a", "b"], exprs)
        sqrt_expr = result["a"].expression.this
        assert isinstance(sqrt_expr, exp.Sqrt)
        # Inside Sqrt is Add of Mul terms
        inner = sqrt_expr.this
        assert isinstance(inner, exp.Add)

    def test_l1_abs_terms(self) -> None:
        """L1 norm uses ABS for each column."""
        norm = self._make_normalizer("l1")
        exprs = {"a": exp.Column(this="a"), "b": exp.Column(this="b")}
        result = norm.expressions(["a", "b"], exprs)
        # Walk the Add tree — one side should be Abs
        inner = result["a"].expression.this  # Nullif -> Add
        assert isinstance(inner, exp.Add)
        assert isinstance(inner.this, exp.Abs) or isinstance(inner.expression, exp.Abs)

    def test_max_greatest_with_abs(self) -> None:
        """Max norm uses GREATEST with ABS terms."""
        norm = self._make_normalizer("max")
        exprs = {"a": exp.Column(this="a"), "b": exp.Column(this="b")}
        result = norm.expressions(["a", "b"], exprs)
        greatest = result["a"].expression.this
        assert isinstance(greatest, exp.Greatest)
        assert isinstance(greatest.this, exp.Abs)


# ── Pipeline integration tests ────────────────────────────────────


class TestPipeline:
    """Test Normalizer integrated with Pipeline (end-to-end)."""

    @pytest.fixture
    def backend(self) -> DuckDBBackend:
        """Create DuckDB backend with numeric test data."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (3.0, 4.0), (1.0, 0.0), (0.0, 5.0), "
            "(6.0, 8.0) t(a, b)"
        )
        return DuckDBBackend(connection=conn)

    def test_fit_transform_l2_row_norms(self, backend: DuckDBBackend) -> None:
        """L2-normalized rows should have Euclidean norm ~1.0."""
        pipe = Pipeline([Normalizer(norm="l2")], backend=backend)
        result = pipe.fit_transform("t")
        for i in range(result.shape[0]):
            row_norm = np.linalg.norm(result[i, :])
            np.testing.assert_allclose(row_norm, 1.0, atol=1e-10)

    def test_fit_transform_l1_row_norms(self, backend: DuckDBBackend) -> None:
        """L1-normalized rows should have L1 norm ~1.0."""
        pipe = Pipeline([Normalizer(norm="l1")], backend=backend)
        result = pipe.fit_transform("t")
        for i in range(result.shape[0]):
            row_norm = np.sum(np.abs(result[i, :]))
            np.testing.assert_allclose(row_norm, 1.0, atol=1e-10)

    def test_fit_transform_max_row_norms(self, backend: DuckDBBackend) -> None:
        """Max-normalized rows should have max abs value ~1.0."""
        pipe = Pipeline([Normalizer(norm="max")], backend=backend)
        result = pipe.fit_transform("t")
        for i in range(result.shape[0]):
            row_max = np.max(np.abs(result[i, :]))
            np.testing.assert_allclose(row_max, 1.0, atol=1e-10)

    def test_fit_transform_shape(self, backend: DuckDBBackend) -> None:
        """Output shape matches input shape."""
        pipe = Pipeline([Normalizer()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (4, 2)

    def test_to_sql_valid(self, backend: DuckDBBackend) -> None:
        """to_sql() produces valid SQL."""
        pipe = Pipeline([Normalizer()], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql()
        assert isinstance(sql, str)
        assert "SELECT" in sql.upper()
        assert "FROM" in sql.upper()

    def test_to_sql_contains_nullif(self, backend: DuckDBBackend) -> None:
        """to_sql() output contains NULLIF for safe division."""
        pipe = Pipeline([Normalizer()], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql()
        assert "NULLIF" in sql.upper()

    def test_to_sql_l2_contains_sqrt(self, backend: DuckDBBackend) -> None:
        """L2 to_sql() output contains SQRT."""
        pipe = Pipeline([Normalizer(norm="l2")], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql()
        assert "SQRT" in sql.upper()

    def test_fit_then_transform_separate(self, backend: DuckDBBackend) -> None:
        """Separate fit() and transform() produce same result as fit_transform()."""
        pipe1 = Pipeline([Normalizer()], backend=backend)
        result1 = pipe1.fit_transform("t")

        pipe2 = Pipeline([Normalizer()], backend=backend)
        pipe2.fit("t")
        result2 = pipe2.transform("t")

        np.testing.assert_array_equal(result1, result2)


# ── sklearn equivalence tests ───────────────────────────────────


class TestSklearnEquivalence:
    """Compare sqlearn Normalizer output to sklearn."""

    def _run_sqlearn(self, data: np.ndarray, norm: str) -> np.ndarray:
        """Run sqlearn Normalizer on numpy data via DuckDB."""
        n_cols = data.shape[1]
        col_names = [f"c{i}" for i in range(n_cols)]
        col_defs = ", ".join(f"{c} DOUBLE" for c in col_names)

        conn = duckdb.connect()
        conn.execute(f"CREATE TABLE t ({col_defs})")
        for row in data:
            placeholders = ", ".join(["?"] * n_cols)
            conn.execute(f"INSERT INTO t VALUES ({placeholders})", [float(x) for x in row])  # noqa: S608

        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Normalizer(norm=norm)], backend=backend)
        return np.array(pipe.fit_transform("t"), dtype=float)

    def test_l2_equivalence(self) -> None:
        """sqlearn matches sklearn for L2 norm."""
        from sklearn.preprocessing import Normalizer as SklearnNormalizer

        data = np.array([[3.0, 4.0], [1.0, 2.0], [5.0, 0.0], [0.6, 0.8]])
        sk_result = SklearnNormalizer(norm="l2").fit_transform(data)
        sq_result = self._run_sqlearn(data, "l2")
        np.testing.assert_allclose(sq_result, sk_result, atol=1e-10)

    def test_l1_equivalence(self) -> None:
        """sqlearn matches sklearn for L1 norm."""
        from sklearn.preprocessing import Normalizer as SklearnNormalizer

        data = np.array([[3.0, 4.0], [1.0, 2.0], [5.0, 0.0], [0.6, 0.8]])
        sk_result = SklearnNormalizer(norm="l1").fit_transform(data)
        sq_result = self._run_sqlearn(data, "l1")
        np.testing.assert_allclose(sq_result, sk_result, atol=1e-10)

    def test_max_equivalence(self) -> None:
        """sqlearn matches sklearn for max norm."""
        from sklearn.preprocessing import Normalizer as SklearnNormalizer

        data = np.array([[3.0, 4.0], [1.0, 2.0], [5.0, 0.0], [0.6, 0.8]])
        sk_result = SklearnNormalizer(norm="max").fit_transform(data)
        sq_result = self._run_sqlearn(data, "max")
        np.testing.assert_allclose(sq_result, sk_result, atol=1e-10)

    def test_multicolumn_equivalence(self) -> None:
        """sqlearn matches sklearn for many columns with varied values."""
        from sklearn.preprocessing import Normalizer as SklearnNormalizer

        rng = np.random.default_rng(42)
        data = rng.standard_normal((50, 5)) * np.array([1, 10, 100, 0.1, 50])

        for norm in ("l1", "l2", "max"):
            sk_result = SklearnNormalizer(norm=norm).fit_transform(data)
            sq_result = self._run_sqlearn(data, norm)
            np.testing.assert_allclose(sq_result, sk_result, atol=1e-10, err_msg=f"norm={norm}")


# ── Clone and pickle tests ──────────────────────────────────────


class TestClonePickle:
    """Deep clone independence and pickle roundtrip."""

    def test_clone_roundtrip(self) -> None:
        """Cloned Normalizer has same params but is independent."""
        norm = Normalizer(norm="l1", columns=["x", "y"])
        cloned = norm.clone()
        assert cloned.norm == "l1"
        assert cloned.columns == ["x", "y"]
        assert cloned is not norm

    def test_clone_independence(self) -> None:
        """Modifying clone does not affect original."""
        norm = Normalizer(norm="l1", columns=["x", "y"])
        cloned = norm.clone()
        cloned.set_params(norm="max")
        assert norm.norm == "l1"
        assert cloned.norm == "max"

    def test_pickle_roundtrip(self) -> None:
        """Pickle a Normalizer preserves params."""
        norm = Normalizer(norm="max", columns=["a", "b"])
        data = pickle.dumps(norm)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.norm == "max"
        assert restored.columns == ["a", "b"]

    def test_pickle_roundtrip_default(self) -> None:
        """Pickle a default Normalizer preserves defaults."""
        norm = Normalizer()
        data = pickle.dumps(norm)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.norm == "l2"
        assert restored.columns is None


# ── Edge cases ──────────────────────────────────────────────────


class TestEdgeCases:
    """Extreme and unusual inputs."""

    def test_single_column(self) -> None:
        """Single column: norm is the column itself, result is 1 or -1."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (3.0), (5.0), (-2.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Normalizer(norm="l2")], backend=backend)
        result = pipe.fit_transform("t")
        # Single column L2: x / |x| = sign(x), i.e. 1.0 or -1.0
        expected = np.array([[1.0], [1.0], [-1.0]])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_all_zero_row(self) -> None:
        """All-zero row produces NULL from NULLIF (norm=0)."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (0.0, 0.0), (3.0, 4.0) t(a, b)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Normalizer(norm="l2")], backend=backend)
        result = pipe.fit_transform("t")
        # Zero row should be None/NaN (NULLIF returns NULL)
        assert result.shape == (2, 2)
        assert result[0, 0] is None or (isinstance(result[0, 0], float) and np.isnan(result[0, 0]))
        # Non-zero row should normalize correctly
        np.testing.assert_allclose(float(result[1, 0]), 0.6, atol=1e-10)
        np.testing.assert_allclose(float(result[1, 1]), 0.8, atol=1e-10)

    def test_single_row(self) -> None:
        """Single row normalizes correctly."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (3.0, 4.0) t(a, b)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Normalizer(norm="l2")], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (1, 2)
        np.testing.assert_allclose(float(result[0, 0]), 0.6, atol=1e-10)
        np.testing.assert_allclose(float(result[0, 1]), 0.8, atol=1e-10)

    def test_negative_values(self) -> None:
        """Negative values handled correctly."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (-3.0, -4.0) t(a, b)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Normalizer(norm="l2")], backend=backend)
        result = pipe.fit_transform("t")
        # L2 norm of (-3, -4) = 5, so result = (-0.6, -0.8)
        np.testing.assert_allclose(float(result[0, 0]), -0.6, atol=1e-10)
        np.testing.assert_allclose(float(result[0, 1]), -0.8, atol=1e-10)

    def test_large_values(self) -> None:
        """Very large values don't crash."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1e50, 0.0), (0.0, 1e50) t(a, b)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Normalizer(norm="l2")], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (2, 2)
        # Row (1e50, 0) -> (1.0, 0.0)
        np.testing.assert_allclose(float(result[0, 0]), 1.0, atol=1e-10)
        np.testing.assert_allclose(float(result[0, 1]), 0.0, atol=1e-10)


# ── Composition tests ───────────────────────────────────────────


class TestComposition:
    """Normalizer composing with other transformers."""

    def test_standard_scaler_then_normalizer(self) -> None:
        """StandardScaler + Normalizer produces valid normalized rows."""
        from sqlearn.scalers.standard import StandardScaler

        conn = duckdb.connect()
        # Use uncorrelated data so no row becomes (0, 0) after scaling
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 50.0), (2.0, 10.0), (3.0, 40.0), "
            "(4.0, 20.0), (5.0, 30.0) t(a, b)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([StandardScaler(), Normalizer()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (5, 2)
        # After L2 normalization, each row should have unit norm
        for i in range(result.shape[0]):
            row_norm = np.linalg.norm(result[i, :].astype(float))
            np.testing.assert_allclose(row_norm, 1.0, atol=1e-10)

    def test_scaler_normalizer_sql_nesting(self) -> None:
        """SQL shows nested StandardScaler expressions inside Normalizer."""
        from sqlearn.scalers.standard import StandardScaler

        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES (1.0, 30.0), (2.0, 10.0), (3.0, 20.0) t(a, b)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([StandardScaler(), Normalizer()], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql().upper()
        # Both NULLIF from StandardScaler and Normalizer should be present
        assert "NULLIF" in sql
        assert "SQRT" in sql
