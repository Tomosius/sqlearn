"""Tests for sqlearn.features.target -- TargetTransform."""

from __future__ import annotations

import pickle
from typing import Any

import duckdb
import numpy as np
import pytest
import sqlglot.expressions as exp

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.pipeline import Pipeline
from sqlearn.core.schema import Schema
from sqlearn.features.target import TargetTransform

# -- Constructor tests --------------------------------------------------------


class TestConstructor:
    """Test TargetTransform constructor and attributes."""

    def test_defaults(self) -> None:
        """Default method=log, lambda_=auto, columns=None."""
        transform = TargetTransform()
        assert transform.method == "log"
        assert transform.lambda_ == "auto"
        assert transform.columns is None

    def test_sqrt_method(self) -> None:
        """Sqrt method accepted."""
        transform = TargetTransform(method="sqrt")
        assert transform.method == "sqrt"

    def test_boxcox_method(self) -> None:
        """Box-Cox method accepted."""
        transform = TargetTransform(method="boxcox")
        assert transform.method == "boxcox"

    def test_invalid_method_raises(self) -> None:
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be"):
            TargetTransform(method="invalid")

    def test_custom_lambda(self) -> None:
        """Custom lambda_ value accepted."""
        transform = TargetTransform(method="boxcox", lambda_=0.5)
        assert transform.lambda_ == 0.5

    def test_lambda_zero(self) -> None:
        """Lambda zero is valid (equivalent to log transform)."""
        transform = TargetTransform(method="boxcox", lambda_=0.0)
        assert transform.lambda_ == 0.0

    def test_custom_columns(self) -> None:
        """Custom columns accepted."""
        transform = TargetTransform(columns=["price"])
        assert transform.columns == ["price"]

    def test_default_columns_is_none(self) -> None:
        """Class default columns is None (requires explicit)."""
        assert TargetTransform._default_columns is None

    def test_get_params(self) -> None:
        """get_params returns constructor kwargs."""
        transform = TargetTransform(method="sqrt", columns=["price"])
        params = transform.get_params()
        assert params == {"method": "sqrt", "lambda_": "auto", "columns": ["price"]}

    def test_set_params(self) -> None:
        """set_params updates attributes and returns self."""
        transform = TargetTransform()
        result = transform.set_params(method="sqrt")
        assert result is transform
        assert transform.method == "sqrt"


# -- Classification tests ----------------------------------------------------


class TestClassification:
    """Test TargetTransform._classification property."""

    def test_log_is_static(self) -> None:
        """Log method is static (no data learning)."""
        transform = TargetTransform(method="log")
        assert transform._classification == "static"

    def test_sqrt_is_static(self) -> None:
        """Sqrt method is static (no data learning)."""
        transform = TargetTransform(method="sqrt")
        assert transform._classification == "static"

    def test_boxcox_auto_is_dynamic(self) -> None:
        """Box-Cox with auto lambda is dynamic."""
        transform = TargetTransform(method="boxcox")
        assert transform._classification == "dynamic"

    def test_boxcox_fixed_is_static(self) -> None:
        """Box-Cox with fixed lambda is static."""
        transform = TargetTransform(method="boxcox", lambda_=0.5)
        assert transform._classification == "static"

    def test_boxcox_zero_lambda_is_static(self) -> None:
        """Box-Cox with lambda=0 is static."""
        transform = TargetTransform(method="boxcox", lambda_=0.0)
        assert transform._classification == "static"


# -- discover() tests --------------------------------------------------------


class TestDiscover:
    """Test TargetTransform.discover() for Box-Cox auto lambda."""

    @pytest.fixture
    def schema(self) -> Schema:
        """Simple numeric schema."""
        return Schema({"price": "DOUBLE"})

    def test_log_discover_empty(self, schema: Schema) -> None:
        """Log method returns empty dict (no learning needed)."""
        transform = TargetTransform(method="log")
        result = transform.discover(["price"], schema)
        assert result == {}

    def test_sqrt_discover_empty(self, schema: Schema) -> None:
        """Sqrt method returns empty dict (no learning needed)."""
        transform = TargetTransform(method="sqrt")
        result = transform.discover(["price"], schema)
        assert result == {}

    def test_boxcox_fixed_discover_empty(self, schema: Schema) -> None:
        """Box-Cox with fixed lambda returns empty dict."""
        transform = TargetTransform(method="boxcox", lambda_=0.5)
        result = transform.discover(["price"], schema)
        assert result == {}

    def test_boxcox_auto_discover_log_mean(self, schema: Schema) -> None:
        """Box-Cox with auto lambda discovers log mean."""
        transform = TargetTransform(method="boxcox")
        result = transform.discover(["price"], schema)
        assert "price__log_mean" in result
        assert len(result) == 1

    def test_boxcox_auto_log_mean_ast_type(self, schema: Schema) -> None:
        """Log mean expression is AVG(LN(col))."""
        transform = TargetTransform(method="boxcox")
        result = transform.discover(["price"], schema)
        avg_expr = result["price__log_mean"]
        assert isinstance(avg_expr, exp.Avg)
        assert isinstance(avg_expr.this, exp.Ln)

    def test_boxcox_auto_multiple_columns(self, schema: Schema) -> None:
        """Box-Cox auto with multiple columns discovers per column."""
        schema = Schema({"price": "DOUBLE", "quantity": "DOUBLE"})
        transform = TargetTransform(method="boxcox")
        result = transform.discover(["price", "quantity"], schema)
        assert "price__log_mean" in result
        assert "quantity__log_mean" in result
        assert len(result) == 2

    def test_boxcox_auto_empty_columns(self, schema: Schema) -> None:
        """Empty columns list returns empty dict."""
        transform = TargetTransform(method="boxcox")
        result = transform.discover([], schema)
        assert result == {}


# -- expressions() tests (log) ------------------------------------------------


class TestExpressionsLog:
    """Test TargetTransform.expressions() for log method."""

    def test_log_expression(self) -> None:
        """Log produces LN(col + 1)."""
        transform = TargetTransform(method="log")
        exprs = {"price": exp.Column(this="price")}
        result = transform.expressions(["price"], exprs)
        assert "price" in result
        assert isinstance(result["price"], exp.Ln)

    def test_log_has_add_one(self) -> None:
        """Log expression adds 1 to the column for zero safety."""
        transform = TargetTransform(method="log")
        exprs = {"price": exp.Column(this="price")}
        result = transform.expressions(["price"], exprs)
        inner = result["price"].this
        assert isinstance(inner, exp.Paren)
        assert isinstance(inner.this, exp.Add)

    def test_log_uses_exprs_not_column(self) -> None:
        """expressions() uses exprs[col] for composition."""
        transform = TargetTransform(method="log")
        prior = exp.Mul(this=exp.Column(this="price"), expression=exp.Literal.number(2))
        exprs = {"price": prior}
        result = transform.expressions(["price"], exprs)
        # LN((price * 2) + 1) -- inner Add should have Mul as first arg
        inner_add = result["price"].this.this  # Ln -> Paren -> Add
        assert isinstance(inner_add.this, exp.Mul)


# -- expressions() tests (sqrt) -----------------------------------------------


class TestExpressionsSqrt:
    """Test TargetTransform.expressions() for sqrt method."""

    def test_sqrt_expression(self) -> None:
        """Sqrt produces SQRT(col)."""
        transform = TargetTransform(method="sqrt")
        exprs = {"price": exp.Column(this="price")}
        result = transform.expressions(["price"], exprs)
        assert "price" in result
        assert isinstance(result["price"], exp.Sqrt)

    def test_sqrt_wraps_column(self) -> None:
        """Sqrt wraps the column expression."""
        transform = TargetTransform(method="sqrt")
        exprs = {"price": exp.Column(this="price")}
        result = transform.expressions(["price"], exprs)
        inner = result["price"].this
        assert isinstance(inner, exp.Column)
        assert inner.name == "price"


# -- expressions() tests (boxcox) ---------------------------------------------


class TestExpressionsBoxCox:
    """Test TargetTransform.expressions() for Box-Cox method."""

    def _make_fitted_transform(
        self,
        params: dict[str, Any],
        *,
        lambda_: str | float = "auto",
    ) -> TargetTransform:
        """Create a fitted TargetTransform with given params."""
        transform = TargetTransform(method="boxcox", lambda_=lambda_)
        transform.params_ = params
        transform._fitted = True
        return transform

    def test_boxcox_fixed_nonzero(self) -> None:
        """Box-Cox with nonzero lambda: (POW(col, lambda) - 1) / lambda."""
        transform = self._make_fitted_transform({}, lambda_=0.5)
        exprs = {"price": exp.Column(this="price")}
        result = transform.expressions(["price"], exprs)
        assert "price" in result
        # Top level is Div
        assert isinstance(result["price"], exp.Div)

    def test_boxcox_fixed_zero(self) -> None:
        """Box-Cox with lambda=0: LN(col)."""
        transform = self._make_fitted_transform({}, lambda_=0.0)
        exprs = {"price": exp.Column(this="price")}
        result = transform.expressions(["price"], exprs)
        assert "price" in result
        assert isinstance(result["price"], exp.Ln)

    def test_boxcox_nonzero_has_pow(self) -> None:
        """Box-Cox with nonzero lambda contains POW."""
        transform = self._make_fitted_transform({}, lambda_=0.5)
        exprs = {"price": exp.Column(this="price")}
        result = transform.expressions(["price"], exprs)
        # Div -> Paren -> Sub -> Pow
        pow_expr = result["price"].this.this.this  # Div -> Paren -> Sub -> Pow
        assert isinstance(pow_expr, exp.Pow)

    def test_boxcox_auto_with_near_zero_log_mean(self) -> None:
        """Auto lambda with near-zero log mean produces LN(col)."""
        transform = self._make_fitted_transform({"price__log_mean": 0.05})
        exprs = {"price": exp.Column(this="price")}
        result = transform.expressions(["price"], exprs)
        assert isinstance(result["price"], exp.Ln)

    def test_boxcox_auto_with_nonzero_log_mean(self) -> None:
        """Auto lambda with nonzero log mean produces pow transform."""
        transform = self._make_fitted_transform({"price__log_mean": 2.0})
        exprs = {"price": exp.Column(this="price")}
        result = transform.expressions(["price"], exprs)
        assert isinstance(result["price"], exp.Div)

    def test_boxcox_multiple_columns(self) -> None:
        """Multiple columns each get their own expression."""
        transform = self._make_fitted_transform({}, lambda_=0.5)
        exprs = {
            "price": exp.Column(this="price"),
            "quantity": exp.Column(this="quantity"),
        }
        result = transform.expressions(["price", "quantity"], exprs)
        assert "price" in result
        assert "quantity" in result

    def test_boxcox_only_modifies_target_columns(self) -> None:
        """Columns not in target list are not in result."""
        transform = self._make_fitted_transform({}, lambda_=0.5)
        exprs = {
            "price": exp.Column(this="price"),
            "other": exp.Column(this="other"),
        }
        result = transform.expressions(["price"], exprs)
        assert "price" in result
        assert "other" not in result


# -- _get_lambda() tests ------------------------------------------------------


class TestGetLambda:
    """Test TargetTransform._get_lambda() method."""

    def test_fixed_int_lambda(self) -> None:
        """Integer lambda returned as float."""
        transform = TargetTransform(method="boxcox", lambda_=1)
        assert transform._get_lambda("price") == 1.0

    def test_fixed_float_lambda(self) -> None:
        """Float lambda returned as-is."""
        transform = TargetTransform(method="boxcox", lambda_=0.5)
        assert transform._get_lambda("price") == 0.5

    def test_auto_near_zero_log_mean(self) -> None:
        """Auto lambda with near-zero log mean returns 0."""
        transform = TargetTransform(method="boxcox")
        transform.params_ = {"price__log_mean": 0.05}
        assert transform._get_lambda("price") == 0.0

    def test_auto_nonzero_log_mean(self) -> None:
        """Auto lambda with nonzero log mean returns positive value."""
        transform = TargetTransform(method="boxcox")
        transform.params_ = {"price__log_mean": 2.0}
        lam = transform._get_lambda("price")
        assert lam > 0
        assert lam < 1


# -- __repr__ tests -----------------------------------------------------------


class TestRepr:
    """Test TargetTransform.__repr__."""

    def test_default_repr(self) -> None:
        """Default params shows no args."""
        transform = TargetTransform()
        assert repr(transform) == "TargetTransform()"

    def test_sqrt_repr(self) -> None:
        """Sqrt method shows in repr."""
        transform = TargetTransform(method="sqrt")
        assert "method='sqrt'" in repr(transform)

    def test_boxcox_fixed_repr(self) -> None:
        """Box-Cox with fixed lambda shows in repr."""
        transform = TargetTransform(method="boxcox", lambda_=0.5)
        r = repr(transform)
        assert "method='boxcox'" in r
        assert "lambda_=0.5" in r

    def test_columns_repr(self) -> None:
        """Columns show in repr."""
        transform = TargetTransform(columns=["price"])
        assert "columns=['price']" in repr(transform)


# -- Pipeline integration tests (log) -----------------------------------------


class TestPipelineLog:
    """Test TargetTransform log method with Pipeline (end-to-end)."""

    @pytest.fixture
    def backend(self) -> DuckDBBackend:
        """Create DuckDB backend with positive numeric data."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES (1.0), (2.0), (3.0), (4.0), (5.0) t(price)"
        )
        return DuckDBBackend(connection=conn)

    def test_log_shape(self, backend: DuckDBBackend) -> None:
        """Log transform preserves shape."""
        pipe = Pipeline([TargetTransform(columns=["price"])], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (5, 1)

    def test_log_values_correct(self, backend: DuckDBBackend) -> None:
        """Log transform values match LN(x+1)."""
        pipe = Pipeline([TargetTransform(columns=["price"])], backend=backend)
        result = pipe.fit_transform("t")
        expected = np.log(np.array([1.0, 2.0, 3.0, 4.0, 5.0]) + 1)
        np.testing.assert_allclose(result[:, 0].astype(float), expected, atol=1e-10)

    def test_log_to_sql_contains_ln(self, backend: DuckDBBackend) -> None:
        """Log SQL contains LN function."""
        pipe = Pipeline([TargetTransform(columns=["price"])], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql()
        assert "LN" in sql.upper()

    def test_log_to_sql_valid(self, backend: DuckDBBackend) -> None:
        """Log SQL is valid and parseable."""
        pipe = Pipeline([TargetTransform(columns=["price"])], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql()
        assert isinstance(sql, str)
        assert "SELECT" in sql.upper()

    def test_log_zero_safe(self) -> None:
        """Log handles zero values via +1 offset."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (0.0), (1.0), (2.0) t(price)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([TargetTransform(columns=["price"])], backend=backend)
        result = pipe.fit_transform("t")
        # LN(0+1) = 0
        assert result[0, 0] == pytest.approx(0.0, abs=1e-10)


# -- Pipeline integration tests (sqrt) ----------------------------------------


class TestPipelineSqrt:
    """Test TargetTransform sqrt method with Pipeline."""

    @pytest.fixture
    def backend(self) -> DuckDBBackend:
        """Create DuckDB backend with positive numeric data."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES (1.0), (4.0), (9.0), (16.0), (25.0) t(price)"
        )
        return DuckDBBackend(connection=conn)

    def test_sqrt_values_correct(self, backend: DuckDBBackend) -> None:
        """Sqrt transform values match SQRT(x)."""
        pipe = Pipeline([TargetTransform(method="sqrt", columns=["price"])], backend=backend)
        result = pipe.fit_transform("t")
        expected = np.sqrt([1.0, 4.0, 9.0, 16.0, 25.0])
        np.testing.assert_allclose(result[:, 0].astype(float), expected, atol=1e-10)

    def test_sqrt_shape(self, backend: DuckDBBackend) -> None:
        """Sqrt preserves shape."""
        pipe = Pipeline([TargetTransform(method="sqrt", columns=["price"])], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (5, 1)

    def test_sqrt_to_sql_contains_sqrt(self, backend: DuckDBBackend) -> None:
        """Sqrt SQL contains SQRT function."""
        pipe = Pipeline([TargetTransform(method="sqrt", columns=["price"])], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql()
        assert "SQRT" in sql.upper()


# -- Pipeline integration tests (boxcox) --------------------------------------


class TestPipelineBoxCox:
    """Test TargetTransform Box-Cox method with Pipeline."""

    @pytest.fixture
    def backend(self) -> DuckDBBackend:
        """Create DuckDB backend with positive numeric data."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES (1.0), (2.0), (3.0), (4.0), (5.0) t(price)"
        )
        return DuckDBBackend(connection=conn)

    def test_boxcox_fixed_shape(self, backend: DuckDBBackend) -> None:
        """Box-Cox with fixed lambda preserves shape."""
        pipe = Pipeline(
            [TargetTransform(method="boxcox", lambda_=0.5, columns=["price"])],
            backend=backend,
        )
        result = pipe.fit_transform("t")
        assert result.shape == (5, 1)

    def test_boxcox_fixed_values(self, backend: DuckDBBackend) -> None:
        """Box-Cox with lambda=0.5: (x^0.5 - 1) / 0.5."""
        pipe = Pipeline(
            [TargetTransform(method="boxcox", lambda_=0.5, columns=["price"])],
            backend=backend,
        )
        result = pipe.fit_transform("t")
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = (data**0.5 - 1) / 0.5
        np.testing.assert_allclose(result[:, 0].astype(float), expected, atol=1e-10)

    def test_boxcox_lambda_zero(self, backend: DuckDBBackend) -> None:
        """Box-Cox with lambda=0 reduces to LN(col)."""
        pipe = Pipeline(
            [TargetTransform(method="boxcox", lambda_=0.0, columns=["price"])],
            backend=backend,
        )
        result = pipe.fit_transform("t")
        expected = np.log([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_allclose(result[:, 0].astype(float), expected, atol=1e-10)

    def test_boxcox_auto_shape(self, backend: DuckDBBackend) -> None:
        """Box-Cox auto lambda preserves shape."""
        pipe = Pipeline(
            [TargetTransform(method="boxcox", columns=["price"])],
            backend=backend,
        )
        result = pipe.fit_transform("t")
        assert result.shape == (5, 1)

    def test_boxcox_auto_params_populated(self, backend: DuckDBBackend) -> None:
        """After fit, params_ contains log_mean for boxcox auto."""
        transform = TargetTransform(method="boxcox", columns=["price"])
        pipe = Pipeline([transform], backend=backend)
        pipe.fit("t")
        assert transform.params_ is not None
        assert "price__log_mean" in transform.params_

    def test_boxcox_to_sql_contains_pow(self, backend: DuckDBBackend) -> None:
        """Box-Cox SQL contains POW for nonzero lambda."""
        pipe = Pipeline(
            [TargetTransform(method="boxcox", lambda_=0.5, columns=["price"])],
            backend=backend,
        )
        pipe.fit("t")
        sql = pipe.to_sql()
        assert "POW" in sql.upper()


# -- Not-fitted guard tests ---------------------------------------------------


class TestNotFittedGuard:
    """Calling transform/to_sql before fit() raises NotFittedError."""

    def test_transform_before_fit_raises(self) -> None:
        """transform() on unfitted pipeline raises NotFittedError."""
        from sqlearn.core.errors import NotFittedError

        pipe = Pipeline([TargetTransform(columns=["price"])])
        with pytest.raises(NotFittedError):
            pipe.transform("t")


# -- Clone and pickle tests ---------------------------------------------------


class TestClonePickle:
    """Deep clone independence and pickle roundtrip."""

    @pytest.fixture
    def fitted_pipe(self) -> Pipeline:
        """Create a fitted pipeline for clone/pickle tests."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES (1.0), (2.0), (3.0), (4.0), (5.0) t(price)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([TargetTransform(columns=["price"])], backend=backend)
        pipe.fit("t")
        return pipe

    def test_clone_produces_same_sql(self, fitted_pipe: Pipeline) -> None:
        """Cloned pipeline produces identical SQL."""
        cloned = fitted_pipe.clone()
        assert fitted_pipe.to_sql() == cloned.to_sql()

    def test_clone_params_independent(self, fitted_pipe: Pipeline) -> None:
        """Cloned step is independent of original."""
        cloned = fitted_pipe.clone()
        original_transform = fitted_pipe.steps[0][1]
        cloned_transform = cloned.steps[0][1]
        assert cloned_transform is not original_transform

    def test_pickle_roundtrip_log(self) -> None:
        """Pickle a log TargetTransform preserves state."""
        transform = TargetTransform(method="log", columns=["price"])
        transform._fitted = True
        data = pickle.dumps(transform)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.method == "log"
        assert restored.columns == ["price"]
        assert restored._fitted is True

    def test_pickle_roundtrip_boxcox(self) -> None:
        """Pickle a boxcox TargetTransform preserves params."""
        transform = TargetTransform(method="boxcox", columns=["price"])
        transform.params_ = {"price__log_mean": 1.5}
        transform._fitted = True
        data = pickle.dumps(transform)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.params_ == {"price__log_mean": 1.5}
        assert restored.method == "boxcox"

    def test_pickle_roundtrip_fixed_lambda(self) -> None:
        """Pickle a fixed-lambda boxcox preserves lambda."""
        transform = TargetTransform(method="boxcox", lambda_=0.5, columns=["price"])
        transform._fitted = True
        data = pickle.dumps(transform)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.lambda_ == 0.5


# -- Composition tests -------------------------------------------------------


class TestComposition:
    """TargetTransform composing with other transformers."""

    def test_log_then_scaler(self) -> None:
        """Log + StandardScaler produces valid results."""
        from sqlearn.scalers.standard import StandardScaler

        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 10.0), (2.0, 20.0), (3.0, 30.0), "
            "(4.0, 40.0), (5.0, 50.0) t(price, quantity)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [TargetTransform(columns=["price"]), StandardScaler()],
            backend=backend,
        )
        result = pipe.fit_transform("t")
        assert result.shape == (5, 2)

    def test_imputer_then_log(self) -> None:
        """Imputer + Log handles NULL values."""
        from sqlearn.imputers.imputer import Imputer

        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES (1.0), (NULL), (3.0), (4.0), (5.0) t(price)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [Imputer(columns=["price"]), TargetTransform(columns=["price"])],
            backend=backend,
        )
        result = pipe.fit_transform("t")
        assert result.shape == (5, 1)
        assert not np.any(np.isnan(result.astype(float)))


# -- Edge cases ---------------------------------------------------------------


class TestEdgeCases:
    """Extreme and unusual inputs."""

    def test_single_row_log(self) -> None:
        """Single-row data works for log."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (42.0) t(price)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([TargetTransform(columns=["price"])], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (1, 1)
        np.testing.assert_allclose(result[0, 0], np.log(43.0), atol=1e-10)

    def test_single_row_sqrt(self) -> None:
        """Single-row data works for sqrt."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (25.0) t(price)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([TargetTransform(method="sqrt", columns=["price"])], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (1, 1)
        np.testing.assert_allclose(result[0, 0], 5.0, atol=1e-10)

    def test_fit_then_transform_separate(self) -> None:
        """Separate fit() and transform() produce same result."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES (1.0), (2.0), (3.0), (4.0), (5.0) t(price)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe1 = Pipeline([TargetTransform(columns=["price"])], backend=backend)
        result1 = pipe1.fit_transform("t")

        pipe2 = Pipeline([TargetTransform(columns=["price"])], backend=backend)
        pipe2.fit("t")
        result2 = pipe2.transform("t")

        np.testing.assert_array_equal(result1, result2)

    def test_large_values_log(self) -> None:
        """Large values don't crash log transform."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1e10), (2e10), (3e10) t(price)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([TargetTransform(columns=["price"])], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (3, 1)
        # Results should be finite
        assert np.all(np.isfinite(result.astype(float)))

    def test_mixed_columns_only_target_transformed(self) -> None:
        """Only specified columns are transformed, others pass through."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 100.0), (2.0, 200.0), (3.0, 300.0) t(price, quantity)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([TargetTransform(columns=["price"])], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (3, 2)
        # quantity should be unchanged
        np.testing.assert_allclose(result[:, 1].astype(float), [100.0, 200.0, 300.0], atol=1e-10)
        # price should be log-transformed: LN(x+1)
        expected_prices = np.log(np.array([1.0, 2.0, 3.0]) + 1)
        np.testing.assert_allclose(result[:, 0].astype(float), expected_prices, atol=1e-10)
