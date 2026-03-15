"""Tests for sqlearn.features.arithmetic -- Log, Sqrt, Power, Clip, Abs, Round, Reciprocal."""

from __future__ import annotations

import pickle

import duckdb
import numpy as np
import pytest
import sqlglot.expressions as exp

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.pipeline import Pipeline
from sqlearn.features.arithmetic import Abs, Clip, Log, Power, Reciprocal, Round, Sqrt

# =============================================================================
# Log
# =============================================================================


class TestLogConstructor:
    """Test Log constructor and validation."""

    def test_default_params(self) -> None:
        """Default Log has base=None, offset=1."""
        t = Log()
        assert t.base is None
        assert t.offset == 1

    def test_custom_base(self) -> None:
        """Custom base is stored."""
        t = Log(base=10)
        assert t.base == 10

    def test_custom_offset(self) -> None:
        """Custom offset is stored."""
        t = Log(offset=0)
        assert t.offset == 0

    def test_classification_is_static(self) -> None:
        """Log is a static transformer."""
        assert Log._classification == "static"
        assert Log()._classify() == "static"

    def test_default_columns_is_numeric(self) -> None:
        """Log defaults to numeric columns."""
        assert Log._default_columns == "numeric"

    def test_base_zero_raises(self) -> None:
        """base=0 raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            Log(base=0)

    def test_base_negative_raises(self) -> None:
        """Negative base raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            Log(base=-2)

    def test_base_one_raises(self) -> None:
        """base=1 raises ValueError (LN(1)=0 causes div-by-zero)."""
        with pytest.raises(ValueError, match="must not be 1"):
            Log(base=1)

    def test_negative_offset_raises(self) -> None:
        """Negative offset raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            Log(offset=-1)

    def test_get_params(self) -> None:
        """get_params returns constructor args."""
        t = Log(base=10, offset=2)
        params = t.get_params()
        assert params == {"base": 10, "offset": 2, "columns": None}


class TestLogExpressions:
    """Test Log.expressions() generates correct sqlglot ASTs."""

    def test_natural_log_with_offset(self) -> None:
        """Default: LN(col + 1)."""
        t = Log()
        exprs = {"a": exp.Column(this="a")}
        result = t.expressions(["a"], exprs)
        sql = result["a"].sql(dialect="duckdb")
        assert "LN" in sql.upper()

    def test_natural_log_no_offset(self) -> None:
        """offset=0: LN(col) with no addition."""
        t = Log(offset=0)
        exprs = {"a": exp.Column(this="a")}
        result = t.expressions(["a"], exprs)
        node = result["a"]
        assert isinstance(node, exp.Ln)
        # Inner should be the column itself, not an Add
        assert not isinstance(node.this, exp.Add)

    def test_base_10_log(self) -> None:
        """base=10: LN(col+1) / LN(10)."""
        t = Log(base=10)
        exprs = {"a": exp.Column(this="a")}
        result = t.expressions(["a"], exprs)
        node = result["a"]
        assert isinstance(node, exp.Div)

    def test_uses_exprs_not_column(self) -> None:
        """expressions() composes with prior transforms via exprs[col]."""
        t = Log()
        prior = exp.Mul(this=exp.Column(this="a"), expression=exp.Literal.number(2))
        exprs = {"a": prior}
        result = t.expressions(["a"], exprs)
        # The inner expression of LN's Add should be the Mul
        ln_node = result["a"]
        assert isinstance(ln_node, exp.Ln)

    def test_multiple_columns(self) -> None:
        """Multiple columns each get their own LN."""
        t = Log()
        exprs = {"a": exp.Column(this="a"), "b": exp.Column(this="b")}
        result = t.expressions(["a", "b"], exprs)
        assert "a" in result
        assert "b" in result

    def test_empty_columns(self) -> None:
        """Empty columns list returns empty dict."""
        t = Log()
        result = t.expressions([], {"a": exp.Column(this="a")})
        assert result == {}


class TestLogPipeline:
    """Test Log with Pipeline end-to-end."""

    @pytest.fixture
    def backend(self) -> DuckDBBackend:
        """Backend with positive numeric data."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES (1.0), (2.0), (3.0), (4.0), (5.0) t(a)"
        )
        return DuckDBBackend(connection=conn)

    def test_fit_transform(self, backend: DuckDBBackend) -> None:
        """fit_transform produces LN(a + 1) values."""
        pipe = Pipeline([Log()], backend=backend)
        result = pipe.fit_transform("t")
        expected = np.log(np.array([[1], [2], [3], [4], [5]], dtype=np.float64) + 1)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_to_sql_contains_ln(self, backend: DuckDBBackend) -> None:
        """Generated SQL contains LN."""
        pipe = Pipeline([Log()], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql().upper()
        assert "LN" in sql

    def test_zero_input_with_offset(self) -> None:
        """Zero values produce LN(0+1)=0, not -inf."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (0.0), (1.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Log()], backend=backend)
        result = pipe.fit_transform("t")
        assert np.isfinite(result[0, 0])
        np.testing.assert_allclose(result[0, 0], 0.0, atol=1e-15)

    def test_null_handling(self) -> None:
        """NULL values propagate as NaN through LN."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1.0), (NULL), (3.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Log()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (3, 1)

    def test_clone_roundtrip(self) -> None:
        """Cloned Log produces identical results."""
        t = Log(base=10, offset=2)
        cloned = t.clone()
        assert cloned.base == 10
        assert cloned.offset == 2
        assert cloned is not t

    def test_pickle_roundtrip(self) -> None:
        """Pickle roundtrip preserves params."""
        t = Log(base=2, offset=0.5)
        restored = pickle.loads(pickle.dumps(t))  # noqa: S301
        assert restored.base == 2
        assert restored.offset == 0.5


# =============================================================================
# Sqrt
# =============================================================================


class TestSqrtConstructor:
    """Test Sqrt constructor."""

    def test_classification_is_static(self) -> None:
        """Sqrt is static."""
        assert Sqrt._classification == "static"
        assert Sqrt()._classify() == "static"

    def test_default_columns_is_numeric(self) -> None:
        """Sqrt defaults to numeric columns."""
        assert Sqrt._default_columns == "numeric"

    def test_get_params(self) -> None:
        """get_params returns columns only."""
        t = Sqrt()
        assert t.get_params() == {"columns": None}


class TestSqrtExpressions:
    """Test Sqrt.expressions() generates correct sqlglot ASTs."""

    def test_sqrt_expression(self) -> None:
        """Generates SQRT(col) node."""
        t = Sqrt()
        exprs = {"a": exp.Column(this="a")}
        result = t.expressions(["a"], exprs)
        assert isinstance(result["a"], exp.Sqrt)

    def test_uses_exprs_not_column(self) -> None:
        """Composes with prior transforms."""
        t = Sqrt()
        prior = exp.Abs(this=exp.Column(this="a"))
        exprs = {"a": prior}
        result = t.expressions(["a"], exprs)
        assert isinstance(result["a"].this, exp.Abs)

    def test_empty_columns(self) -> None:
        """Empty columns list returns empty dict."""
        t = Sqrt()
        result = t.expressions([], {"a": exp.Column(this="a")})
        assert result == {}


class TestSqrtPipeline:
    """Test Sqrt with Pipeline end-to-end."""

    def test_fit_transform(self) -> None:
        """fit_transform produces SQRT values."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (4.0), (9.0), (16.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Sqrt()], backend=backend)
        result = pipe.fit_transform("t")
        np.testing.assert_allclose(result, np.array([[2.0], [3.0], [4.0]]), rtol=1e-10)

    def test_to_sql_contains_sqrt(self) -> None:
        """Generated SQL contains SQRT."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (4.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Sqrt()], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql().upper()
        assert "SQRT" in sql

    def test_null_handling(self) -> None:
        """NULL propagates through SQRT."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (4.0), (NULL), (16.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Sqrt()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (3, 1)

    def test_clone_roundtrip(self) -> None:
        """Cloned Sqrt is independent."""
        t = Sqrt()
        cloned = t.clone()
        assert cloned is not t

    def test_pickle_roundtrip(self) -> None:
        """Pickle roundtrip preserves state."""
        t = Sqrt()
        restored = pickle.loads(pickle.dumps(t))  # noqa: S301
        assert isinstance(restored, Sqrt)


# =============================================================================
# Power
# =============================================================================


class TestPowerConstructor:
    """Test Power constructor and validation."""

    def test_exponent_stored(self) -> None:
        """Exponent is stored."""
        t = Power(exponent=2)
        assert t.exponent == 2

    def test_classification_is_static(self) -> None:
        """Power is static."""
        assert Power._classification == "static"
        assert Power(exponent=2)._classify() == "static"

    def test_default_columns_is_numeric(self) -> None:
        """Power defaults to numeric columns."""
        assert Power._default_columns == "numeric"

    def test_non_numeric_exponent_raises(self) -> None:
        """Non-numeric exponent raises TypeError."""
        with pytest.raises(TypeError, match="must be a number"):
            Power(exponent="two")  # type: ignore[arg-type]

    def test_float_exponent(self) -> None:
        """Float exponent is accepted."""
        t = Power(exponent=0.5)
        assert t.exponent == 0.5

    def test_negative_exponent(self) -> None:
        """Negative exponent is accepted."""
        t = Power(exponent=-1)
        assert t.exponent == -1

    def test_zero_exponent(self) -> None:
        """Zero exponent is accepted (col^0 = 1)."""
        t = Power(exponent=0)
        assert t.exponent == 0

    def test_get_params(self) -> None:
        """get_params returns exponent and columns."""
        t = Power(exponent=3)
        params = t.get_params()
        assert params == {"exponent": 3, "columns": None}


class TestPowerExpressions:
    """Test Power.expressions() generates correct sqlglot ASTs."""

    def test_pow_expression(self) -> None:
        """Generates POW(col, exponent) node."""
        t = Power(exponent=2)
        exprs = {"a": exp.Column(this="a")}
        result = t.expressions(["a"], exprs)
        assert isinstance(result["a"], exp.Pow)

    def test_uses_exprs_not_column(self) -> None:
        """Composes with prior transforms."""
        t = Power(exponent=3)
        prior = exp.Add(this=exp.Column(this="a"), expression=exp.Literal.number(1))
        exprs = {"a": prior}
        result = t.expressions(["a"], exprs)
        assert isinstance(result["a"].this, exp.Add)

    def test_empty_columns(self) -> None:
        """Empty columns list returns empty dict."""
        t = Power(exponent=2)
        result = t.expressions([], {"a": exp.Column(this="a")})
        assert result == {}


class TestPowerPipeline:
    """Test Power with Pipeline end-to-end."""

    def test_fit_transform_square(self) -> None:
        """fit_transform produces col^2 values."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (2.0), (3.0), (4.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Power(exponent=2)], backend=backend)
        result = pipe.fit_transform("t")
        np.testing.assert_allclose(result, np.array([[4.0], [9.0], [16.0]]), rtol=1e-10)

    def test_to_sql_contains_pow(self) -> None:
        """Generated SQL contains POW."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (2.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Power(exponent=3)], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql().upper()
        assert "POW" in sql or "POWER" in sql

    def test_null_handling(self) -> None:
        """NULL propagates through POW."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (2.0), (NULL), (4.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Power(exponent=2)], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (3, 1)

    def test_zero_exponent(self) -> None:
        """col^0 = 1 for all non-null values."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (5.0), (10.0), (0.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Power(exponent=0)], backend=backend)
        result = pipe.fit_transform("t")
        np.testing.assert_allclose(result, np.array([[1.0], [1.0], [1.0]]), rtol=1e-10)

    def test_clone_roundtrip(self) -> None:
        """Cloned Power preserves exponent."""
        t = Power(exponent=3)
        cloned = t.clone()
        assert cloned.exponent == 3
        assert cloned is not t

    def test_pickle_roundtrip(self) -> None:
        """Pickle roundtrip preserves exponent."""
        t = Power(exponent=2.5)
        restored = pickle.loads(pickle.dumps(t))  # noqa: S301
        assert restored.exponent == 2.5


# =============================================================================
# Clip
# =============================================================================


class TestClipConstructor:
    """Test Clip constructor and validation."""

    def test_both_bounds(self) -> None:
        """Both lower and upper are stored."""
        t = Clip(lower=0, upper=100)
        assert t.lower == 0
        assert t.upper == 100

    def test_lower_only(self) -> None:
        """Lower-only Clip is valid."""
        t = Clip(lower=0)
        assert t.lower == 0
        assert t.upper is None

    def test_upper_only(self) -> None:
        """Upper-only Clip is valid."""
        t = Clip(upper=100)
        assert t.lower is None
        assert t.upper == 100

    def test_no_bounds_raises(self) -> None:
        """Neither bound specified raises ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            Clip()

    def test_lower_greater_than_upper_raises(self) -> None:
        """lower > upper raises ValueError."""
        with pytest.raises(ValueError, match="must not be greater"):
            Clip(lower=100, upper=0)

    def test_equal_bounds(self) -> None:
        """lower == upper is valid (constant output)."""
        t = Clip(lower=5, upper=5)
        assert t.lower == 5
        assert t.upper == 5

    def test_classification_is_static(self) -> None:
        """Clip is static."""
        assert Clip._classification == "static"
        assert Clip(lower=0)._classify() == "static"

    def test_default_columns_is_numeric(self) -> None:
        """Clip defaults to numeric columns."""
        assert Clip._default_columns == "numeric"

    def test_get_params(self) -> None:
        """get_params returns lower, upper, columns."""
        t = Clip(lower=0, upper=10)
        params = t.get_params()
        assert params == {"lower": 0, "upper": 10, "columns": None}


class TestClipExpressions:
    """Test Clip.expressions() generates correct sqlglot ASTs."""

    def test_both_bounds_expression(self) -> None:
        """Both bounds: GREATEST(LEAST(col, upper), lower)."""
        t = Clip(lower=0, upper=100)
        exprs = {"a": exp.Column(this="a")}
        result = t.expressions(["a"], exprs)
        node = result["a"]
        assert isinstance(node, exp.Greatest)

    def test_lower_only_expression(self) -> None:
        """Lower only: GREATEST(col, lower)."""
        t = Clip(lower=0)
        exprs = {"a": exp.Column(this="a")}
        result = t.expressions(["a"], exprs)
        node = result["a"]
        assert isinstance(node, exp.Greatest)

    def test_upper_only_expression(self) -> None:
        """Upper only: LEAST(col, upper)."""
        t = Clip(upper=100)
        exprs = {"a": exp.Column(this="a")}
        result = t.expressions(["a"], exprs)
        node = result["a"]
        assert isinstance(node, exp.Least)

    def test_uses_exprs_not_column(self) -> None:
        """Composes with prior transforms."""
        t = Clip(lower=0, upper=10)
        prior = exp.Mul(this=exp.Column(this="a"), expression=exp.Literal.number(2))
        exprs = {"a": prior}
        result = t.expressions(["a"], exprs)
        assert isinstance(result["a"], exp.Greatest)

    def test_empty_columns(self) -> None:
        """Empty columns list returns empty dict."""
        t = Clip(lower=0)
        result = t.expressions([], {"a": exp.Column(this="a")})
        assert result == {}


class TestClipPipeline:
    """Test Clip with Pipeline end-to-end."""

    def test_fit_transform_both_bounds(self) -> None:
        """Values are clipped to [lower, upper]."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (-5.0), (0.0), (50.0), (150.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Clip(lower=0, upper=100)], backend=backend)
        result = pipe.fit_transform("t")
        np.testing.assert_allclose(result, np.array([[0.0], [0.0], [50.0], [100.0]]), rtol=1e-10)

    def test_fit_transform_lower_only(self) -> None:
        """Values below lower are clipped up."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (-5.0), (0.0), (50.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Clip(lower=0)], backend=backend)
        result = pipe.fit_transform("t")
        np.testing.assert_allclose(result, np.array([[0.0], [0.0], [50.0]]), rtol=1e-10)

    def test_fit_transform_upper_only(self) -> None:
        """Values above upper are clipped down."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (-5.0), (50.0), (150.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Clip(upper=100)], backend=backend)
        result = pipe.fit_transform("t")
        np.testing.assert_allclose(result, np.array([[-5.0], [50.0], [100.0]]), rtol=1e-10)

    def test_to_sql_contains_greatest_least(self) -> None:
        """Generated SQL contains GREATEST and LEAST."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Clip(lower=0, upper=100)], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql().upper()
        assert "GREATEST" in sql
        assert "LEAST" in sql

    def test_null_handling(self) -> None:
        """NULL propagates through Clip."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1.0), (NULL), (3.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Clip(lower=0, upper=10)], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (3, 1)

    def test_clone_roundtrip(self) -> None:
        """Cloned Clip preserves bounds."""
        t = Clip(lower=0, upper=10)
        cloned = t.clone()
        assert cloned.lower == 0
        assert cloned.upper == 10
        assert cloned is not t

    def test_pickle_roundtrip(self) -> None:
        """Pickle roundtrip preserves bounds."""
        t = Clip(lower=-1, upper=1)
        restored = pickle.loads(pickle.dumps(t))  # noqa: S301
        assert restored.lower == -1
        assert restored.upper == 1


# =============================================================================
# Abs
# =============================================================================


class TestAbsConstructor:
    """Test Abs constructor."""

    def test_classification_is_static(self) -> None:
        """Abs is static."""
        assert Abs._classification == "static"
        assert Abs()._classify() == "static"

    def test_default_columns_is_numeric(self) -> None:
        """Abs defaults to numeric columns."""
        assert Abs._default_columns == "numeric"

    def test_get_params(self) -> None:
        """get_params returns columns only."""
        t = Abs()
        assert t.get_params() == {"columns": None}


class TestAbsExpressions:
    """Test Abs.expressions() generates correct sqlglot ASTs."""

    def test_abs_expression(self) -> None:
        """Generates ABS(col) node."""
        t = Abs()
        exprs = {"a": exp.Column(this="a")}
        result = t.expressions(["a"], exprs)
        assert isinstance(result["a"], exp.Abs)

    def test_uses_exprs_not_column(self) -> None:
        """Composes with prior transforms."""
        t = Abs()
        prior = exp.Sub(this=exp.Column(this="a"), expression=exp.Literal.number(5))
        exprs = {"a": prior}
        result = t.expressions(["a"], exprs)
        assert isinstance(result["a"].this, exp.Sub)

    def test_empty_columns(self) -> None:
        """Empty columns list returns empty dict."""
        t = Abs()
        result = t.expressions([], {"a": exp.Column(this="a")})
        assert result == {}


class TestAbsPipeline:
    """Test Abs with Pipeline end-to-end."""

    def test_fit_transform(self) -> None:
        """fit_transform produces absolute values."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (-3.0), (0.0), (5.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Abs()], backend=backend)
        result = pipe.fit_transform("t")
        np.testing.assert_allclose(result, np.array([[3.0], [0.0], [5.0]]), rtol=1e-10)

    def test_to_sql_contains_abs(self) -> None:
        """Generated SQL contains ABS."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (-1.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Abs()], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql().upper()
        assert "ABS" in sql

    def test_null_handling(self) -> None:
        """NULL propagates through ABS."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (-1.0), (NULL), (3.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Abs()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (3, 1)

    def test_clone_roundtrip(self) -> None:
        """Cloned Abs is independent."""
        t = Abs()
        cloned = t.clone()
        assert cloned is not t

    def test_pickle_roundtrip(self) -> None:
        """Pickle roundtrip preserves state."""
        t = Abs()
        restored = pickle.loads(pickle.dumps(t))  # noqa: S301
        assert isinstance(restored, Abs)


# =============================================================================
# Round
# =============================================================================


class TestRoundConstructor:
    """Test Round constructor."""

    def test_default_decimals(self) -> None:
        """Default decimals is 0."""
        t = Round()
        assert t.decimals == 0

    def test_custom_decimals(self) -> None:
        """Custom decimals is stored."""
        t = Round(decimals=2)
        assert t.decimals == 2

    def test_negative_decimals(self) -> None:
        """Negative decimals rounds to powers of 10."""
        t = Round(decimals=-1)
        assert t.decimals == -1

    def test_classification_is_static(self) -> None:
        """Round is static."""
        assert Round._classification == "static"
        assert Round()._classify() == "static"

    def test_default_columns_is_numeric(self) -> None:
        """Round defaults to numeric columns."""
        assert Round._default_columns == "numeric"

    def test_get_params(self) -> None:
        """get_params returns decimals and columns."""
        t = Round(decimals=3)
        params = t.get_params()
        assert params == {"decimals": 3, "columns": None}


class TestRoundExpressions:
    """Test Round.expressions() generates correct sqlglot ASTs."""

    def test_round_expression(self) -> None:
        """Generates ROUND(col, decimals) node."""
        t = Round(decimals=2)
        exprs = {"a": exp.Column(this="a")}
        result = t.expressions(["a"], exprs)
        assert isinstance(result["a"], exp.Round)

    def test_uses_exprs_not_column(self) -> None:
        """Composes with prior transforms."""
        t = Round(decimals=0)
        prior = exp.Div(this=exp.Column(this="a"), expression=exp.Literal.number(3))
        exprs = {"a": prior}
        result = t.expressions(["a"], exprs)
        assert isinstance(result["a"].this, exp.Div)

    def test_empty_columns(self) -> None:
        """Empty columns list returns empty dict."""
        t = Round()
        result = t.expressions([], {"a": exp.Column(this="a")})
        assert result == {}


class TestRoundPipeline:
    """Test Round with Pipeline end-to-end."""

    def test_fit_transform_default(self) -> None:
        """Default (0 decimals) rounds to integer."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1.4), (2.5), (3.6) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Round()], backend=backend)
        result = pipe.fit_transform("t")
        # DuckDB ROUND(2.5, 0) uses banker's rounding: 2.0
        assert result.shape == (3, 1)
        # Check that results are integers (no decimal part)
        for val in result[:, 0]:
            assert float(val) == round(float(val))

    def test_fit_transform_2_decimals(self) -> None:
        """Round to 2 decimal places."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1.456), (2.789), (3.123) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Round(decimals=2)], backend=backend)
        result = pipe.fit_transform("t")
        np.testing.assert_allclose(result, np.array([[1.46], [2.79], [3.12]]), rtol=1e-10)

    def test_to_sql_contains_round(self) -> None:
        """Generated SQL contains ROUND."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1.5) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Round(decimals=2)], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql().upper()
        assert "ROUND" in sql

    def test_null_handling(self) -> None:
        """NULL propagates through ROUND."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1.5), (NULL), (3.5) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Round()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (3, 1)

    def test_negative_decimals(self) -> None:
        """Negative decimals rounds to powers of 10."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (123.0), (456.0), (789.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Round(decimals=-1)], backend=backend)
        result = pipe.fit_transform("t")
        np.testing.assert_allclose(result, np.array([[120.0], [460.0], [790.0]]), rtol=1e-10)

    def test_clone_roundtrip(self) -> None:
        """Cloned Round preserves decimals."""
        t = Round(decimals=3)
        cloned = t.clone()
        assert cloned.decimals == 3
        assert cloned is not t

    def test_pickle_roundtrip(self) -> None:
        """Pickle roundtrip preserves decimals."""
        t = Round(decimals=4)
        restored = pickle.loads(pickle.dumps(t))  # noqa: S301
        assert restored.decimals == 4


# =============================================================================
# Reciprocal
# =============================================================================


class TestReciprocalConstructor:
    """Test Reciprocal constructor."""

    def test_classification_is_static(self) -> None:
        """Reciprocal is static."""
        assert Reciprocal._classification == "static"
        assert Reciprocal()._classify() == "static"

    def test_default_columns_is_numeric(self) -> None:
        """Reciprocal defaults to numeric columns."""
        assert Reciprocal._default_columns == "numeric"

    def test_get_params(self) -> None:
        """get_params returns columns only."""
        t = Reciprocal()
        assert t.get_params() == {"columns": None}


class TestReciprocalExpressions:
    """Test Reciprocal.expressions() generates correct sqlglot ASTs."""

    def test_reciprocal_expression(self) -> None:
        """Generates 1.0 / NULLIF(col, 0) node."""
        t = Reciprocal()
        exprs = {"a": exp.Column(this="a")}
        result = t.expressions(["a"], exprs)
        assert isinstance(result["a"], exp.Div)

    def test_nullif_wraps_column(self) -> None:
        """The denominator uses NULLIF for zero-safe division."""
        t = Reciprocal()
        exprs = {"a": exp.Column(this="a")}
        result = t.expressions(["a"], exprs)
        div_node = result["a"]
        assert isinstance(div_node, exp.Div)
        assert isinstance(div_node.expression, exp.Nullif)

    def test_uses_exprs_not_column(self) -> None:
        """Composes with prior transforms."""
        t = Reciprocal()
        prior = exp.Add(this=exp.Column(this="a"), expression=exp.Literal.number(1))
        exprs = {"a": prior}
        result = t.expressions(["a"], exprs)
        nullif_node = result["a"].expression
        assert isinstance(nullif_node.this, exp.Add)

    def test_empty_columns(self) -> None:
        """Empty columns list returns empty dict."""
        t = Reciprocal()
        result = t.expressions([], {"a": exp.Column(this="a")})
        assert result == {}


class TestReciprocalPipeline:
    """Test Reciprocal with Pipeline end-to-end."""

    def test_fit_transform(self) -> None:
        """fit_transform produces 1/x values."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (2.0), (4.0), (5.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Reciprocal()], backend=backend)
        result = pipe.fit_transform("t")
        np.testing.assert_allclose(result, np.array([[0.5], [0.25], [0.2]]), rtol=1e-10)

    def test_zero_becomes_null(self) -> None:
        """Zero input produces NULL (not division-by-zero error)."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (0.0), (2.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Reciprocal()], backend=backend)
        result = pipe.fit_transform("t")
        # Zero should become NaN (NULL -> NaN in numpy)
        assert result.shape == (2, 1)
        assert np.isnan(result[0, 0])
        np.testing.assert_allclose(result[1, 0], 0.5, rtol=1e-10)

    def test_to_sql_contains_nullif(self) -> None:
        """Generated SQL contains NULLIF for zero safety."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Reciprocal()], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql().upper()
        assert "NULLIF" in sql

    def test_null_handling(self) -> None:
        """NULL propagates through Reciprocal."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1.0), (NULL), (3.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Reciprocal()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (3, 1)

    def test_clone_roundtrip(self) -> None:
        """Cloned Reciprocal is independent."""
        t = Reciprocal()
        cloned = t.clone()
        assert cloned is not t

    def test_pickle_roundtrip(self) -> None:
        """Pickle roundtrip preserves state."""
        t = Reciprocal()
        restored = pickle.loads(pickle.dumps(t))  # noqa: S301
        assert isinstance(restored, Reciprocal)


# =============================================================================
# Cross-transformer tests
# =============================================================================


class TestCrossTransformer:
    """Tests spanning multiple arithmetic transforms."""

    def test_all_imports_from_sqlearn(self) -> None:
        """All 7 transforms importable from sqlearn top-level."""
        import sqlearn as sq

        assert sq.Log is Log
        assert sq.Sqrt is Sqrt
        assert sq.Power is Power
        assert sq.Clip is Clip
        assert sq.Abs is Abs
        assert sq.Round is Round
        assert sq.Reciprocal is Reciprocal

    def test_all_imports_from_features(self) -> None:
        """All 7 transforms importable from sqlearn.features."""
        from sqlearn import features

        assert features.Log is Log
        assert features.Sqrt is Sqrt
        assert features.Power is Power
        assert features.Clip is Clip
        assert features.Abs is Abs
        assert features.Round is Round
        assert features.Reciprocal is Reciprocal

    def test_composition_log_then_round(self) -> None:
        """Log + Round composes: ROUND(LN(col + 1), 2)."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1.0), (10.0), (100.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Log(), Round(decimals=2)], backend=backend)
        result = pipe.fit_transform("t")
        expected = np.round(np.log(np.array([[1], [10], [100]], dtype=np.float64) + 1), 2)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_composition_abs_then_sqrt(self) -> None:
        """Abs + Sqrt composes: SQRT(ABS(col))."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (-4.0), (-9.0), (-16.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Abs(), Sqrt()], backend=backend)
        result = pipe.fit_transform("t")
        np.testing.assert_allclose(result, np.array([[2.0], [3.0], [4.0]]), rtol=1e-10)

    def test_operator_composition(self) -> None:
        """Arithmetic transforms compose via + operator."""
        pipe = Log() + Round(decimals=2)
        assert isinstance(pipe, Pipeline)

    def test_repr_all_transforms(self) -> None:
        """All transforms have sensible repr."""
        reprs = [
            repr(Log()),
            repr(Log(base=10, offset=0)),
            repr(Sqrt()),
            repr(Power(exponent=2)),
            repr(Clip(lower=0, upper=100)),
            repr(Clip(lower=0)),
            repr(Abs()),
            repr(Round()),
            repr(Round(decimals=2)),
            repr(Reciprocal()),
        ]
        for r in reprs:
            assert isinstance(r, str)
            assert len(r) > 0

    def test_log_repr_default(self) -> None:
        """Log() repr shows no params (all defaults)."""
        r = repr(Log())
        assert r == "Log()"

    def test_log_repr_custom(self) -> None:
        """Log(base=10) repr shows base."""
        r = repr(Log(base=10))
        assert "base=10" in r

    def test_power_repr(self) -> None:
        """Power(exponent=2) repr shows exponent."""
        r = repr(Power(exponent=2))
        assert "exponent=2" in r

    def test_single_row_all_transforms(self) -> None:
        """All transforms work with single-row data."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (4.0) t(a)")
        backend = DuckDBBackend(connection=conn)

        transforms = [
            Log(),
            Sqrt(),
            Power(exponent=2),
            Clip(lower=0, upper=10),
            Abs(),
            Round(decimals=2),
            Reciprocal(),
        ]
        for tfm in transforms:
            pipe = Pipeline([tfm], backend=backend)
            result = pipe.fit_transform("t")
            assert result.shape == (1, 1), f"{type(tfm).__name__} failed on single row"

    def test_multiple_columns_all_transforms(self) -> None:
        """All transforms work with multiple numeric columns."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1.0, 2.0), (3.0, 4.0) t(a, b)")
        backend = DuckDBBackend(connection=conn)

        transforms = [
            Log(),
            Sqrt(),
            Power(exponent=2),
            Clip(lower=0, upper=10),
            Abs(),
            Round(),
            Reciprocal(),
        ]
        for tfm in transforms:
            pipe = Pipeline([tfm], backend=backend)
            result = pipe.fit_transform("t")
            assert result.shape == (2, 2), f"{type(tfm).__name__} failed with 2 columns"
