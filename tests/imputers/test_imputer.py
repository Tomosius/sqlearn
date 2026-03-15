"""Tests for sqlearn.imputers.imputer — Imputer transformer."""

from __future__ import annotations

from typing import Any

import duckdb
import numpy as np
import pytest
import sqlglot.expressions as exp

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.pipeline import Pipeline
from sqlearn.core.schema import Schema
from sqlearn.imputers.imputer import Imputer

# ── Constructor tests ──────────────────────────────────────────────


class TestConstructor:
    """Test Imputer constructor and attributes."""

    def test_defaults(self) -> None:
        """Default strategy='auto', columns=None."""
        imp = Imputer()
        assert imp.strategy == "auto"
        assert imp.columns is None

    def test_single_strategy(self) -> None:
        """Single strategy string is stored."""
        imp = Imputer(strategy="mean")
        assert imp.strategy == "mean"

    def test_dict_mode(self) -> None:
        """Dict strategy stores the dict as-is."""
        mapping: dict[str, str | int | float] = {"a": "mean", "b": 0}
        imp = Imputer(strategy=mapping)
        assert imp.strategy == mapping

    def test_custom_columns(self) -> None:
        """Explicit columns override."""
        imp = Imputer(strategy="median", columns=["a", "b"])
        assert imp.columns == ["a", "b"]
        assert imp.strategy == "median"

    def test_default_columns_is_all(self) -> None:
        """Class default routes to all columns."""
        assert Imputer._default_columns == "all"

    def test_classification_is_dynamic(self) -> None:
        """Class is dynamic (needs to learn fill values)."""
        assert Imputer._classification == "dynamic"

    def test_get_params_default(self) -> None:
        """get_params returns constructor kwargs for defaults."""
        imp = Imputer()
        params = imp.get_params()
        assert params == {"strategy": "auto", "columns": None}

    def test_get_params_dict_mode(self) -> None:
        """get_params round-trip for dict mode preserves the dict."""
        mapping: dict[str, str | int | float] = {"price": "mean", "qty": 0}
        imp = Imputer(strategy=mapping)
        params = imp.get_params()
        assert params["strategy"] == mapping
        assert params["columns"] is None

    def test_get_params_with_columns(self) -> None:
        """get_params with explicit columns."""
        imp = Imputer(strategy="median", columns=["x"])
        params = imp.get_params()
        assert params == {"strategy": "median", "columns": ["x"]}


# ── _resolve_columns_spec tests ───────────────────────────────────


class TestResolveColumnsSpec:
    """Test column resolution override for dict mode."""

    def test_dict_mode_returns_dict_keys(self) -> None:
        """Dict mode returns dict keys as column list."""
        imp = Imputer(strategy={"a": "mean", "b": 0, "c": "active"})
        result = imp._resolve_columns_spec()
        assert result == ["a", "b", "c"]

    def test_non_dict_delegates_to_super(self) -> None:
        """Non-dict mode delegates to superclass."""
        imp = Imputer(strategy="median")
        result = imp._resolve_columns_spec()
        # columns=None, so falls through to _default_columns = "all"
        assert result == "all"

    def test_non_dict_with_columns(self) -> None:
        """Non-dict with explicit columns returns those columns."""
        imp = Imputer(strategy="mean", columns=["x", "y"])
        result = imp._resolve_columns_spec()
        assert result == ["x", "y"]


# ── _resolve_strategy tests ───────────────────────────────────────


class TestResolveStrategy:
    """Test strategy resolution per column."""

    @pytest.fixture
    def mixed_schema(self) -> Schema:
        """Schema with numeric and categorical columns."""
        return Schema({"num": "DOUBLE", "cat": "VARCHAR", "ts": "TIMESTAMP"})

    def test_auto_numeric_to_median(self, mixed_schema: Schema) -> None:
        """auto + numeric column -> median."""
        imp = Imputer()
        assert imp._resolve_strategy("num", mixed_schema) == "median"

    def test_auto_categorical_to_most_frequent(self, mixed_schema: Schema) -> None:
        """auto + categorical column -> most_frequent."""
        imp = Imputer()
        assert imp._resolve_strategy("cat", mixed_schema) == "most_frequent"

    def test_auto_other_to_median(self, mixed_schema: Schema) -> None:
        """auto + temporal/other column -> median (safe default)."""
        imp = Imputer()
        assert imp._resolve_strategy("ts", mixed_schema) == "median"

    def test_explicit_mean(self, mixed_schema: Schema) -> None:
        """Explicit mean strategy passes through."""
        imp = Imputer(strategy="mean")
        assert imp._resolve_strategy("num", mixed_schema) == "mean"

    def test_explicit_median(self, mixed_schema: Schema) -> None:
        """Explicit median strategy passes through."""
        imp = Imputer(strategy="median")
        assert imp._resolve_strategy("num", mixed_schema) == "median"

    def test_explicit_most_frequent(self, mixed_schema: Schema) -> None:
        """Explicit most_frequent strategy passes through."""
        imp = Imputer(strategy="most_frequent")
        assert imp._resolve_strategy("cat", mixed_schema) == "most_frequent"

    def test_explicit_zero(self, mixed_schema: Schema) -> None:
        """Explicit zero strategy passes through."""
        imp = Imputer(strategy="zero")
        assert imp._resolve_strategy("num", mixed_schema) == "zero"

    def test_dict_known_strategy(self, mixed_schema: Schema) -> None:
        """Dict value that is a known strategy string."""
        imp = Imputer(strategy={"num": "mean"})
        assert imp._resolve_strategy("num", mixed_schema) == "mean"

    def test_dict_constant_int(self, mixed_schema: Schema) -> None:
        """Dict value that is an int constant."""
        imp = Imputer(strategy={"num": 0})
        assert imp._resolve_strategy("num", mixed_schema) == 0

    def test_dict_constant_float(self, mixed_schema: Schema) -> None:
        """Dict value that is a float constant."""
        imp = Imputer(strategy={"num": -1.5})
        assert imp._resolve_strategy("num", mixed_schema) == -1.5

    def test_dict_constant_string(self, mixed_schema: Schema) -> None:
        """Dict value that is an unknown string -> constant fill."""
        imp = Imputer(strategy={"cat": "active"})
        assert imp._resolve_strategy("cat", mixed_schema) == "active"


# ── discover() tests ──────────────────────────────────────────────


class TestDiscover:
    """Test Imputer.discover() returns correct sqlglot aggregates."""

    @pytest.fixture
    def mixed_schema(self) -> Schema:
        """Schema with numeric and categorical columns."""
        return Schema({"num": "DOUBLE", "cat": "VARCHAR"})

    def test_auto_numeric_returns_median(self, mixed_schema: Schema) -> None:
        """auto mode: numeric column -> MEDIAN aggregate."""
        imp = Imputer()
        imp.input_schema_ = mixed_schema
        result = imp.discover(["num"], mixed_schema)
        assert "num__value" in result
        agg = result["num__value"]
        assert isinstance(agg, exp.Anonymous)
        assert agg.this == "MEDIAN"

    def test_auto_categorical_returns_mode(self, mixed_schema: Schema) -> None:
        """auto mode: categorical column -> MODE aggregate."""
        imp = Imputer()
        imp.input_schema_ = mixed_schema
        result = imp.discover(["cat"], mixed_schema)
        assert "cat__value" in result
        agg = result["cat__value"]
        assert isinstance(agg, exp.Anonymous)
        assert agg.this == "MODE"

    def test_auto_mixed_columns(self, mixed_schema: Schema) -> None:
        """auto mode: mixed columns get appropriate aggregates."""
        imp = Imputer()
        imp.input_schema_ = mixed_schema
        result = imp.discover(["num", "cat"], mixed_schema)
        assert len(result) == 2
        assert isinstance(result["num__value"], exp.Anonymous)
        assert result["num__value"].this == "MEDIAN"
        assert isinstance(result["cat__value"], exp.Anonymous)
        assert result["cat__value"].this == "MODE"

    def test_mean_strategy_returns_avg(self, mixed_schema: Schema) -> None:
        """mean strategy -> AVG aggregate."""
        imp = Imputer(strategy="mean")
        result = imp.discover(["num"], mixed_schema)
        assert "num__value" in result
        assert isinstance(result["num__value"], exp.Avg)

    def test_median_strategy_returns_median(self, mixed_schema: Schema) -> None:
        """median strategy -> MEDIAN (Anonymous) aggregate."""
        imp = Imputer(strategy="median")
        result = imp.discover(["num"], mixed_schema)
        agg = result["num__value"]
        assert isinstance(agg, exp.Anonymous)
        assert agg.this == "MEDIAN"

    def test_most_frequent_returns_mode(self, mixed_schema: Schema) -> None:
        """most_frequent strategy -> MODE (Anonymous) aggregate."""
        imp = Imputer(strategy="most_frequent")
        result = imp.discover(["cat"], mixed_schema)
        agg = result["cat__value"]
        assert isinstance(agg, exp.Anonymous)
        assert agg.this == "MODE"

    def test_zero_strategy_returns_empty(self, mixed_schema: Schema) -> None:
        """zero strategy -> no aggregate needed."""
        imp = Imputer(strategy="zero")
        result = imp.discover(["num"], mixed_schema)
        assert result == {}

    def test_constant_int_returns_empty(self, mixed_schema: Schema) -> None:
        """Constant int fill -> no aggregate needed."""
        imp = Imputer(strategy={"num": 42})
        result = imp.discover(["num"], mixed_schema)
        assert result == {}

    def test_constant_float_returns_empty(self, mixed_schema: Schema) -> None:
        """Constant float fill -> no aggregate needed."""
        imp = Imputer(strategy={"num": 3.14})
        result = imp.discover(["num"], mixed_schema)
        assert result == {}

    def test_constant_string_returns_empty(self, mixed_schema: Schema) -> None:
        """Constant string fill -> no aggregate needed."""
        imp = Imputer(strategy={"cat": "unknown"})
        result = imp.discover(["cat"], mixed_schema)
        assert result == {}

    def test_dict_mixed_strategies(self, mixed_schema: Schema) -> None:
        """Dict mode with mix of strategies and constants."""
        imp = Imputer(strategy={"num": "mean", "cat": "active"})
        result = imp.discover(["num", "cat"], mixed_schema)
        # Only num needs an aggregate (mean); cat is constant
        assert "num__value" in result
        assert "cat__value" not in result
        assert isinstance(result["num__value"], exp.Avg)

    def test_discover_uses_source_column(self, mixed_schema: Schema) -> None:
        """discover() aggregates use exp.Column(this=col), not exprs."""
        imp = Imputer(strategy="mean")
        result = imp.discover(["num"], mixed_schema)
        inner = result["num__value"].this
        assert isinstance(inner, exp.Column)
        assert inner.name == "num"

    def test_empty_columns_returns_empty(self, mixed_schema: Schema) -> None:
        """Empty columns list returns empty dict."""
        imp = Imputer()
        imp.input_schema_ = mixed_schema
        result = imp.discover([], mixed_schema)
        assert result == {}


# ── expressions() tests ───────────────────────────────────────────


class TestExpressions:
    """Test Imputer.expressions() generates correct COALESCE ASTs."""

    def _make_fitted_imputer(
        self,
        strategy: str | dict[str, str | int | float] = "auto",
        params: dict[str, Any] | None = None,
        schema: Schema | None = None,
    ) -> Imputer:
        """Create a fitted Imputer with given params and schema."""
        imp = Imputer(strategy=strategy)
        imp.params_ = params or {}
        imp._fitted = True
        imp.input_schema_ = schema or Schema({"num": "DOUBLE", "cat": "VARCHAR"})
        return imp

    def test_coalesce_numeric_fill(self) -> None:
        """COALESCE with numeric fill from params_."""
        imp = self._make_fitted_imputer(
            strategy="median",
            params={"num__value": 3.0},
        )
        exprs = {"num": exp.Column(this="num")}
        result = imp.expressions(["num"], exprs)
        assert "num" in result
        assert isinstance(result["num"], exp.Coalesce)
        # Check the fill value is a number literal
        fill = result["num"].expressions[0]
        assert isinstance(fill, exp.Literal)

    def test_coalesce_string_fill(self) -> None:
        """COALESCE with string fill from params_."""
        imp = self._make_fitted_imputer(
            strategy="most_frequent",
            params={"cat__value": "active"},
        )
        exprs = {"cat": exp.Column(this="cat")}
        result = imp.expressions(["cat"], exprs)
        assert isinstance(result["cat"], exp.Coalesce)
        fill = result["cat"].expressions[0]
        assert isinstance(fill, exp.Literal)
        assert fill.is_string

    def test_constant_int_fill(self) -> None:
        """COALESCE with constant int (no params_ needed)."""
        imp = self._make_fitted_imputer(strategy={"num": 0})
        exprs = {"num": exp.Column(this="num")}
        result = imp.expressions(["num"], exprs)
        assert isinstance(result["num"], exp.Coalesce)

    def test_constant_float_fill(self) -> None:
        """COALESCE with constant float (no params_ needed)."""
        imp = self._make_fitted_imputer(strategy={"num": -1.5})
        exprs = {"num": exp.Column(this="num")}
        result = imp.expressions(["num"], exprs)
        assert isinstance(result["num"], exp.Coalesce)

    def test_constant_string_fill(self) -> None:
        """COALESCE with constant string (no params_ needed)."""
        imp = self._make_fitted_imputer(strategy={"cat": "unknown"})
        exprs = {"cat": exp.Column(this="cat")}
        result = imp.expressions(["cat"], exprs)
        assert isinstance(result["cat"], exp.Coalesce)
        fill = result["cat"].expressions[0]
        assert isinstance(fill, exp.Literal)
        assert fill.is_string

    def test_zero_fill(self) -> None:
        """Zero strategy fills with 0."""
        imp = self._make_fitted_imputer(strategy="zero")
        exprs = {"num": exp.Column(this="num")}
        result = imp.expressions(["num"], exprs)
        assert isinstance(result["num"], exp.Coalesce)
        fill = result["num"].expressions[0]
        assert isinstance(fill, exp.Literal)
        assert fill.is_number

    def test_multiple_columns(self) -> None:
        """Multiple columns each get their own COALESCE."""
        imp = self._make_fitted_imputer(
            strategy="auto",
            params={"num__value": 2.5, "cat__value": "x"},
        )
        exprs = {"num": exp.Column(this="num"), "cat": exp.Column(this="cat")}
        result = imp.expressions(["num", "cat"], exprs)
        assert "num" in result
        assert "cat" in result
        assert isinstance(result["num"], exp.Coalesce)
        assert isinstance(result["cat"], exp.Coalesce)

    def test_uses_exprs_not_column(self) -> None:
        """expressions() uses exprs[col], not exp.Column(this=col)."""
        imp = self._make_fitted_imputer(
            strategy="median",
            params={"num__value": 5.0},
        )
        # Simulate a prior transform: num is already (num * 2)
        prior = exp.Mul(this=exp.Column(this="num"), expression=exp.Literal.number(2))
        exprs = {"num": prior}
        result = imp.expressions(["num"], exprs)
        # The COALESCE's 'this' should be the prior Mul expression
        coalesce = result["num"]
        assert isinstance(coalesce, exp.Coalesce)
        assert isinstance(coalesce.this, exp.Mul)

    def test_dict_mixed_strategies(self) -> None:
        """Dict mode with mixed data-learned and constant strategies."""
        imp = self._make_fitted_imputer(
            strategy={"num": "mean", "cat": "unknown"},
            params={"num__value": 10.0},
        )
        exprs = {"num": exp.Column(this="num"), "cat": exp.Column(this="cat")}
        result = imp.expressions(["num", "cat"], exprs)
        assert isinstance(result["num"], exp.Coalesce)
        assert isinstance(result["cat"], exp.Coalesce)
        # cat fill should be string "unknown"
        cat_fill = result["cat"].expressions[0]
        assert cat_fill.is_string


# ── __repr__ tests ────────────────────────────────────────────────


class TestRepr:
    """Test Imputer.__repr__."""

    def test_default_repr(self) -> None:
        """Default params shows no args."""
        imp = Imputer()
        assert repr(imp) == "Imputer()"

    def test_strategy_repr(self) -> None:
        """Non-default strategy shows in repr."""
        imp = Imputer(strategy="mean")
        assert repr(imp) == "Imputer(strategy='mean')"

    def test_dict_mode_repr(self) -> None:
        """Dict mode shows the dict in repr."""
        imp = Imputer(strategy={"a": "mean", "b": 0})
        r = repr(imp)
        assert r.startswith("Imputer(strategy=")
        assert "'a': 'mean'" in r
        assert "'b': 0" in r

    def test_columns_repr(self) -> None:
        """Explicit columns show in repr."""
        imp = Imputer(strategy="median", columns=["x"])
        assert repr(imp) == "Imputer(strategy='median', columns=['x'])"


# ── Pipeline integration tests ────────────────────────────────────


class TestPipeline:
    """Test Imputer integrated with Pipeline (end-to-end)."""

    @pytest.fixture
    def backend(self) -> DuckDBBackend:
        """Create DuckDB backend with test data containing NULLs."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 'a'), (NULL, 'b'), (3.0, NULL), (4.0, 'a') t(num, cat)"
        )
        return DuckDBBackend(connection=conn)

    def test_fit_transform_no_nulls(self, backend: DuckDBBackend) -> None:
        """After imputation, output should contain no NULLs."""
        pipe = Pipeline([Imputer()], backend=backend)
        result = pipe.fit_transform("t")
        # No None values in the result
        for row in result:
            for val in row:
                assert val is not None

    def test_auto_fill_values(self, backend: DuckDBBackend) -> None:
        """Auto strategy: numeric uses median, categorical uses mode."""
        imp = Imputer()
        pipe = Pipeline([imp], backend=backend)
        pipe.fit_transform("t")
        # num values: 1.0, NULL, 3.0, 4.0 -> median of [1,3,4] = 3.0
        # cat values: 'a', 'b', NULL, 'a' -> mode = 'a'
        assert imp.params_ is not None
        num_fill = float(imp.params_["num__value"])
        cat_fill = imp.params_["cat__value"]
        np.testing.assert_allclose(num_fill, 3.0, atol=1e-10)
        assert cat_fill == "a"

    def test_dict_mode_fill(self, backend: DuckDBBackend) -> None:
        """Dict mode applies per-column strategies correctly."""
        imp = Imputer(strategy={"num": "mean", "cat": "unknown"})
        pipe = Pipeline([imp], backend=backend)
        result = pipe.fit_transform("t")
        # num: mean of [1,3,4] = 8/3 ~= 2.6667
        # cat: constant "unknown"
        assert imp.params_ is not None
        num_fill = imp.params_["num__value"]
        np.testing.assert_allclose(num_fill, 8.0 / 3.0, atol=1e-10)
        # Check row 1 (was NULL num) has the mean value
        np.testing.assert_allclose(float(result[1, 0]), 8.0 / 3.0, atol=1e-10)
        # Check row 2 (was NULL cat) has "unknown"
        assert result[2, 1] == "unknown"

    def test_to_sql_contains_coalesce(self, backend: DuckDBBackend) -> None:
        """to_sql() output contains COALESCE."""
        pipe = Pipeline([Imputer()], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql()
        assert "COALESCE" in sql.upper()

    def test_output_shape(self, backend: DuckDBBackend) -> None:
        """Output shape matches input shape."""
        pipe = Pipeline([Imputer()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (4, 2)

    def test_fit_then_transform_separate(self, backend: DuckDBBackend) -> None:
        """Separate fit() and transform() produce same result as fit_transform()."""
        pipe1 = Pipeline([Imputer()], backend=backend)
        result1 = pipe1.fit_transform("t")

        pipe2 = Pipeline([Imputer()], backend=backend)
        pipe2.fit("t")
        result2 = pipe2.transform("t")

        np.testing.assert_array_equal(result1, result2)

    def test_zero_strategy_pipeline(self, backend: DuckDBBackend) -> None:
        """Zero strategy fills NULLs with 0 (numeric columns)."""
        imp = Imputer(strategy="zero", columns=["num"])
        pipe = Pipeline([imp], backend=backend)
        result = pipe.fit_transform("t")
        # Row 1 (was NULL) should be 0
        np.testing.assert_allclose(float(result[1, 0]), 0.0, atol=1e-10)

    def test_constant_fill_pipeline(self, backend: DuckDBBackend) -> None:
        """Constant fill via dict mode."""
        imp = Imputer(strategy={"num": -1, "cat": "MISSING"})
        pipe = Pipeline([imp], backend=backend)
        result = pipe.fit_transform("t")
        # Row 1 num (was NULL) -> -1
        np.testing.assert_allclose(float(result[1, 0]), -1.0, atol=1e-10)
        # Row 2 cat (was NULL) -> "MISSING"
        assert result[2, 1] == "MISSING"


# ── Not-fitted guard tests ───────────────────────────────────────


class TestNotFittedGuard:
    """Calling transform/to_sql before fit() raises NotFittedError."""

    def test_transform_before_fit_raises(self) -> None:
        """transform() on unfitted pipeline raises NotFittedError."""
        from sqlearn.core.errors import NotFittedError

        pipe = Pipeline([Imputer()])
        with pytest.raises(NotFittedError):
            pipe.transform("t")

    def test_to_sql_before_fit_raises(self) -> None:
        """to_sql() on unfitted pipeline raises NotFittedError."""
        from sqlearn.core.errors import NotFittedError

        pipe = Pipeline([Imputer()])
        with pytest.raises(NotFittedError):
            pipe.to_sql()


# ── Clone and pickle tests ──────────────────────────────────────


class TestCloneAndPickle:
    """Deep clone independence and pickle roundtrip."""

    @pytest.fixture
    def fitted_pipe(self) -> Pipeline:
        """Create a fitted pipeline for clone/pickle tests."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 'a'), (NULL, 'b'), (3.0, NULL) t(num, cat)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Imputer()], backend=backend)
        pipe.fit("t")
        return pipe

    def test_clone_produces_same_sql(self, fitted_pipe: Pipeline) -> None:
        """Cloned pipeline produces identical SQL."""
        cloned = fitted_pipe.clone()
        assert fitted_pipe.to_sql() == cloned.to_sql()

    def test_clone_params_independent(self, fitted_pipe: Pipeline) -> None:
        """Modifying cloned step params does not affect original."""
        cloned = fitted_pipe.clone()
        original_imp = fitted_pipe.steps[0][1]
        cloned_imp = cloned.steps[0][1]
        assert cloned_imp.params_ is not None
        cloned_imp.params_["num__value"] = 999.0
        assert original_imp.params_ is not None
        assert original_imp.params_["num__value"] != 999.0

    def test_pickle_roundtrip(self) -> None:
        """Pickle an individual imputer preserves params."""
        import pickle

        imp = Imputer(strategy="mean")
        imp.params_ = {"num__value": 42.5}
        imp._fitted = True
        data = pickle.dumps(imp)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.params_ == {"num__value": 42.5}
        assert restored._fitted is True


# ── Edge cases ──────────────────────────────────────────────────


class TestEdgeCases:
    """Extreme and unusual inputs."""

    def test_mostly_nulls_column(self) -> None:
        """Column with mostly NULLs — fill value from single non-null."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (CAST(NULL AS DOUBLE)), (42.0), (CAST(NULL AS DOUBLE)) t(a)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Imputer(strategy="mean")], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (3, 1)
        # All values should be 42.0 (nulls filled with mean of [42.0] = 42.0)
        np.testing.assert_allclose(result.flatten(), [42.0, 42.0, 42.0], atol=1e-10)

    def test_no_nulls_passthrough(self) -> None:
        """Data with no NULLs passes through unchanged."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1.0), (2.0), (3.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Imputer(strategy="mean")], backend=backend)
        result = pipe.fit_transform("t")
        np.testing.assert_array_equal(result.flatten(), [1.0, 2.0, 3.0])

    def test_two_rows_one_null(self) -> None:
        """Two rows, one NULL — fill with mean of single non-null."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (10.0), (CAST(NULL AS DOUBLE)) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Imputer(strategy="mean")], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (2, 1)
        np.testing.assert_allclose(result.flatten(), [10.0, 10.0], atol=1e-10)

    def test_single_row_no_null(self) -> None:
        """Single row without NULL passes through."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (42.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Imputer(strategy="mean")], backend=backend)
        result = pipe.fit_transform("t")
        np.testing.assert_allclose(result[0, 0], 42.0, atol=1e-10)

    def test_many_columns(self) -> None:
        """10 columns with mixed NULLs all imputed."""
        conn = duckdb.connect()
        cols = ", ".join([f"c{i} DOUBLE" for i in range(10)])
        conn.execute(f"CREATE TABLE t ({cols})")
        for i in range(5):
            vals = ", ".join(
                [str(float(i + j)) if (i + j) % 3 != 0 else "NULL" for j in range(10)]
            )
            conn.execute(f"INSERT INTO t VALUES ({vals})")  # noqa: S608
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Imputer(strategy="mean")], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (5, 10)


# ── SQL snapshot tests ──────────────────────────────────────────


class TestSqlSnapshot:
    """Verify SQL output structure and patterns."""

    @pytest.fixture
    def fitted_pipe(self) -> Pipeline:
        """Create fitted pipeline for SQL verification."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 'a'), (NULL, 'b'), (3.0, NULL) t(price, city)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Imputer()], backend=backend)
        pipe.fit("t")
        return pipe

    def test_sql_has_coalesce_per_column(self, fitted_pipe: Pipeline) -> None:
        """Each column gets a COALESCE in the SQL."""
        sql = fitted_pipe.to_sql().upper()
        assert sql.count("COALESCE") == 2  # one per column

    def test_sql_has_select_from(self, fitted_pipe: Pipeline) -> None:
        """SQL contains SELECT and FROM."""
        sql = fitted_pipe.to_sql().upper()
        assert "SELECT" in sql
        assert "FROM" in sql

    def test_sql_custom_table(self, fitted_pipe: Pipeline) -> None:
        """to_sql(table=...) uses custom table name."""
        sql = fitted_pipe.to_sql(table="raw_data")
        assert "raw_data" in sql


# ── Composition tests ───────────────────────────────────────────


class TestComposition:
    """Imputer composing with other transformers."""

    def test_imputer_then_scaler(self) -> None:
        """Imputer + StandardScaler: COALESCE nested in subtraction."""
        from sqlearn.scalers.standard import StandardScaler

        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1.0), (NULL), (3.0), (4.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Imputer(), StandardScaler()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (4, 1)
        assert not np.any(np.isnan(result.astype(float)))

    def test_imputer_then_encoder(self) -> None:
        """Imputer + OneHotEncoder: NULLs filled before encoding."""
        from sqlearn.encoders.onehot import OneHotEncoder

        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES ('A'), ('B'), (NULL), ('A') t(cat)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Imputer(), OneHotEncoder()], backend=backend)
        result = pipe.fit_transform("t")
        # 4 rows, 2 categories (A, B) -> 2 binary columns
        assert result.shape == (4, 2)

    def test_three_step_pipeline(self) -> None:
        """Imputer + StandardScaler + OneHotEncoder: full pipeline."""
        from sqlearn.encoders.onehot import OneHotEncoder
        from sqlearn.scalers.standard import StandardScaler

        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 'A'), (NULL, 'B'), (3.0, NULL), (4.0, 'A') t(num, cat)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Imputer(), StandardScaler(), OneHotEncoder()], backend=backend)
        result = pipe.fit_transform("t")
        # num (scaled) + cat_a + cat_b = 3 columns
        assert result.shape == (4, 3)
