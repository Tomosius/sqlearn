"""Tests for sqlearn.encoders.frequency -- FrequencyEncoder."""

from __future__ import annotations

from typing import Any

import duckdb
import numpy as np
import pytest
import sqlglot.expressions as exp

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.pipeline import Pipeline
from sqlearn.core.schema import Schema
from sqlearn.encoders.frequency import FrequencyEncoder

# -- Constructor tests ----------------------------------------------------------


class TestConstructor:
    """Test FrequencyEncoder constructor and attributes."""

    def test_defaults(self) -> None:
        """Default normalize=True, handle_unknown='value', fill_value=0.0, columns=None."""
        enc = FrequencyEncoder()
        assert enc.normalize is True
        assert enc.handle_unknown == "value"
        assert enc.fill_value == 0.0
        assert enc.columns is None

    def test_default_columns_is_categorical(self) -> None:
        """Class default routes to categorical columns."""
        assert FrequencyEncoder._default_columns == "categorical"

    def test_classification_is_dynamic(self) -> None:
        """Class is dynamic (needs to learn frequencies)."""
        assert FrequencyEncoder._classification == "dynamic"

    def test_custom_normalize_false(self) -> None:
        """normalize=False switches to raw counts."""
        enc = FrequencyEncoder(normalize=False)
        assert enc.normalize is False

    def test_custom_fill_value(self) -> None:
        """Custom fill_value is stored."""
        enc = FrequencyEncoder(fill_value=-1.0)
        assert enc.fill_value == -1.0

    def test_custom_handle_unknown(self) -> None:
        """handle_unknown='error' is stored."""
        enc = FrequencyEncoder(handle_unknown="error")
        assert enc.handle_unknown == "error"

    def test_invalid_handle_unknown_raises(self) -> None:
        """Invalid handle_unknown value raises ValueError."""
        with pytest.raises(ValueError, match="handle_unknown"):
            FrequencyEncoder(handle_unknown="ignore")

    def test_custom_columns(self) -> None:
        """Explicit columns override."""
        enc = FrequencyEncoder(columns=["city", "color"])
        assert enc.columns == ["city", "color"]

    def test_get_params(self) -> None:
        """get_params returns constructor kwargs."""
        enc = FrequencyEncoder(normalize=False, fill_value=-1.0, columns=["city"])
        params = enc.get_params()
        assert params == {
            "normalize": False,
            "handle_unknown": "value",
            "fill_value": -1.0,
            "columns": ["city"],
        }

    def test_set_params(self) -> None:
        """set_params updates attributes and returns self."""
        enc = FrequencyEncoder()
        result = enc.set_params(normalize=False)
        assert result is enc
        assert enc.normalize is False


# -- discover() tests ----------------------------------------------------------


class TestDiscover:
    """Test FrequencyEncoder.discover() returns total count aggregate."""

    @pytest.fixture
    def schema(self) -> Schema:
        """Schema with categorical columns."""
        return Schema({"city": "VARCHAR", "color": "VARCHAR", "price": "DOUBLE"})

    def test_returns_total_count(self, schema: Schema) -> None:
        """discover() returns a single __total_count entry."""
        enc = FrequencyEncoder()
        result = enc.discover(["city", "color"], schema)
        assert "__total_count" in result
        assert len(result) == 1

    def test_total_count_is_count_star(self, schema: Schema) -> None:
        """__total_count is COUNT(*)."""
        enc = FrequencyEncoder()
        result = enc.discover(["city"], schema)
        count_expr = result["__total_count"]
        assert isinstance(count_expr, exp.Count)
        assert isinstance(count_expr.this, exp.Star)

    def test_empty_columns_still_returns_count(self, schema: Schema) -> None:
        """Even with empty columns list, total count is returned."""
        enc = FrequencyEncoder()
        result = enc.discover([], schema)
        assert "__total_count" in result


# -- discover_sets() tests -----------------------------------------------------


class TestDiscoverSets:
    """Test FrequencyEncoder.discover_sets() returns frequency queries."""

    @pytest.fixture
    def schema(self) -> Schema:
        """Schema with categorical columns."""
        return Schema({"city": "VARCHAR", "color": "VARCHAR", "price": "DOUBLE"})

    def test_returns_freq_query_per_column(self, schema: Schema) -> None:
        """Each column gets a frequency query."""
        enc = FrequencyEncoder()
        result = enc.discover_sets(["city", "color"], schema)
        assert "city__freq" in result
        assert "color__freq" in result
        assert len(result) == 2

    def test_ast_is_select_with_group_by(self, schema: Schema) -> None:
        """Each query is a Select with GROUP BY."""
        enc = FrequencyEncoder()
        result = enc.discover_sets(["city"], schema)
        query = result["city__freq"]
        assert isinstance(query, exp.Select)
        # Should have a GROUP BY clause
        group = query.find(exp.Group)
        assert group is not None

    def test_select_has_count(self, schema: Schema) -> None:
        """Each query selects COUNT(*)."""
        enc = FrequencyEncoder()
        result = enc.discover_sets(["city"], schema)
        query = result["city__freq"]
        assert isinstance(query, exp.Select)
        count_nodes = list(query.find_all(exp.Count))
        assert len(count_nodes) > 0

    def test_empty_columns_returns_empty(self, schema: Schema) -> None:
        """Empty columns list returns empty dict."""
        enc = FrequencyEncoder()
        result = enc.discover_sets([], schema)
        assert result == {}


# -- _get_frequencies() tests ---------------------------------------------------


class TestGetFrequencies:
    """Test FrequencyEncoder._get_frequencies() helper."""

    def _make_fitted_encoder(
        self,
        params: dict[str, Any],
        sets: dict[str, list[dict[str, Any]]],
        *,
        normalize: bool = True,
    ) -> FrequencyEncoder:
        """Create a fitted FrequencyEncoder with given params and sets."""
        enc = FrequencyEncoder(normalize=normalize)
        enc.params_ = params
        enc.sets_ = sets
        enc._fitted = True
        return enc

    def test_sorted_alphabetically(self) -> None:
        """Categories are sorted alphabetically."""
        enc = self._make_fitted_encoder(
            params={"__total_count": 6},
            sets={
                "city__freq": [
                    {"city": "Paris", "_count": 2},
                    {"city": "London", "_count": 3},
                    {"city": "Berlin", "_count": 1},
                ]
            },
        )
        pairs = enc._get_frequencies("city")
        cats = [p[0] for p in pairs]
        assert cats == ["Berlin", "London", "Paris"]

    def test_proportions_correct(self) -> None:
        """Proportions are count / total."""
        enc = self._make_fitted_encoder(
            params={"__total_count": 10},
            sets={
                "city__freq": [
                    {"city": "A", "_count": 3},
                    {"city": "B", "_count": 7},
                ]
            },
        )
        pairs = enc._get_frequencies("city")
        assert pairs == [("A", 0.3), ("B", 0.7)]

    def test_raw_counts(self) -> None:
        """normalize=False returns raw counts."""
        enc = self._make_fitted_encoder(
            params={"__total_count": 10},
            sets={
                "city__freq": [
                    {"city": "A", "_count": 3},
                    {"city": "B", "_count": 7},
                ]
            },
            normalize=False,
        )
        pairs = enc._get_frequencies("city")
        assert pairs == [("A", 3.0), ("B", 7.0)]

    def test_nulls_excluded(self) -> None:
        """None values are excluded from frequencies."""
        enc = self._make_fitted_encoder(
            params={"__total_count": 6},
            sets={
                "city__freq": [
                    {"city": "A", "_count": 3},
                    {"city": None, "_count": 1},
                    {"city": "B", "_count": 2},
                ]
            },
        )
        pairs = enc._get_frequencies("city")
        cats = [p[0] for p in pairs]
        assert None not in cats
        assert len(pairs) == 2


# -- expressions() tests -------------------------------------------------------


class TestExpressions:
    """Test FrequencyEncoder.expressions() generates correct CASE WHEN ASTs."""

    def _make_fitted_encoder(
        self,
        params: dict[str, Any],
        sets: dict[str, list[dict[str, Any]]],
        *,
        normalize: bool = True,
        fill_value: float = 0.0,
    ) -> FrequencyEncoder:
        """Create a fitted FrequencyEncoder with given params and sets."""
        enc = FrequencyEncoder(normalize=normalize, fill_value=fill_value)
        enc.params_ = params
        enc.sets_ = sets
        enc._fitted = True
        enc.columns_ = list({k.replace("__freq", "") for k in sets})
        return enc

    def test_case_when_with_proportions(self) -> None:
        """Each category gets a CASE WHEN with its proportion."""
        enc = self._make_fitted_encoder(
            params={"__total_count": 4},
            sets={
                "city__freq": [
                    {"city": "London", "_count": 2},
                    {"city": "Paris", "_count": 2},
                ]
            },
        )
        exprs = {"city": exp.Column(this="city")}
        result = enc.expressions(["city"], exprs)
        assert "city" in result
        assert isinstance(result["city"], exp.Case)

    def test_result_replaces_column_in_place(self) -> None:
        """FrequencyEncoder replaces columns in-place (same key name)."""
        enc = self._make_fitted_encoder(
            params={"__total_count": 4},
            sets={
                "city__freq": [
                    {"city": "A", "_count": 2},
                    {"city": "B", "_count": 2},
                ]
            },
        )
        exprs = {"city": exp.Column(this="city")}
        result = enc.expressions(["city"], exprs)
        # Key is the original column name, not a new column
        assert "city" in result
        assert len(result) == 1

    def test_multiple_columns(self) -> None:
        """Multiple columns each get their own CASE expression."""
        enc = self._make_fitted_encoder(
            params={"__total_count": 4},
            sets={
                "city__freq": [{"city": "A", "_count": 2}, {"city": "B", "_count": 2}],
                "color__freq": [{"color": "R", "_count": 1}, {"color": "G", "_count": 3}],
            },
        )
        exprs = {"city": exp.Column(this="city"), "color": exp.Column(this="color")}
        result = enc.expressions(["city", "color"], exprs)
        assert "city" in result
        assert "color" in result
        assert isinstance(result["city"], exp.Case)
        assert isinstance(result["color"], exp.Case)

    def test_fill_value_in_else(self) -> None:
        """ELSE clause uses fill_value."""
        enc = self._make_fitted_encoder(
            params={"__total_count": 4},
            sets={"city__freq": [{"city": "A", "_count": 4}]},
            fill_value=-1.0,
        )
        exprs = {"city": exp.Column(this="city")}
        result = enc.expressions(["city"], exprs)
        case_expr = result["city"]
        assert isinstance(case_expr, exp.Case)
        default = case_expr.args["default"]
        # Negative literals are wrapped in exp.Neg by sqlglot
        assert isinstance(default, exp.Neg)
        assert isinstance(default.this, exp.Literal)
        assert float(default.this.this) == 1.0  # Neg wraps positive value

    def test_uses_exprs_not_column(self) -> None:
        """expressions() uses exprs[col], not exp.Column(this=col)."""
        enc = self._make_fitted_encoder(
            params={"__total_count": 4},
            sets={"city__freq": [{"city": "A", "_count": 4}]},
        )
        # Simulate a prior transform (e.g., UPPER)
        prior = exp.Upper(this=exp.Column(this="city"))
        exprs = {"city": prior}
        result = enc.expressions(["city"], exprs)
        case_expr = result["city"]
        assert isinstance(case_expr, exp.Case)
        if_expr = case_expr.args["ifs"][0]
        eq_expr = if_expr.this
        assert isinstance(eq_expr, exp.EQ)
        # Should reference the prior expression (Upper), not raw Column
        assert isinstance(eq_expr.this, exp.Upper)

    def test_categories_sorted_in_case(self) -> None:
        """CASE WHEN branches are in alphabetical order of categories."""
        enc = self._make_fitted_encoder(
            params={"__total_count": 6},
            sets={
                "city__freq": [
                    {"city": "Zurich", "_count": 1},
                    {"city": "Amsterdam", "_count": 2},
                    {"city": "Berlin", "_count": 3},
                ]
            },
        )
        exprs = {"city": exp.Column(this="city")}
        result = enc.expressions(["city"], exprs)
        case_expr = result["city"]
        ifs = case_expr.args["ifs"]
        # Extract category strings from EQ expressions
        cats = []
        for if_node in ifs:
            eq = if_node.this
            cats.append(eq.expression.this)
        assert cats == ["Amsterdam", "Berlin", "Zurich"]


# -- Pipeline integration tests ------------------------------------------------


class TestPipeline:
    """Test FrequencyEncoder integrated with Pipeline (end-to-end)."""

    @pytest.fixture
    def backend(self) -> DuckDBBackend:
        """Create DuckDB backend with categorical test data."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 'London'), (2.0, 'Paris'), "
            "(3.0, 'London'), (4.0, 'Tokyo') t(price, city)"
        )
        return DuckDBBackend(connection=conn)

    def test_fit_transform_shape(self, backend: DuckDBBackend) -> None:
        """Output shape matches input: 4 rows x 2 cols (price + city freq)."""
        pipe = Pipeline([FrequencyEncoder()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (4, 2)

    def test_frequency_values_correct(self, backend: DuckDBBackend) -> None:
        """London=2/4=0.5, Paris=1/4=0.25, Tokyo=1/4=0.25."""
        pipe = Pipeline([FrequencyEncoder()], backend=backend)
        result = pipe.fit_transform("t")
        # Column order: price, city
        city_col = result[:, 1].astype(float)
        # Row 0: London -> 0.5
        assert city_col[0] == pytest.approx(0.5)
        # Row 1: Paris -> 0.25
        assert city_col[1] == pytest.approx(0.25)
        # Row 2: London -> 0.5
        assert city_col[2] == pytest.approx(0.5)
        # Row 3: Tokyo -> 0.25
        assert city_col[3] == pytest.approx(0.25)

    def test_frequencies_sum_correctly(self, backend: DuckDBBackend) -> None:
        """All proportions are count/total and consistent with actual data.

        Data: London(x2), Paris(x1), Tokyo(x1) -- total 4.
        Proportions: London=0.5, Paris=0.25, Tokyo=0.25.
        Product of each row's frequency and total should equal that category's count.
        """
        pipe = Pipeline([FrequencyEncoder()], backend=backend)
        result = pipe.fit_transform("t")
        city_col = result[:, 1].astype(float)
        total = 4
        # London rows (0, 2) should have freq * total = 2
        assert city_col[0] * total == pytest.approx(2.0)
        assert city_col[2] * total == pytest.approx(2.0)
        # Paris row (1) should have freq * total = 1
        assert city_col[1] * total == pytest.approx(1.0)
        # Tokyo row (3) should have freq * total = 1
        assert city_col[3] * total == pytest.approx(1.0)

    def test_normalize_false_raw_counts(self, backend: DuckDBBackend) -> None:
        """normalize=False outputs raw counts (London=2, Paris=1, Tokyo=1)."""
        pipe = Pipeline([FrequencyEncoder(normalize=False)], backend=backend)
        result = pipe.fit_transform("t")
        city_col = result[:, 1].astype(float)
        # Row 0: London -> 2
        assert city_col[0] == pytest.approx(2.0)
        # Row 1: Paris -> 1
        assert city_col[1] == pytest.approx(1.0)
        # Row 2: London -> 2
        assert city_col[2] == pytest.approx(2.0)
        # Row 3: Tokyo -> 1
        assert city_col[3] == pytest.approx(1.0)

    def test_to_sql_contains_case_when(self, backend: DuckDBBackend) -> None:
        """to_sql() output contains CASE WHEN."""
        pipe = Pipeline([FrequencyEncoder()], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql()
        assert "CASE" in sql.upper()
        assert "WHEN" in sql.upper()

    def test_get_feature_names_out(self, backend: DuckDBBackend) -> None:
        """get_feature_names_out returns original column names (in-place replacement)."""
        pipe = Pipeline([FrequencyEncoder()], backend=backend)
        pipe.fit("t")
        names = pipe.get_feature_names_out()
        assert "price" in names
        assert "city" in names
        assert len(names) == 2

    def test_fit_then_transform_separate(self, backend: DuckDBBackend) -> None:
        """Separate fit() and transform() produce same result as fit_transform()."""
        pipe1 = Pipeline([FrequencyEncoder()], backend=backend)
        result1 = pipe1.fit_transform("t")

        pipe2 = Pipeline([FrequencyEncoder()], backend=backend)
        pipe2.fit("t")
        result2 = pipe2.transform("t")

        np.testing.assert_array_equal(result1, result2)

    def test_output_values_between_0_and_1_normalized(self, backend: DuckDBBackend) -> None:
        """When normalized, all output values are in [0, 1]."""
        pipe = Pipeline([FrequencyEncoder()], backend=backend)
        result = pipe.fit_transform("t")
        city_col = result[:, 1].astype(float)
        assert np.all(city_col >= 0.0)
        assert np.all(city_col <= 1.0)


# -- handle_unknown / fill_value tests -----------------------------------------


class TestHandleUnknown:
    """Test handle_unknown and fill_value behavior."""

    def test_fill_value_in_sql(self) -> None:
        """Custom fill_value appears in ELSE clause of generated SQL."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES ('A'), ('B'), ('A') t(cat)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([FrequencyEncoder(fill_value=-1.0)], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql()
        assert "-1.0" in sql

    def test_default_fill_value_zero(self) -> None:
        """Default fill_value=0.0 appears in ELSE clause."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES ('A'), ('B') t(cat)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([FrequencyEncoder()], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql()
        # ELSE clause with 0.0
        assert "ELSE" in sql.upper()


# -- Not-fitted guard tests -----------------------------------------------------


class TestNotFittedGuard:
    """Calling transform/to_sql before fit() raises NotFittedError."""

    def test_transform_before_fit_raises(self) -> None:
        """transform() on unfitted pipeline raises NotFittedError."""
        from sqlearn.core.errors import NotFittedError

        pipe = Pipeline([FrequencyEncoder()])
        with pytest.raises(NotFittedError):
            pipe.transform("t")

    def test_to_sql_before_fit_raises(self) -> None:
        """to_sql() on unfitted pipeline raises NotFittedError."""
        from sqlearn.core.errors import NotFittedError

        pipe = Pipeline([FrequencyEncoder()])
        with pytest.raises(NotFittedError):
            pipe.to_sql()

    def test_get_feature_names_before_fit_raises(self) -> None:
        """get_feature_names_out() on unfitted pipeline raises NotFittedError."""
        from sqlearn.core.errors import NotFittedError

        pipe = Pipeline([FrequencyEncoder()])
        with pytest.raises(NotFittedError):
            pipe.get_feature_names_out()


# -- Clone and pickle tests -----------------------------------------------------


class TestCloneAndPickle:
    """Deep clone independence and pickle roundtrip."""

    @pytest.fixture
    def fitted_pipe(self) -> Pipeline:
        """Create a fitted pipeline for clone/pickle tests."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 'London'), (2.0, 'Paris'), (3.0, 'Tokyo') t(price, city)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([FrequencyEncoder()], backend=backend)
        pipe.fit("t")
        return pipe

    def test_clone_produces_same_sql(self, fitted_pipe: Pipeline) -> None:
        """Cloned pipeline produces identical SQL."""
        cloned = fitted_pipe.clone()
        assert fitted_pipe.to_sql() == cloned.to_sql()

    def test_clone_sets_independent(self, fitted_pipe: Pipeline) -> None:
        """Modifying cloned step sets does not affect original."""
        cloned = fitted_pipe.clone()
        original_enc = fitted_pipe.steps[0][1]
        cloned_enc = cloned.steps[0][1]
        assert cloned_enc.sets_ is not None
        cloned_enc.sets_["city__freq"] = []
        assert original_enc.sets_ is not None
        assert len(original_enc.sets_["city__freq"]) == 3

    def test_pickle_roundtrip(self) -> None:
        """Pickle an individual encoder preserves sets and params."""
        import pickle

        enc = FrequencyEncoder()
        enc.params_ = {"__total_count": 10}
        enc.sets_ = {"city__freq": [{"city": "A", "_count": 3}, {"city": "B", "_count": 7}]}
        enc._fitted = True
        enc.columns_ = ["city"]
        enc.input_schema_ = Schema({"price": "DOUBLE", "city": "VARCHAR"})
        enc.output_schema_ = enc.output_schema(enc.input_schema_)
        data = pickle.dumps(enc)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.params_ == enc.params_
        assert restored.sets_ == enc.sets_
        assert restored._fitted is True


# -- Edge cases -----------------------------------------------------------------


class TestEdgeCases:
    """Extreme and unusual inputs."""

    def test_single_category(self) -> None:
        """Column with one unique category has frequency=1.0."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES ('A'), ('A'), ('A') t(cat)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([FrequencyEncoder()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (3, 1)
        # All rows should be 1.0 (100% frequency)
        np.testing.assert_allclose(result.flatten().astype(float), [1.0, 1.0, 1.0])

    def test_all_same_value(self) -> None:
        """All identical values produce frequency=1.0."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES (1.0, 'X'), (2.0, 'X'), (3.0, 'X') t(num, cat)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([FrequencyEncoder()], backend=backend)
        result = pipe.fit_transform("t")
        cat_col = result[:, 1].astype(float)
        np.testing.assert_allclose(cat_col, [1.0, 1.0, 1.0])

    def test_null_only_category_column(self) -> None:
        """Column with all NULLs produces fill_value for every row."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, CAST(NULL AS VARCHAR)), "
            "(2.0, CAST(NULL AS VARCHAR)) t(price, cat)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([FrequencyEncoder()], backend=backend)
        result = pipe.fit_transform("t")
        # cat column should be fill_value=0.0 for all rows (no known categories)
        cat_col = result[:, 1].astype(float)
        np.testing.assert_allclose(cat_col, [0.0, 0.0])

    def test_null_mixed_with_values(self) -> None:
        """NULLs mixed with values: NULLs get fill_value, known categories get frequency."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES ('A'), ('B'), (NULL), ('A') t(cat)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([FrequencyEncoder()], backend=backend)
        result = pipe.fit_transform("t")
        cat_col = result[:, 0].astype(float)
        # A appears 2/4=0.5, B appears 1/4=0.25, NULL gets fill_value=0.0
        assert cat_col[0] == pytest.approx(0.5)
        assert cat_col[1] == pytest.approx(0.25)
        assert cat_col[2] == pytest.approx(0.0)
        assert cat_col[3] == pytest.approx(0.5)

    def test_numeric_passthrough(self) -> None:
        """Numeric columns are not encoded (pass through unchanged)."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 100, 'A'), (2.0, 200, 'B') t(price, qty, color)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([FrequencyEncoder()], backend=backend)
        result = pipe.fit_transform("t")
        # price and qty pass through, color encoded = 3 columns
        assert result.shape == (2, 3)
        np.testing.assert_allclose(result[:, 0].astype(float), [1.0, 2.0])
        np.testing.assert_allclose(result[:, 1].astype(float), [100.0, 200.0])

    def test_many_categories(self) -> None:
        """Column with many unique categories all get correct frequencies."""
        conn = duckdb.connect()
        values = ", ".join([f"('{chr(65 + i)}')" for i in range(26)])
        conn.execute(f"CREATE TABLE t AS SELECT * FROM VALUES {values} t(letter)")  # noqa: S608
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([FrequencyEncoder()], backend=backend)
        result = pipe.fit_transform("t")
        # 26 rows x 1 column
        assert result.shape == (26, 1)
        # Each letter appears once out of 26 = 1/26
        expected_freq = 1.0 / 26.0
        np.testing.assert_allclose(result.flatten().astype(float), [expected_freq] * 26)


# -- SQL snapshot tests ---------------------------------------------------------


class TestSqlSnapshot:
    """Verify SQL output structure and patterns."""

    @pytest.fixture
    def fitted_pipe(self) -> Pipeline:
        """Create fitted pipeline for SQL verification."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 'London'), (2.0, 'Paris'), (3.0, 'Tokyo') t(price, city)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([FrequencyEncoder()], backend=backend)
        pipe.fit("t")
        return pipe

    def test_sql_has_case_when(self, fitted_pipe: Pipeline) -> None:
        """SQL contains CASE WHEN for frequency encoding."""
        sql = fitted_pipe.to_sql().upper()
        assert "CASE" in sql
        assert "WHEN" in sql

    def test_sql_has_select_from(self, fitted_pipe: Pipeline) -> None:
        """SQL contains SELECT and FROM."""
        sql = fitted_pipe.to_sql().upper()
        assert "SELECT" in sql
        assert "FROM" in sql

    def test_sql_has_else_clause(self, fitted_pipe: Pipeline) -> None:
        """SQL contains ELSE for unknown category handling."""
        sql = fitted_pipe.to_sql().upper()
        assert "ELSE" in sql

    def test_sql_custom_table(self, fitted_pipe: Pipeline) -> None:
        """to_sql(table=...) uses custom table name."""
        sql = fitted_pipe.to_sql(table="raw_data")
        assert "raw_data" in sql

    def test_sql_preserves_original_column_name(self, fitted_pipe: Pipeline) -> None:
        """Output SQL uses original column name (in-place replacement)."""
        sql = fitted_pipe.to_sql()
        # Should have city as output name, unlike OneHotEncoder which creates new columns
        assert "city" in sql.lower()


# -- Composition tests ----------------------------------------------------------


class TestComposition:
    """FrequencyEncoder composing with other transformers."""

    def test_imputer_then_encoder(self) -> None:
        """Imputer + FrequencyEncoder: NULLs filled before encoding."""
        from sqlearn.imputers.imputer import Imputer

        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES ('A'), ('B'), (NULL), ('A') t(cat)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Imputer(), FrequencyEncoder()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (4, 1)
        # No None/NaN in output
        result_float = result.flatten().astype(float)
        assert not np.any(np.isnan(result_float))

    def test_encoder_sql_composes_with_imputer(self) -> None:
        """SQL shows COALESCE nested inside CASE WHEN from composition."""
        from sqlearn.imputers.imputer import Imputer

        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES ('A'), ('B'), (NULL) t(cat)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Imputer(), FrequencyEncoder()], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql().upper()
        assert "COALESCE" in sql
        assert "CASE" in sql

    def test_full_pipeline_with_scaler(self) -> None:
        """Imputer + StandardScaler + FrequencyEncoder processes mixed columns."""
        from sqlearn.imputers.imputer import Imputer
        from sqlearn.scalers.standard import StandardScaler

        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 'A'), (NULL, 'B'), (3.0, NULL), (4.0, 'A') t(num, cat)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Imputer(), StandardScaler(), FrequencyEncoder()], backend=backend)
        result = pipe.fit_transform("t")
        # num (scaled) + cat (frequency encoded) = 2 columns
        assert result.shape == (4, 2)


# -- Repr tests -----------------------------------------------------------------


class TestRepr:
    """Test FrequencyEncoder.__repr__."""

    def test_default_repr(self) -> None:
        """Default params shows no args."""
        enc = FrequencyEncoder()
        assert repr(enc) == "FrequencyEncoder()"

    def test_normalize_false_repr(self) -> None:
        """Non-default normalize shows in repr."""
        enc = FrequencyEncoder(normalize=False)
        assert repr(enc) == "FrequencyEncoder(normalize=False)"

    def test_custom_fill_value_repr(self) -> None:
        """Non-default fill_value shows in repr."""
        enc = FrequencyEncoder(fill_value=-1.0)
        assert repr(enc) == "FrequencyEncoder(fill_value=-1.0)"

    def test_all_custom_repr(self) -> None:
        """All non-default params show in repr."""
        enc = FrequencyEncoder(normalize=False, handle_unknown="error", fill_value=-1.0)
        assert repr(enc) == (
            "FrequencyEncoder(normalize=False, handle_unknown='error', fill_value=-1.0)"
        )
