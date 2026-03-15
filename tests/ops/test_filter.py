"""Tests for sqlearn.ops.filter -- Filter."""

from __future__ import annotations

import pickle

import duckdb
import numpy as np
import pytest
import sqlglot
import sqlglot.expressions as exp

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.pipeline import Pipeline
from sqlearn.ops.filter import Filter

# -- Constructor tests --------------------------------------------------------


class TestConstructor:
    """Test Filter constructor and validation."""

    def test_valid_condition(self) -> None:
        """Valid SQL condition is accepted and stored."""
        f = Filter(condition="price > 0")
        assert f.condition == "price > 0"

    def test_condition_parsed_at_init(self) -> None:
        """Condition is parsed into sqlglot AST at init time."""
        f = Filter(condition="price > 0")
        assert isinstance(f._condition_ast, exp.Expression)

    def test_empty_condition_raises(self) -> None:
        """Empty string condition raises ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            Filter(condition="")

    def test_whitespace_only_condition_raises(self) -> None:
        """Whitespace-only condition raises ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            Filter(condition="   ")

    def test_invalid_sql_raises(self) -> None:
        """Invalid SQL condition raises ParseError at init."""
        with pytest.raises(sqlglot.errors.ParseError):
            Filter(condition=">>> NOT VALID SQL <<<")

    def test_non_string_condition_raises(self) -> None:
        """Non-string condition raises ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            Filter(condition=42)  # type: ignore[arg-type]

    def test_classification_is_static(self) -> None:
        """Filter is classified as static (no data learning needed)."""
        assert Filter._classification == "static"

    def test_default_columns_is_none(self) -> None:
        """Filter has no default column routing."""
        assert Filter._default_columns is None

    def test_get_params(self) -> None:
        """get_params returns constructor kwargs."""
        f = Filter(condition="price > 0")
        params = f.get_params()
        assert params == {"condition": "price > 0"}

    def test_set_params(self) -> None:
        """set_params updates condition and returns self."""
        f = Filter(condition="price > 0")
        result = f.set_params(condition="price > 10")
        assert result is f
        assert f.condition == "price > 10"

    def test_repr(self) -> None:
        """repr shows the condition string."""
        f = Filter(condition="price > 0")
        assert repr(f) == "Filter(condition='price > 0')"


# -- expressions() tests -----------------------------------------------------


class TestExpressions:
    """Test Filter.expressions() returns empty dict."""

    def test_returns_empty_dict(self) -> None:
        """expressions() returns empty dict since Filter uses query()."""
        f = Filter(condition="price > 0")
        exprs = {"a": exp.Column(this="a"), "b": exp.Column(this="b")}
        result = f.expressions(["a", "b"], exprs)
        assert result == {}


# -- query() tests -----------------------------------------------------------


class TestQuery:
    """Test Filter.query() generates correct SQL AST."""

    def test_generates_select_star(self) -> None:
        """query() generates SELECT * FROM (input) WHERE condition."""
        f = Filter(condition="price > 0")
        input_q = exp.select(exp.Star()).from_("t")
        result = f.query(input_q)
        assert isinstance(result, exp.Select)
        # Should have a Star expression
        star_found = any(isinstance(e, exp.Star) for e in result.expressions)
        assert star_found

    def test_has_where_clause(self) -> None:
        """query() output contains a WHERE clause."""
        f = Filter(condition="price > 0")
        input_q = exp.select(exp.Star()).from_("t")
        result = f.query(input_q)
        where = result.find(exp.Where)
        assert where is not None

    def test_wraps_input_as_subquery(self) -> None:
        """query() wraps input query in a subquery."""
        f = Filter(condition="price > 0")
        input_q = exp.select(exp.Star()).from_("t")
        result = f.query(input_q)
        subquery = result.find(exp.Subquery)
        assert subquery is not None

    def test_subquery_alias_is_input(self) -> None:
        """Subquery is aliased as __input__."""
        f = Filter(condition="price > 0")
        input_q = exp.select(exp.Star()).from_("t")
        result = f.query(input_q)
        subquery = result.find(exp.Subquery)
        assert subquery is not None
        alias = subquery.alias
        assert alias == "__input__"

    def test_condition_in_where(self) -> None:
        """WHERE clause contains the parsed condition."""
        f = Filter(condition="price > 0")
        input_q = exp.select(exp.Star()).from_("t")
        result = f.query(input_q)
        sql = result.sql(dialect="duckdb")
        assert "price > 0" in sql.lower()

    def test_condition_ast_copied(self) -> None:
        """Condition AST is copied to avoid node sharing bugs."""
        f = Filter(condition="price > 0")
        input_q1 = exp.select(exp.Star()).from_("t1")
        input_q2 = exp.select(exp.Star()).from_("t2")
        r1 = f.query(input_q1)
        r2 = f.query(input_q2)
        # Both results should be independent
        w1 = r1.find(exp.Where)
        w2 = r2.find(exp.Where)
        assert w1 is not None
        assert w2 is not None
        assert w1.this is not w2.this  # different AST nodes

    def test_sql_output_valid(self) -> None:
        """Generated SQL is valid and parseable."""
        f = Filter(condition="price > 0")
        input_q = exp.select(exp.Star()).from_("t")
        result = f.query(input_q)
        sql = result.sql(dialect="duckdb")
        parsed = sqlglot.parse_one(sql)
        assert isinstance(parsed, exp.Select)


# -- Various condition types --------------------------------------------------


class TestConditionTypes:
    """Test various SQL condition patterns."""

    def _query_sql(self, condition: str) -> str:
        """Generate SQL for a given condition."""
        f = Filter(condition=condition)
        input_q = exp.select(exp.Star()).from_("t")
        result = f.query(input_q)
        return result.sql(dialect="duckdb")

    def test_comparison(self) -> None:
        """Simple comparison: price > 0."""
        sql = self._query_sql("price > 0")
        assert "WHERE" in sql.upper()

    def test_is_not_null(self) -> None:
        """IS NOT NULL condition."""
        sql = self._query_sql("city IS NOT NULL")
        assert "NOT" in sql.upper()
        assert "NULL" in sql.upper()

    def test_and_condition(self) -> None:
        """AND compound condition."""
        sql = self._query_sql("price > 0 AND quantity > 10")
        assert "AND" in sql.upper()

    def test_or_condition(self) -> None:
        """OR compound condition."""
        sql = self._query_sql("price > 0 OR quantity > 10")
        assert "OR" in sql.upper()

    def test_between_condition(self) -> None:
        """BETWEEN condition."""
        sql = self._query_sql("age BETWEEN 18 AND 65")
        assert "BETWEEN" in sql.upper()

    def test_in_condition(self) -> None:
        """IN condition."""
        sql = self._query_sql("city IN ('London', 'Paris')")
        assert "IN" in sql.upper()

    def test_like_condition(self) -> None:
        """LIKE condition."""
        sql = self._query_sql("name LIKE '%smith%'")
        assert "LIKE" in sql.upper()

    def test_not_equal(self) -> None:
        """Not-equal condition."""
        sql = self._query_sql("status <> 'deleted'")
        assert "WHERE" in sql.upper()


# -- Pipeline integration tests -----------------------------------------------


class TestPipeline:
    """Test Filter integrated with Pipeline (end-to-end)."""

    @pytest.fixture
    def backend(self) -> DuckDBBackend:
        """Create DuckDB backend with test data."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 10.0), (2.0, 20.0), (-1.0, 30.0), "
            "(3.0, 40.0), (0.0, 50.0), (-2.0, 60.0) t(price, quantity)"
        )
        return DuckDBBackend(connection=conn)

    def test_fit_transform_filters_rows(self, backend: DuckDBBackend) -> None:
        """Rows not matching condition are excluded from output."""
        pipe = Pipeline([Filter(condition="price > 0")], backend=backend)
        result = pipe.fit_transform("t")
        # Only rows with price > 0: (1, 10), (2, 20), (3, 40)
        assert result.shape[0] == 3

    def test_fit_transform_preserves_columns(self, backend: DuckDBBackend) -> None:
        """All columns pass through unchanged."""
        pipe = Pipeline([Filter(condition="price > 0")], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape[1] == 2

    def test_fit_transform_correct_values(self, backend: DuckDBBackend) -> None:
        """Filtered values are correct."""
        pipe = Pipeline([Filter(condition="price > 0")], backend=backend)
        result = pipe.fit_transform("t")
        prices = sorted(result[:, 0].tolist())
        assert prices == [1.0, 2.0, 3.0]

    def test_to_sql_contains_where(self, backend: DuckDBBackend) -> None:
        """to_sql() output contains WHERE clause."""
        pipe = Pipeline([Filter(condition="price > 0")], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql()
        assert "WHERE" in sql.upper()
        assert "price > 0" in sql.lower()

    def test_to_sql_valid_sql(self, backend: DuckDBBackend) -> None:
        """to_sql() produces valid, parseable SQL."""
        pipe = Pipeline([Filter(condition="price > 0")], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql()
        parsed = sqlglot.parse_one(sql)
        assert isinstance(parsed, exp.Select)

    def test_fit_then_transform_separate(self, backend: DuckDBBackend) -> None:
        """Separate fit() and transform() produce same result as fit_transform()."""
        pipe1 = Pipeline([Filter(condition="price > 0")], backend=backend)
        result1 = pipe1.fit_transform("t")

        pipe2 = Pipeline([Filter(condition="price > 0")], backend=backend)
        pipe2.fit("t")
        result2 = pipe2.transform("t")

        np.testing.assert_array_equal(result1, result2)

    def test_row_count_reduced(self, backend: DuckDBBackend) -> None:
        """Filtered table has fewer rows than original."""
        pipe = Pipeline([Filter(condition="price > 0")], backend=backend)
        result = pipe.fit_transform("t")
        # Original has 6 rows, filter keeps 3
        assert result.shape[0] < 6

    def test_condition_matches_all_rows(self, backend: DuckDBBackend) -> None:
        """Condition matching all rows returns all data."""
        pipe = Pipeline([Filter(condition="quantity > 0")], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape[0] == 6

    def test_condition_matches_no_rows(self, backend: DuckDBBackend) -> None:
        """Condition matching no rows returns empty array."""
        pipe = Pipeline([Filter(condition="price > 1000")], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape[0] == 0
        assert result.shape[1] == 2

    def test_output_shape_unchanged_columns(self, backend: DuckDBBackend) -> None:
        """Schema (column count/names) is unchanged after filter."""
        pipe = Pipeline([Filter(condition="price > 0")], backend=backend)
        pipe.fit("t")
        names = pipe.get_feature_names_out()
        assert names == ["price", "quantity"]


# -- Composition tests --------------------------------------------------------


class TestComposition:
    """Test Filter composing with other transformers."""

    def test_filter_before_scaler(self) -> None:
        """Filter followed by StandardScaler produces valid results."""
        from sqlearn.scalers.standard import StandardScaler

        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 10.0), (2.0, 20.0), (-1.0, 30.0), "
            "(3.0, 40.0), (0.0, 50.0) t(a, b)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [Filter(condition="a > 0"), StandardScaler()],
            backend=backend,
        )
        result = pipe.fit_transform("t")
        # After filtering: 3 rows with a > 0
        assert result.shape == (3, 2)

    def test_filter_after_scaler(self) -> None:
        """StandardScaler followed by Filter compiles and runs."""
        from sqlearn.scalers.standard import StandardScaler

        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 10.0), (2.0, 20.0), (3.0, 30.0), "
            "(4.0, 40.0), (5.0, 50.0) t(a, b)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [StandardScaler(), Filter(condition="a > 0")],
            backend=backend,
        )
        result = pipe.fit_transform("t")
        # StandardScaler then Filter -- rows where standardized a > 0
        assert result.shape[1] == 2
        # At least some rows should pass (a=4,5 have standardized a > 0)
        assert result.shape[0] > 0

    def test_multiple_filters(self) -> None:
        """Multiple sequential filters narrow results progressively."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 10.0), (2.0, 20.0), (3.0, 30.0), "
            "(4.0, 40.0), (5.0, 50.0) t(a, b)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [Filter(condition="a > 1"), Filter(condition="a < 5")],
            backend=backend,
        )
        result = pipe.fit_transform("t")
        # a > 1 AND a < 5 => rows (2, 3, 4)
        assert result.shape[0] == 3

    def test_filter_sql_shows_cte(self) -> None:
        """Filter in pipeline produces CTE in SQL output."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1.0), (2.0), (3.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Filter(condition="a > 0")], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql()
        # Filter uses query() which promotes to CTE
        assert "WHERE" in sql.upper()


# -- Clone and pickle tests ---------------------------------------------------


class TestClonePickle:
    """Deep clone independence and pickle roundtrip."""

    def test_clone_roundtrip(self) -> None:
        """Cloned Filter has same condition but is independent."""
        f = Filter(condition="price > 0")
        cloned = f.clone()
        assert cloned.condition == "price > 0"
        assert cloned is not f

    def test_clone_independence(self) -> None:
        """Modifying clone does not affect original."""
        f = Filter(condition="price > 0")
        cloned = f.clone()
        cloned.set_params(condition="price > 10")
        assert f.condition == "price > 0"
        assert cloned.condition == "price > 10"

    def test_pickle_roundtrip(self) -> None:
        """Pickle a Filter preserves condition."""
        f = Filter(condition="price > 0 AND quantity IS NOT NULL")
        data = pickle.dumps(f)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.condition == "price > 0 AND quantity IS NOT NULL"

    def test_pickle_condition_ast_reparsed(self) -> None:
        """Pickled Filter still has valid condition AST."""
        f = Filter(condition="price > 0")
        data = pickle.dumps(f)
        restored = pickle.loads(data)  # noqa: S301
        assert isinstance(restored._condition_ast, exp.Expression)


# -- Edge cases ---------------------------------------------------------------


class TestEdgeCases:
    """Extreme and unusual inputs."""

    def test_equality_condition(self) -> None:
        """Equality filter works."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1.0), (2.0), (1.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Filter(condition="a = 1")], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape[0] == 2

    def test_is_null_condition(self) -> None:
        """IS NULL filter works."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1.0), (NULL), (3.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Filter(condition="a IS NULL")], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape[0] == 1

    def test_single_row_table(self) -> None:
        """Filter on single-row table works."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (5.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Filter(condition="a > 0")], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (1, 1)

    def test_mixed_types_table(self) -> None:
        """Filter with mixed column types preserves all columns."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES (1.0, 'a'), (2.0, 'b'), (3.0, 'c') t(num, cat)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Filter(condition="num > 1")], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape[0] == 2
        assert result.shape[1] == 2

    def test_complex_condition(self) -> None:
        """Complex compound condition works end-to-end."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 10.0), (2.0, 20.0), (3.0, 5.0), "
            "(4.0, 40.0), (5.0, 50.0) t(a, b)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [Filter(condition="a > 2 AND b > 10")],
            backend=backend,
        )
        result = pipe.fit_transform("t")
        # a > 2 AND b > 10: (4, 40), (5, 50) -- (3, 5) excluded by b > 10
        assert result.shape[0] == 2

    def test_not_fitted_guard(self) -> None:
        """Accessing transform before fit raises NotFittedError."""
        from sqlearn.core.errors import NotFittedError

        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1.0) t(a)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Filter(condition="a > 0")], backend=backend)
        with pytest.raises(NotFittedError):
            pipe.transform("t")
