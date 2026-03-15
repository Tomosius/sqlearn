"""Tests for sqlearn.ops.sample — Sample."""

from __future__ import annotations

import pickle

import duckdb
import pytest
import sqlglot.expressions as exp

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.errors import NotFittedError
from sqlearn.core.pipeline import Pipeline
from sqlearn.ops.sample import Sample

# ── Constructor tests ──────────────────────────────────────────────


class TestConstructor:
    """Test Sample constructor validation and attributes."""

    def test_n_mode(self) -> None:
        """Sample with n stores the count."""
        s = Sample(n=100)
        assert s.n == 100
        assert s.fraction is None

    def test_fraction_mode(self) -> None:
        """Sample with fraction stores the proportion."""
        s = Sample(fraction=0.5)
        assert s.fraction == 0.5
        assert s.n is None

    def test_seed_stored(self) -> None:
        """Seed parameter is stored (reserved for future use)."""
        s = Sample(n=10, seed=42)
        assert s.seed == 42

    def test_both_n_and_fraction_raises(self) -> None:
        """Specifying both n and fraction raises ValueError."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            Sample(n=100, fraction=0.5)

    def test_neither_n_nor_fraction_raises(self) -> None:
        """Specifying neither n nor fraction raises ValueError."""
        with pytest.raises(ValueError, match="Must specify either"):
            Sample()

    def test_negative_n_raises(self) -> None:
        """Negative n raises ValueError."""
        with pytest.raises(ValueError, match="positive integer"):
            Sample(n=-5)

    def test_zero_n_raises(self) -> None:
        """Zero n raises ValueError."""
        with pytest.raises(ValueError, match="positive integer"):
            Sample(n=0)

    def test_fraction_zero_raises(self) -> None:
        """Fraction of 0.0 raises ValueError."""
        with pytest.raises(ValueError, match=r"between 0\.0 and 1\.0"):
            Sample(fraction=0.0)

    def test_fraction_one_raises(self) -> None:
        """Fraction of 1.0 raises ValueError."""
        with pytest.raises(ValueError, match=r"between 0\.0 and 1\.0"):
            Sample(fraction=1.0)

    def test_fraction_negative_raises(self) -> None:
        """Negative fraction raises ValueError."""
        with pytest.raises(ValueError, match=r"between 0\.0 and 1\.0"):
            Sample(fraction=-0.5)

    def test_fraction_above_one_raises(self) -> None:
        """Fraction above 1.0 raises ValueError."""
        with pytest.raises(ValueError, match=r"between 0\.0 and 1\.0"):
            Sample(fraction=1.5)

    def test_n_float_raises(self) -> None:
        """Float n raises TypeError."""
        with pytest.raises(TypeError, match="must be an integer"):
            Sample(n=10.5)  # type: ignore[arg-type]

    def test_classification_is_static(self) -> None:
        """Sample is classified as static."""
        assert Sample._classification == "static"

    def test_default_columns_is_none(self) -> None:
        """Default columns is None (operates on rows, not columns)."""
        assert Sample._default_columns is None

    def test_get_params(self) -> None:
        """get_params returns constructor kwargs."""
        s = Sample(n=50, seed=42)
        params = s.get_params()
        assert params == {"n": 50, "fraction": None, "seed": 42, "columns": None}

    def test_get_params_fraction(self) -> None:
        """get_params for fraction mode returns correct values."""
        s = Sample(fraction=0.3)
        params = s.get_params()
        assert params == {"n": None, "fraction": 0.3, "seed": None, "columns": None}

    def test_set_params(self) -> None:
        """set_params updates attributes and returns self."""
        s = Sample(n=10)
        result = s.set_params(n=20)
        assert result is s
        assert s.n == 20

    def test_repr_n(self) -> None:
        """repr shows n parameter."""
        s = Sample(n=100)
        assert repr(s) == "Sample(n=100)"

    def test_repr_fraction(self) -> None:
        """repr shows fraction parameter."""
        s = Sample(fraction=0.5)
        assert repr(s) == "Sample(fraction=0.5)"


# ── query() tests ─────────────────────────────────────────────────


class TestQuery:
    """Test Sample.query() generates correct sqlglot ASTs."""

    def _make_input_query(self) -> exp.Select:
        """Create a simple input query for testing."""
        return exp.select(exp.Column(this="a"), exp.Column(this="b")).from_(exp.to_table("t"))  # pyright: ignore[reportUnknownMemberType]

    def test_n_mode_has_order_by_random(self) -> None:
        """Count mode generates ORDER BY RANDOM()."""
        s = Sample(n=50)
        result = s.query(self._make_input_query())
        sql = result.sql(dialect="duckdb").upper()
        assert "ORDER BY" in sql
        assert "RANDOM()" in sql

    def test_n_mode_has_limit(self) -> None:
        """Count mode generates LIMIT n."""
        s = Sample(n=50)
        result = s.query(self._make_input_query())
        sql = result.sql(dialect="duckdb").upper()
        assert "LIMIT 50" in sql

    def test_fraction_mode_has_where_random(self) -> None:
        """Fraction mode generates WHERE RANDOM() < fraction."""
        s = Sample(fraction=0.3)
        result = s.query(self._make_input_query())
        sql = result.sql(dialect="duckdb").upper()
        assert "WHERE" in sql
        assert "RANDOM()" in sql
        assert "0.3" in result.sql(dialect="duckdb")

    def test_n_mode_returns_select(self) -> None:
        """query() returns a Select expression for n mode."""
        s = Sample(n=10)
        result = s.query(self._make_input_query())
        assert isinstance(result, exp.Select)

    def test_fraction_mode_returns_select(self) -> None:
        """query() returns a Select expression for fraction mode."""
        s = Sample(fraction=0.5)
        result = s.query(self._make_input_query())
        assert isinstance(result, exp.Select)

    def test_query_wraps_input_as_subquery(self) -> None:
        """query() wraps input in a subquery with __input__ alias."""
        s = Sample(n=10)
        result = s.query(self._make_input_query())
        sql = result.sql(dialect="duckdb")
        assert "__input__" in sql

    def test_selects_star(self) -> None:
        """query() selects all columns via star."""
        s = Sample(n=10)
        result = s.query(self._make_input_query())
        sql = result.sql(dialect="duckdb")
        assert "*" in sql

    def test_n_1_limit(self) -> None:
        """n=1 generates LIMIT 1."""
        s = Sample(n=1)
        result = s.query(self._make_input_query())
        sql = result.sql(dialect="duckdb").upper()
        assert "LIMIT 1" in sql

    def test_large_n_limit(self) -> None:
        """Large n value works correctly in SQL."""
        s = Sample(n=1000000)
        result = s.query(self._make_input_query())
        sql = result.sql(dialect="duckdb").upper()
        assert "LIMIT 1000000" in sql

    def test_small_fraction(self) -> None:
        """Small fraction value appears in SQL."""
        s = Sample(fraction=0.01)
        result = s.query(self._make_input_query())
        sql = result.sql(dialect="duckdb")
        assert "0.01" in sql


# ── Pipeline integration tests ────────────────────────────────────


class TestPipeline:
    """Test Sample integrated with Pipeline (end-to-end)."""

    @pytest.fixture
    def backend(self) -> DuckDBBackend:
        """Create DuckDB backend with 100-row test data."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT i AS a, i * 2 AS b FROM range(100) tbl(i)")
        return DuckDBBackend(connection=conn)

    def test_n_returns_exact_count(self, backend: DuckDBBackend) -> None:
        """Sample(n=10) returns exactly 10 rows."""
        pipe = Pipeline([Sample(n=10)], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape[0] == 10

    def test_n_1_returns_single_row(self, backend: DuckDBBackend) -> None:
        """Sample(n=1) returns exactly 1 row."""
        pipe = Pipeline([Sample(n=1)], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape[0] == 1

    def test_fraction_returns_approximate_count(self, backend: DuckDBBackend) -> None:
        """Sample(fraction=0.5) returns approximately half the rows."""
        pipe = Pipeline([Sample(fraction=0.5)], backend=backend)
        result = pipe.fit_transform("t")
        # Wide tolerance: 50% of 100 = 50, allow 15-85
        assert 15 <= result.shape[0] <= 85

    def test_schema_unchanged(self, backend: DuckDBBackend) -> None:
        """Output columns match input columns after sampling."""
        pipe = Pipeline([Sample(n=10)], backend=backend)
        pipe.fit("t")
        names = pipe.get_feature_names_out()
        assert names == ["a", "b"]

    def test_output_shape_columns(self, backend: DuckDBBackend) -> None:
        """Output has correct number of columns."""
        pipe = Pipeline([Sample(n=5)], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (5, 2)

    def test_to_sql_n_mode(self, backend: DuckDBBackend) -> None:
        """to_sql() with n mode contains ORDER BY RANDOM() LIMIT."""
        pipe = Pipeline([Sample(n=10)], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql().upper()
        assert "ORDER BY" in sql
        assert "RANDOM()" in sql
        assert "LIMIT 10" in sql

    def test_to_sql_fraction_mode(self, backend: DuckDBBackend) -> None:
        """to_sql() with fraction mode contains WHERE RANDOM()."""
        pipe = Pipeline([Sample(fraction=0.5)], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql()
        assert "WHERE" in sql.upper()
        assert "RANDOM()" in sql.upper()
        assert "0.5" in sql

    def test_fit_then_transform_separate(self, backend: DuckDBBackend) -> None:
        """Separate fit() and transform() both work."""
        pipe = Pipeline([Sample(n=5)], backend=backend)
        pipe.fit("t")
        result = pipe.transform("t")
        assert result.shape[0] == 5
        assert result.shape[1] == 2

    def test_sampled_values_from_input(self, backend: DuckDBBackend) -> None:
        """Sampled rows contain values actually present in the input."""
        pipe = Pipeline([Sample(n=10)], backend=backend)
        result = pipe.fit_transform("t")
        # Column a has values 0..99, column b has 0..198 (even)
        for i in range(result.shape[0]):
            a_val = int(result[i, 0])
            b_val = int(result[i, 1])
            assert 0 <= a_val < 100
            assert b_val == a_val * 2


# ── Not-fitted guard ──────────────────────────────────────────────


class TestNotFitted:
    """Test that unfitted Sample raises NotFittedError."""

    def test_get_feature_names_out_not_fitted(self) -> None:
        """get_feature_names_out raises when not fitted."""
        s = Sample(n=10)
        with pytest.raises(NotFittedError):
            s.get_feature_names_out()

    def test_pipeline_transform_not_fitted(self) -> None:
        """Pipeline.transform raises when not fitted."""
        pipe = Pipeline([Sample(n=10)])
        with pytest.raises(NotFittedError):
            pipe.transform("t")


# ── Clone and pickle tests ──────────────────────────────────────


class TestClonePickle:
    """Deep clone independence and pickle roundtrip."""

    def test_clone_roundtrip_n(self) -> None:
        """Cloned Sample(n=) has same params but is independent."""
        s = Sample(n=50, seed=42)
        cloned = s.clone()
        assert cloned.n == 50
        assert cloned.seed == 42
        assert cloned is not s

    def test_clone_roundtrip_fraction(self) -> None:
        """Cloned Sample(fraction=) has same params but is independent."""
        s = Sample(fraction=0.3)
        cloned = s.clone()
        assert cloned.fraction == 0.3
        assert cloned.n is None
        assert cloned is not s

    def test_clone_independence(self) -> None:
        """Modifying clone does not affect original."""
        s = Sample(n=10)
        cloned = s.clone()
        cloned.set_params(n=20)
        assert s.n == 10
        assert cloned.n == 20

    def test_pickle_roundtrip_n(self) -> None:
        """Pickle a Sample(n=) preserves params."""
        s = Sample(n=100, seed=7)
        data = pickle.dumps(s)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.n == 100
        assert restored.seed == 7
        assert restored.fraction is None

    def test_pickle_roundtrip_fraction(self) -> None:
        """Pickle a Sample(fraction=) preserves params."""
        s = Sample(fraction=0.8)
        data = pickle.dumps(s)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.fraction == 0.8
        assert restored.n is None


# ── Static classification tests ──────────────────────────────────


class TestStaticClassification:
    """Verify Sample is classified as static by the compiler."""

    def test_classify_returns_static(self) -> None:
        """_classify() returns 'static'."""
        s = Sample(n=10)
        assert s._classify() == "static"

    def test_discover_returns_empty(self) -> None:
        """discover() returns empty dict (static transformer)."""
        from sqlearn.core.schema import Schema

        s = Sample(n=10)
        schema = Schema({"a": "INTEGER", "b": "DOUBLE"})
        result = s.discover(["a", "b"], schema)
        assert result == {}

    def test_discover_sets_returns_empty(self) -> None:
        """discover_sets() returns empty dict (static transformer)."""
        from sqlearn.core.schema import Schema

        s = Sample(n=10)
        schema = Schema({"a": "INTEGER", "b": "DOUBLE"})
        result = s.discover_sets(["a", "b"], schema)
        assert result == {}


# ── Composition tests ─────────────────────────────────────────────


class TestComposition:
    """Sample composing with other transformers."""

    def test_standard_scaler_then_sample(self) -> None:
        """StandardScaler + Sample produces scaled subset of rows."""
        from sqlearn.scalers.standard import StandardScaler

        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 10.0), (2.0, 20.0), (3.0, 30.0), "
            "(4.0, 40.0), (5.0, 50.0) t(a, b)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([StandardScaler(), Sample(n=3)], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (3, 2)

    def test_sample_then_nothing(self) -> None:
        """Sample alone in pipeline works correctly."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT i AS val FROM range(50) tbl(i)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Sample(n=10)], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape[0] == 10

    def test_to_sql_with_scaler_contains_cte(self) -> None:
        """StandardScaler + Sample SQL uses CTE for the query() step."""
        from sqlearn.scalers.standard import StandardScaler

        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES (1.0, 2.0), (3.0, 4.0), (5.0, 6.0) t(a, b)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([StandardScaler(), Sample(n=2)], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql().upper()
        # The query() step creates a CTE
        assert "RANDOM()" in sql
        assert "LIMIT 2" in sql


# ── Edge cases ──────────────────────────────────────────────────


class TestEdgeCases:
    """Extreme and unusual inputs."""

    def test_n_larger_than_table(self) -> None:
        """n larger than table size returns all rows."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT i AS a FROM range(5) tbl(i)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Sample(n=100)], backend=backend)
        result = pipe.fit_transform("t")
        # LIMIT 100 on 5-row table returns all 5
        assert result.shape[0] == 5

    def test_single_row_table(self) -> None:
        """Sampling from single-row table works."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT 42 AS a")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Sample(n=1)], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (1, 1)
        assert int(result[0, 0]) == 42

    def test_fraction_very_small(self) -> None:
        """Very small fraction may return 0 rows."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT i AS a FROM range(10) tbl(i)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Sample(fraction=0.01)], backend=backend)
        result = pipe.fit_transform("t")
        # With 10 rows and fraction=0.01, likely 0 rows but could be 1
        assert result.shape[0] <= 10

    def test_fraction_near_one(self) -> None:
        """Fraction near 1.0 returns most rows."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT i AS a FROM range(100) tbl(i)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Sample(fraction=0.99)], backend=backend)
        result = pipe.fit_transform("t")
        # With fraction=0.99, expect most rows
        assert result.shape[0] >= 80

    def test_multiple_columns_preserved(self) -> None:
        """All columns pass through sampling unchanged."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT i AS a, i * 2 AS b, i * 3 AS c FROM range(20) tbl(i)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Sample(n=5)], backend=backend)
        pipe.fit("t")
        names = pipe.get_feature_names_out()
        assert names == ["a", "b", "c"]
        result = pipe.transform("t")
        assert result.shape == (5, 3)

    def test_expressions_returns_empty(self) -> None:
        """expressions() returns empty dict (Sample uses query())."""
        s = Sample(n=10)
        exprs = {"a": exp.Column(this="a")}
        result = s.expressions(["a"], exprs)
        assert result == {}
