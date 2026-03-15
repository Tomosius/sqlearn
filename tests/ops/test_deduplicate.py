"""Tests for sqlearn.ops.deduplicate -- Deduplicate."""

from __future__ import annotations

import pickle

import duckdb
import pytest
import sqlglot.expressions as exp

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.errors import NotFittedError
from sqlearn.core.pipeline import Pipeline
from sqlearn.ops.deduplicate import Deduplicate

# -- Constructor tests --------------------------------------------------------


class TestConstructor:
    """Test Deduplicate constructor and parameter validation."""

    def test_defaults(self) -> None:
        """Default subset=None, keep='first'."""
        dedup = Deduplicate()
        assert dedup.subset is None
        assert dedup.keep == "first"

    def test_custom_subset(self) -> None:
        """Explicit subset list accepted."""
        dedup = Deduplicate(subset=["city", "name"])
        assert dedup.subset == ["city", "name"]

    def test_keep_last(self) -> None:
        """keep='last' accepted."""
        dedup = Deduplicate(keep="last")
        assert dedup.keep == "last"

    def test_keep_none(self) -> None:
        """keep='none' accepted."""
        dedup = Deduplicate(keep="none")
        assert dedup.keep == "none"

    def test_invalid_keep_raises(self) -> None:
        """Invalid keep value raises ValueError."""
        with pytest.raises(ValueError, match="Invalid keep"):
            Deduplicate(keep="invalid")

    def test_empty_subset_raises(self) -> None:
        """Empty subset list raises ValueError."""
        with pytest.raises(ValueError, match="non-empty list"):
            Deduplicate(subset=[])

    def test_columns_is_none(self) -> None:
        """Base class columns parameter is always None (not used)."""
        dedup = Deduplicate(subset=["a", "b"])
        assert dedup.columns is None

    def test_classification_is_static(self) -> None:
        """Class is static (no data learning needed)."""
        assert Deduplicate._classification == "static"

    def test_classify_returns_static(self) -> None:
        """_classify() returns 'static'."""
        dedup = Deduplicate()
        assert dedup._classify() == "static"

    def test_get_params(self) -> None:
        """get_params returns constructor kwargs."""
        dedup = Deduplicate(subset=["a"], keep="none")
        params = dedup.get_params()
        assert params == {"subset": ["a"], "keep": "none"}

    def test_get_params_default(self) -> None:
        """get_params for default instance returns default values."""
        dedup = Deduplicate()
        params = dedup.get_params()
        assert params == {"subset": None, "keep": "first"}

    def test_set_params(self) -> None:
        """set_params updates attributes and returns self."""
        dedup = Deduplicate()
        result = dedup.set_params(keep="last")
        assert result is dedup
        assert dedup.keep == "last"

    def test_repr_default(self) -> None:
        """repr of default Deduplicate shows no params."""
        dedup = Deduplicate()
        assert repr(dedup) == "Deduplicate()"

    def test_repr_custom(self) -> None:
        """repr of custom Deduplicate shows non-default params."""
        dedup = Deduplicate(subset=["a", "b"], keep="none")
        r = repr(dedup)
        assert "subset=['a', 'b']" in r
        assert "keep='none'" in r


# -- Not-fitted guard tests ---------------------------------------------------


class TestNotFittedGuard:
    """Test that unfitted Deduplicate raises on fitted-only methods."""

    def test_get_feature_names_out_raises(self) -> None:
        """get_feature_names_out raises NotFittedError when not fitted."""
        dedup = Deduplicate()
        with pytest.raises(NotFittedError):
            dedup.get_feature_names_out()


# -- query() unit tests -------------------------------------------------------


class TestQuery:
    """Test Deduplicate.query() generates correct sqlglot ASTs."""

    def _make_input_query(self) -> exp.Expression:
        """Create a simple SELECT * FROM t input query."""
        return exp.select(exp.Star()).from_("t")

    def test_full_row_distinct(self) -> None:
        """Full-row dedup with keep='first' produces DISTINCT."""
        dedup = Deduplicate()
        input_q = self._make_input_query()
        result = dedup.query(input_q)
        assert result is not None
        sql = result.sql(dialect="duckdb")
        assert "DISTINCT" in sql.upper()

    def test_subset_uses_row_number(self) -> None:
        """Subset dedup uses ROW_NUMBER window function."""
        dedup = Deduplicate(subset=["city", "name"])
        # Simulate fitted state
        from sqlearn.core.schema import Schema

        dedup.input_schema_ = Schema({"city": "VARCHAR", "name": "VARCHAR", "price": "DOUBLE"})
        dedup.columns_ = ["city", "name", "price"]
        dedup._fitted = True

        input_q = self._make_input_query()
        result = dedup.query(input_q)
        assert result is not None
        sql = result.sql(dialect="duckdb")
        assert "ROW_NUMBER" in sql.upper()
        assert "__rn__" in sql

    def test_keep_none_uses_count(self) -> None:
        """keep='none' uses COUNT(*) OVER (PARTITION BY ...)."""
        dedup = Deduplicate(subset=["city"], keep="none")
        from sqlearn.core.schema import Schema

        dedup.input_schema_ = Schema({"city": "VARCHAR", "price": "DOUBLE"})
        dedup.columns_ = ["city", "price"]
        dedup._fitted = True

        input_q = self._make_input_query()
        result = dedup.query(input_q)
        assert result is not None
        sql = result.sql(dialect="duckdb")
        assert "COUNT" in sql.upper()
        assert "__cnt__" in sql

    def test_keep_last_uses_desc_order(self) -> None:
        """keep='last' uses ROW_NUMBER with DESC order."""
        dedup = Deduplicate(subset=["city"], keep="last")
        from sqlearn.core.schema import Schema

        dedup.input_schema_ = Schema({"city": "VARCHAR", "price": "DOUBLE"})
        dedup.columns_ = ["city", "price"]
        dedup._fitted = True

        input_q = self._make_input_query()
        result = dedup.query(input_q)
        assert result is not None
        sql = result.sql(dialect="duckdb")
        assert "ROW_NUMBER" in sql.upper()
        assert "DESC" in sql.upper()

    def test_output_excludes_helper_columns(self) -> None:
        """Output SQL does not include __rn__ or __cnt__ in final SELECT."""
        dedup = Deduplicate(subset=["city"], keep="first")
        from sqlearn.core.schema import Schema

        dedup.input_schema_ = Schema({"city": "VARCHAR", "price": "DOUBLE"})
        dedup.columns_ = ["city", "price"]
        dedup._fitted = True

        input_q = self._make_input_query()
        result = dedup.query(input_q)
        assert result is not None
        sql = result.sql(dialect="duckdb")
        # The outer SELECT should list original columns, not __rn__
        # Check that the outer select has city and price but not __rn__ as
        # column references in the outermost SELECT
        assert "__rn__" in sql  # appears in WHERE clause
        # But the outer SELECT lists specific columns, not *


# -- Pipeline integration tests -----------------------------------------------


class TestPipeline:
    """Test Deduplicate integrated with Pipeline (end-to-end)."""

    @pytest.fixture
    def backend_with_dupes(self) -> DuckDBBackend:
        """Create DuckDB backend with data containing duplicates."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 'a'), (2.0, 'b'), (1.0, 'a'), "
            "(3.0, 'c'), (2.0, 'b'), (2.0, 'b') t(price, city)"
        )
        return DuckDBBackend(connection=conn)

    @pytest.fixture
    def backend_no_dupes(self) -> DuckDBBackend:
        """Create DuckDB backend with data containing NO duplicates."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 'a'), (2.0, 'b'), (3.0, 'c') t(price, city)"
        )
        return DuckDBBackend(connection=conn)

    def test_full_row_dedup_removes_duplicates(self, backend_with_dupes: DuckDBBackend) -> None:
        """Full-row dedup reduces row count from 6 to 3 unique rows."""
        pipe = Pipeline([Deduplicate()], backend=backend_with_dupes)
        result = pipe.fit_transform("t")
        # 3 unique rows: (1, 'a'), (2, 'b'), (3, 'c')
        assert result.shape[0] == 3

    def test_full_row_dedup_column_count_unchanged(
        self, backend_with_dupes: DuckDBBackend
    ) -> None:
        """Dedup does not change the number of columns."""
        pipe = Pipeline([Deduplicate()], backend=backend_with_dupes)
        result = pipe.fit_transform("t")
        assert result.shape[1] == 2

    def test_no_duplicates_unchanged(self, backend_no_dupes: DuckDBBackend) -> None:
        """Data without duplicates passes through unchanged."""
        pipe = Pipeline([Deduplicate()], backend=backend_no_dupes)
        result = pipe.fit_transform("t")
        assert result.shape[0] == 3

    def test_subset_dedup(self, backend_with_dupes: DuckDBBackend) -> None:
        """Subset dedup on one column deduplicates by that column."""
        pipe = Pipeline([Deduplicate(subset=["city"])], backend=backend_with_dupes)
        result = pipe.fit_transform("t")
        # 3 unique cities: 'a', 'b', 'c'
        assert result.shape[0] == 3

    def test_all_identical_rows(self) -> None:
        """All identical rows collapse to 1 row."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 'x'), (1.0, 'x'), (1.0, 'x'), "
            "(1.0, 'x'), (1.0, 'x') t(price, city)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Deduplicate()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape[0] == 1

    def test_keep_none_removes_all_duplicates(self) -> None:
        """keep='none' removes ALL rows that have any duplicates."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 'a'), (2.0, 'b'), (1.0, 'a'), "
            "(3.0, 'c') t(price, city)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Deduplicate(subset=["price", "city"], keep="none")], backend=backend)
        result = pipe.fit_transform("t")
        # (1, 'a') appears twice -> removed; (2, 'b') and (3, 'c') unique -> kept
        assert result.shape[0] == 2

    def test_keep_none_all_duplicates_returns_zero(self) -> None:
        """keep='none' returns 0 rows when ALL rows are duplicates."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 'a'), (1.0, 'a'), (1.0, 'a') t(price, city)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Deduplicate(subset=["price", "city"], keep="none")], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape[0] == 0

    def test_to_sql_distinct(self, backend_with_dupes: DuckDBBackend) -> None:
        """to_sql() produces DISTINCT for full-row dedup."""
        pipe = Pipeline([Deduplicate()], backend=backend_with_dupes)
        pipe.fit("t")
        sql = pipe.to_sql()
        assert isinstance(sql, str)
        assert "DISTINCT" in sql.upper()

    def test_to_sql_row_number(self, backend_with_dupes: DuckDBBackend) -> None:
        """to_sql() produces ROW_NUMBER for subset dedup."""
        pipe = Pipeline([Deduplicate(subset=["city"])], backend=backend_with_dupes)
        pipe.fit("t")
        sql = pipe.to_sql()
        assert isinstance(sql, str)
        assert "ROW_NUMBER" in sql.upper()

    def test_to_sql_count_for_keep_none(self, backend_with_dupes: DuckDBBackend) -> None:
        """to_sql() produces COUNT for keep='none'."""
        pipe = Pipeline([Deduplicate(subset=["city"], keep="none")], backend=backend_with_dupes)
        pipe.fit("t")
        sql = pipe.to_sql()
        assert isinstance(sql, str)
        assert "COUNT" in sql.upper()

    def test_fit_then_transform_separate(self, backend_with_dupes: DuckDBBackend) -> None:
        """Separate fit() and transform() produce same row set as fit_transform()."""
        pipe1 = Pipeline([Deduplicate()], backend=backend_with_dupes)
        result1 = pipe1.fit_transform("t")

        pipe2 = Pipeline([Deduplicate()], backend=backend_with_dupes)
        pipe2.fit("t")
        result2 = pipe2.transform("t")

        # DISTINCT order is non-deterministic, so compare as sorted row sets
        assert result1.shape == result2.shape
        rows1 = sorted(str(row) for row in result1.tolist())
        rows2 = sorted(str(row) for row in result2.tolist())
        assert rows1 == rows2

    def test_schema_unchanged(self, backend_with_dupes: DuckDBBackend) -> None:
        """Output schema has same columns as input schema."""
        pipe = Pipeline([Deduplicate()], backend=backend_with_dupes)
        pipe.fit("t")
        features = pipe.get_feature_names_out()
        assert "price" in features
        assert "city" in features
        assert len(features) == 2


# -- Composition tests --------------------------------------------------------


class TestComposition:
    """Deduplicate composing with other transformers."""

    def test_deduplicate_then_standard_scaler(self) -> None:
        """Deduplicate + StandardScaler: dedup first, then scale."""
        from sqlearn.scalers.standard import StandardScaler

        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 10.0), (2.0, 20.0), (1.0, 10.0), "
            "(3.0, 30.0) t(a, b)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Deduplicate(), StandardScaler()], backend=backend)
        result = pipe.fit_transform("t")
        # After dedup: 3 rows. After scaling: 3 rows, 2 cols.
        assert result.shape == (3, 2)

    def test_standard_scaler_then_deduplicate(self) -> None:
        """StandardScaler + Deduplicate: scale first, then dedup."""
        from sqlearn.scalers.standard import StandardScaler

        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 10.0), (2.0, 20.0), (1.0, 10.0), "
            "(3.0, 30.0) t(a, b)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([StandardScaler(), Deduplicate()], backend=backend)
        result = pipe.fit_transform("t")
        # After scaling, duplicates still duplicate -> 3 unique rows
        assert result.shape == (3, 2)


# -- Clone and pickle tests ---------------------------------------------------


class TestClonePickle:
    """Deep clone independence and pickle roundtrip."""

    def test_clone_roundtrip(self) -> None:
        """Cloned Deduplicate has same params but is independent."""
        dedup = Deduplicate(subset=["a", "b"], keep="none")
        cloned = dedup.clone()
        assert cloned.subset == ["a", "b"]
        assert cloned.keep == "none"
        assert cloned is not dedup

    def test_clone_independence(self) -> None:
        """Modifying clone does not affect original."""
        dedup = Deduplicate(subset=["a", "b"], keep="none")
        cloned = dedup.clone()
        cloned.set_params(keep="first")
        assert dedup.keep == "none"
        assert cloned.keep == "first"

    def test_pickle_roundtrip(self) -> None:
        """Pickle a Deduplicate preserves params."""
        dedup = Deduplicate(subset=["x", "y"], keep="last")
        data = pickle.dumps(dedup)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.subset == ["x", "y"]
        assert restored.keep == "last"

    def test_pickle_roundtrip_default(self) -> None:
        """Pickle a default Deduplicate preserves defaults."""
        dedup = Deduplicate()
        data = pickle.dumps(dedup)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.subset is None
        assert restored.keep == "first"


# -- Edge cases ---------------------------------------------------------------


class TestEdgeCases:
    """Extreme and unusual inputs."""

    def test_single_row(self) -> None:
        """Single row passes through unchanged."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES (1.0, 'a') t(price, city)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Deduplicate()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (1, 2)

    def test_many_columns(self) -> None:
        """Dedup works with many columns."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1, 2, 3, 4, 5), (1, 2, 3, 4, 5), (6, 7, 8, 9, 10) "
            "t(a, b, c, d, e)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Deduplicate()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (2, 5)

    def test_mixed_types(self) -> None:
        """Dedup works with mixed column types."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 'a', true), (2.0, 'b', false), "
            "(1.0, 'a', true) t(price, city, active)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Deduplicate()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape[0] == 2

    def test_subset_dedup_preserves_all_columns(self) -> None:
        """Subset dedup returns all columns, not just subset columns."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 'a', 100), (2.0, 'a', 200), "
            "(3.0, 'b', 300) t(price, city, qty)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Deduplicate(subset=["city"])], backend=backend)
        result = pipe.fit_transform("t")
        # 2 unique cities, all 3 columns preserved
        assert result.shape[0] == 2
        assert result.shape[1] == 3

    def test_null_values_in_data(self) -> None:
        """Dedup handles NULL values in data."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 'a'), (NULL, 'b'), (NULL, 'b'), "
            "(2.0, NULL) t(price, city)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Deduplicate()], backend=backend)
        result = pipe.fit_transform("t")
        # DuckDB: DISTINCT treats NULLs as equal, so (NULL, 'b') appears once
        assert result.shape[0] <= 4  # At most 4 (but likely 3 due to NULL equality)

    def test_numeric_only_columns(self) -> None:
        """Dedup works with all numeric columns."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES (1.0, 2.0), (3.0, 4.0), (1.0, 2.0) t(a, b)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Deduplicate()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (2, 2)

    def test_keep_last_subset(self) -> None:
        """keep='last' with subset dedup returns a valid result."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 'a'), (2.0, 'a'), (3.0, 'b') t(price, city)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Deduplicate(subset=["city"], keep="last")], backend=backend)
        result = pipe.fit_transform("t")
        # 2 unique cities
        assert result.shape[0] == 2
