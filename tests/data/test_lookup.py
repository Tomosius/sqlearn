"""Tests for sqlearn.data.lookup -- Lookup mid-pipeline JOIN transformer."""

from __future__ import annotations

import pickle

import duckdb
import numpy as np
import pytest
import sqlglot.expressions as exp

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.errors import SchemaError
from sqlearn.core.pipeline import Pipeline
from sqlearn.core.schema import Schema
from sqlearn.data.lookup import Lookup

# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def backend() -> DuckDBBackend:
    """Create DuckDB backend with test tables for Lookup tests."""
    conn = duckdb.connect()
    conn.execute(
        "CREATE TABLE products AS SELECT * FROM VALUES "
        "(1, 10, 50.0), (2, 20, 30.0), (3, 10, 70.0) "
        "t(product_id, category_id, price)"
    )
    conn.execute(
        "CREATE TABLE categories AS SELECT * FROM VALUES "
        "(10, 'Electronics', 'Tech'), (20, 'Books', 'Media') "
        "t(category_id, category_name, department)"
    )
    return DuckDBBackend(connection=conn)


# ── Constructor tests ─────────────────────────────────────────────


class TestConstructor:
    """Test Lookup constructor validation and attributes."""

    def test_basic_init(self) -> None:
        """Basic initialization with required args."""
        lookup = Lookup("categories", on="category_id")
        assert lookup.source == "categories"
        assert lookup.on == "category_id"
        assert lookup.select is None
        assert lookup.how == "left"
        assert lookup.suffix == "_lookup"

    def test_init_with_select(self) -> None:
        """Initialization with explicit select columns."""
        lookup = Lookup("categories", on="category_id", select=["category_name"])
        assert lookup.select == ["category_name"]

    def test_init_multi_key(self) -> None:
        """Initialization with multiple join keys."""
        lookup = Lookup("ref", on=["k1", "k2"])
        assert lookup.on == ["k1", "k2"]

    def test_init_inner_join(self) -> None:
        """Initialization with how='inner'."""
        lookup = Lookup("ref", on="id", how="inner")
        assert lookup.how == "inner"

    def test_init_custom_suffix(self) -> None:
        """Initialization with custom suffix."""
        lookup = Lookup("ref", on="id", suffix="_ref")
        assert lookup.suffix == "_ref"

    def test_non_string_source_raises(self) -> None:
        """Non-string source raises TypeError."""
        with pytest.raises(TypeError, match="source must be a string"):
            Lookup(123, on="id")  # type: ignore[arg-type]

    def test_non_string_on_raises(self) -> None:
        """Non-string/list on raises TypeError."""
        with pytest.raises(TypeError, match="on must be a string or list"):
            Lookup("ref", on=123)  # type: ignore[arg-type]

    def test_non_string_on_list_element_raises(self) -> None:
        """Non-string element in on list raises TypeError."""
        with pytest.raises(TypeError, match="on keys must be strings"):
            Lookup("ref", on=[123])  # type: ignore[list-item]

    def test_empty_on_list_raises(self) -> None:
        """Empty on list raises ValueError."""
        with pytest.raises(ValueError, match="on must not be an empty list"):
            Lookup("ref", on=[])

    def test_invalid_how_raises(self) -> None:
        """Invalid how raises ValueError."""
        with pytest.raises(ValueError, match="Invalid join type"):
            Lookup("ref", on="id", how="cross")

    def test_empty_select_raises(self) -> None:
        """Empty select list raises ValueError."""
        with pytest.raises(ValueError, match="select must not be an empty list"):
            Lookup("ref", on="id", select=[])

    def test_classification_is_static(self) -> None:
        """Lookup is classified as static."""
        assert Lookup._classification == "static"

    def test_default_columns_is_none(self) -> None:
        """Lookup has no default column routing."""
        assert Lookup._default_columns is None

    def test_get_params(self) -> None:
        """get_params returns constructor kwargs."""
        lookup = Lookup("ref", on="id", select=["name"], how="inner", suffix="_r")
        params = lookup.get_params()
        assert params["source"] == "ref"
        assert params["on"] == "id"
        assert params["select"] == ["name"]
        assert params["how"] == "inner"
        assert params["suffix"] == "_r"

    def test_set_params(self) -> None:
        """set_params updates and returns self."""
        lookup = Lookup("ref", on="id")
        result = lookup.set_params(source="new_ref")
        assert result is lookup
        assert lookup.source == "new_ref"

    def test_repr(self) -> None:
        """repr shows key parameters."""
        lookup = Lookup("categories", on="category_id")
        r = repr(lookup)
        assert "Lookup" in r
        assert "categories" in r


# ── output_schema tests ──────────────────────────────────────────


class TestOutputSchema:
    """Test Lookup.output_schema() adds lookup columns."""

    def _make_lookup(
        self,
        source: str,
        on: str | list[str],
        select: list[str] | None = None,
        suffix: str = "_lookup",
    ) -> Lookup:
        """Create a Lookup with resolved lookup schema."""
        return Lookup(source, on=on, select=select, suffix=suffix)

    def test_adds_selected_columns(self, backend: DuckDBBackend) -> None:
        """Selected lookup columns are added to output schema."""
        lookup = self._make_lookup("categories", on="category_id", select=["category_name"])
        lookup._lookup_schema = backend.describe("categories")
        lookup._select_columns = ["category_name"]

        schema = Schema({"product_id": "INTEGER", "category_id": "INTEGER", "price": "DOUBLE"})
        out = lookup.output_schema(schema)
        assert "category_name" in out.columns
        assert "product_id" in out.columns

    def test_adds_all_non_key_columns(self, backend: DuckDBBackend) -> None:
        """Without select, all non-key lookup columns are added."""
        lookup = self._make_lookup("categories", on="category_id")
        lookup._lookup_schema = backend.describe("categories")
        lookup._select_columns = ["category_name", "department"]

        schema = Schema({"product_id": "INTEGER", "category_id": "INTEGER", "price": "DOUBLE"})
        out = lookup.output_schema(schema)
        assert "category_name" in out.columns
        assert "department" in out.columns

    def test_suffix_for_overlap(self) -> None:
        """Overlapping non-key columns get suffix."""
        lookup = self._make_lookup("ref", on="id", select=["name"], suffix="_ref")
        lookup._lookup_schema = Schema({"id": "INTEGER", "name": "VARCHAR"})
        lookup._select_columns = ["name"]

        schema = Schema({"id": "INTEGER", "name": "VARCHAR", "value": "DOUBLE"})
        out = lookup.output_schema(schema)
        assert "name_ref" in out.columns
        assert "name" in out.columns  # original preserved

    def test_no_suffix_for_non_overlapping(self, backend: DuckDBBackend) -> None:
        """Non-overlapping lookup columns don't get suffix."""
        lookup = self._make_lookup("categories", on="category_id", select=["category_name"])
        lookup._lookup_schema = backend.describe("categories")
        lookup._select_columns = ["category_name"]

        schema = Schema({"product_id": "INTEGER", "category_id": "INTEGER", "price": "DOUBLE"})
        out = lookup.output_schema(schema)
        assert "category_name" in out.columns
        assert "category_name_lookup" not in out.columns

    def test_returns_input_schema_without_lookup_schema(self) -> None:
        """Without lookup_schema, returns input schema unchanged."""
        lookup = Lookup("ref", on="id")
        schema = Schema({"id": "INTEGER", "name": "VARCHAR"})
        out = lookup.output_schema(schema)
        assert out == schema


# ── query() tests ─────────────────────────────────────────────────


class TestQuery:
    """Test Lookup.query() generates correct sqlglot ASTs."""

    def _make_fitted_lookup(  # noqa: PLR0913
        self,
        source: str,
        on: str | list[str],
        schema: Schema,
        lookup_schema: Schema,
        select: list[str] | None = None,
        how: str = "left",
        suffix: str = "_lookup",
    ) -> Lookup:
        """Create a fitted Lookup with schemas set."""
        lookup = Lookup(source, on=on, select=select, how=how, suffix=suffix)
        lookup.input_schema_ = schema
        lookup._lookup_schema = lookup_schema
        keys = [on] if isinstance(on, str) else on
        if select is not None:
            lookup._select_columns = select
        else:
            lookup._select_columns = [c for c in lookup_schema.columns if c not in set(keys)]
        lookup._fitted = True
        return lookup

    def test_produces_select_with_join(self) -> None:
        """query() produces a SELECT with a JOIN."""
        schema = Schema({"id": "INTEGER", "val": "DOUBLE"})
        lookup_schema = Schema({"id": "INTEGER", "info": "VARCHAR"})
        lookup = self._make_fitted_lookup(
            "ref", on="id", schema=schema, lookup_schema=lookup_schema, select=["info"]
        )
        input_q = exp.select(exp.Star()).from_("t")
        result = lookup.query(input_q)
        assert isinstance(result, exp.Select)
        assert result.find(exp.Join) is not None

    def test_wraps_input_as_subquery(self) -> None:
        """query() wraps input query as a subquery."""
        schema = Schema({"id": "INTEGER", "val": "DOUBLE"})
        lookup_schema = Schema({"id": "INTEGER", "info": "VARCHAR"})
        lookup = self._make_fitted_lookup(
            "ref", on="id", schema=schema, lookup_schema=lookup_schema, select=["info"]
        )
        input_q = exp.select(exp.Star()).from_("t")
        result = lookup.query(input_q)
        assert result.find(exp.Subquery) is not None

    def test_left_join_default(self) -> None:
        """Default join type is LEFT JOIN."""
        schema = Schema({"id": "INTEGER", "val": "DOUBLE"})
        lookup_schema = Schema({"id": "INTEGER", "info": "VARCHAR"})
        lookup = self._make_fitted_lookup(
            "ref",
            on="id",
            schema=schema,
            lookup_schema=lookup_schema,
            select=["info"],
            how="left",
        )
        input_q = exp.select(exp.Star()).from_("t")
        result = lookup.query(input_q)
        sql = result.sql(dialect="duckdb").upper()
        assert "LEFT" in sql

    def test_inner_join(self) -> None:
        """how='inner' produces JOIN without LEFT."""
        schema = Schema({"id": "INTEGER", "val": "DOUBLE"})
        lookup_schema = Schema({"id": "INTEGER", "info": "VARCHAR"})
        lookup = self._make_fitted_lookup(
            "ref",
            on="id",
            schema=schema,
            lookup_schema=lookup_schema,
            select=["info"],
            how="inner",
        )
        input_q = exp.select(exp.Star()).from_("t")
        result = lookup.query(input_q)
        sql = result.sql(dialect="duckdb").upper()
        assert "JOIN" in sql

    def test_multi_key_join_condition(self) -> None:
        """Multi-key join produces AND conditions."""
        schema = Schema({"k1": "INTEGER", "k2": "VARCHAR", "val": "DOUBLE"})
        lookup_schema = Schema({"k1": "INTEGER", "k2": "VARCHAR", "info": "VARCHAR"})
        lookup = self._make_fitted_lookup(
            "ref",
            on=["k1", "k2"],
            schema=schema,
            lookup_schema=lookup_schema,
            select=["info"],
        )
        input_q = exp.select(exp.Star()).from_("t")
        result = lookup.query(input_q)
        sql = result.sql(dialect="duckdb").upper()
        assert "AND" in sql

    def test_file_path_source(self) -> None:
        """File path source is properly quoted."""
        schema = Schema({"id": "INTEGER", "val": "DOUBLE"})
        lookup_schema = Schema({"id": "INTEGER", "info": "VARCHAR"})
        lookup = self._make_fitted_lookup(
            "data/ref.parquet",
            on="id",
            schema=schema,
            lookup_schema=lookup_schema,
            select=["info"],
        )
        input_q = exp.select(exp.Star()).from_("t")
        result = lookup.query(input_q)
        sql = result.sql(dialect="duckdb")
        assert "data/ref.parquet" in sql


# ── Pipeline integration tests ────────────────────────────────────


class TestPipelineIntegration:
    """Test Lookup integrated with Pipeline (end-to-end)."""

    def test_basic_lookup_pipeline(self, backend: DuckDBBackend) -> None:
        """Lookup adds columns in a pipeline."""
        pipe = Pipeline(
            [Lookup("categories", on="category_id", select=["category_name", "department"])],
            backend=backend,
        )
        result = pipe.fit_transform("products")
        # products has 3 cols + 2 from lookup = 5
        assert result.shape == (3, 5)

    def test_lookup_output_names(self, backend: DuckDBBackend) -> None:
        """get_feature_names_out includes lookup columns."""
        pipe = Pipeline(
            [Lookup("categories", on="category_id", select=["category_name"])],
            backend=backend,
        )
        pipe.fit("products")
        names = pipe.get_feature_names_out()
        assert "category_name" in names
        assert "product_id" in names
        assert "category_id" in names
        assert "price" in names

    def test_lookup_preserves_row_count_left_join(self, backend: DuckDBBackend) -> None:
        """Left join lookup preserves all input rows."""
        pipe = Pipeline(
            [Lookup("categories", on="category_id", select=["category_name"])],
            backend=backend,
        )
        result = pipe.fit_transform("products")
        assert result.shape[0] == 3  # All 3 products preserved

    def test_lookup_inner_join_filters_rows(self) -> None:
        """Inner join lookup drops unmatched rows."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE main AS SELECT * FROM VALUES (1, 10), (2, 20), (3, 99) t(id, cat_id)"
        )
        conn.execute(
            "CREATE TABLE ref AS SELECT * FROM VALUES (10, 'A'), (20, 'B') t(cat_id, name)"
        )
        be = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [Lookup("ref", on="cat_id", select=["name"], how="inner")],
            backend=be,
        )
        result = pipe.fit_transform("main")
        assert result.shape[0] == 2  # cat_id=99 not in ref

    def test_lookup_all_non_key_columns(self, backend: DuckDBBackend) -> None:
        """select=None adds all non-key lookup columns."""
        pipe = Pipeline(
            [Lookup("categories", on="category_id")],
            backend=backend,
        )
        pipe.fit("products")
        names = pipe.get_feature_names_out()
        assert "category_name" in names
        assert "department" in names

    def test_lookup_to_sql(self, backend: DuckDBBackend) -> None:
        """to_sql produces valid SQL with JOIN."""
        pipe = Pipeline(
            [Lookup("categories", on="category_id", select=["category_name"])],
            backend=backend,
        )
        pipe.fit("products")
        sql = pipe.to_sql()
        assert isinstance(sql, str)
        upper = sql.upper()
        assert "JOIN" in upper

    def test_lookup_sql_executable(self, backend: DuckDBBackend) -> None:
        """to_sql output is executable DuckDB SQL."""
        pipe = Pipeline(
            [Lookup("categories", on="category_id", select=["category_name"])],
            backend=backend,
        )
        pipe.fit("products")
        sql = pipe.to_sql().replace("__input__", "products")
        conn = backend._get_connection()
        rows = conn.execute(sql).fetchall()
        assert len(rows) == 3

    def test_fit_then_transform(self, backend: DuckDBBackend) -> None:
        """Separate fit() and transform() match fit_transform()."""
        pipe1 = Pipeline(
            [Lookup("categories", on="category_id", select=["category_name"])],
            backend=backend,
        )
        result1 = pipe1.fit_transform("products")

        pipe2 = Pipeline(
            [Lookup("categories", on="category_id", select=["category_name"])],
            backend=backend,
        )
        pipe2.fit("products")
        result2 = pipe2.transform("products")

        np.testing.assert_array_equal(result1, result2)


# ── Classification tests ──────────────────────────────────────────


class TestClassification:
    """Test Lookup transformer classification."""

    def test_static_classification(self) -> None:
        """Lookup._classification is 'static'."""
        assert Lookup._classification == "static"

    def test_classify_returns_static(self) -> None:
        """_classify() returns 'static'."""
        lookup = Lookup("ref", on="id")
        assert lookup._classify() == "static"

    def test_discover_returns_empty(self) -> None:
        """discover() returns empty dict (static transformer)."""
        lookup = Lookup("ref", on="id")
        schema = Schema({"id": "INTEGER"})
        assert lookup.discover(["id"], schema) == {}

    def test_discover_sets_returns_empty(self) -> None:
        """discover_sets() returns empty dict."""
        lookup = Lookup("ref", on="id")
        schema = Schema({"id": "INTEGER"})
        assert lookup.discover_sets(["id"], schema) == {}

    def test_expressions_returns_empty(self) -> None:
        """expressions() returns empty dict (uses query instead)."""
        lookup = Lookup("ref", on="id")
        assert lookup.expressions([], {}) == {}


# ── Clone tests ──────────────────────────────────────────────────


class TestClone:
    """Test clone() produces independent copies."""

    def test_clone_preserves_params(self) -> None:
        """Cloned Lookup has same parameters."""
        lookup = Lookup("ref", on="id", select=["name"], how="inner", suffix="_r")
        cloned = lookup.clone()
        assert cloned.source == "ref"
        assert cloned.on == "id"
        assert cloned.select == ["name"]
        assert cloned.how == "inner"
        assert cloned.suffix == "_r"

    def test_clone_is_independent(self) -> None:
        """Modifying clone does not affect original."""
        lookup = Lookup("ref", on="id")
        cloned = lookup.clone()
        cloned.set_params(source="new_ref")
        assert lookup.source == "ref"
        assert cloned.source == "new_ref"

    def test_clone_is_different_object(self) -> None:
        """Clone is a different object."""
        lookup = Lookup("ref", on="id")
        cloned = lookup.clone()
        assert cloned is not lookup


# ── Pickle tests ─────────────────────────────────────────────────


class TestPickle:
    """Test pickle serialization roundtrip."""

    def test_pickle_roundtrip(self) -> None:
        """Pickle preserves parameters."""
        lookup = Lookup("ref", on="id", select=["name"], how="inner")
        data = pickle.dumps(lookup)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.source == "ref"
        assert restored.on == "id"
        assert restored.select == ["name"]
        assert restored.how == "inner"

    def test_pickle_fitted(self, backend: DuckDBBackend) -> None:
        """Pickle preserves fitted state."""
        pipe = Pipeline(
            [Lookup("categories", on="category_id", select=["category_name"])],
            backend=backend,
        )
        pipe.fit("products")
        step = pipe.named_steps["step_00"]
        data = pickle.dumps(step)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.source == "categories"
        assert restored.is_fitted


# ── Resolve lookup validation tests ──────────────────────────────


class TestResolveLookup:
    """Test _resolve_lookup validation."""

    def test_missing_key_in_lookup_raises(self) -> None:
        """Missing join key in lookup source raises SchemaError."""
        lookup = Lookup("ref", on="nonexistent")
        lookup._lookup_schema = Schema({"id": "INTEGER", "name": "VARCHAR"})
        with pytest.raises(SchemaError, match="Join key"):
            lookup._resolve_lookup()

    def test_missing_select_column_raises(self) -> None:
        """Missing select column in lookup source raises SchemaError."""
        lookup = Lookup("ref", on="id", select=["nonexistent"])
        lookup._lookup_schema = Schema({"id": "INTEGER", "name": "VARCHAR"})
        with pytest.raises(SchemaError, match="Select column"):
            lookup._resolve_lookup()

    def test_no_lookup_schema_raises(self) -> None:
        """Calling _resolve_lookup without schema raises SchemaError."""
        lookup = Lookup("ref", on="id")
        with pytest.raises(SchemaError, match="not resolved"):
            lookup._resolve_lookup()


# ── Edge cases ───────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases and unusual inputs."""

    def test_single_column_lookup(self) -> None:
        """Lookup that adds a single column."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE main AS SELECT * FROM VALUES (1), (2) t(id)")
        conn.execute("CREATE TABLE ref AS SELECT * FROM VALUES (1, 'A'), (2, 'B') t(id, name)")
        be = DuckDBBackend(connection=conn)
        pipe = Pipeline([Lookup("ref", on="id", select=["name"])], backend=be)
        result = pipe.fit_transform("main")
        assert result.shape == (2, 2)

    def test_lookup_with_nulls(self) -> None:
        """Lookup handles NULL values in join keys (no match)."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE main AS SELECT * FROM VALUES (1, 10), (2, NULL::INTEGER) t(id, cat_id)"
        )
        conn.execute("CREATE TABLE ref AS SELECT * FROM VALUES (10, 'A') t(cat_id, name)")
        be = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [Lookup("ref", on="cat_id", select=["name"])],
            backend=be,
        )
        result = pipe.fit_transform("main")
        assert result.shape == (2, 3)  # All rows preserved (left join)

    def test_lookup_preserves_data(self, backend: DuckDBBackend) -> None:
        """Lookup preserves original column values."""
        pipe = Pipeline(
            [Lookup("categories", on="category_id", select=["category_name"])],
            backend=backend,
        )
        result = pipe.fit_transform("products")
        # price column (index 2) should be unchanged
        prices = result[:, 2].astype(float)
        np.testing.assert_array_equal(sorted(prices), [30.0, 50.0, 70.0])

    def test_lookup_multi_key_pipeline(self) -> None:
        """Multi-key lookup works in pipeline."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE main AS SELECT * FROM VALUES "
            "(1, 'a', 100.0), (2, 'b', 200.0) t(k1, k2, val)"
        )
        conn.execute(
            "CREATE TABLE ref AS SELECT * FROM VALUES (1, 'a', 'X'), (2, 'b', 'Y') t(k1, k2, info)"
        )
        be = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [Lookup("ref", on=["k1", "k2"], select=["info"])],
            backend=be,
        )
        result = pipe.fit_transform("main")
        assert result.shape == (2, 4)  # k1, k2, val, info
