"""Tests for sqlearn.encoders.onehot — OneHotEncoder."""

from __future__ import annotations

from typing import Any

import duckdb
import numpy as np
import pytest
import sqlglot.expressions as exp

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.compiler import _collect_set_queries
from sqlearn.core.pipeline import Pipeline
from sqlearn.core.schema import Schema
from sqlearn.encoders.onehot import OneHotEncoder

# ── Constructor tests ──────────────────────────────────────────────


class TestConstructor:
    """Test OneHotEncoder constructor and attributes."""

    def test_defaults(self) -> None:
        """Default max_categories=30, columns=None."""
        enc = OneHotEncoder()
        assert enc.max_categories == 30
        assert enc.columns is None

    def test_default_columns_is_categorical(self) -> None:
        """Class default routes to categorical columns."""
        assert OneHotEncoder._default_columns == "categorical"

    def test_classification_is_dynamic(self) -> None:
        """Class is dynamic (needs to learn categories)."""
        assert OneHotEncoder._classification == "dynamic"

    def test_custom_max_categories(self) -> None:
        """Custom max_categories is stored."""
        enc = OneHotEncoder(max_categories=10)
        assert enc.max_categories == 10

    def test_custom_columns(self) -> None:
        """Explicit columns override."""
        enc = OneHotEncoder(columns=["city", "color"])
        assert enc.columns == ["city", "color"]

    def test_get_params(self) -> None:
        """get_params returns constructor kwargs."""
        enc = OneHotEncoder(max_categories=5, columns=["city"])
        params = enc.get_params()
        assert params == {"max_categories": 5, "columns": ["city"]}


# ── discover_sets() tests ─────────────────────────────────────────


class TestDiscoverSets:
    """Test OneHotEncoder.discover_sets() returns correct set queries."""

    @pytest.fixture
    def schema(self) -> Schema:
        """Schema with categorical columns."""
        return Schema({"city": "VARCHAR", "color": "VARCHAR", "price": "DOUBLE"})

    def test_returns_select_distinct_per_column(self, schema: Schema) -> None:
        """Each column gets a SELECT DISTINCT query."""
        enc = OneHotEncoder()
        result = enc.discover_sets(["city", "color"], schema)
        assert "city__categories" in result
        assert "color__categories" in result
        assert len(result) == 2

    def test_ast_is_select_with_distinct(self, schema: Schema) -> None:
        """Each query is a Select containing a Distinct node."""
        enc = OneHotEncoder()
        result = enc.discover_sets(["city"], schema)
        query = result["city__categories"]
        assert isinstance(query, exp.Select)
        # Find a Distinct node in the expression tree
        distinct_nodes = list(query.find_all(exp.Distinct))
        assert len(distinct_nodes) > 0

    def test_empty_columns_returns_empty(self, schema: Schema) -> None:
        """Empty columns list returns empty dict."""
        enc = OneHotEncoder()
        result = enc.discover_sets([], schema)
        assert result == {}


# ── _get_categories() tests ───────────────────────────────────────


class TestGetCategories:
    """Test OneHotEncoder._get_categories() helper."""

    def _make_fitted_encoder(
        self,
        sets: dict[str, list[dict[str, Any]]],
        max_categories: int = 30,
    ) -> OneHotEncoder:
        """Create a fitted OneHotEncoder with given sets."""
        enc = OneHotEncoder(max_categories=max_categories)
        enc.sets_ = sets
        enc._fitted = True
        return enc

    def test_sorted_alphabetically(self) -> None:
        """Categories are sorted alphabetically."""
        enc = self._make_fitted_encoder(
            {"city__categories": [{"city": "Paris"}, {"city": "London"}, {"city": "Berlin"}]}
        )
        assert enc._get_categories("city") == ["Berlin", "London", "Paris"]

    def test_nulls_excluded(self) -> None:
        """None values are excluded from categories."""
        enc = self._make_fitted_encoder(
            {"city__categories": [{"city": "Paris"}, {"city": None}, {"city": "London"}]}
        )
        cats = enc._get_categories("city")
        assert None not in cats
        assert len(cats) == 2

    def test_max_categories_truncation(self) -> None:
        """Categories are truncated to max_categories."""
        rows = [{"city": chr(65 + i)} for i in range(10)]  # A through J
        enc = self._make_fitted_encoder({"city__categories": rows}, max_categories=3)
        cats = enc._get_categories("city")
        assert len(cats) == 3
        # Should be first 3 alphabetically: A, B, C
        assert cats == ["A", "B", "C"]


# ── _category_col_name() tests ────────────────────────────────────


class TestCategoryColName:
    """Test OneHotEncoder._category_col_name() helper."""

    def test_basic_name(self) -> None:
        """Basic category name: col_category."""
        enc = OneHotEncoder()
        assert enc._category_col_name("city", "London") == "city_london"

    def test_spaces_to_underscores(self) -> None:
        """Spaces are converted to underscores."""
        enc = OneHotEncoder()
        assert enc._category_col_name("city", "New York") == "city_new_york"

    def test_lowercase(self) -> None:
        """Category name is lowercased."""
        enc = OneHotEncoder()
        assert enc._category_col_name("color", "RED") == "color_red"


# ── expressions() tests ──────────────────────────────────────────


class TestExpressions:
    """Test OneHotEncoder.expressions() generates correct CASE WHEN ASTs."""

    def _make_fitted_encoder(
        self,
        sets: dict[str, list[dict[str, Any]]],
        max_categories: int = 30,
    ) -> OneHotEncoder:
        """Create a fitted OneHotEncoder with given sets."""
        enc = OneHotEncoder(max_categories=max_categories)
        enc.sets_ = sets
        enc._fitted = True
        enc.columns_ = list({k.replace("__categories", "") for k in sets})
        return enc

    def test_case_when_per_category(self) -> None:
        """Each category gets a CASE WHEN expression."""
        enc = self._make_fitted_encoder(
            {"city__categories": [{"city": "London"}, {"city": "Paris"}]}
        )
        exprs = {"city": exp.Column(this="city")}
        result = enc.expressions(["city"], exprs)
        assert "city_london" in result
        assert "city_paris" in result
        assert isinstance(result["city_london"], exp.Case)
        assert isinstance(result["city_paris"], exp.Case)

    def test_multiple_columns(self) -> None:
        """Multiple columns generate all category combinations."""
        enc = self._make_fitted_encoder(
            {
                "city__categories": [{"city": "London"}, {"city": "Paris"}],
                "color__categories": [{"color": "Red"}, {"color": "Blue"}],
            }
        )
        exprs = {"city": exp.Column(this="city"), "color": exp.Column(this="color")}
        result = enc.expressions(["city", "color"], exprs)
        assert "city_london" in result
        assert "city_paris" in result
        assert "color_red" in result
        assert "color_blue" in result
        assert len(result) == 4

    def test_column_naming(self) -> None:
        """Column names are lowercase with spaces as underscores."""
        enc = self._make_fitted_encoder({"city__categories": [{"city": "New York"}]})
        exprs = {"city": exp.Column(this="city")}
        result = enc.expressions(["city"], exprs)
        assert "city_new_york" in result

    def test_uses_exprs_not_column(self) -> None:
        """expressions() uses exprs[col], not exp.Column(this=col)."""
        enc = self._make_fitted_encoder({"city__categories": [{"city": "London"}]})
        # Simulate a prior transform
        prior = exp.Upper(this=exp.Column(this="city"))
        exprs = {"city": prior}
        result = enc.expressions(["city"], exprs)
        # The CASE WHEN's EQ should reference the prior expression (Upper), not Column
        case_expr = result["city_london"]
        assert isinstance(case_expr, exp.Case)
        if_expr = case_expr.args["ifs"][0]
        eq_expr = if_expr.this
        assert isinstance(eq_expr, exp.EQ)
        assert isinstance(eq_expr.this, exp.Upper)


# ── output_schema() tests ────────────────────────────────────────


class TestOutputSchema:
    """Test OneHotEncoder.output_schema() in pre-fit and post-fit modes."""

    def test_pre_fit_drops_target_columns(self) -> None:
        """Pre-fit (sets_=None): drops target categorical columns."""
        enc = OneHotEncoder()
        schema = Schema({"price": "DOUBLE", "city": "VARCHAR"})
        result = enc.output_schema(schema)
        assert "city" not in result.columns
        assert "price" in result.columns

    def test_post_fit_drops_originals_adds_binary(self) -> None:
        """Post-fit: drops originals + adds binary INTEGER columns."""
        enc = OneHotEncoder()
        enc.sets_ = {"city__categories": [{"city": "London"}, {"city": "Paris"}]}
        schema = Schema({"price": "DOUBLE", "city": "VARCHAR"})
        result = enc.output_schema(schema)
        assert "city" not in result.columns
        assert "price" in result.columns
        assert "city_london" in result.columns
        assert "city_paris" in result.columns
        assert result.columns["city_london"] == "INTEGER"
        assert result.columns["city_paris"] == "INTEGER"

    def test_preserves_non_target_columns(self) -> None:
        """Non-target columns (numeric) pass through unchanged."""
        enc = OneHotEncoder()
        enc.sets_ = {"city__categories": [{"city": "London"}]}
        schema = Schema({"price": "DOUBLE", "qty": "INTEGER", "city": "VARCHAR"})
        result = enc.output_schema(schema)
        assert result.columns["price"] == "DOUBLE"
        assert result.columns["qty"] == "INTEGER"

    def test_no_target_columns_returns_schema(self) -> None:
        """Schema with no categorical columns returns unchanged."""
        enc = OneHotEncoder()
        schema = Schema({"price": "DOUBLE", "qty": "INTEGER"})
        result = enc.output_schema(schema)
        assert result.columns == schema.columns


# ── Pipeline integration tests ────────────────────────────────────


class TestPipeline:
    """Test OneHotEncoder integrated with Pipeline (end-to-end)."""

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
        """Output shape: 4 rows x 4 cols (price + 3 city binary cols)."""
        pipe = Pipeline([OneHotEncoder()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (4, 4)

    def test_binary_encoding_correct(self, backend: DuckDBBackend) -> None:
        """Binary encoding: London=[1,0,0], Paris=[0,1,0], Tokyo=[0,0,1]."""
        pipe = Pipeline([OneHotEncoder()], backend=backend)
        result = pipe.fit_transform("t")
        # Columns: price, city_london, city_paris, city_tokyo
        # Row 0: price=1.0, city=London -> [1,0,0]
        assert result[0, 1] == 1  # london
        assert result[0, 2] == 0  # paris
        assert result[0, 3] == 0  # tokyo
        # Row 1: price=2.0, city=Paris -> [0,1,0]
        assert result[1, 1] == 0  # london
        assert result[1, 2] == 1  # paris
        assert result[1, 3] == 0  # tokyo
        # Row 3: price=4.0, city=Tokyo -> [0,0,1]
        assert result[3, 1] == 0  # london
        assert result[3, 2] == 0  # paris
        assert result[3, 3] == 1  # tokyo

    def test_to_sql_contains_case_when(self, backend: DuckDBBackend) -> None:
        """to_sql() output contains CASE WHEN."""
        pipe = Pipeline([OneHotEncoder()], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql()
        assert "CASE" in sql.upper()
        assert "WHEN" in sql.upper()

    def test_get_feature_names_out(self, backend: DuckDBBackend) -> None:
        """get_feature_names_out returns correct names."""
        pipe = Pipeline([OneHotEncoder()], backend=backend)
        pipe.fit("t")
        names = pipe.get_feature_names_out()
        assert "price" in names
        assert "city_london" in names
        assert "city_paris" in names
        assert "city_tokyo" in names
        assert len(names) == 4

    def test_max_categories_limits_output(self, backend: DuckDBBackend) -> None:
        """max_categories=2 limits output to 2 binary columns."""
        pipe = Pipeline([OneHotEncoder(max_categories=2)], backend=backend)
        result = pipe.fit_transform("t")
        # price + 2 binary columns = 3
        assert result.shape == (4, 3)

    def test_fit_then_transform_separate(self, backend: DuckDBBackend) -> None:
        """Separate fit() and transform() produce same result as fit_transform()."""
        pipe1 = Pipeline([OneHotEncoder()], backend=backend)
        result1 = pipe1.fit_transform("t")

        pipe2 = Pipeline([OneHotEncoder()], backend=backend)
        pipe2.fit("t")
        result2 = pipe2.transform("t")

        np.testing.assert_array_equal(result1, result2)


# ── Compiler fix tests ────────────────────────────────────────────


class TestCompilerFix:
    """Test _collect_set_queries adds FROM clause to queries."""

    def test_adds_from_to_queries_without_one(self) -> None:
        """Queries without FROM get FROM source added."""
        from sqlearn.core.compiler import (
            SchemaChangeResult,
            StepClassification,
            StepInfo,
        )

        schema = Schema({"city": "VARCHAR"})
        enc = OneHotEncoder()
        step_info = StepInfo(
            step=enc,
            classification=StepClassification(kind="dynamic", tier=1, reason="test"),
            schema_change=SchemaChangeResult(changes=True, reason="test"),
            columns=["city"],
            input_schema=schema,
            step_output_schema=schema,
        )

        set_queries: dict[str, exp.Expression] = {}
        _collect_set_queries(step_info, 0, "my_table", set_queries)

        # Should have added FROM
        for query in set_queries.values():
            assert isinstance(query, exp.Select)
            from_clause = query.find(exp.From)
            assert from_clause is not None

    def test_queries_with_existing_from_not_modified(self) -> None:
        """Queries that already have FROM are not modified."""
        from sqlearn.core.compiler import (
            SchemaChangeResult,
            StepClassification,
            StepInfo,
        )

        schema = Schema({"city": "VARCHAR"})

        # Create a custom transformer that returns a query with FROM already
        class EncoderWithFrom(OneHotEncoder):
            """Test encoder that includes FROM in discover_sets."""

            def discover_sets(
                self,
                columns: list[str],
                schema: Schema,
                y_column: str | None = None,
            ) -> dict[str, exp.Expression]:
                """Return queries with FROM already included."""
                result: dict[str, exp.Expression] = {}
                for col in columns:
                    query = exp.Select(
                        expressions=[exp.Distinct(expressions=[exp.Column(this=col)])]
                    ).from_(exp.to_table("existing_table"))
                    result[f"{col}__categories"] = query
                return result

        enc = EncoderWithFrom()
        step_info = StepInfo(
            step=enc,
            classification=StepClassification(kind="dynamic", tier=1, reason="test"),
            schema_change=SchemaChangeResult(changes=True, reason="test"),
            columns=["city"],
            input_schema=schema,
            step_output_schema=schema,
        )

        set_queries: dict[str, exp.Expression] = {}
        _collect_set_queries(step_info, 0, "my_table", set_queries)

        # FROM should still reference existing_table, not my_table
        for query in set_queries.values():
            assert isinstance(query, exp.Select)
            from_clause = query.find(exp.From)
            assert from_clause is not None
            table = from_clause.find(exp.Table)
            assert table is not None
            assert table.name == "existing_table"


# ── Pipeline fix tests ────────────────────────────────────────────


class TestPipelineFix:
    """Test pipeline correctly computes output_schema_ after fit."""

    def test_output_schema_recomputed_after_fit(self) -> None:
        """output_schema_ is correct after fit for schema-changing steps."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES (1.0, 'London'), (2.0, 'Paris') t(price, city)"
        )
        backend = DuckDBBackend(connection=conn)

        enc = OneHotEncoder()
        pipe = Pipeline([enc], backend=backend)
        pipe.fit("t")

        # output_schema_ should have the binary columns
        assert enc.output_schema_ is not None
        assert "city" not in enc.output_schema_.columns
        assert "price" in enc.output_schema_.columns
        assert "city_london" in enc.output_schema_.columns
        assert "city_paris" in enc.output_schema_.columns

    def test_pipeline_schema_out_correct(self) -> None:
        """Pipeline._schema_out uses post-fit schema."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES (1.0, 'London'), (2.0, 'Paris') t(price, city)"
        )
        backend = DuckDBBackend(connection=conn)

        pipe = Pipeline([OneHotEncoder()], backend=backend)
        pipe.fit("t")

        names = pipe.get_feature_names_out()
        assert "city_london" in names
        assert "city_paris" in names
        assert "city" not in names


# ── Not-fitted guard tests ───────────────────────────────────────


class TestNotFittedGuard:
    """Calling transform/to_sql before fit() raises NotFittedError."""

    def test_transform_before_fit_raises(self) -> None:
        """transform() on unfitted pipeline raises NotFittedError."""
        from sqlearn.core.errors import NotFittedError

        pipe = Pipeline([OneHotEncoder()])
        with pytest.raises(NotFittedError):
            pipe.transform("t")

    def test_to_sql_before_fit_raises(self) -> None:
        """to_sql() on unfitted pipeline raises NotFittedError."""
        from sqlearn.core.errors import NotFittedError

        pipe = Pipeline([OneHotEncoder()])
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
            "VALUES (1.0, 'London'), (2.0, 'Paris'), (3.0, 'Tokyo') t(price, city)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([OneHotEncoder()], backend=backend)
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
        cloned_enc.sets_["city__categories"] = []
        assert original_enc.sets_ is not None
        assert len(original_enc.sets_["city__categories"]) == 3

    def test_pickle_roundtrip(self) -> None:
        """Pickle an individual encoder preserves sets."""
        import pickle

        enc = OneHotEncoder()
        enc.sets_ = {"city__categories": [{"city": "London"}, {"city": "Paris"}]}
        enc._fitted = True
        enc.columns_ = ["city"]
        enc.input_schema_ = Schema({"price": "DOUBLE", "city": "VARCHAR"})
        enc.output_schema_ = enc.output_schema(enc.input_schema_)
        data = pickle.dumps(enc)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.sets_ == enc.sets_
        assert restored._fitted is True


# ── Edge cases ──────────────────────────────────────────────────


class TestEdgeCases:
    """Extreme and unusual inputs."""

    def test_single_category(self) -> None:
        """Column with one unique category produces one binary column."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES ('A'), ('A'), ('A') t(cat)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([OneHotEncoder()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (3, 1)
        # All rows should be 1
        np.testing.assert_array_equal(result.flatten(), [1, 1, 1])

    def test_many_categories(self) -> None:
        """Column with many unique categories."""
        conn = duckdb.connect()
        values = ", ".join([f"('{chr(65 + i)}')" for i in range(26)])
        conn.execute(f"CREATE TABLE t AS SELECT * FROM VALUES {values} t(letter)")  # noqa: S608
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([OneHotEncoder(max_categories=10)], backend=backend)
        result = pipe.fit_transform("t")
        # 26 rows x 10 categories (truncated)
        assert result.shape == (26, 10)

    def test_categories_with_spaces(self) -> None:
        """Categories containing spaces produce valid column names."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES ('New York'), ('Los Angeles'), ('New York') t(city)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([OneHotEncoder()], backend=backend)
        pipe.fit("t")
        names = pipe.get_feature_names_out()
        assert "city_new_york" in names
        assert "city_los_angeles" in names

    def test_categories_with_mixed_case(self) -> None:
        """Mixed case categories normalized in column names."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM VALUES ('RED'), ('Blue'), ('GREEN') t(color)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([OneHotEncoder()], backend=backend)
        pipe.fit("t")
        names = pipe.get_feature_names_out()
        assert "color_red" in names
        assert "color_blue" in names
        assert "color_green" in names

    def test_null_only_category_column(self) -> None:
        """Column with all NULLs produces no binary columns."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, CAST(NULL AS VARCHAR)), "
            "(2.0, CAST(NULL AS VARCHAR)) t(price, cat)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([OneHotEncoder()], backend=backend)
        result = pipe.fit_transform("t")
        # Only price remains (cat had no non-null categories)
        assert result.shape == (2, 1)

    def test_numeric_passthrough(self) -> None:
        """Numeric columns are not encoded (pass through unchanged)."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "VALUES (1.0, 100, 'A'), (2.0, 200, 'B') t(price, qty, color)"
        )
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([OneHotEncoder()], backend=backend)
        result = pipe.fit_transform("t")
        # price, qty pass through + color_a, color_b = 4 columns
        assert result.shape == (2, 4)
        # price and qty should be unchanged
        np.testing.assert_allclose(result[:, 0].astype(float), [1.0, 2.0])
        np.testing.assert_allclose(result[:, 1].astype(float), [100.0, 200.0])


# ── SQL snapshot tests ──────────────────────────────────────────


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
        pipe = Pipeline([OneHotEncoder()], backend=backend)
        pipe.fit("t")
        return pipe

    def test_sql_has_case_when_per_category(self, fitted_pipe: Pipeline) -> None:
        """Each category gets a CASE WHEN in the SQL."""
        sql = fitted_pipe.to_sql().upper()
        assert sql.count("CASE") == 3  # London, Paris, Tokyo

    def test_sql_has_select_from(self, fitted_pipe: Pipeline) -> None:
        """SQL contains SELECT and FROM."""
        sql = fitted_pipe.to_sql().upper()
        assert "SELECT" in sql
        assert "FROM" in sql

    def test_sql_no_original_column(self, fitted_pipe: Pipeline) -> None:
        """Original 'city' column is not in SELECT (only binary cols)."""
        sql = fitted_pipe.to_sql()
        # Should have city_london, city_paris, city_tokyo but NOT bare city AS city
        assert "city_london" in sql.lower()
        assert "city_paris" in sql.lower()

    def test_sql_custom_table(self, fitted_pipe: Pipeline) -> None:
        """to_sql(table=...) uses custom table name."""
        sql = fitted_pipe.to_sql(table="raw_data")
        assert "raw_data" in sql


# ── Composition tests ───────────────────────────────────────────


class TestComposition:
    """OneHotEncoder composing with other transformers."""

    def test_imputer_then_encoder(self) -> None:
        """Imputer + OneHotEncoder: NULLs filled before encoding."""
        from sqlearn.imputers.imputer import Imputer

        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES ('A'), ('B'), (NULL), ('A') t(cat)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Imputer(), OneHotEncoder()], backend=backend)
        result = pipe.fit_transform("t")
        assert result.shape == (4, 2)

    def test_full_pipeline(self) -> None:
        """Imputer + StandardScaler + OneHotEncoder: full pipeline."""
        from sqlearn.imputers.imputer import Imputer
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

    def test_encoder_sql_composes_with_imputer(self) -> None:
        """SQL shows COALESCE nested inside CASE WHEN from composition."""
        from sqlearn.imputers.imputer import Imputer

        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM VALUES ('A'), ('B'), (NULL) t(cat)")
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Imputer(), OneHotEncoder()], backend=backend)
        pipe.fit("t")
        sql = pipe.to_sql().upper()
        assert "COALESCE" in sql
        assert "CASE" in sql


class TestOutputSchemaDefensive:
    """Defensive code paths in OneHotEncoder.output_schema()."""

    def test_output_schema_no_columns_spec(self) -> None:
        """output_schema() with _resolve_columns_spec() → None returns schema.

        When both user columns and _default_columns resolve to None, output_schema
        cannot determine which columns to encode and returns the schema unchanged.
        This is a defensive fallback for unusual subclass configurations.
        """
        encoder = OneHotEncoder()
        schema = Schema({"price": "DOUBLE", "city": "VARCHAR"})
        # Force _resolve_columns_spec to return None
        encoder._default_columns = None  # type: ignore[assignment]
        encoder.columns = None
        result = encoder.output_schema(schema)
        assert result is schema

    def test_output_schema_resolve_columns_raises(self) -> None:
        """output_schema() with invalid column spec returns schema unchanged.

        When resolve_columns() raises because the specified columns don't exist
        in the schema, output_schema gracefully returns the input schema instead
        of crashing. This handles cases like pipeline reconfiguration where the
        schema no longer has the expected columns.
        """
        encoder = OneHotEncoder(columns=["nonexistent"])
        schema = Schema({"price": "DOUBLE"})
        result = encoder.output_schema(schema)
        assert result is schema
