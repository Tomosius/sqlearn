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
