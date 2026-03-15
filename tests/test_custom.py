"""Tests for sq.Expression() and sq.custom() custom transformer factories."""

from __future__ import annotations

import duckdb
import numpy as np
import pytest

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.compiler import compose_transform
from sqlearn.core.pipeline import Pipeline
from sqlearn.custom import Expression, _CustomTransformer, _ExpressionTransformer, custom
from sqlearn.imputers.imputer import Imputer
from sqlearn.scalers.standard import StandardScaler

# ── Helpers ──────────────────────────────────────────────────────────


def _make_numeric_conn() -> duckdb.DuckDBPyConnection:
    """Create DuckDB connection with a 5-row numeric table."""
    conn = duckdb.connect()
    conn.execute("""
        CREATE TABLE data AS SELECT * FROM VALUES
            (1.0, 10.0),
            (2.0, 20.0),
            (3.0, 30.0),
            (4.0, 40.0),
            (5.0, 50.0)
        t(price, quantity)
    """)
    return conn


def _make_mixed_conn() -> duckdb.DuckDBPyConnection:
    """Create DuckDB connection with numeric + categorical + NULLs."""
    conn = duckdb.connect()
    conn.execute("""
        CREATE TABLE data AS SELECT * FROM VALUES
            (1.0, 10.0, 'London'),
            (2.0, NULL, 'Paris'),
            (NULL, 30.0, 'London'),
            (4.0, 40.0, 'Tokyo'),
            (5.0, 50.0, 'Paris')
        t(price, quantity, city)
    """)
    return conn


# ══════════════════════════════════════════════════════════════════════
# Expression Tests
# ══════════════════════════════════════════════════════════════════════


class TestExpressionCreation:
    """Expression() factory validates SQL at creation time."""

    def test_valid_expression(self) -> None:
        """Valid SQL with AS alias creates an _ExpressionTransformer without error."""
        t = Expression("price * quantity AS revenue")
        assert isinstance(t, _ExpressionTransformer)

    def test_rejects_missing_alias(self) -> None:
        """Expression without AS clause raises ValueError."""
        with pytest.raises(ValueError, match="must contain AS"):
            Expression("price * quantity")

    def test_rejects_invalid_sql(self) -> None:
        """Expression with unparseable SQL raises ValueError at creation time."""
        with pytest.raises(ValueError, match="Invalid SQL"):
            Expression("SELECT FROM WHERE AS broken")

    def test_repr(self) -> None:
        """Repr includes the class name for identification in pipeline listings."""
        t = Expression("price * 2 AS double_price")
        assert "Expression" in repr(t) or "_ExpressionTransformer" in repr(t)

    def test_classification_is_static(self) -> None:
        """Expression transformers are always static since they learn no stats."""
        t = Expression("price * 2 AS double_price")
        assert t._classify() == "static"

    def test_clone_roundtrip(self) -> None:
        """Clone produces an independent copy that preserves the SQL template."""
        t = Expression("price * 2 AS double_price")
        cloned = t.clone()
        assert isinstance(cloned, _ExpressionTransformer)
        assert cloned.sql == t.sql

    def test_pickle_roundtrip(self) -> None:
        """Pickle serialize and deserialize preserves SQL template and alias."""
        import pickle

        t = Expression("price * 2 AS double_price")
        data = pickle.dumps(t)
        restored = pickle.loads(data)  # noqa: S301
        assert isinstance(restored, _ExpressionTransformer)
        assert restored.sql == t.sql
        assert restored._alias == t._alias


class TestExpressionInPipeline:
    """Expression works in a Pipeline: adds column, composes with prior steps."""

    def test_adds_column(self) -> None:
        """Expression adds a new derived column while preserving all originals."""
        conn = _make_numeric_conn()
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Expression("price * quantity AS revenue")], backend=backend)
        pipe.fit("data", backend=backend)
        result = pipe.transform("data", backend=backend)

        # Output should have 3 columns: price, quantity, revenue
        names = pipe.get_feature_names_out()
        assert "revenue" in names
        assert "price" in names
        assert "quantity" in names
        assert result.shape == (5, 3)

    def test_revenue_values(self) -> None:
        """Derived column computes correct arithmetic values row by row."""
        conn = _make_numeric_conn()
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Expression("price * quantity AS revenue")], backend=backend)
        pipe.fit("data", backend=backend)
        result = pipe.transform("data", backend=backend)

        names = pipe.get_feature_names_out()
        rev_idx = names.index("revenue")
        expected_revenue = np.array([10.0, 40.0, 90.0, 160.0, 250.0])
        np.testing.assert_allclose(result[:, rev_idx], expected_revenue)

    def test_to_sql_contains_expression(self) -> None:
        """Generated SQL includes the alias and multiplication operator."""
        conn = _make_numeric_conn()
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline([Expression("price * quantity AS revenue")], backend=backend)
        pipe.fit("data", backend=backend)
        sql = pipe.to_sql()

        assert "revenue" in sql.lower()
        assert "*" in sql  # multiplication operator

    def test_composes_with_imputer(self) -> None:
        """Expression after Imputer uses imputed values."""
        conn = _make_mixed_conn()
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [
                Imputer(strategy="mean", columns=["price", "quantity"]),
                Expression("price * quantity AS revenue"),
            ],
            backend=backend,
        )
        pipe.fit("data", backend=backend)
        result = pipe.transform("data", backend=backend)

        # No NaN in revenue (imputer filled NULLs before multiplication)
        names = pipe.get_feature_names_out()
        rev_idx = names.index("revenue")
        assert not np.any(np.isnan(result[:, rev_idx].astype(float)))

    def test_case_expression(self) -> None:
        """CASE WHEN expression produces a named output column."""
        conn = _make_numeric_conn()
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [
                Expression("CASE WHEN price > 3 THEN 'high' ELSE 'low' END AS tier"),
            ],
            backend=backend,
        )
        pipe.fit("data", backend=backend)
        pipe.transform("data", backend=backend)

        names = pipe.get_feature_names_out()
        assert "tier" in names


# ══════════════════════════════════════════════════════════════════════
# custom() Static Tests
# ══════════════════════════════════════════════════════════════════════


class TestCustomCreation:
    """custom() factory validates templates at creation time."""

    def test_valid_static_template(self) -> None:
        """Valid SQL template with {col} placeholder creates a _CustomTransformer."""
        t = custom("LN({col} + 1)", columns="numeric")
        assert isinstance(t, _CustomTransformer)

    def test_classification_static_without_learn(self) -> None:
        """Custom transformer without learn= is classified as static."""
        t = custom("{col} * 2", columns="numeric")
        assert t._classify() == "static"

    def test_classification_dynamic_with_learn(self) -> None:
        """Custom transformer with learn= is classified as dynamic."""
        t = custom("{col} - {mean}", columns="numeric", learn={"mean": "AVG({col})"})
        assert t._classify() == "dynamic"

    def test_rejects_invalid_sql(self) -> None:
        """Unparseable SQL template raises ValueError at creation time."""
        with pytest.raises(ValueError, match="Invalid SQL"):
            custom("SELECT FROM WHERE {col}", columns="numeric")

    def test_rejects_invalid_learn_sql(self) -> None:
        """Unparseable SQL in learn= dict raises ValueError at creation time."""
        with pytest.raises(ValueError, match="Invalid SQL in learn"):
            custom("{col} - {mean}", columns="numeric", learn={"mean": "AVG({col})) EXTRA"})

    def test_rejects_combine_with_col_placeholder(self) -> None:
        """Combine mode with {col} placeholder raises ValueError since combine is cross-column."""
        with pytest.raises(ValueError, match=r"combine.*cannot use.*col"):
            custom("{col} * 2 AS doubled", columns=["a"], mode="combine")

    def test_rejects_unknown_mode(self) -> None:
        """Unrecognized mode string raises ValueError."""
        with pytest.raises(ValueError, match="mode must be"):
            custom("{col} * 2", columns="numeric", mode="invalid")

    def test_clone_roundtrip(self) -> None:
        """Clone produces an independent copy preserving SQL template and mode."""
        t = custom("{col} * 2", columns="numeric")
        cloned = t.clone()
        assert isinstance(cloned, _CustomTransformer)
        assert cloned.sql == t.sql
        assert cloned.mode == "per_column"


class TestCustomStaticPerColumn:
    """custom() static per_column mode: {col} iterates over target columns."""

    def test_replace_in_place(self) -> None:
        """Template without AS replaces columns in-place."""
        conn = _make_numeric_conn()
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [
                custom("{col} * 2", columns=["price", "quantity"]),
            ],
            backend=backend,
        )
        pipe.fit("data", backend=backend)
        result = pipe.transform("data", backend=backend)

        # Same columns, values doubled
        names = pipe.get_feature_names_out()
        assert names == ["price", "quantity"]
        np.testing.assert_allclose(result[:, 0], [2.0, 4.0, 6.0, 8.0, 10.0])
        np.testing.assert_allclose(result[:, 1], [20.0, 40.0, 60.0, 80.0, 100.0])

    def test_new_column_with_alias(self) -> None:
        """Template with AS {col}_suffix adds new columns."""
        conn = _make_numeric_conn()
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [
                custom("LN({col} + 1) AS {col}_log", columns=["price"]),
            ],
            backend=backend,
        )
        pipe.fit("data", backend=backend)
        names = pipe.get_feature_names_out()

        assert "price_log" in names
        assert "price" in names  # original preserved
        assert "quantity" in names  # untouched passthrough

    def test_to_sql(self) -> None:
        """Generated SQL includes the column name and arithmetic operator."""
        conn = _make_numeric_conn()
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [
                custom("{col} * 2", columns=["price"]),
            ],
            backend=backend,
        )
        pipe.fit("data", backend=backend)
        sql = pipe.to_sql()

        assert "price" in sql.lower()
        assert "*" in sql or "2" in sql

    def test_composes_with_imputer(self) -> None:
        """custom() after Imputer uses imputed values (COALESCE nested)."""
        conn = _make_mixed_conn()
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [
                Imputer(strategy="mean", columns=["price"]),
                custom("{col} * 2", columns=["price"]),
            ],
            backend=backend,
        )
        pipe.fit("data", backend=backend)
        result = pipe.transform("data", backend=backend)

        # price column should have no NaN (imputed then doubled)
        names = pipe.get_feature_names_out()
        price_idx = names.index("price")
        assert not np.any(np.isnan(result[:, price_idx].astype(float)))

    def test_single_column(self) -> None:
        """Apply SQRT to a single column and verify correct values."""
        conn = _make_numeric_conn()
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [
                custom("SQRT({col})", columns=["price"]),
            ],
            backend=backend,
        )
        pipe.fit("data", backend=backend)
        result = pipe.transform("data", backend=backend)

        names = pipe.get_feature_names_out()
        price_idx = names.index("price")
        np.testing.assert_allclose(result[:, price_idx], np.sqrt([1, 2, 3, 4, 5]), rtol=1e-6)


class TestCustomCombineMode:
    """custom() combine mode: single expression with multiple columns."""

    def test_combine_cross_column(self) -> None:
        """Combine mode multiplies two columns into a new derived column with correct values."""
        conn = _make_numeric_conn()
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [
                custom(
                    "price * quantity AS revenue",
                    columns=["price", "quantity"],
                    mode="combine",
                ),
            ],
            backend=backend,
        )
        pipe.fit("data", backend=backend)
        result = pipe.transform("data", backend=backend)

        names = pipe.get_feature_names_out()
        assert "revenue" in names
        rev_idx = names.index("revenue")
        np.testing.assert_allclose(result[:, rev_idx], [10.0, 40.0, 90.0, 160.0, 250.0])

    def test_combine_preserves_originals(self) -> None:
        """Combine mode adds the derived column while keeping original columns intact."""
        conn = _make_numeric_conn()
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [
                custom(
                    "price / NULLIF(quantity, 0) AS ratio",
                    columns=["price", "quantity"],
                    mode="combine",
                ),
            ],
            backend=backend,
        )
        pipe.fit("data", backend=backend)
        names = pipe.get_feature_names_out()

        assert "ratio" in names
        assert "price" in names
        assert "quantity" in names


# ══════════════════════════════════════════════════════════════════════
# custom() Dynamic Tests
# ══════════════════════════════════════════════════════════════════════


class TestCustomDynamic:
    """custom() with learn= makes it dynamic (learns from data during fit)."""

    def test_centering(self) -> None:
        """Learn mean, subtract it."""
        conn = _make_numeric_conn()
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [
                custom(
                    "{col} - {mean}",
                    columns=["price"],
                    learn={"mean": "AVG({col})"},
                ),
            ],
            backend=backend,
        )
        pipe.fit("data", backend=backend)
        result = pipe.transform("data", backend=backend)

        names = pipe.get_feature_names_out()
        price_idx = names.index("price")
        # Mean of [1,2,3,4,5] = 3.0
        np.testing.assert_allclose(result[:, price_idx], [-2.0, -1.0, 0.0, 1.0, 2.0], rtol=1e-6)

    def test_z_score_normalization(self) -> None:
        """Learn mean and std, compute z-score."""
        conn = _make_numeric_conn()
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [
                custom(
                    "({col} - {mean}) / NULLIF({std}, 0)",
                    columns=["price"],
                    learn={
                        "mean": "AVG({col})",
                        "std": "STDDEV_POP({col})",
                    },
                ),
            ],
            backend=backend,
        )
        pipe.fit("data", backend=backend)
        result = pipe.transform("data", backend=backend)

        # Should match StandardScaler output
        sk_pipe = Pipeline([StandardScaler(columns=["price"])], backend=backend)
        sk_pipe.fit("data", backend=backend)
        sk_result = sk_pipe.transform("data", backend=backend)

        names = pipe.get_feature_names_out()
        price_idx = names.index("price")
        sk_names = sk_pipe.get_feature_names_out()
        sk_price_idx = sk_names.index("price")
        np.testing.assert_allclose(result[:, price_idx], sk_result[:, sk_price_idx], rtol=1e-6)

    def test_dynamic_new_column(self) -> None:
        """Dynamic with AS clause adds new column."""
        conn = _make_numeric_conn()
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [
                custom(
                    "({col} - {min_val}) / NULLIF({max_val} - {min_val}, 0) AS {col}_pct",
                    columns=["price"],
                    learn={
                        "min_val": "MIN({col})",
                        "max_val": "MAX({col})",
                    },
                ),
            ],
            backend=backend,
        )
        pipe.fit("data", backend=backend)
        result = pipe.transform("data", backend=backend)

        names = pipe.get_feature_names_out()
        assert "price_pct" in names
        assert "price" in names  # original preserved

        pct_idx = names.index("price_pct")
        # (1-1)/(5-1)=0, (2-1)/(5-1)=0.25, ..., (5-1)/(5-1)=1
        np.testing.assert_allclose(result[:, pct_idx], [0.0, 0.25, 0.5, 0.75, 1.0], rtol=1e-6)

    def test_dynamic_composes_with_imputer(self) -> None:
        """Dynamic custom after Imputer: COALESCE nested in arithmetic."""
        conn = _make_mixed_conn()
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [
                Imputer(strategy="mean", columns=["price"]),
                custom(
                    "{col} - {mean}",
                    columns=["price"],
                    learn={"mean": "AVG({col})"},
                ),
            ],
            backend=backend,
        )
        pipe.fit("data", backend=backend)
        result = pipe.transform("data", backend=backend)

        names = pipe.get_feature_names_out()
        price_idx = names.index("price")
        assert not np.any(np.isnan(result[:, price_idx].astype(float)))

    def test_to_sql_contains_learned_values(self) -> None:
        """Generated SQL embeds the learned statistic as a literal value, not an aggregate call."""
        conn = _make_numeric_conn()
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [
                custom(
                    "{col} - {mean}",
                    columns=["price"],
                    learn={"mean": "AVG({col})"},
                ),
            ],
            backend=backend,
        )
        pipe.fit("data", backend=backend)
        sql = pipe.to_sql()

        # Mean of [1,2,3,4,5] = 3.0 should appear in SQL
        assert "3.0" in sql


# ══════════════════════════════════════════════════════════════════════
# AST Composition Tests
# ══════════════════════════════════════════════════════════════════════


class TestASTComposition:
    """Verify expression composition at the AST level."""

    def test_expression_composes_with_imputer_ast(self) -> None:
        """Expression after Imputer: Column refs replaced with Coalesce."""
        conn = _make_mixed_conn()
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [
                Imputer(strategy="mean", columns=["price", "quantity"]),
                Expression("price * quantity AS revenue"),
            ],
            backend=backend,
        )
        pipe.fit("data", backend=backend)

        transformers = [step for _, step in pipe.steps]
        select = compose_transform(transformers, "__input__")
        sql = select.sql(dialect="duckdb")

        # COALESCE should appear (from Imputer), nested in multiplication
        assert "COALESCE" in sql.upper()

    def test_custom_per_column_no_cte(self) -> None:
        """Static custom per_column produces no CTEs."""
        conn = _make_numeric_conn()
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [
                custom("{col} * 2", columns=["price"]),
            ],
            backend=backend,
        )
        pipe.fit("data", backend=backend)
        sql = pipe.to_sql()

        assert "WITH" not in sql.upper()


# ══════════════════════════════════════════════════════════════════════
# Edge Case Tests
# ══════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases from spec: empty columns, single column, no-placeholder combine."""

    def test_empty_column_list(self) -> None:
        """custom() with empty column list is a no-op."""
        conn = _make_numeric_conn()
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [custom("{col} * 2", columns=[])],
            backend=backend,
        )
        pipe.fit("data", backend=backend)
        result = pipe.transform("data", backend=backend)

        # No columns targeted, so values unchanged
        names = pipe.get_feature_names_out()
        assert names == ["price", "quantity"]
        np.testing.assert_allclose(result[:, 0], [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_combine_no_placeholders(self) -> None:
        """Combine mode with literal expression (no column refs)."""
        conn = _make_numeric_conn()
        backend = DuckDBBackend(connection=conn)
        pipe = Pipeline(
            [custom("42 AS constant", columns=["price"], mode="combine")],
            backend=backend,
        )
        pipe.fit("data", backend=backend)
        result = pipe.transform("data", backend=backend)

        names = pipe.get_feature_names_out()
        assert "constant" in names
        const_idx = names.index("constant")
        np.testing.assert_allclose(result[:, const_idx], [42.0] * 5)


class TestCustomDefensiveEdgeCases:
    """Edge cases for defensive code paths in custom.py."""

    def test_discover_no_learn_returns_empty(self) -> None:
        """Static custom (no learn=) discover() returns empty dict.

        When learn= is not provided, the custom transformer is static and
        discover() should return {} immediately since there are no stats to
        learn from the data.
        """
        t = custom("{col} * 2", columns=["price"])
        schema = __import__("sqlearn.core.schema", fromlist=["Schema"]).Schema({"price": "DOUBLE"})
        result = t.discover(["price"], schema)
        assert result == {}

    def test_discover_learn_alias_unwrap(self) -> None:
        """Learn template with AS alias: discover() unwraps the alias.

        When a learn template like 'AVG({col}) AS avg_val' is parsed, the
        result is an Alias node. discover() must unwrap it to get the raw
        aggregate expression (AVG), not the aliased version, so the compiler
        can batch aggregates correctly.
        """
        t = custom(
            "{col} - {mean}",
            columns=["price"],
            learn={"mean": "AVG({col}) AS avg_val"},
        )
        from sqlearn.core.schema import Schema

        schema = Schema({"price": "DOUBLE"})
        result = t.discover(["price"], schema)
        assert "price__mean" in result
        # Should be Avg, not Alias
        import sqlglot.expressions as exp

        assert isinstance(result["price__mean"], exp.Avg)

    def test_combine_without_alias_returns_empty(self) -> None:
        """Combine mode SQL without AS alias → expressions() returns empty dict.

        When the combine mode SQL has no alias (e.g., 'price + quantity' without
        'AS total'), there's no way to name the output column, so expressions()
        must return an empty dict (no-op passthrough).
        """
        t = custom("price + quantity", columns=["price", "quantity"], mode="combine")
        import sqlglot.expressions as exp

        exprs = {
            "price": exp.Column(this="price"),
            "quantity": exp.Column(this="quantity"),
        }
        t.columns_ = ["price", "quantity"]
        t._fitted = True
        t.params_ = {}
        result = t.expressions(["price", "quantity"], exprs)
        assert result == {}

    def test_combine_output_schema_without_alias(self) -> None:
        """Combine mode without alias → _has_alias is False, early return.

        When there's no alias, output_schema returns at line 309 since
        _has_alias is False. The combine-mode branch at 312-317 is not reached.
        """
        t = custom("price + quantity", columns=["price", "quantity"], mode="combine")
        from sqlearn.core.schema import Schema

        schema = Schema({"price": "DOUBLE", "quantity": "DOUBLE"})
        result = t.output_schema(schema)
        assert result is schema

    def test_combine_output_schema_with_alias(self) -> None:
        """Combine mode with alias → output_schema adds the aliased column.

        The output schema should include the original columns plus the new
        column defined by the AS alias in the combine expression.
        Covers lines 314-316 (combine mode with alias in output_schema).
        """
        t = custom("price + quantity AS total", columns=["price", "quantity"], mode="combine")
        from sqlearn.core.schema import Schema

        schema = Schema({"price": "DOUBLE", "quantity": "DOUBLE"})
        result = t.output_schema(schema)
        assert "total" in result.columns

    def test_output_schema_no_columns_spec(self) -> None:
        """Per-column mode with alias but no columns → passthrough.

        _has_alias is True (SQL has AS clause), but _resolve_columns_spec()
        returns None (no columns and no _default_columns), so output_schema
        cannot determine target columns and returns schema unchanged.
        Covers line 322.
        """
        t = custom("{col} * 2 AS doubled")
        from sqlearn.core.schema import Schema

        schema = Schema({"price": "DOUBLE"})
        result = t.output_schema(schema)
        assert result is schema

    def test_output_schema_resolve_columns_raises(self) -> None:
        """output_schema with alias but invalid columns → returns unchanged.

        _has_alias is True, but resolve_columns() raises because the column
        spec references columns that don't exist. output_schema gracefully
        returns the unchanged schema instead of crashing.
        Covers lines 325-326.
        """
        t = custom("{col} * 2 AS doubled", columns=["nonexistent_col"])
        from sqlearn.core.schema import Schema

        schema = Schema({"price": "DOUBLE"})
        result = t.output_schema(schema)
        assert result is schema

    def test_output_schema_per_column_no_alias_in_result(self) -> None:
        """Per-column template without per-column alias → no new columns added.

        _has_alias is True from validation (because {col} becomes __col__
        which doesn't affect AS clause), but per-column substitution produces
        an expression without alias → new_cols is empty → returns schema.
        Covers branch 335->329.
        """
        t = custom("{col} * 2 AS doubled", columns=["price"])
        from sqlearn.core.schema import Schema

        schema = Schema({"price": "DOUBLE"})
        result = t.output_schema(schema)
        # Should add "doubled" column
        assert "doubled" in result.columns

    def test_parse_select_expr_empty_sql(self) -> None:
        """_parse_select_expr with empty string → ValueError.

        An empty SQL template produces SELECT FROM __t__ which parses
        to a Select with no expressions, triggering the validation at
        lines 45-47.
        """
        from sqlearn.custom import _parse_select_expr

        with pytest.raises(ValueError, match="Invalid SQL template"):
            _parse_select_expr("")
