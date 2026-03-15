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
        t = Expression("price * quantity AS revenue")
        assert isinstance(t, _ExpressionTransformer)

    def test_rejects_missing_alias(self) -> None:
        with pytest.raises(ValueError, match="must contain AS"):
            Expression("price * quantity")

    def test_rejects_invalid_sql(self) -> None:
        with pytest.raises(ValueError, match="Invalid SQL"):
            Expression("SELECT FROM WHERE AS broken")

    def test_repr(self) -> None:
        t = Expression("price * 2 AS double_price")
        assert "Expression" in repr(t) or "_ExpressionTransformer" in repr(t)

    def test_classification_is_static(self) -> None:
        t = Expression("price * 2 AS double_price")
        assert t._classify() == "static"

    def test_clone_roundtrip(self) -> None:
        t = Expression("price * 2 AS double_price")
        cloned = t.clone()
        assert isinstance(cloned, _ExpressionTransformer)
        assert cloned.sql == t.sql

    def test_pickle_roundtrip(self) -> None:
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
        t = custom("LN({col} + 1)", columns="numeric")
        assert isinstance(t, _CustomTransformer)

    def test_classification_static_without_learn(self) -> None:
        t = custom("{col} * 2", columns="numeric")
        assert t._classify() == "static"

    def test_classification_dynamic_with_learn(self) -> None:
        t = custom("{col} - {mean}", columns="numeric", learn={"mean": "AVG({col})"})
        assert t._classify() == "dynamic"

    def test_rejects_invalid_sql(self) -> None:
        with pytest.raises(ValueError, match="Invalid SQL"):
            custom("SELECT FROM WHERE {col}", columns="numeric")

    def test_rejects_invalid_learn_sql(self) -> None:
        with pytest.raises(ValueError, match="Invalid SQL in learn"):
            custom("{col} - {mean}", columns="numeric", learn={"mean": "AVG({col})) EXTRA"})

    def test_rejects_combine_with_col_placeholder(self) -> None:
        with pytest.raises(ValueError, match=r"combine.*cannot use.*col"):
            custom("{col} * 2 AS doubled", columns=["a"], mode="combine")

    def test_rejects_unknown_mode(self) -> None:
        with pytest.raises(ValueError, match="mode must be"):
            custom("{col} * 2", columns="numeric", mode="invalid")

    def test_clone_roundtrip(self) -> None:
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
