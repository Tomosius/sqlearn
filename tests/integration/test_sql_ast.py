"""AST structure tests — verify sqlglot expression tree shape.

Tests use compose_transform() to get the raw expression tree and assert
specific node types at specific positions, proving composition works
correctly at the AST level.
"""

from __future__ import annotations

import duckdb
import sqlglot.expressions as exp

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.compiler import compose_transform
from sqlearn.core.pipeline import Pipeline
from sqlearn.encoders.onehot import OneHotEncoder
from sqlearn.imputers.imputer import Imputer
from sqlearn.scalers.standard import StandardScaler

# ── Helpers ──────────────────────────────────────────────────────────


def _fit_and_compose(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    steps: list,
    source: str = "__input__",
) -> exp.Select:
    """Fit a pipeline and return compose_transform() output."""
    backend = DuckDBBackend(connection=conn)
    pipe = Pipeline(steps, backend=backend)
    pipe.fit(table, backend=backend)
    transformers = [step for _, step in pipe.steps]
    return compose_transform(transformers, source)


def _get_output_exprs(select: exp.Select) -> dict[str, exp.Expression]:
    """Extract {alias: expression} from a SELECT's output columns."""
    result: dict[str, exp.Expression] = {}
    for col_expr in select.expressions:
        if isinstance(col_expr, exp.Alias):
            result[col_expr.alias] = col_expr.this
        elif isinstance(col_expr, exp.Column):
            result[col_expr.name] = col_expr
    return result


# ── StandardScaler AST ───────────────────────────────────────────────


class TestStandardScalerAST:
    """StandardScaler output expressions are Div(Paren(Sub(...)), Nullif(...))."""

    def test_expression_structure(self) -> None:
        """Each column is (col - mean) / NULLIF(std, 0)."""
        conn = duckdb.connect()
        conn.execute("""
            CREATE TABLE data AS SELECT * FROM VALUES
                (1.0, 10.0), (2.0, 20.0), (3.0, 30.0)
            t(a, b)
        """)
        select = _fit_and_compose(conn, "data", [StandardScaler()])
        output = _get_output_exprs(select)

        for col in ["a", "b"]:
            expr = output[col]
            # Top level: Div
            assert isinstance(expr, exp.Div), f"{col}: expected Div, got {type(expr).__name__}"

            # Numerator: Paren(Sub(...))
            numerator = expr.this
            assert isinstance(numerator, exp.Paren), (
                f"{col}: expected Paren numerator, got {type(numerator).__name__}"
            )
            sub = numerator.this
            assert isinstance(sub, exp.Sub), (
                f"{col}: expected Sub inside Paren, got {type(sub).__name__}"
            )

            # Denominator: Nullif(std_literal, 0)
            denominator = expr.expression
            assert isinstance(denominator, exp.Nullif), (
                f"{col}: expected Nullif denominator, got {type(denominator).__name__}"
            )


# ── Imputer AST ──────────────────────────────────────────────────────


class TestImputerAST:
    """Imputer output expressions are Coalesce(Column, Literal)."""

    def test_expression_structure(self) -> None:
        """Each column is COALESCE(col, fill_value)."""
        conn = duckdb.connect()
        conn.execute("""
            CREATE TABLE data AS SELECT * FROM VALUES
                (1.0, 10.0), (NULL, 20.0), (3.0, NULL)
            t(a, b)
        """)
        select = _fit_and_compose(conn, "data", [Imputer(strategy="mean")])
        output = _get_output_exprs(select)

        for col in ["a", "b"]:
            expr = output[col]
            assert isinstance(expr, exp.Coalesce), (
                f"{col}: expected Coalesce, got {type(expr).__name__}"
            )
            # First arg is Column reference
            assert isinstance(expr.this, exp.Column), (
                f"{col}: expected Column inside Coalesce, got {type(expr.this).__name__}"
            )


# ── OneHotEncoder AST ────────────────────────────────────────────────


class TestOneHotEncoderAST:
    """OneHotEncoder output expressions are Case(If(EQ(...), 1), default=0)."""

    def test_expression_structure(self) -> None:
        """Each one-hot column is CASE WHEN col = 'cat' THEN 1 ELSE 0 END."""
        conn = duckdb.connect()
        conn.execute("""
            CREATE TABLE data AS SELECT * FROM VALUES
                ('red',), ('blue',), ('green',)
            t(color)
        """)
        select = _fit_and_compose(conn, "data", [OneHotEncoder()])
        output = _get_output_exprs(select)

        expected_cols = ["color_blue", "color_green", "color_red"]
        for col in expected_cols:
            assert col in output, f"Missing expected one-hot column: {col}"
            expr = output[col]

            # Top level: Case
            assert isinstance(expr, exp.Case), f"{col}: expected Case, got {type(expr).__name__}"

            # Has exactly one If branch
            ifs = [x for x in expr.args.get("ifs", []) if isinstance(x, exp.If)]
            assert len(ifs) == 1, f"{col}: expected 1 If branch, got {len(ifs)}"

            # The If condition is EQ
            condition = ifs[0].this
            assert isinstance(condition, exp.EQ), (
                f"{col}: expected EQ condition, got {type(condition).__name__}"
            )

            # THEN value is Literal(1)
            true_val = ifs[0].args.get("true")
            assert isinstance(true_val, exp.Literal), (
                f"{col}: expected Literal for THEN, got {type(true_val).__name__}"
            )

            # Default is Literal(0)
            default = expr.args.get("default")
            assert isinstance(default, exp.Literal), (
                f"{col}: expected Literal for ELSE, got {type(default).__name__}"
            )

        # Original 'color' column should NOT be in output
        assert "color" not in output, "Original column should be removed"


# ── Composition AST ──────────────────────────────────────────────────


class TestImputerScalerCompositionAST:
    """Imputer+Scaler composition: Div(Paren(Sub(Coalesce(...),...)), Nullif(...))."""

    def test_nested_expression(self) -> None:
        """COALESCE is nested inside scaler arithmetic, not separate."""
        conn = duckdb.connect()
        conn.execute("""
            CREATE TABLE data AS SELECT * FROM VALUES
                (1.0, 10.0), (NULL, 20.0), (3.0, NULL)
            t(a, b)
        """)
        select = _fit_and_compose(conn, "data", [Imputer(strategy="mean"), StandardScaler()])
        output = _get_output_exprs(select)

        for col in ["a", "b"]:
            expr = output[col]

            # Top: Div (from scaler)
            assert isinstance(expr, exp.Div), (
                f"{col}: expected Div at top, got {type(expr).__name__}"
            )

            # Numerator: Paren(Sub(...))
            paren = expr.this
            assert isinstance(paren, exp.Paren)
            sub = paren.this
            assert isinstance(sub, exp.Sub)

            # The left side of Sub should contain Coalesce (from imputer)
            coalesce = sub.this
            assert isinstance(coalesce, exp.Coalesce), (
                f"{col}: expected Coalesce nested in Sub, got {type(coalesce).__name__}"
            )

            # Denominator: Nullif
            assert isinstance(expr.expression, exp.Nullif)

    def test_no_cte(self) -> None:
        """Expression-only pipeline (Imputer+Scaler) produces no CTEs."""
        conn = duckdb.connect()
        conn.execute("""
            CREATE TABLE data AS SELECT * FROM VALUES
                (1.0, 10.0), (NULL, 20.0), (3.0, NULL)
            t(a, b)
        """)
        select = _fit_and_compose(conn, "data", [Imputer(strategy="mean"), StandardScaler()])

        # No CTE nodes
        ctes = list(select.find_all(exp.CTE))
        assert len(ctes) == 0, f"Expected no CTEs, found {len(ctes)}"


# ── Expression Depth ─────────────────────────────────────────────────


class TestExpressionDepth:
    """Expression depth stays within expected bounds."""

    def test_single_transformer_depth(self) -> None:
        """Single StandardScaler should have reasonable depth."""
        conn = duckdb.connect()
        conn.execute("""
            CREATE TABLE data AS SELECT * FROM VALUES
                (1.0,), (2.0,), (3.0,)
            t(a)
        """)
        select = _fit_and_compose(conn, "data", [StandardScaler()])
        output = _get_output_exprs(select)

        from sqlearn.core.compiler import _expression_depth

        depth = _expression_depth(output["a"])
        # Div(Paren(Sub(Column, Literal)), Nullif(Literal, Literal))
        # Expected depth around 4-5
        assert depth <= 8, f"Unexpected depth {depth} for single StandardScaler"

    def test_two_step_composition_depth(self) -> None:
        """Imputer+Scaler composition should stay under CTE threshold."""
        conn = duckdb.connect()
        conn.execute("""
            CREATE TABLE data AS SELECT * FROM VALUES
                (1.0,), (NULL,), (3.0,)
            t(a)
        """)
        select = _fit_and_compose(conn, "data", [Imputer(strategy="mean"), StandardScaler()])
        output = _get_output_exprs(select)

        from sqlearn.core.compiler import _expression_depth

        depth = _expression_depth(output["a"])
        # Div(Paren(Sub(Coalesce(Column, Literal), Literal)), Nullif(...))
        # Expected depth around 5-6
        assert depth <= 8, (
            f"Unexpected depth {depth} for Imputer+Scaler — "
            "should stay under CTE promotion threshold"
        )
