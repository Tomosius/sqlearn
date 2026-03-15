"""Custom transformer factories — Expression() and custom().

Level 1: ``sq.Expression("price * qty AS revenue")`` — static one-liner
Level 2: ``sq.custom("{col} - {mean}", learn={"mean": "AVG({col})"})`` — template
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlglot
import sqlglot.expressions as exp

from sqlearn.core.schema import resolve_columns
from sqlearn.core.transformer import Transformer

if TYPE_CHECKING:
    from sqlearn.core.schema import Schema

# Placeholder for {col} in templates — distinctive to avoid collisions
_COL_PLACEHOLDER = "__sqcol__"


# ── Helpers ──────────────────────────────────────────────────────────


def _parse_select_expr(sql: str) -> exp.Expression:
    """Parse a SQL expression by wrapping in SELECT ... FROM __t__.

    Args:
        sql: SQL expression string (e.g. ``"LN(price + 1) AS price_log"``).

    Returns:
        The parsed sqlglot expression (may be Alias, Column, etc.).

    Raises:
        ValueError: If the SQL cannot be parsed.
    """
    wrapped = f"SELECT {sql} FROM __t__"  # noqa: S608
    try:
        select = sqlglot.parse_one(wrapped, dialect="duckdb")  # pyright: ignore[reportUnknownMemberType]
    except Exception as e:
        msg = f"Invalid SQL template: {e}"
        raise ValueError(msg) from e
    if not isinstance(select, exp.Select) or not select.expressions:
        msg = f"Invalid SQL template: could not parse '{sql}'"
        raise ValueError(msg)
    result: exp.Expression = select.expressions[0]
    return result


def _substitute_column_refs(
    expr: exp.Expression,
    exprs: dict[str, exp.Expression],
) -> exp.Expression:
    """Replace bare Column references with composed expressions from prior steps.

    Walks the AST and replaces ``Column("price")`` with ``exprs["price"]``
    (which may be ``Coalesce(Column("price"), Literal(3.0))`` from an Imputer).

    Only replaces columns without a table qualifier (bare column refs).

    Args:
        expr: Expression to transform (modified in place on copy).
        exprs: Current composed expressions from prior pipeline steps.

    Returns:
        New expression with column references substituted.
    """
    expr = expr.copy()
    for node in expr.walk():
        if isinstance(node, exp.Column) and not node.table and node.name in exprs:
            node.replace(exprs[node.name].copy())
    return expr


# ── Expression ───────────────────────────────────────────────────────


class _ExpressionTransformer(Transformer):
    """Internal transformer for sq.Expression() — static one-liner.

    Not for direct use. Use ``sq.Expression(sql)`` factory function.
    """

    _classification: str = "static"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(self, sql: str, *, columns: None = None) -> None:
        super().__init__(columns=columns)
        self.sql = sql
        # Parse and validate
        self._alias: str = ""
        self._expr: exp.Expression = exp.Null()
        self._parse()

    def _parse(self) -> None:
        """Parse SQL and extract alias + expression."""
        parsed = _parse_select_expr(self.sql)
        if not isinstance(parsed, exp.Alias):
            msg = f"Expression must contain AS <name> to define output column. Got: '{self.sql}'"
            raise ValueError(msg)  # noqa: TRY004
        self._alias = parsed.alias
        self._expr = parsed.this

    def expressions(
        self,
        columns: list[str],  # noqa: ARG002
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        """Return the single new column expression.

        Args:
            columns: Target columns (unused — Expression has no column routing).
            exprs: Current composed expressions from prior steps.

        Returns:
            Dict with one entry: ``{alias: substituted_expression}``.
        """
        expr = _substitute_column_refs(self._expr, exprs)
        return {self._alias: expr}

    def output_schema(self, schema: Schema) -> Schema:
        """Add the new column to the schema.

        Args:
            schema: Input schema.

        Returns:
            Schema with the new column added.
        """
        return schema.add({self._alias: "VARCHAR"})


def Expression(sql: str) -> Transformer:  # noqa: N802
    """Create a static one-liner transformer from a SQL expression.

    The SQL string must contain ``AS <name>`` to define the output column.
    Original columns pass through unchanged. The expression is parsed
    through sqlglot at creation time for validation and multi-database safety.

    Args:
        sql: SQL expression with ``AS <name>`` clause.
            Examples: ``"price * qty AS revenue"``,
            ``"COALESCE(nickname, first_name) AS display_name"``.

    Returns:
        A static Transformer that adds one new column.

    Raises:
        ValueError: If SQL is invalid or missing ``AS <name>``.

    Example::

        import sqlearn as sq

        pipe = sq.Pipeline(
            [
                sq.Imputer(),
                sq.Expression("price * quantity AS revenue"),
                sq.StandardScaler(),
            ]
        )
    """
    return _ExpressionTransformer(sql)


# ── custom() ─────────────────────────────────────────────────────────


class _CustomTransformer(Transformer):
    """Internal transformer for sq.custom() — template-based.

    Not for direct use. Use ``sq.custom(sql, ...)`` factory function.
    """

    def __init__(
        self,
        sql: str,
        *,
        columns: str | list[str] | None = None,
        learn: dict[str, str] | None = None,
        mode: str = "per_column",
    ) -> None:
        super().__init__(columns=columns)
        self.sql = sql
        self.learn = learn
        self.mode = mode
        # Set classification based on learn
        self._classification = "dynamic" if learn else "static"
        # Parsed state
        self._has_alias: bool = False
        self._learn_keys: list[str] = list((learn or {}).keys())
        self._validate()

    def _validate(self) -> None:
        """Validate template at creation time."""
        if self.mode not in ("per_column", "combine"):
            msg = f"mode must be 'per_column' or 'combine', got '{self.mode}'"
            raise ValueError(msg)
        if self.mode == "combine" and "{col}" in self.sql:
            msg = "mode='combine' cannot use {col} placeholder — reference columns by name"
            raise ValueError(msg)

        # Substitute placeholders with valid SQL tokens
        test_sql = self.sql.replace("{col}", _COL_PLACEHOLDER)
        for key in self._learn_keys:
            test_sql = test_sql.replace(f"{{{key}}}", "0")

        parsed = _parse_select_expr(test_sql)
        self._has_alias = isinstance(parsed, exp.Alias)

        # Validate learn templates
        for key, agg_sql in (self.learn or {}).items():
            test_agg = agg_sql.replace("{col}", _COL_PLACEHOLDER)
            try:
                _parse_select_expr(test_agg)
            except ValueError as e:
                msg = f"Invalid SQL in learn['{key}']: {e}"
                raise ValueError(msg) from e

    def discover(
        self,
        columns: list[str],
        schema: Schema,  # noqa: ARG002
        y_column: str | None = None,  # noqa: ARG002
    ) -> dict[str, exp.Expression]:
        """Expand learn templates into aggregate expressions per column.

        Args:
            columns: Target columns.
            schema: Current schema (unused).
            y_column: Target column (unused).

        Returns:
            Dict of ``{col}__param`` to sqlglot aggregate expressions.
        """
        if not self.learn:
            return {}
        result: dict[str, exp.Expression] = {}
        for col in columns:
            for param_name, agg_template in self.learn.items():
                agg_sql = agg_template.replace("{col}", col)
                parsed = _parse_select_expr(agg_sql)
                # Unwrap alias if present (SELECT AVG(x) AS avg → just AVG(x))
                if isinstance(parsed, exp.Alias):
                    parsed = parsed.this
                result[f"{col}__{param_name}"] = parsed
        return result

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        """Expand template per column with learned values substituted.

        Args:
            columns: Target columns.
            exprs: Current composed expressions from prior steps.

        Returns:
            Dict of output column expressions.
        """
        params = self.params_ or {}
        result: dict[str, exp.Expression] = {}

        if self.mode == "combine":
            # Single expression, columns referenced by name directly
            expr_sql = self.sql
            parsed = _parse_select_expr(expr_sql)
            if isinstance(parsed, exp.Alias):
                alias = parsed.alias
                expr_node = parsed.this
            else:
                # combine mode without alias: no-op passthrough
                return {}
            expr_node = _substitute_column_refs(expr_node, exprs)
            result[alias] = expr_node
        else:
            # Per-column iteration
            for col in columns:
                col_sql = self.sql.replace("{col}", col)
                for param_name in self._learn_keys:
                    value = params.get(f"{col}__{param_name}", 0)
                    col_sql = col_sql.replace(f"{{{param_name}}}", str(value))

                parsed = _parse_select_expr(col_sql)
                if isinstance(parsed, exp.Alias):
                    alias = parsed.alias
                    expr_node = parsed.this
                else:
                    alias = col  # replace in-place
                    expr_node = parsed

                expr_node = _substitute_column_refs(expr_node, exprs)
                result[alias] = expr_node

        return result

    def output_schema(self, schema: Schema) -> Schema:
        """Add new columns when template has AS clause.

        Args:
            schema: Input schema.

        Returns:
            Schema, possibly with new columns added.
        """
        if not self._has_alias:
            return schema

        if self.mode == "combine":
            # Single new column from combine mode
            parsed = _parse_select_expr(self.sql)
            if isinstance(parsed, exp.Alias):
                return schema.add({parsed.alias: "VARCHAR"})
            return schema

        # Per-column: resolve target columns and add one new column per target
        col_spec = self._resolve_columns_spec()
        if col_spec is None:
            return schema
        try:
            target_cols = resolve_columns(schema, col_spec)
        except (ValueError, Exception):
            return schema

        new_cols: dict[str, str] = {}
        for col in target_cols:
            # Derive alias by substituting {col} and parsing
            col_sql = self.sql.replace("{col}", col)
            for key in self._learn_keys:
                col_sql = col_sql.replace(f"{{{key}}}", "0")
            parsed = _parse_select_expr(col_sql)
            if isinstance(parsed, exp.Alias):
                new_cols[parsed.alias] = "VARCHAR"

        return schema.add(new_cols) if new_cols else schema


def custom(
    sql: str,
    *,
    columns: str | list[str] | None = None,
    learn: dict[str, str] | None = None,
    mode: str = "per_column",
) -> Transformer:
    """Create a template-based transformer from a SQL expression.

    Covers 90% of custom transformer needs without subclassing.
    Use ``{col}`` to reference the current column (in per_column mode)
    and ``{param}`` placeholders for learned values (from ``learn=``).

    Args:
        sql: SQL template with ``{col}`` and ``{param}`` placeholders.
        columns: Column selector (``"numeric"``, ``["a", "b"]``, etc.).
        learn: Dict of ``{param: aggregate_sql}`` for dynamic learning.
            When present, the transformer becomes dynamic.
        mode: ``"per_column"`` (default) iterates template per column.
            ``"combine"`` applies template once for cross-column expressions.

    Returns:
        A Transformer (static if no learn, dynamic if learn provided).

    Raises:
        ValueError: If SQL is invalid, learn SQL is invalid, or mode constraints violated.

    Example::

        import sqlearn as sq

        # Static per-column
        log = sq.custom("LN({col} + 1)", columns="numeric")

        # Dynamic per-column
        z = sq.custom(
            "({col} - {mean}) / NULLIF({std}, 0)",
            columns="numeric",
            learn={"mean": "AVG({col})", "std": "STDDEV_POP({col})"},
        )

        # Static cross-column (combine)
        bmi = sq.custom(
            "weight / (height * height) * 703 AS bmi",
            columns=["weight", "height"],
            mode="combine",
        )
    """
    return _CustomTransformer(sql, columns=columns, learn=learn, mode=mode)
