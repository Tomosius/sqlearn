"""Imputer — fill missing values via SQL COALESCE with learned or constant fills.

Supports four calling conventions:

1. ``Imputer()`` — auto-detect strategy per column type
2. ``Imputer(strategy="mean")`` — single strategy for all columns
3. ``Imputer(columns=["a","b"], strategy="median")`` — explicit columns
4. ``Imputer({"price": "mean", "qty": 0, "status": "active"})`` — per-column dict
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlglot.expressions as exp

from sqlearn.core.transformer import Transformer

if TYPE_CHECKING:
    from sqlearn.core.schema import ColumnSelector, Schema

_STRATEGIES: frozenset[str] = frozenset({"mean", "median", "most_frequent", "zero"})


class Imputer(Transformer):
    """Fill missing values via SQL COALESCE with learned or constant fill values.

    Automatically selects imputation strategy per column type when
    ``strategy="auto"`` (default): numeric columns use median, categorical
    columns use most_frequent.

    Four calling conventions are supported:

    1. ``Imputer()`` — auto strategy per column type
    2. ``Imputer(strategy="mean")`` — single strategy for all target columns
    3. ``Imputer(columns=["a","b"], strategy="median")`` — explicit columns
    4. ``Imputer({"price": "mean", "qty": 0})`` — per-column dict mode

    Args:
        strategy: Imputation strategy. One of ``"auto"``, ``"mean"``,
            ``"median"``, ``"most_frequent"``, ``"zero"``, or a dict
            mapping column names to strategies or constant fill values.
        columns: Column specification override. Defaults to all columns
            via ``_default_columns = "all"``. Ignored in dict mode
            (columns come from dict keys).

    Example::

        import sqlearn as sq

        pipe = sq.Pipeline([sq.Imputer()])
        pipe.fit("train.parquet")
        result = pipe.transform("test.parquet")  # no NULLs
        sql = pipe.to_sql()  # contains COALESCE(...)
    """

    _default_columns: str = "all"  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "dynamic"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        strategy: str | dict[str, str | int | float] = "auto",
        *,
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        super().__init__(columns=columns)
        self.strategy = strategy

    # --- Column resolution ---

    def _resolve_columns_spec(self) -> str | list[str] | ColumnSelector | None:
        """Return column spec, using dict keys in dict mode.

        In dict mode, columns come from the dict keys regardless of
        the ``columns`` parameter. Otherwise delegates to superclass.

        Returns:
            Column specification for resolution.
        """
        if isinstance(self.strategy, dict):
            return list(self.strategy.keys())
        return super()._resolve_columns_spec()

    # --- Strategy resolution ---

    def _resolve_strategy(self, col: str, schema: Schema) -> str | int | float:
        """Resolve the fill strategy for a single column.

        For dict mode, returns the dict value directly (strategy string
        or constant). For ``"auto"``, uses column type category to pick
        median (numeric) or most_frequent (categorical). Other strategies
        pass through unchanged.

        Args:
            col: Column name.
            schema: Current table schema for type inspection.

        Returns:
            Strategy string (``"mean"``, ``"median"``, ``"most_frequent"``,
            ``"zero"``) or a constant fill value (int, float, or string).
        """
        if isinstance(self.strategy, dict):
            value = self.strategy[col]
            # Known strategy strings are strategies, not constants
            if isinstance(value, str) and value in _STRATEGIES:
                return value
            return value
        if self.strategy == "auto":
            category = schema.column_category(col)
            if category == "numeric":
                return "median"
            if category == "categorical":
                return "most_frequent"
            # Safe default for temporal, boolean, other
            return "median"
        return self.strategy

    # --- Subclass overrides ---

    def discover(
        self,
        columns: list[str],
        schema: Schema,
        y_column: str | None = None,  # noqa: ARG002
    ) -> dict[str, exp.Expression]:
        """Learn fill values from data via SQL aggregates.

        Only emits aggregates for strategies that need data: ``"mean"``
        uses AVG, ``"median"`` uses MEDIAN, ``"most_frequent"`` uses MODE.
        Constant and ``"zero"`` strategies need no aggregation.

        Args:
            columns: Target columns to compute fill values for.
            schema: Current table schema.
            y_column: Target column name (unused, required by protocol).

        Returns:
            Dict mapping ``'{col}__value'`` to sqlglot aggregate expressions.
            Empty for columns with constant fill values.
        """
        result: dict[str, exp.Expression] = {}
        for col in columns:
            strategy = self._resolve_strategy(col, schema)

            if strategy == "mean":
                result[f"{col}__value"] = exp.Avg(this=exp.Column(this=col))
            elif strategy == "median":
                result[f"{col}__value"] = exp.Anonymous(
                    this="MEDIAN",
                    expressions=[exp.Column(this=col)],
                )
            elif strategy == "most_frequent":
                result[f"{col}__value"] = exp.Anonymous(
                    this="MODE",
                    expressions=[exp.Column(this=col)],
                )
            # "zero" and constants: no aggregate needed
        return result

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        """Generate COALESCE expressions for null replacement.

        Wraps each target column in ``COALESCE(exprs[col], fill_value)``
        where fill_value is either learned from data (via params_) or
        a constant specified in the strategy.

        Args:
            columns: Target columns to transform.
            exprs: Current expression dict for ALL columns from prior steps.

        Returns:
            Dict of modified column expressions with COALESCE wrapping.
        """
        params = self.params_ or {}
        schema = self.input_schema_
        result: dict[str, exp.Expression] = {}

        for col in columns:
            # Resolve fill value
            if schema is not None:
                strategy = self._resolve_strategy(col, schema)
            else:
                strategy = self.strategy if isinstance(self.strategy, str) else self.strategy[col]

            if isinstance(strategy, str) and strategy in _STRATEGIES:
                if strategy == "zero":
                    fill_value: str | int | float = 0
                else:
                    fill_value = params.get(f"{col}__value", 0)
            else:
                # Constant fill value (int, float, or string)
                fill_value = strategy

            # Build literal
            if isinstance(fill_value, str):
                fill_expr: exp.Expression = exp.Literal.string(fill_value)  # pyright: ignore[reportUnknownMemberType]
            else:
                fill_expr = exp.Literal.number(fill_value)  # pyright: ignore[reportUnknownMemberType]

            result[col] = exp.Coalesce(
                this=exprs[col],
                expressions=[fill_expr],
            )

        return result
