"""OneHotEncoder — encode categorical columns as binary indicator columns via SQL.

Compiles to inline SQL: ``CASE WHEN col = 'category' THEN 1 ELSE 0 END``
per category per column. Original categorical columns are dropped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sqlglot.expressions as exp

from sqlearn.core.errors import SchemaError
from sqlearn.core.schema import resolve_columns
from sqlearn.core.transformer import Transformer

if TYPE_CHECKING:
    from sqlearn.core.schema import ColumnSelector, Schema


class OneHotEncoder(Transformer):
    """Encode categorical columns as binary indicator columns via SQL.

    Each unique category value in a categorical column becomes a new binary
    column with values 0 or 1. The original categorical column is dropped.
    Categories are learned during fit() via ``SELECT DISTINCT``.

    Safe defaults: categories are sorted alphabetically and truncated to
    ``max_categories`` to prevent column explosion on high-cardinality columns.

    Args:
        max_categories: Maximum number of categories per column.
            Categories beyond this limit are silently dropped (alphabetical
            order). Defaults to 30.
        columns: Column specification override. Defaults to all categorical
            columns via ``_default_columns = "categorical"``.

    Example::

        import sqlearn as sq

        pipe = sq.Pipeline([sq.OneHotEncoder()])
        pipe.fit("train.parquet")
        result = pipe.transform("test.parquet")  # binary indicator columns
        sql = pipe.to_sql()  # contains CASE WHEN ...
    """

    _default_columns: str = "categorical"  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "dynamic"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        max_categories: int = 30,
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        super().__init__(columns=columns)
        self.max_categories = max_categories

    def discover_sets(
        self,
        columns: list[str],
        schema: Schema,  # noqa: ARG002
        y_column: str | None = None,  # noqa: ARG002
    ) -> dict[str, exp.Expression]:
        """Learn distinct categories per column via SELECT DISTINCT.

        Returns sqlglot SELECT queries that the compiler executes during
        fit(). The compiler adds a FROM clause automatically.

        Args:
            columns: Target columns to discover categories for.
            schema: Current table schema (unused, required by protocol).
            y_column: Target column name (unused, required by protocol).

        Returns:
            Dict mapping ``'{col}__categories'`` to ``SELECT DISTINCT col``
            queries for each column.
        """
        result: dict[str, exp.Expression] = {}
        for col in columns:
            # SELECT DISTINCT col — compiler adds FROM source
            result[f"{col}__categories"] = exp.Select(
                expressions=[exp.Distinct(expressions=[exp.Column(this=col)])]
            )
        return result

    def _get_categories(self, col: str) -> list[str]:
        """Get sorted, truncated category list for a column.

        Args:
            col: Column name to get categories for.

        Returns:
            Sorted list of category strings, truncated to max_categories.
        """
        raw: list[dict[str, Any]] = self.sets_[f"{col}__categories"]  # type: ignore[index]
        values: list[str] = []
        for row in raw:  # pyright: ignore[reportUnknownVariableType]
            val: Any = row[col]  # pyright: ignore[reportUnknownVariableType]
            if val is not None:
                values.append(str(val))  # pyright: ignore[reportUnknownArgumentType]
        values.sort()
        return values[: self.max_categories]

    def _category_col_name(self, col: str, category: str) -> str:
        """Generate safe column name: col_category (lowercase, spaces to underscores).

        Args:
            col: Original column name.
            category: Category value.

        Returns:
            Safe column name like ``col_category``.
        """
        safe = category.lower().replace(" ", "_")
        return f"{col}_{safe}"

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        """Generate CASE WHEN expressions for one-hot encoding.

        Creates one ``CASE WHEN col = 'category' THEN 1 ELSE 0 END``
        expression per category per column. Returns ONLY new binary
        columns; original columns are excluded via output_schema().

        Args:
            columns: Target columns to encode.
            exprs: Current expression dict for ALL columns from prior steps.

        Returns:
            Dict of new binary column expressions. Does not include
            original columns (those are removed by output_schema).
        """
        result: dict[str, exp.Expression] = {}
        for col in columns:
            categories = self._get_categories(col)
            for cat in categories:
                col_name = self._category_col_name(col, cat)
                result[col_name] = exp.Case(
                    ifs=[
                        exp.If(
                            this=exp.EQ(
                                this=exprs[col],
                                expression=exp.Literal.string(cat),  # pyright: ignore[reportUnknownMemberType]
                            ),
                            true=exp.Literal.number(1),  # pyright: ignore[reportUnknownMemberType]
                        ),
                    ],
                    default=exp.Literal.number(0),  # pyright: ignore[reportUnknownMemberType]
                )
        return result

    def output_schema(self, schema: Schema) -> Schema:
        """Declare output schema: drop originals, add binary columns.

        Works in two modes:

        1. **Pre-fit** (sets_ is None): drops target columns to signal
           schema change. Exact binary columns unknown yet.
        2. **Post-fit** (sets_ populated): drops originals, adds exact
           binary INTEGER columns.

        Args:
            schema: Input schema.

        Returns:
            Output schema with original categorical columns replaced by
            binary indicator columns.
        """
        col_spec = self._resolve_columns_spec()
        if col_spec is None:
            return schema
        try:
            target_cols = resolve_columns(schema, col_spec)
        except (ValueError, SchemaError):
            return schema

        if not target_cols:
            return schema

        result = schema.drop(target_cols)

        if self.sets_ is not None:
            new_cols: dict[str, str] = {}
            for col in target_cols:
                for cat in self._get_categories(col):
                    col_name = self._category_col_name(col, cat)
                    new_cols[col_name] = "INTEGER"
            result = result.add(new_cols)

        return result
