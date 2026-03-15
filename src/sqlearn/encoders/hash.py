"""HashEncoder — map categorical columns to fixed-size hash buckets via SQL.

Compiles to inline SQL: ``CASE WHEN ABS(HASH(col)) % n = i THEN 1 ELSE 0 END``
per bucket per column. Original categorical columns are dropped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlglot.expressions as exp

from sqlearn.core.errors import SchemaError
from sqlearn.core.schema import resolve_columns
from sqlearn.core.transformer import Transformer

if TYPE_CHECKING:
    from sqlearn.core.schema import ColumnSelector, Schema


class HashEncoder(Transformer):
    """Map categorical columns to fixed-size hash buckets via SQL.

    Each categorical column is replaced by ``n_features`` binary columns,
    where exactly one column is 1 per row (the bucket the hashed value
    falls into). Unlike :class:`~sqlearn.encoders.onehot.OneHotEncoder`,
    HashEncoder does **not** need to learn categories from data --- it is
    a **static** transformer.

    This is useful for high-cardinality columns where one-hot encoding
    would create too many columns, or when you want a fixed-width output
    regardless of the number of unique values.

    Trade-off: hash collisions are possible. Two different category values
    may hash to the same bucket. Increase ``n_features`` to reduce
    collision probability.

    Generated SQL (n_features=8)::

        SELECT
          CASE WHEN ABS(HASH(city)) % 8 = 0 THEN 1 ELSE 0 END AS city_hash_0,
          CASE WHEN ABS(HASH(city)) % 8 = 1 THEN 1 ELSE 0 END AS city_hash_1,
          ...
          CASE WHEN ABS(HASH(city)) % 8 = 7 THEN 1 ELSE 0 END AS city_hash_7
        FROM __input__

    Args:
        n_features: Number of hash buckets (output columns per input column).
            Defaults to 8. Higher values reduce collision risk but increase
            dimensionality.
        columns: Column specification override. Defaults to all categorical
            columns via ``_default_columns = "categorical"``.

    Raises:
        ValueError: If ``n_features`` is less than 1.

    Examples:
        Basic usage --- hash all categorical columns into 8 buckets:

        >>> import sqlearn as sq
        >>> pipe = sq.Pipeline([sq.HashEncoder()])
        >>> pipe.fit("train.parquet")
        >>> pipe.to_sql()
        ... # CASE WHEN ABS(HASH(city)) % 8 = 0 THEN 1 ELSE 0 END AS city_hash_0, ...

        Fewer buckets for compact representation:

        >>> encoder = sq.HashEncoder(n_features=4)
        >>> pipe = sq.Pipeline([encoder])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()  # 4 binary columns per categorical column

        Hash specific columns only:

        >>> encoder = sq.HashEncoder(columns=["city", "color"])
        >>> pipe = sq.Pipeline([encoder])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()  # only city and color are hashed

        Full pipeline --- impute, scale numerics, hash categoricals:

        >>> pipe = sq.Pipeline(
        ...     [
        ...         sq.Imputer(),
        ...         sq.StandardScaler(),
        ...         sq.HashEncoder(),
        ...     ]
        ... )
        >>> pipe.fit("data.parquet", y="target")
        >>> pipe.to_sql()
        ... # COALESCE + scaling for numerics, HASH for categoricals

    See Also:
        :class:`~sqlearn.encoders.onehot.OneHotEncoder`: Exact encoding
            (one column per category, requires fit).
        :class:`~sqlearn.imputers.imputer.Imputer`: Fill NULLs before encoding.
        :class:`~sqlearn.core.pipeline.Pipeline`: Compose transformers.
    """

    _default_columns: str = "categorical"  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "static"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        n_features: int = 8,
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        super().__init__(columns=columns)
        if n_features < 1:
            msg = f"n_features must be >= 1, got {n_features}"
            raise ValueError(msg)
        self.n_features = n_features

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        """Generate CASE WHEN expressions for hash bucket assignment.

        Creates ``n_features`` binary columns per input column using
        ``CASE WHEN ABS(HASH(col)) % n_features = i THEN 1 ELSE 0 END``.
        Returns ONLY new hash columns; original columns are excluded
        via output_schema().

        Uses ``.copy()`` on the hash/mod sub-expression for each bucket
        to avoid sqlglot AST node sharing issues.

        Args:
            columns: Target columns to encode.
            exprs: Current expression dict for ALL columns from prior steps.

        Returns:
            Dict of new binary column expressions. Does not include
            original columns (those are removed by output_schema).
        """
        result: dict[str, exp.Expression] = {}
        for col in columns:
            # ABS(HASH(col)) % n_features — built once, .copy()-ed per bucket
            hash_expr = exp.Anonymous(this="HASH", expressions=[exprs[col]])
            abs_hash = exp.Abs(this=hash_expr)
            mod_expr = exp.Mod(
                this=abs_hash,
                expression=exp.Literal.number(self.n_features),  # pyright: ignore[reportUnknownMemberType]
            )

            for i in range(self.n_features):
                col_name = f"{col}_hash_{i}"
                result[col_name] = exp.Case(
                    ifs=[
                        exp.If(
                            this=exp.EQ(
                                this=mod_expr.copy(),
                                expression=exp.Literal.number(i),  # pyright: ignore[reportUnknownMemberType]
                            ),
                            true=exp.Literal.number(1),  # pyright: ignore[reportUnknownMemberType]
                        ),
                    ],
                    default=exp.Literal.number(0),  # pyright: ignore[reportUnknownMemberType]
                )
        return result

    def output_schema(self, schema: Schema) -> Schema:
        """Declare output schema: drop originals, add hash bucket columns.

        Removes target categorical columns and adds ``n_features`` INTEGER
        columns per original column, named ``{col}_hash_{i}``.

        Args:
            schema: Input schema.

        Returns:
            Output schema with original categorical columns replaced by
            hash bucket columns.
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

        new_cols: dict[str, str] = {}
        for col in target_cols:
            for i in range(self.n_features):
                new_cols[f"{col}_hash_{i}"] = "INTEGER"

        return schema.drop(target_cols).add(new_cols)
