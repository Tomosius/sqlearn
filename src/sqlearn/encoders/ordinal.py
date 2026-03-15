"""OrdinalEncoder — encode categorical columns as integer codes via SQL.

Compiles to inline SQL: ``CASE WHEN col = 'A' THEN 0 WHEN col = 'B' THEN 1 ... END``
per column. Original columns are replaced in-place with integer codes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sqlglot.expressions as exp

from sqlearn.core.transformer import Transformer

if TYPE_CHECKING:
    from sqlearn.core.schema import ColumnSelector, Schema


class OrdinalEncoder(Transformer):
    """Encode categorical columns as integer codes via SQL.

    Each unique category value in a categorical column is mapped to a
    contiguous integer starting from 0. Categories are sorted alphabetically
    for deterministic ordering. Unlike :class:`~sqlearn.encoders.onehot.OneHotEncoder`,
    the original column is replaced in-place (no new columns added).

    Categories are learned during ``fit()`` via ``SELECT DISTINCT``.

    Generated SQL::

        SELECT
          CASE WHEN city = 'Berlin' THEN 0
               WHEN city = 'London' THEN 1
               WHEN city = 'Paris' THEN 2
               ELSE -1 END AS city
        FROM __input__

    Args:
        categories: Category discovery mode. ``"auto"`` (default) learns
            categories from the training data via ``SELECT DISTINCT``.
        handle_unknown: Strategy for unseen categories at transform time.
            ``"error"`` (default) raises at validation time. When
            ``"use_encoded_value"`` is set, unseen categories are mapped
            to ``unknown_value``.
        unknown_value: Integer value assigned to unseen categories.
            Required when ``handle_unknown="use_encoded_value"``. Must
            not collide with valid ordinal codes (typically use ``-1``).
        columns: Column specification override. Defaults to all categorical
            columns via ``_default_columns = "categorical"``.

    Raises:
        NotFittedError: If ``transform()`` or ``to_sql()`` is called
            before ``fit()``.
        ValueError: If ``handle_unknown="use_encoded_value"`` and
            ``unknown_value`` is None, or if ``handle_unknown="error"``
            and ``unknown_value`` is not None.

    Examples:
        Basic usage --- encode all categorical columns:

        >>> import sqlearn as sq
        >>> pipe = sq.Pipeline([sq.OrdinalEncoder()])
        >>> pipe.fit("train.parquet")
        >>> result = pipe.transform("test.parquet")
        >>> pipe.to_sql()
        ... # CASE WHEN city = 'Berlin' THEN 0 WHEN city = 'London' THEN 1 ...

        Handle unknown categories with a fallback value:

        >>> encoder = sq.OrdinalEncoder(
        ...     handle_unknown="use_encoded_value", unknown_value=-1
        ... )
        >>> pipe = sq.Pipeline([encoder])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # CASE WHEN ... ELSE -1 END AS city

        Encode specific columns only:

        >>> encoder = sq.OrdinalEncoder(columns=["city", "color"])
        >>> pipe = sq.Pipeline([encoder])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()  # only city and color are encoded

        Full pipeline --- impute, encode, then scale numerics:

        >>> pipe = sq.Pipeline(
        ...     [
        ...         sq.Imputer(),
        ...         sq.OrdinalEncoder(),
        ...         sq.StandardScaler(),
        ...     ]
        ... )
        >>> pipe.fit("data.parquet", y="target")
        >>> pipe.to_sql()
        ... # COALESCE + CASE WHEN for categoricals, then scaling

    See Also:
        :class:`~sqlearn.encoders.onehot.OneHotEncoder`: Binary indicator
            encoding (one column per category).
        :class:`~sqlearn.imputers.imputer.Imputer`: Fill NULLs before encoding.
        :class:`~sqlearn.core.pipeline.Pipeline`: Compose transformers.
    """

    _default_columns: str = "categorical"  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "dynamic"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        categories: str = "auto",
        handle_unknown: str = "error",
        unknown_value: int | None = None,
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        super().__init__(columns=columns)
        self.categories = categories
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self._validate_params()

    def _validate_params(self) -> None:
        """Validate handle_unknown / unknown_value consistency.

        Raises:
            ValueError: If parameter combination is invalid.
        """
        if self.handle_unknown == "use_encoded_value" and self.unknown_value is None:
            msg = (
                "unknown_value must be set when handle_unknown='use_encoded_value'. "
                "Typical choice: unknown_value=-1."
            )
            raise ValueError(msg)
        if self.handle_unknown == "error" and self.unknown_value is not None:
            msg = (
                "unknown_value must be None when handle_unknown='error'. "
                "Set handle_unknown='use_encoded_value' to use a fallback value."
            )
            raise ValueError(msg)
        valid_handle_unknown = ("error", "use_encoded_value")
        if self.handle_unknown not in valid_handle_unknown:
            msg = (
                f"Invalid handle_unknown={self.handle_unknown!r}. "
                f"Must be one of {valid_handle_unknown!r}."
            )
            raise ValueError(msg)

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
        """Get sorted category list for a column.

        Args:
            col: Column name to get categories for.

        Returns:
            Sorted list of category strings (NULLs excluded).
        """
        raw: list[dict[str, Any]] = self.sets_[f"{col}__categories"]  # type: ignore[index]
        values: list[str] = []
        for row in raw:  # pyright: ignore[reportUnknownVariableType]
            val: Any = row[col]  # pyright: ignore[reportUnknownVariableType]
            if val is not None:
                values.append(str(val))  # pyright: ignore[reportUnknownArgumentType]
        values.sort()
        return values

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        """Generate CASE WHEN expressions for ordinal encoding.

        Creates a single ``CASE WHEN col = 'A' THEN 0 WHEN col = 'B' THEN 1
        ... END`` expression per column, mapping each category to its
        alphabetical index. The original column is replaced in-place.

        Args:
            columns: Target columns to encode.
            exprs: Current expression dict for ALL columns from prior steps.

        Returns:
            Dict of column expressions with CASE WHEN ordinal mappings.
            Replaces original columns (same keys, new values).
        """
        result: dict[str, exp.Expression] = {}
        for col in columns:
            categories = self._get_categories(col)

            # No categories (e.g. all-NULL column): emit NULL
            if not categories:
                result[col] = exp.Null()
                continue

            ifs: list[exp.If] = []
            for idx, cat in enumerate(categories):
                ifs.append(
                    exp.If(
                        this=exp.EQ(
                            this=exprs[col],
                            expression=exp.Literal.string(cat),  # pyright: ignore[reportUnknownMemberType]
                        ),
                        true=exp.Literal.number(idx),  # pyright: ignore[reportUnknownMemberType]
                    ),
                )

            # Default for unseen categories
            if self.handle_unknown == "use_encoded_value":
                default: exp.Expression = exp.Literal.number(  # pyright: ignore[reportUnknownMemberType]
                    self.unknown_value,
                )
            else:
                # handle_unknown="error": use NULL as sentinel
                default = exp.Null()

            result[col] = exp.Case(ifs=ifs, default=default)
        return result
