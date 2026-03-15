"""Datetime feature extraction transformers.

Static transformers for extracting date/time components into numeric features:
DateParts, DateDiff, IsWeekend, Quarter. All operate on temporal columns and
compile to inline SQL via sqlglot ASTs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlglot.expressions as exp

from sqlearn.core.errors import SchemaError
from sqlearn.core.schema import resolve_columns
from sqlearn.core.transformer import Transformer

if TYPE_CHECKING:
    from sqlearn.core.schema import ColumnSelector, Schema

# Valid date parts for EXTRACT
_VALID_PARTS: frozenset[str] = frozenset(
    {
        "year",
        "month",
        "day",
        "dayofweek",
        "hour",
        "minute",
        "second",
        "quarter",
        "week",
        "dayofyear",
    }
)

# Mapping from user-facing part names to SQL EXTRACT keywords
_PART_TO_SQL: dict[str, str] = {
    "year": "YEAR",
    "month": "MONTH",
    "day": "DAY",
    "dayofweek": "DOW",
    "hour": "HOUR",
    "minute": "MINUTE",
    "second": "SECOND",
    "quarter": "QUARTER",
    "week": "WEEK",
    "dayofyear": "DOY",
}


class DateParts(Transformer):
    """Extract multiple date parts into separate integer columns.

    A **static** transformer -- no statistics are learned during ``fit()``.
    Each temporal column produces one new column per requested part
    (e.g. ``ts_year``, ``ts_month``, ``ts_day``). The original column
    is preserved in the output.

    Generated SQL::

        SELECT
          EXTRACT(YEAR FROM ts) AS ts_year,
          EXTRACT(MONTH FROM ts) AS ts_month,
          EXTRACT(DAY FROM ts) AS ts_day,
          EXTRACT(DOW FROM ts) AS ts_dayofweek,
          EXTRACT(HOUR FROM ts) AS ts_hour,
          ts
        FROM __input__

    Args:
        parts: List of date parts to extract. Supported values:
            ``"year"``, ``"month"``, ``"day"``, ``"dayofweek"``,
            ``"hour"``, ``"minute"``, ``"second"``, ``"quarter"``,
            ``"week"``, ``"dayofyear"``.
            Defaults to ``["year", "month", "day", "dayofweek", "hour"]``.
        columns: Column specification override. Defaults to all temporal
            columns via ``_default_columns = "temporal"``.

    Raises:
        ValueError: If ``parts`` is empty or contains invalid part names.

    Examples:
        Extract year and month from timestamps:

        >>> import sqlearn as sq
        >>> pipe = sq.Pipeline([sq.DateParts(parts=["year", "month"])])
        >>> pipe.fit("events.parquet")
        >>> pipe.to_sql()
        ... # EXTRACT(YEAR FROM ts) AS ts_year, EXTRACT(MONTH FROM ts) AS ts_month

        Extract day of week for feature engineering:

        >>> pipe = sq.Pipeline([sq.DateParts(parts=["dayofweek"])])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # EXTRACT(DOW FROM ts) AS ts_dayofweek

    See Also:
        :class:`~sqlearn.features.datetime.IsWeekend`: Binary weekend flag.
        :class:`~sqlearn.features.datetime.Quarter`: Extract quarter (1-4).
        :class:`~sqlearn.core.transformer.Transformer`: Base class.
    """

    _default_columns: str = "temporal"  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "static"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        parts: list[str] | None = None,
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        """Initialize DateParts with parts to extract.

        Args:
            parts: List of date parts to extract. Defaults to
                ``["year", "month", "day", "dayofweek", "hour"]``.
            columns: Column specification override. If provided, takes
                precedence over ``_default_columns``.

        Raises:
            ValueError: If ``parts`` is empty or contains invalid part names.
        """
        super().__init__(columns=columns)
        if parts is None:
            parts = ["year", "month", "day", "dayofweek", "hour"]
        if len(parts) == 0:
            msg = "parts must be non-empty"
            raise ValueError(msg)
        invalid = set(parts) - _VALID_PARTS
        if invalid:
            msg = f"Invalid date parts: {sorted(invalid)}. Valid: {sorted(_VALID_PARTS)}"
            raise ValueError(msg)
        self.parts = parts

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        """Generate EXTRACT expressions for each part for each column.

        Creates ``EXTRACT(part FROM col)`` for each requested part and
        each target column. Returns new columns named ``{col}_{part}``.

        Args:
            columns: Target temporal columns to extract from.
            exprs: Current expression dict for ALL columns from prior steps.

        Returns:
            Dict of new column expressions. Original columns are not
            included (they pass through automatically).
        """
        result: dict[str, exp.Expression] = {}
        for col in columns:
            for part in self.parts:
                col_name = f"{col}_{part}"
                sql_part = _PART_TO_SQL[part]
                result[col_name] = exp.Extract(
                    this=exp.Var(this=sql_part),
                    expression=exprs[col],
                )
        return result

    def output_schema(self, schema: Schema) -> Schema:
        """Declare output schema: original columns plus new part columns.

        Args:
            schema: Input schema.

        Returns:
            Schema with new INTEGER columns for each extracted part.
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
            for part in self.parts:
                col_name = f"{col}_{part}"
                new_cols[col_name] = "INTEGER"
        return schema.add(new_cols)


class DateDiff(Transformer):
    """Compute date difference from a reference date or between columns.

    A **static** transformer -- no statistics are learned during ``fit()``.
    Computes the difference between each temporal column and a reference
    date (ISO string) or another column, in the specified time unit.

    Generated SQL (reference date)::

        SELECT
          DATEDIFF('DAY', '2020-01-01', ts) AS ts
        FROM __input__

    Generated SQL (reference column)::

        SELECT
          DATEDIFF('DAY', start_date, ts) AS ts
        FROM __input__

    Args:
        reference: ISO date string (e.g. ``"2020-01-01"``) or column name.
            Used as the starting point for the difference calculation.
        unit: Time unit for the difference. One of ``"day"``, ``"hour"``,
            ``"month"``, ``"year"``, ``"minute"``, ``"second"``, ``"week"``.
            Defaults to ``"day"``.
        columns: Column specification override. Defaults to all temporal
            columns via ``_default_columns = "temporal"``.

    Raises:
        ValueError: If ``unit`` is not a recognized time unit.

    Examples:
        Days since a reference date:

        >>> import sqlearn as sq
        >>> pipe = sq.Pipeline([sq.DateDiff(reference="2020-01-01")])
        >>> pipe.fit("events.parquet")
        >>> pipe.to_sql()
        ... # DATEDIFF('DAY', '2020-01-01', ts) AS ts

        Hours between two columns:

        >>> pipe = sq.Pipeline(
        ...     [sq.DateDiff(reference="start_time", unit="hour", columns=["end_time"])]
        ... )
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # DATEDIFF('HOUR', start_time, end_time) AS end_time

    See Also:
        :class:`~sqlearn.features.datetime.DateParts`: Extract date components.
        :class:`~sqlearn.core.transformer.Transformer`: Base class.
    """

    _default_columns: str = "temporal"  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "static"  # pyright: ignore[reportIncompatibleVariableOverride]

    _VALID_UNITS: frozenset[str] = frozenset(
        {"day", "hour", "month", "year", "minute", "second", "week"}
    )

    def __init__(
        self,
        *,
        reference: str,
        unit: str = "day",
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        """Initialize DateDiff with reference and unit.

        Args:
            reference: ISO date string or column name for the reference
                point of the difference calculation.
            unit: Time unit for the difference (default: ``"day"``).
            columns: Column specification override.

        Raises:
            ValueError: If ``unit`` is not a recognized time unit.
        """
        super().__init__(columns=columns)
        if unit not in self._VALID_UNITS:
            msg = f"Invalid unit: {unit!r}. Valid: {sorted(self._VALID_UNITS)}"
            raise ValueError(msg)
        self.reference = reference
        self.unit = unit

    def _is_column_reference(self) -> bool:
        """Check if reference looks like a column name (not an ISO date).

        Returns:
            True if reference does not match ISO date pattern.
        """
        # ISO dates contain hyphens and are all digits/hyphens/colons/spaces/T
        ref = self.reference
        # Simple heuristic: ISO dates match YYYY-MM-DD pattern
        return not (len(ref) >= 8 and ref[4:5] == "-")  # noqa: PLR2004

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        """Generate DATEDIFF expressions for each target column.

        Args:
            columns: Target temporal columns.
            exprs: Current expression dict for ALL columns from prior steps.

        Returns:
            Dict of column expressions with DATEDIFF applied.
        """
        result: dict[str, exp.Expression] = {}
        unit_var = exp.Var(this=self.unit.upper())
        for col in columns:
            if self._is_column_reference():
                ref_expr = exprs.get(self.reference, exp.Column(this=self.reference))
            else:
                ref_expr = exp.Literal.string(self.reference)  # pyright: ignore[reportUnknownMemberType]
            result[col] = exp.DateDiff(
                this=exprs[col],
                expression=ref_expr,
                unit=unit_var,
            )
        return result

    def output_schema(self, schema: Schema) -> Schema:
        """Declare output schema: temporal columns become INTEGER.

        Args:
            schema: Input schema.

        Returns:
            Schema with target columns changed to INTEGER type.
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

        cast_updates = {col: "INTEGER" for col in target_cols if col in schema.columns}
        if not cast_updates:
            return schema
        return schema.cast(cast_updates)


class IsWeekend(Transformer):
    """Binary flag: 1 if Saturday or Sunday, 0 otherwise.

    A **static** transformer -- no statistics are learned during ``fit()``.
    Each temporal column is replaced by a binary integer column indicating
    whether the date falls on a weekend day.

    Generated SQL::

        SELECT
          CASE
            WHEN EXTRACT(DOW FROM ts) IN (0, 6)
            THEN 1
            ELSE 0
          END AS ts
        FROM __input__

    .. note::
        DuckDB uses ISO day-of-week numbering where Sunday = 0
        and Saturday = 6.

    Args:
        columns: Column specification override. Defaults to all temporal
            columns via ``_default_columns = "temporal"``.

    Examples:
        Add weekend flag:

        >>> import sqlearn as sq
        >>> pipe = sq.Pipeline([sq.IsWeekend()])
        >>> pipe.fit("events.parquet")
        >>> pipe.to_sql()
        ... # CASE WHEN EXTRACT(DOW FROM ts) IN (0, 6) THEN 1 ELSE 0 END AS ts

        Apply only to specific columns:

        >>> pipe = sq.Pipeline([sq.IsWeekend(columns=["created_at"])])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # CASE WHEN EXTRACT(DOW FROM created_at) IN (0, 6) THEN 1 ELSE 0 END

    See Also:
        :class:`~sqlearn.features.datetime.DateParts`: Extract date components.
        :class:`~sqlearn.features.datetime.Quarter`: Extract quarter (1-4).
        :class:`~sqlearn.core.transformer.Transformer`: Base class.
    """

    _default_columns: str = "temporal"  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "static"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        """Initialize IsWeekend.

        Args:
            columns: Column specification override.
        """
        super().__init__(columns=columns)

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        """Generate CASE WHEN weekend expressions.

        Produces ``CASE WHEN EXTRACT(DOW FROM col) IN (0, 6) THEN 1 ELSE 0 END``
        for each target column.

        Args:
            columns: Target temporal columns.
            exprs: Current expression dict for ALL columns from prior steps.

        Returns:
            Dict of column expressions with weekend flag applied.
        """
        result: dict[str, exp.Expression] = {}
        for col in columns:
            dow_extract = exp.Extract(
                this=exp.Var(this="DOW"),
                expression=exprs[col],
            )
            result[col] = exp.Case(
                ifs=[
                    exp.If(
                        this=exp.In(
                            this=dow_extract,
                            expressions=[
                                exp.Literal.number(0),  # pyright: ignore[reportUnknownMemberType]
                                exp.Literal.number(6),  # pyright: ignore[reportUnknownMemberType]
                            ],
                        ),
                        true=exp.Literal.number(1),  # pyright: ignore[reportUnknownMemberType]
                    ),
                ],
                default=exp.Literal.number(0),  # pyright: ignore[reportUnknownMemberType]
            )
        return result

    def output_schema(self, schema: Schema) -> Schema:
        """Declare output schema: temporal columns become INTEGER.

        Args:
            schema: Input schema.

        Returns:
            Schema with target columns changed to INTEGER type.
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

        cast_updates = {col: "INTEGER" for col in target_cols if col in schema.columns}
        if not cast_updates:
            return schema
        return schema.cast(cast_updates)


class Quarter(Transformer):
    """Extract quarter (1-4) from temporal columns.

    A **static** transformer -- no statistics are learned during ``fit()``.
    Each temporal column is replaced by an integer column containing
    the quarter number (1 through 4).

    Generated SQL::

        SELECT
          EXTRACT(QUARTER FROM ts) AS ts
        FROM __input__

    Args:
        columns: Column specification override. Defaults to all temporal
            columns via ``_default_columns = "temporal"``.

    Examples:
        Extract quarter from timestamps:

        >>> import sqlearn as sq
        >>> pipe = sq.Pipeline([sq.Quarter()])
        >>> pipe.fit("events.parquet")
        >>> pipe.to_sql()
        ... # EXTRACT(QUARTER FROM ts) AS ts

        Apply only to specific columns:

        >>> pipe = sq.Pipeline([sq.Quarter(columns=["order_date"])])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # EXTRACT(QUARTER FROM order_date) AS order_date

    See Also:
        :class:`~sqlearn.features.datetime.DateParts`: Extract multiple parts.
        :class:`~sqlearn.features.datetime.IsWeekend`: Binary weekend flag.
        :class:`~sqlearn.core.transformer.Transformer`: Base class.
    """

    _default_columns: str = "temporal"  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "static"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        """Initialize Quarter.

        Args:
            columns: Column specification override.
        """
        super().__init__(columns=columns)

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        """Generate EXTRACT(QUARTER FROM col) expressions.

        Args:
            columns: Target temporal columns.
            exprs: Current expression dict for ALL columns from prior steps.

        Returns:
            Dict of column expressions with QUARTER extraction applied.
        """
        return {
            col: exp.Extract(
                this=exp.Var(this="QUARTER"),
                expression=exprs[col],
            )
            for col in columns
        }

    def output_schema(self, schema: Schema) -> Schema:
        """Declare output schema: temporal columns become INTEGER.

        Args:
            schema: Input schema.

        Returns:
            Schema with target columns changed to INTEGER type.
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

        cast_updates = {col: "INTEGER" for col in target_cols if col in schema.columns}
        if not cast_updates:
            return schema
        return schema.cast(cast_updates)
