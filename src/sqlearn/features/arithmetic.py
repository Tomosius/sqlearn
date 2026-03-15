"""Arithmetic transforms -- Log, Sqrt, Power, Clip, Abs, Round, Reciprocal.

All are **static** transformers (no statistics learned during ``fit()``).
Each compiles to a single inline SQL expression per column.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlglot.expressions as exp

from sqlearn.core.transformer import Transformer

if TYPE_CHECKING:
    from sqlearn.core.schema import ColumnSelector


class Log(Transformer):
    """Natural (or custom-base) logarithm with configurable offset.

    Compiles to inline SQL: ``LN(col + offset)`` (natural log) or
    ``LN(col + offset) / LN(base)`` (change-of-base formula).

    The default ``offset=1`` ensures ``LN(0 + 1) = 0`` instead of
    ``LN(0) = -Infinity``. Set ``offset=0`` if your data has no zeros.

    Generated SQL (natural log, default)::

        SELECT LN(price + 1) AS price FROM __input__

    Generated SQL (base-10)::

        SELECT LN(price + 1) / LN(10) AS price FROM __input__

    Args:
        base: Logarithm base. ``None`` (default) uses natural log (base *e*).
            Common choices: ``10`` (common log), ``2`` (binary log).
        offset: Value added before taking the log. Default ``1`` to handle
            zeros safely.
        columns: Column specification override. Defaults to all numeric
            columns via ``_default_columns = "numeric"``.

    Raises:
        ValueError: If ``base`` is not positive or equals 1 (``LN(1) = 0``
            causes division by zero in the change-of-base formula).
        ValueError: If ``offset`` is negative.

    Examples:
        Natural log with default offset (handles zeros):

        >>> import sqlearn as sq
        >>> pipe = sq.Pipeline([sq.Log()])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # LN(price + 1) AS price

        Base-10 logarithm:

        >>> pipe = sq.Pipeline([sq.Log(base=10)])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # LN(price + 1) / LN(10) AS price

        No offset (data guaranteed positive):

        >>> pipe = sq.Pipeline([sq.Log(offset=0)])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # LN(price) AS price

    See Also:
        :class:`Sqrt`: Square root transform.
        :class:`Power`: Raise to arbitrary power.
        :class:`Reciprocal`: Inverse (1/x) transform.
    """

    _default_columns: str = "numeric"  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "static"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        base: float | None = None,
        offset: float = 1,
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        """Initialize Log transform.

        Args:
            base: Logarithm base. ``None`` for natural log.
            offset: Value added before log. Default ``1``.
            columns: Column specification override.

        Raises:
            ValueError: If ``base`` is not positive or equals 1.
            ValueError: If ``offset`` is negative.
        """
        if base is not None:
            if base <= 0:
                msg = f"base must be positive, got {base}"
                raise ValueError(msg)
            if base == 1:
                msg = "base must not be 1 (LN(1) = 0 causes division by zero)"
                raise ValueError(msg)
        if offset < 0:
            msg = f"offset must be non-negative, got {offset}"
            raise ValueError(msg)

        super().__init__(columns=columns)
        self.base = base
        self.offset = offset

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        """Generate inline LN expressions for each target column.

        Args:
            columns: Target columns to transform.
            exprs: Current expression dict for ALL columns from prior steps.

        Returns:
            Dict of modified column expressions with LN applied.
        """
        result: dict[str, exp.Expression] = {}
        for col in columns:
            inner = exprs[col]
            if self.offset != 0:
                inner = exp.Add(
                    this=inner,
                    expression=exp.Literal.number(self.offset),  # pyright: ignore[reportUnknownMemberType]
                )
            ln_expr: exp.Expression = exp.Ln(this=inner)
            if self.base is not None:
                ln_expr = exp.Div(
                    this=ln_expr,
                    expression=exp.Ln(
                        this=exp.Literal.number(self.base),  # pyright: ignore[reportUnknownMemberType]
                    ),
                )
            result[col] = ln_expr
        return result


class Sqrt(Transformer):
    """Square root transform.

    Compiles to inline SQL: ``SQRT(col)``.

    Generated SQL::

        SELECT SQRT(price) AS price FROM __input__

    Args:
        columns: Column specification override. Defaults to all numeric
            columns via ``_default_columns = "numeric"``.

    Examples:
        Apply square root to all numeric columns:

        >>> import sqlearn as sq
        >>> pipe = sq.Pipeline([sq.Sqrt()])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # SQRT(price) AS price

        Compose with Imputer (impute NULLs, then sqrt):

        >>> pipe = sq.Pipeline([sq.Imputer(), sq.Sqrt()])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # SQRT(COALESCE(price, 42.0)) AS price

    See Also:
        :class:`Log`: Logarithmic transform.
        :class:`Power`: Raise to arbitrary power.
    """

    _default_columns: str = "numeric"  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "static"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        """Initialize Sqrt transform.

        Args:
            columns: Column specification override.
        """
        super().__init__(columns=columns)

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        """Generate inline SQRT expressions for each target column.

        Args:
            columns: Target columns to transform.
            exprs: Current expression dict for ALL columns from prior steps.

        Returns:
            Dict of modified column expressions with SQRT applied.
        """
        return {col: exp.Sqrt(this=exprs[col]) for col in columns}


class Power(Transformer):
    """Raise columns to a fixed power.

    Compiles to inline SQL: ``POW(col, exponent)``.

    Generated SQL::

        SELECT POW(price, 2) AS price FROM __input__

    Args:
        exponent: The power to raise each column to.
        columns: Column specification override. Defaults to all numeric
            columns via ``_default_columns = "numeric"``.

    Raises:
        TypeError: If ``exponent`` is not a number.

    Examples:
        Square all numeric columns:

        >>> import sqlearn as sq
        >>> pipe = sq.Pipeline([sq.Power(exponent=2)])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # POW(price, 2) AS price

        Cube root (exponent=1/3):

        >>> pipe = sq.Pipeline([sq.Power(exponent=1 / 3)])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # POW(price, 0.3333...) AS price

    See Also:
        :class:`Sqrt`: Square root (equivalent to ``Power(exponent=0.5)``).
        :class:`Log`: Logarithmic transform.
    """

    _default_columns: str = "numeric"  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "static"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        exponent: float,
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        """Initialize Power transform.

        Args:
            exponent: The power to raise each column to.
            columns: Column specification override.

        Raises:
            TypeError: If ``exponent`` is not int or float.
        """
        if not isinstance(exponent, int | float):
            msg = f"exponent must be a number, got {type(exponent).__name__}"
            raise TypeError(msg)

        super().__init__(columns=columns)
        self.exponent = exponent

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        """Generate inline POW expressions for each target column.

        Args:
            columns: Target columns to transform.
            exprs: Current expression dict for ALL columns from prior steps.

        Returns:
            Dict of modified column expressions with POW applied.
        """
        return {
            col: exp.Pow(
                this=exprs[col],
                expression=exp.Literal.number(self.exponent),  # pyright: ignore[reportUnknownMemberType]
            )
            for col in columns
        }


class Clip(Transformer):
    """Clip (clamp) column values to a range.

    Compiles to inline SQL using ``GREATEST`` and ``LEAST``:

    - Both bounds: ``GREATEST(LEAST(col, upper), lower)``
    - Lower only: ``GREATEST(col, lower)``
    - Upper only: ``LEAST(col, upper)``

    Generated SQL (both bounds)::

        SELECT GREATEST(LEAST(price, 100), 0) AS price FROM __input__

    Args:
        lower: Minimum value. Values below this are clipped up.
        upper: Maximum value. Values above this are clipped down.
        columns: Column specification override. Defaults to all numeric
            columns via ``_default_columns = "numeric"``.

    Raises:
        ValueError: If both ``lower`` and ``upper`` are None.
        ValueError: If ``lower`` is greater than ``upper``.

    Examples:
        Clip prices to [0, 100]:

        >>> import sqlearn as sq
        >>> pipe = sq.Pipeline([sq.Clip(lower=0, upper=100)])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # GREATEST(LEAST(price, 100), 0) AS price

        Floor only (no upper bound):

        >>> pipe = sq.Pipeline([sq.Clip(lower=0)])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # GREATEST(price, 0) AS price

        Cap only (no lower bound):

        >>> pipe = sq.Pipeline([sq.Clip(upper=1000)])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # LEAST(price, 1000) AS price

    See Also:
        :class:`Abs`: Absolute value (clips sign).
        :class:`Round`: Round to N decimals.
    """

    _default_columns: str = "numeric"  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "static"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        lower: float | None = None,
        upper: float | None = None,
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        """Initialize Clip transform.

        Args:
            lower: Minimum value (floor). None for no lower bound.
            upper: Maximum value (ceiling). None for no upper bound.
            columns: Column specification override.

        Raises:
            ValueError: If both ``lower`` and ``upper`` are None.
            ValueError: If ``lower > upper``.
        """
        if lower is None and upper is None:
            msg = "at least one of lower or upper must be specified"
            raise ValueError(msg)
        if lower is not None and upper is not None and lower > upper:
            msg = f"lower ({lower}) must not be greater than upper ({upper})"
            raise ValueError(msg)

        super().__init__(columns=columns)
        self.lower = lower
        self.upper = upper

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        """Generate inline GREATEST/LEAST expressions for each target column.

        Args:
            columns: Target columns to transform.
            exprs: Current expression dict for ALL columns from prior steps.

        Returns:
            Dict of modified column expressions with clipping applied.
        """
        result: dict[str, exp.Expression] = {}
        for col in columns:
            expr = exprs[col]
            if self.upper is not None:
                expr = exp.Least(
                    this=expr,
                    expressions=[exp.Literal.number(self.upper)],  # pyright: ignore[reportUnknownMemberType]
                )
            if self.lower is not None:
                expr = exp.Greatest(
                    this=expr,
                    expressions=[exp.Literal.number(self.lower)],  # pyright: ignore[reportUnknownMemberType]
                )
            result[col] = expr
        return result


class Abs(Transformer):
    """Absolute value transform.

    Compiles to inline SQL: ``ABS(col)``.

    Generated SQL::

        SELECT ABS(price) AS price FROM __input__

    Args:
        columns: Column specification override. Defaults to all numeric
            columns via ``_default_columns = "numeric"``.

    Examples:
        Absolute value of all numeric columns:

        >>> import sqlearn as sq
        >>> pipe = sq.Pipeline([sq.Abs()])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # ABS(price) AS price

        Compose with StandardScaler (scale, then take absolute value):

        >>> pipe = sq.Pipeline([sq.StandardScaler(), sq.Abs()])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # ABS((price - 3.0) / NULLIF(1.41, 0)) AS price

    See Also:
        :class:`Clip`: Clip values to a range.
        :class:`Round`: Round to N decimals.
    """

    _default_columns: str = "numeric"  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "static"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        """Initialize Abs transform.

        Args:
            columns: Column specification override.
        """
        super().__init__(columns=columns)

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        """Generate inline ABS expressions for each target column.

        Args:
            columns: Target columns to transform.
            exprs: Current expression dict for ALL columns from prior steps.

        Returns:
            Dict of modified column expressions with ABS applied.
        """
        return {col: exp.Abs(this=exprs[col]) for col in columns}


class Round(Transformer):
    """Round columns to a fixed number of decimal places.

    Compiles to inline SQL: ``ROUND(col, decimals)``.

    Generated SQL::

        SELECT ROUND(price, 2) AS price FROM __input__

    Args:
        decimals: Number of decimal places. Default ``0`` (round to integer).
            Negative values round to powers of 10 (e.g. ``-1`` rounds to
            nearest 10).
        columns: Column specification override. Defaults to all numeric
            columns via ``_default_columns = "numeric"``.

    Examples:
        Round to 2 decimal places:

        >>> import sqlearn as sq
        >>> pipe = sq.Pipeline([sq.Round(decimals=2)])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # ROUND(price, 2) AS price

        Round to integer (default):

        >>> pipe = sq.Pipeline([sq.Round()])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # ROUND(price, 0) AS price

    See Also:
        :class:`Clip`: Clip values to a range.
        :class:`Abs`: Absolute value.
    """

    _default_columns: str = "numeric"  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "static"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        decimals: int = 0,
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        """Initialize Round transform.

        Args:
            decimals: Number of decimal places. Default ``0``.
            columns: Column specification override.
        """
        super().__init__(columns=columns)
        self.decimals = decimals

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        """Generate inline ROUND expressions for each target column.

        Args:
            columns: Target columns to transform.
            exprs: Current expression dict for ALL columns from prior steps.

        Returns:
            Dict of modified column expressions with ROUND applied.
        """
        return {
            col: exp.Round(
                this=exprs[col],
                decimals=exp.Literal.number(self.decimals),  # pyright: ignore[reportUnknownMemberType]
            )
            for col in columns
        }


class Reciprocal(Transformer):
    """Reciprocal (inverse) transform with safe zero handling.

    Compiles to inline SQL: ``1.0 / NULLIF(col, 0)``.

    ``NULLIF`` returns NULL when the column value is zero, avoiding
    division-by-zero errors.

    Generated SQL::

        SELECT 1.0 / NULLIF(price, 0) AS price FROM __input__

    Args:
        columns: Column specification override. Defaults to all numeric
            columns via ``_default_columns = "numeric"``.

    Examples:
        Reciprocal of all numeric columns:

        >>> import sqlearn as sq
        >>> pipe = sq.Pipeline([sq.Reciprocal()])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # 1.0 / NULLIF(price, 0) AS price

        Compose with Imputer (impute NULLs, then take reciprocal):

        >>> pipe = sq.Pipeline([sq.Imputer(), sq.Reciprocal()])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # 1.0 / NULLIF(COALESCE(price, 42.0), 0) AS price

    See Also:
        :class:`Log`: Logarithmic transform.
        :class:`Power`: Raise to arbitrary power.
    """

    _default_columns: str = "numeric"  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "static"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        """Initialize Reciprocal transform.

        Args:
            columns: Column specification override.
        """
        super().__init__(columns=columns)

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        """Generate inline reciprocal expressions for each target column.

        Args:
            columns: Target columns to transform.
            exprs: Current expression dict for ALL columns from prior steps.

        Returns:
            Dict of modified column expressions with 1/NULLIF(col, 0).
        """
        return {
            col: exp.Div(
                this=exp.Literal.number(1.0),  # pyright: ignore[reportUnknownMemberType]
                expression=exp.Nullif(
                    this=exprs[col],
                    expression=exp.Literal.number(0),  # pyright: ignore[reportUnknownMemberType]
                ),
            )
            for col in columns
        }
