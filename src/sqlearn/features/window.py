"""Window function transformers -- Lag, Lead, RollingMean, RollingStd, Rank, RowNumber.

All window transforms use ``query()`` (not ``expressions()``) because SQL window
functions require their own SELECT level and cannot be nested inside column
expressions.

Each transformer wraps the input query in a new SELECT that adds window columns
alongside all original columns via ``SELECT *, <window_col> FROM (...)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlglot.expressions as exp

from sqlearn.core.transformer import Transformer

if TYPE_CHECKING:
    from sqlearn.core.schema import ColumnSelector, Schema

_VALID_RANK_METHODS = ("rank", "dense_rank", "row_number")
_MIN_ROLLING_WINDOW = 2


def _build_order(
    order_by: str | list[str],
) -> exp.Order:
    """Build an ORDER BY clause from column name(s).

    Args:
        order_by: Single column name or list of column names.

    Returns:
        sqlglot Order expression.
    """
    cols = [order_by] if isinstance(order_by, str) else order_by
    return exp.Order(
        expressions=[exp.Ordered(this=exp.Column(this=exp.to_identifier(c))) for c in cols],
    )


def _build_partition(
    partition_by: str | list[str] | None,
) -> list[exp.Expression]:
    """Build PARTITION BY column list.

    Args:
        partition_by: Column name(s) to partition by, or None.

    Returns:
        List of sqlglot Column expressions (empty if None).
    """
    if partition_by is None:
        return []
    cols = [partition_by] if isinstance(partition_by, str) else partition_by
    return [exp.Column(this=exp.to_identifier(c)) for c in cols]


def _wrap_input(input_query: exp.Expression) -> exp.Subquery:
    """Wrap an input query as a subquery aliased ``__input__``.

    Args:
        input_query: The query to wrap.

    Returns:
        Subquery expression.
    """
    return exp.Subquery(
        this=input_query,
        alias=exp.TableAlias(this=exp.to_identifier("__input__")),
    )


class Lag(Transformer):
    """Access a previous row's value via SQL ``LAG()`` window function.

    Creates new columns ``{col}_lag{N}`` for each target column, containing
    the value from ``N`` rows before the current row (ordered by
    ``order_by``). Uses ``query()`` because window functions need their
    own SELECT level.

    This is a **static** transformer -- no statistics are learned during
    ``fit()``.

    Generated SQL (periods=1, order_by='ts')::

        SELECT
          *, LAG(value, 1) OVER (ORDER BY ts) AS value_lag1
        FROM (__input__) AS __input__

    Generated SQL (periods=2, order_by='ts', partition_by='category')::

        SELECT
          *, LAG(value, 2) OVER (PARTITION BY category ORDER BY ts) AS value_lag2
        FROM (__input__) AS __input__

    Args:
        periods: Number of rows to lag (offset). Must be a positive integer.
            Defaults to 1.
        order_by: Column(s) defining row order. Required.
        partition_by: Column(s) to partition the window. Optional.
        columns: Target columns to lag. Defaults to numeric columns.

    Raises:
        ValueError: If ``periods`` is not positive or ``order_by`` is empty.

    Examples:
        Create lag features for time series:

        >>> import sqlearn as sq
        >>> pipe = sq.Pipeline([sq.Lag(order_by="ts")])
        >>> pipe.fit("data.parquet")
        >>> pipe.to_sql()
        ... # SELECT *, LAG(value, 1) OVER (ORDER BY ts) AS value_lag1

        Lag with partitioning:

        >>> lag = sq.Lag(periods=3, order_by="ts", partition_by="category")

    See Also:
        :class:`Lead`: Access future row values.
        :class:`RollingMean`: Rolling average over a window.
    """

    _default_columns: str = "numeric"  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "static"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        periods: int = 1,
        order_by: str | list[str],
        partition_by: str | list[str] | None = None,
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        super().__init__(columns=columns)
        if not isinstance(periods, int) or periods < 1:
            msg = f"'periods' must be a positive integer, got {periods!r}."
            raise ValueError(msg)
        order_list = [order_by] if isinstance(order_by, str) else order_by
        if len(order_list) == 0:
            msg = "'order_by' must be a non-empty string or list of strings."
            raise ValueError(msg)
        self.periods = periods
        self.order_by = order_by
        self.partition_by = partition_by

    def query(self, input_query: exp.Expression) -> exp.Expression:
        """Generate a SELECT wrapping input with LAG window columns.

        Builds ``SELECT *, LAG(col, periods) OVER (...) AS col_lagN``
        for each target column.

        Args:
            input_query: The input query to wrap.

        Returns:
            A new sqlglot SELECT with lag columns added.
        """
        target_cols = self.columns_ or []
        window_exprs: list[exp.Expression] = []
        for col in target_cols:
            lag_fn = exp.Anonymous(
                this="LAG",
                expressions=[
                    exp.Column(this=exp.to_identifier(col)),
                    exp.Literal.number(self.periods),  # pyright: ignore[reportUnknownMemberType]
                ],
            )
            window = exp.Window(
                this=lag_fn,
                partition_by=_build_partition(self.partition_by),
                order=_build_order(self.order_by),
            )
            alias = f"{col}_lag{self.periods}"
            window_exprs.append(
                window.as_(alias)  # pyright: ignore[reportUnknownMemberType]
            )

        return exp.Select(
            expressions=[exp.Star(), *window_exprs],
        ).from_(_wrap_input(input_query))  # pyright: ignore[reportUnknownMemberType]

    def output_schema(self, schema: Schema) -> Schema:
        """Add lag columns to the schema.

        For each target column, adds ``{col}_lag{N}`` with type ``DOUBLE``.

        Args:
            schema: Input schema.

        Returns:
            New schema with lag columns appended.
        """
        target_cols = self.columns_ or []
        new_cols: dict[str, str] = {}
        for col in target_cols:
            alias = f"{col}_lag{self.periods}"
            new_cols[alias] = schema.columns.get(col, "DOUBLE")
        return schema.add(new_cols)

    def expressions(
        self,
        columns: list[str],  # noqa: ARG002
        exprs: dict[str, exp.Expression],  # noqa: ARG002
    ) -> dict[str, exp.Expression]:
        """Passthrough no-op since Lag uses query-level SQL instead.

        Args:
            columns: Target columns (unused).
            exprs: Current expression dict (unused).

        Returns:
            Empty dict.
        """
        return {}


class Lead(Transformer):
    """Access a future row's value via SQL ``LEAD()`` window function.

    Creates new columns ``{col}_lead{N}`` for each target column, containing
    the value from ``N`` rows after the current row (ordered by
    ``order_by``). Uses ``query()`` because window functions need their
    own SELECT level.

    This is a **static** transformer -- no statistics are learned during
    ``fit()``.

    Generated SQL (periods=1, order_by='ts')::

        SELECT
          *, LEAD(value, 1) OVER (ORDER BY ts) AS value_lead1
        FROM (__input__) AS __input__

    Args:
        periods: Number of rows to lead (offset). Must be a positive integer.
            Defaults to 1.
        order_by: Column(s) defining row order. Required.
        partition_by: Column(s) to partition the window. Optional.
        columns: Target columns to lead. Defaults to numeric columns.

    Raises:
        ValueError: If ``periods`` is not positive or ``order_by`` is empty.

    Examples:
        Create lead features for time series:

        >>> import sqlearn as sq
        >>> lead = sq.Lead(order_by="ts", columns=["value"])

        Lead with partitioning:

        >>> lead = sq.Lead(periods=2, order_by="ts", partition_by="group")

    See Also:
        :class:`Lag`: Access previous row values.
        :class:`RollingMean`: Rolling average over a window.
    """

    _default_columns: str = "numeric"  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "static"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        periods: int = 1,
        order_by: str | list[str],
        partition_by: str | list[str] | None = None,
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        super().__init__(columns=columns)
        if not isinstance(periods, int) or periods < 1:
            msg = f"'periods' must be a positive integer, got {periods!r}."
            raise ValueError(msg)
        order_list = [order_by] if isinstance(order_by, str) else order_by
        if len(order_list) == 0:
            msg = "'order_by' must be a non-empty string or list of strings."
            raise ValueError(msg)
        self.periods = periods
        self.order_by = order_by
        self.partition_by = partition_by

    def query(self, input_query: exp.Expression) -> exp.Expression:
        """Generate a SELECT wrapping input with LEAD window columns.

        Builds ``SELECT *, LEAD(col, periods) OVER (...) AS col_leadN``
        for each target column.

        Args:
            input_query: The input query to wrap.

        Returns:
            A new sqlglot SELECT with lead columns added.
        """
        target_cols = self.columns_ or []
        window_exprs: list[exp.Expression] = []
        for col in target_cols:
            lead_fn = exp.Anonymous(
                this="LEAD",
                expressions=[
                    exp.Column(this=exp.to_identifier(col)),
                    exp.Literal.number(self.periods),  # pyright: ignore[reportUnknownMemberType]
                ],
            )
            window = exp.Window(
                this=lead_fn,
                partition_by=_build_partition(self.partition_by),
                order=_build_order(self.order_by),
            )
            alias = f"{col}_lead{self.periods}"
            window_exprs.append(
                window.as_(alias)  # pyright: ignore[reportUnknownMemberType]
            )

        return exp.Select(
            expressions=[exp.Star(), *window_exprs],
        ).from_(_wrap_input(input_query))  # pyright: ignore[reportUnknownMemberType]

    def output_schema(self, schema: Schema) -> Schema:
        """Add lead columns to the schema.

        For each target column, adds ``{col}_lead{N}`` with the same type.

        Args:
            schema: Input schema.

        Returns:
            New schema with lead columns appended.
        """
        target_cols = self.columns_ or []
        new_cols: dict[str, str] = {}
        for col in target_cols:
            alias = f"{col}_lead{self.periods}"
            new_cols[alias] = schema.columns.get(col, "DOUBLE")
        return schema.add(new_cols)

    def expressions(
        self,
        columns: list[str],  # noqa: ARG002
        exprs: dict[str, exp.Expression],  # noqa: ARG002
    ) -> dict[str, exp.Expression]:
        """Passthrough no-op since Lead uses query-level SQL instead.

        Args:
            columns: Target columns (unused).
            exprs: Current expression dict (unused).

        Returns:
            Empty dict.
        """
        return {}


class RollingMean(Transformer):
    """Rolling average via SQL ``AVG() OVER (ROWS ...)`` window function.

    Creates new columns ``{col}_rmean{W}`` for each target column, containing
    the rolling mean of the preceding ``W`` rows (including the current row).
    Uses ``query()`` because window functions need their own SELECT level.

    This is a **static** transformer -- no statistics are learned during
    ``fit()``.

    Generated SQL (window=3, order_by='ts')::

        SELECT
          *, AVG(value) OVER (
            ORDER BY ts
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
          ) AS value_rmean3
        FROM (__input__) AS __input__

    The window frame is ``ROWS BETWEEN (window-1) PRECEDING AND CURRENT ROW``,
    so the first ``window-1`` rows will compute the mean over fewer values
    (partial window).

    Args:
        window: Number of rows in the rolling window (including current row).
            Must be a positive integer >= 2.
        order_by: Column(s) defining row order. Required.
        partition_by: Column(s) to partition the window. Optional.
        columns: Target columns. Defaults to numeric columns.

    Raises:
        ValueError: If ``window`` < 2 or ``order_by`` is empty.

    Examples:
        3-period rolling mean:

        >>> import sqlearn as sq
        >>> pipe = sq.Pipeline([sq.RollingMean(window=3, order_by="ts")])
        >>> pipe.fit("data.parquet")

        Rolling mean with partitioning:

        >>> rm = sq.RollingMean(window=5, order_by="ts", partition_by="group")

    See Also:
        :class:`RollingStd`: Rolling standard deviation.
        :class:`Lag`: Access previous row values.
    """

    _default_columns: str = "numeric"  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "static"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        window: int,
        order_by: str | list[str],
        partition_by: str | list[str] | None = None,
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        super().__init__(columns=columns)
        if not isinstance(window, int) or window < _MIN_ROLLING_WINDOW:
            msg = f"'window' must be an integer >= {_MIN_ROLLING_WINDOW}, got {window!r}."
            raise ValueError(msg)
        order_list = [order_by] if isinstance(order_by, str) else order_by
        if len(order_list) == 0:
            msg = "'order_by' must be a non-empty string or list of strings."
            raise ValueError(msg)
        self.window = window
        self.order_by = order_by
        self.partition_by = partition_by

    def query(self, input_query: exp.Expression) -> exp.Expression:
        """Generate a SELECT wrapping input with rolling mean columns.

        Builds ``SELECT *, AVG(col) OVER (ORDER BY ... ROWS BETWEEN
        (window-1) PRECEDING AND CURRENT ROW) AS col_rmeanW``.

        Args:
            input_query: The input query to wrap.

        Returns:
            A new sqlglot SELECT with rolling mean columns added.
        """
        target_cols = self.columns_ or []
        window_exprs: list[exp.Expression] = []
        for col in target_cols:
            avg_fn = exp.Avg(this=exp.Column(this=exp.to_identifier(col)))
            spec = exp.WindowSpec(
                kind="ROWS",
                start=exp.Literal.number(self.window - 1),  # pyright: ignore[reportUnknownMemberType]
                start_side="PRECEDING",
                end="CURRENT ROW",
            )
            window = exp.Window(
                this=avg_fn,
                partition_by=_build_partition(self.partition_by),
                order=_build_order(self.order_by),
                spec=spec,
            )
            alias = f"{col}_rmean{self.window}"
            window_exprs.append(
                window.as_(alias)  # pyright: ignore[reportUnknownMemberType]
            )

        return exp.Select(
            expressions=[exp.Star(), *window_exprs],
        ).from_(_wrap_input(input_query))  # pyright: ignore[reportUnknownMemberType]

    def output_schema(self, schema: Schema) -> Schema:
        """Add rolling mean columns to the schema.

        For each target column, adds ``{col}_rmean{W}`` with type ``DOUBLE``.

        Args:
            schema: Input schema.

        Returns:
            New schema with rolling mean columns appended.
        """
        target_cols = self.columns_ or []
        new_cols = {f"{col}_rmean{self.window}": "DOUBLE" for col in target_cols}
        return schema.add(new_cols)

    def expressions(
        self,
        columns: list[str],  # noqa: ARG002
        exprs: dict[str, exp.Expression],  # noqa: ARG002
    ) -> dict[str, exp.Expression]:
        """Passthrough no-op since RollingMean uses query-level SQL instead.

        Args:
            columns: Target columns (unused).
            exprs: Current expression dict (unused).

        Returns:
            Empty dict.
        """
        return {}


class RollingStd(Transformer):
    """Rolling standard deviation via SQL ``STDDEV_POP() OVER (ROWS ...)`` window function.

    Creates new columns ``{col}_rstd{W}`` for each target column, containing
    the rolling population standard deviation of the preceding ``W`` rows
    (including the current row). Uses ``query()`` because window functions
    need their own SELECT level.

    This is a **static** transformer -- no statistics are learned during
    ``fit()``.

    Generated SQL (window=3, order_by='ts')::

        SELECT
          *, STDDEV_POP(value) OVER (
            ORDER BY ts
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
          ) AS value_rstd3
        FROM (__input__) AS __input__

    Args:
        window: Number of rows in the rolling window (including current row).
            Must be a positive integer >= 2.
        order_by: Column(s) defining row order. Required.
        partition_by: Column(s) to partition the window. Optional.
        columns: Target columns. Defaults to numeric columns.

    Raises:
        ValueError: If ``window`` < 2 or ``order_by`` is empty.

    Examples:
        3-period rolling standard deviation:

        >>> import sqlearn as sq
        >>> pipe = sq.Pipeline([sq.RollingStd(window=3, order_by="ts")])
        >>> pipe.fit("data.parquet")

        Rolling std with partitioning:

        >>> rs = sq.RollingStd(window=5, order_by="ts", partition_by="group")

    See Also:
        :class:`RollingMean`: Rolling average.
        :class:`Lag`: Access previous row values.
    """

    _default_columns: str = "numeric"  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "static"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        window: int,
        order_by: str | list[str],
        partition_by: str | list[str] | None = None,
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        super().__init__(columns=columns)
        if not isinstance(window, int) or window < _MIN_ROLLING_WINDOW:
            msg = f"'window' must be an integer >= {_MIN_ROLLING_WINDOW}, got {window!r}."
            raise ValueError(msg)
        order_list = [order_by] if isinstance(order_by, str) else order_by
        if len(order_list) == 0:
            msg = "'order_by' must be a non-empty string or list of strings."
            raise ValueError(msg)
        self.window = window
        self.order_by = order_by
        self.partition_by = partition_by

    def query(self, input_query: exp.Expression) -> exp.Expression:
        """Generate a SELECT wrapping input with rolling stddev columns.

        Builds ``SELECT *, STDDEV_POP(col) OVER (ORDER BY ... ROWS BETWEEN
        (window-1) PRECEDING AND CURRENT ROW) AS col_rstdW``.

        Args:
            input_query: The input query to wrap.

        Returns:
            A new sqlglot SELECT with rolling stddev columns added.
        """
        target_cols = self.columns_ or []
        window_exprs: list[exp.Expression] = []
        for col in target_cols:
            stddev_fn = exp.Anonymous(
                this="STDDEV_POP",
                expressions=[exp.Column(this=exp.to_identifier(col))],
            )
            spec = exp.WindowSpec(
                kind="ROWS",
                start=exp.Literal.number(self.window - 1),  # pyright: ignore[reportUnknownMemberType]
                start_side="PRECEDING",
                end="CURRENT ROW",
            )
            window = exp.Window(
                this=stddev_fn,
                partition_by=_build_partition(self.partition_by),
                order=_build_order(self.order_by),
                spec=spec,
            )
            alias = f"{col}_rstd{self.window}"
            window_exprs.append(
                window.as_(alias)  # pyright: ignore[reportUnknownMemberType]
            )

        return exp.Select(
            expressions=[exp.Star(), *window_exprs],
        ).from_(_wrap_input(input_query))  # pyright: ignore[reportUnknownMemberType]

    def output_schema(self, schema: Schema) -> Schema:
        """Add rolling stddev columns to the schema.

        For each target column, adds ``{col}_rstd{W}`` with type ``DOUBLE``.

        Args:
            schema: Input schema.

        Returns:
            New schema with rolling stddev columns appended.
        """
        target_cols = self.columns_ or []
        new_cols = {f"{col}_rstd{self.window}": "DOUBLE" for col in target_cols}
        return schema.add(new_cols)

    def expressions(
        self,
        columns: list[str],  # noqa: ARG002
        exprs: dict[str, exp.Expression],  # noqa: ARG002
    ) -> dict[str, exp.Expression]:
        """Passthrough no-op since RollingStd uses query-level SQL instead.

        Args:
            columns: Target columns (unused).
            exprs: Current expression dict (unused).

        Returns:
            Empty dict.
        """
        return {}


class Rank(Transformer):
    """Window rank via SQL ``RANK()``, ``DENSE_RANK()``, or ``ROW_NUMBER()``.

    Creates new columns ``{col}_rank`` for each target column (or a single
    ``rank`` column when no target columns are specified). Uses ``query()``
    because window functions need their own SELECT level.

    This is a **static** transformer -- no statistics are learned during
    ``fit()``.

    Generated SQL (method='rank', order_by='score')::

        SELECT
          *, RANK() OVER (ORDER BY score) AS score_rank
        FROM (__input__) AS __input__

    Generated SQL (method='dense_rank', partition_by='group')::

        SELECT
          *, DENSE_RANK() OVER (PARTITION BY group ORDER BY score) AS score_rank
        FROM (__input__) AS __input__

    Args:
        order_by: Column(s) defining rank order. Required.
        partition_by: Column(s) to partition the window. Optional.
        method: Ranking function: ``'rank'`` (default), ``'dense_rank'``,
            or ``'row_number'``.
        columns: Target columns to rank. If None, ranks by ``order_by``
            and creates a single ``rank`` column.

    Raises:
        ValueError: If ``method`` is invalid or ``order_by`` is empty.

    Examples:
        Rank by score:

        >>> import sqlearn as sq
        >>> rank = sq.Rank(order_by="score", columns=["score"])

        Dense rank with partitioning:

        >>> rank = sq.Rank(order_by="score", partition_by="group", method="dense_rank")

    See Also:
        :class:`RowNumber`: Simple row numbering.
        :class:`Lag`: Access previous row values.
    """

    _default_columns: str | None = None  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "static"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        order_by: str | list[str],
        partition_by: str | list[str] | None = None,
        method: str = "rank",
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        super().__init__(columns=columns)
        if method not in _VALID_RANK_METHODS:
            msg = f"Invalid method {method!r}. Must be one of {_VALID_RANK_METHODS!r}."
            raise ValueError(msg)
        order_list = [order_by] if isinstance(order_by, str) else order_by
        if len(order_list) == 0:
            msg = "'order_by' must be a non-empty string or list of strings."
            raise ValueError(msg)
        self.order_by = order_by
        self.partition_by = partition_by
        self.method = method
        # Track whether user explicitly provided columns for per-column vs
        # single-column mode decision in query() and output_schema().
        self._has_explicit_columns = columns is not None

    def _rank_fn_name(self) -> str:
        """Return the SQL function name for the chosen method.

        Returns:
            SQL function name string.
        """
        return self.method.upper()

    def _use_per_column_mode(self) -> bool:
        """Determine if Rank should create per-column rank columns.

        When the user explicitly passes ``columns``, each column gets its
        own ``{col}_rank`` output. When no columns are specified (relying on
        auto-resolution), a single ``rank`` column is created using
        ``order_by``.

        Returns:
            True for per-column mode, False for single rank column.
        """
        return self._has_explicit_columns

    def query(self, input_query: exp.Expression) -> exp.Expression:
        """Generate a SELECT wrapping input with rank columns.

        If columns were explicitly provided, creates ``{col}_rank`` for each.
        Otherwise, creates a single ``rank`` column ordered by ``order_by``.

        Args:
            input_query: The input query to wrap.

        Returns:
            A new sqlglot SELECT with rank columns added.
        """
        fn_name = self._rank_fn_name()
        target_cols = self.columns_ or []
        window_exprs: list[exp.Expression] = []

        if self._use_per_column_mode() and target_cols:
            for col in target_cols:
                rank_fn = exp.Anonymous(this=fn_name, expressions=[])
                order = exp.Order(
                    expressions=[exp.Ordered(this=exp.Column(this=exp.to_identifier(col)))],
                )
                window = exp.Window(
                    this=rank_fn,
                    partition_by=_build_partition(self.partition_by),
                    order=order,
                )
                alias = f"{col}_rank"
                window_exprs.append(
                    window.as_(alias)  # pyright: ignore[reportUnknownMemberType]
                )
        else:
            # No explicit columns -- rank by order_by, single "rank" column
            rank_fn = exp.Anonymous(this=fn_name, expressions=[])
            window = exp.Window(
                this=rank_fn,
                partition_by=_build_partition(self.partition_by),
                order=_build_order(self.order_by),
            )
            window_exprs.append(
                window.as_("rank")  # pyright: ignore[reportUnknownMemberType]
            )

        return exp.Select(
            expressions=[exp.Star(), *window_exprs],
        ).from_(_wrap_input(input_query))  # pyright: ignore[reportUnknownMemberType]

    def output_schema(self, schema: Schema) -> Schema:
        """Add rank columns to the schema.

        Adds ``{col}_rank`` for each target column (per-column mode), or
        ``rank`` (single-column mode when columns not explicitly provided).

        Args:
            schema: Input schema.

        Returns:
            New schema with rank columns appended.
        """
        target_cols = self.columns_ or []
        if self._use_per_column_mode() and target_cols:
            new_cols = {f"{col}_rank": "BIGINT" for col in target_cols}
        else:
            new_cols = {"rank": "BIGINT"}
        return schema.add(new_cols)

    def expressions(
        self,
        columns: list[str],  # noqa: ARG002
        exprs: dict[str, exp.Expression],  # noqa: ARG002
    ) -> dict[str, exp.Expression]:
        """Passthrough no-op since Rank uses query-level SQL instead.

        Args:
            columns: Target columns (unused).
            exprs: Current expression dict (unused).

        Returns:
            Empty dict.
        """
        return {}


class RowNumber(Transformer):
    """Row numbering via SQL ``ROW_NUMBER()`` window function.

    Creates a single ``row_number`` column. Uses ``query()`` because window
    functions need their own SELECT level.

    This is a **static** transformer -- no statistics are learned during
    ``fit()``.

    Generated SQL (order_by='ts')::

        SELECT
          *, ROW_NUMBER() OVER (ORDER BY ts) AS row_number
        FROM (__input__) AS __input__

    Generated SQL (order_by='ts', partition_by='group')::

        SELECT
          *, ROW_NUMBER() OVER (PARTITION BY group ORDER BY ts) AS row_number
        FROM (__input__) AS __input__

    Args:
        order_by: Column(s) defining row order. Required.
        partition_by: Column(s) to partition the window. Optional.

    Raises:
        ValueError: If ``order_by`` is empty.

    Examples:
        Add row numbers:

        >>> import sqlearn as sq
        >>> pipe = sq.Pipeline([sq.RowNumber(order_by="ts")])
        >>> pipe.fit("data.parquet")

        Row numbers within groups:

        >>> rn = sq.RowNumber(order_by="ts", partition_by="group")

    See Also:
        :class:`Rank`: Window rank functions.
        :class:`Lag`: Access previous row values.
    """

    _default_columns: None = None  # pyright: ignore[reportIncompatibleVariableOverride]
    _classification: str = "static"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        order_by: str | list[str],
        partition_by: str | list[str] | None = None,
    ) -> None:
        super().__init__(columns=None)
        order_list = [order_by] if isinstance(order_by, str) else order_by
        if len(order_list) == 0:
            msg = "'order_by' must be a non-empty string or list of strings."
            raise ValueError(msg)
        self.order_by = order_by
        self.partition_by = partition_by

    def query(self, input_query: exp.Expression) -> exp.Expression:
        """Generate a SELECT wrapping input with a row_number column.

        Builds ``SELECT *, ROW_NUMBER() OVER (...) AS row_number``.

        Args:
            input_query: The input query to wrap.

        Returns:
            A new sqlglot SELECT with row_number column added.
        """
        rn_fn = exp.Anonymous(this="ROW_NUMBER", expressions=[])
        window = exp.Window(
            this=rn_fn,
            partition_by=_build_partition(self.partition_by),
            order=_build_order(self.order_by),
        )
        return exp.Select(
            expressions=[
                exp.Star(),
                window.as_("row_number"),  # pyright: ignore[reportUnknownMemberType]
            ],
        ).from_(_wrap_input(input_query))  # pyright: ignore[reportUnknownMemberType]

    def output_schema(self, schema: Schema) -> Schema:
        """Add row_number column to the schema.

        Args:
            schema: Input schema.

        Returns:
            New schema with ``row_number`` column of type ``BIGINT``.
        """
        return schema.add({"row_number": "BIGINT"})

    def expressions(
        self,
        columns: list[str],  # noqa: ARG002
        exprs: dict[str, exp.Expression],  # noqa: ARG002
    ) -> dict[str, exp.Expression]:
        """Passthrough no-op since RowNumber uses query-level SQL instead.

        Args:
            columns: Target columns (unused).
            exprs: Current expression dict (unused).

        Returns:
            Empty dict.
        """
        return {}
