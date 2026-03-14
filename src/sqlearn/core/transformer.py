"""Transformer base class for all sqlearn transformers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import sqlglot.expressions as exp

    from sqlearn.core.schema import ColumnSelector, Schema


class Transformer:
    """Base class for all sqlearn transformers.

    Subclasses override discover(), expressions(), and optionally
    discover_sets(), query(), and output_schema() to define behavior.
    """

    # --- Class attributes (set by subclasses) ---
    _default_columns: str | None = None
    _classification: str | None = None

    def __init__(
        self,
        *,
        columns: str | list[str] | ColumnSelector | None = None,
    ) -> None:
        """Initialize transformer.

        Args:
            columns: Column specification override. If provided, takes
                precedence over _default_columns. Accepts column names,
                type strings, or ColumnSelector objects. Resolved against
                schema at fit time.
        """
        # Init params (stored as-is for get_params compatibility)
        self.columns = columns

        # Internal state
        self._fitted: bool = False
        self._owner_thread: int | None = None
        self._owner_pid: int | None = None
        self._connection: Any = None  # DuckDB connection, lazy

        # Fitted attributes (set by Pipeline.fit)
        self.params_: dict[str, Any] | None = None
        self.sets_: dict[str, list[dict[str, Any]]] | None = None
        self.columns_: list[str] | None = None
        self.input_schema_: Schema | None = None
        self.output_schema_: Schema | None = None
        self._y_column: str | None = None

    # --- Properties ---

    @property
    def is_fitted(self) -> bool:
        """Whether this transformer has been fitted.

        Returns:
            True if fit() has been called successfully.
        """
        return self._fitted

    def __sklearn_is_fitted__(self) -> bool:
        """Sklearn compatibility: enables sklearn's check_is_fitted().

        Returns:
            True if this transformer has been fitted.
        """
        return self._fitted

    # --- Subclass overrides ---

    def discover(
        self,
        columns: list[str],
        schema: Schema,
        y_column: str | None = None,
    ) -> dict[str, exp.Expression]:
        """Learn scalar statistics from data via SQL aggregates.

        Override to return {param_name: sqlglot_aggregate} mappings.
        Results are executed as SQL and stored in self.params_.

        Default returns {} (static -- no learning).

        Param naming convention: '{col}__{stat}' (e.g. 'price__mean').
        Must return sqlglot AST nodes, never raw strings or Python values.

        Args:
            columns: Target columns this transformer operates on.
            schema: Current table schema.
            y_column: Target column name, if provided to fit().

        Returns:
            Mapping of parameter names to sqlglot aggregate expressions.
        """
        return {}

    def discover_sets(
        self,
        columns: list[str],
        schema: Schema,
        y_column: str | None = None,
    ) -> dict[str, exp.Expression]:
        """Learn set-valued (multi-row) data via SQL queries.

        Override to return {param_name: sqlglot_select_query} mappings.
        Results are executed and stored in self.sets_ as lists of dicts.

        Default returns {} (no set learning).

        Args:
            columns: Target columns this transformer operates on.
            schema: Current table schema.
            y_column: Target column name, if provided to fit().

        Returns:
            Mapping of parameter names to sqlglot SELECT queries.
        """
        return {}

    def expressions(
        self,
        columns: list[str],
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        """Generate inline SQL column expressions.

        Subclasses must override this or query(). Returns ONLY modified/new
        columns. Unmentioned columns pass through automatically via
        _apply_expressions().

        Must return sqlglot AST nodes, never raw strings.

        Args:
            columns: Target columns this transformer operates on.
            exprs: Current expression dict for ALL columns.

        Returns:
            Dict of modified/new column expressions.

        Raises:
            NotImplementedError: If not overridden by subclass.
        """
        raise NotImplementedError

    def query(
        self,
        input_query: exp.Expression,
    ) -> exp.Expression | None:
        """Generate a full query wrapping input.

        Alternative to expressions() for transforms needing query-level
        control (window functions, joins, CTEs). Returns None to fall
        back to expressions().

        Default returns None.

        Args:
            input_query: The input query to wrap.

        Returns:
            A new sqlglot SELECT wrapping the input, or None.
        """
        return None

    def output_schema(self, schema: Schema) -> Schema:
        """Declare output schema after this step.

        Override when adding, removing, renaming, or retyping columns.
        Default returns input schema unchanged.

        Args:
            schema: Input schema.

        Returns:
            Output schema after this transformation step.
        """
        return schema
