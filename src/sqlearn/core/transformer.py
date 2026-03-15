"""Transformer base class for all sqlearn transformers."""

from __future__ import annotations

import copy
import inspect
import os
import threading
import warnings
from typing import TYPE_CHECKING, Any

from sqlearn.core.errors import NotFittedError

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

    # --- Column resolution ---

    def _resolve_columns_spec(self) -> str | list[str] | ColumnSelector | None:
        """Return the effective column spec (user override or class default).

        Returns self.columns if user passed columns=, else _default_columns.
        Actual resolution against schema happens at fit time via
        resolve_columns().

        Returns:
            Column specification to resolve, or None.
        """
        if self.columns is not None:
            return self.columns
        return self._default_columns

    # --- Classification ---

    def _classify(self) -> str:
        """Classify this transformer as 'static' or 'dynamic'.

        Tier 1: If _classification is set, trust it.
        Tier 3: Check if discover() or discover_sets() are overridden.

        Safety rule: if in doubt, classify as dynamic.

        Returns:
            ``'static'`` or ``'dynamic'``.
        """
        # Tier 1: explicit declaration
        if self._classification is not None:
            return self._classification

        # Tier 3: auto-detect by checking method overrides
        has_discover = type(self).discover is not Transformer.discover
        has_discover_sets = type(self).discover_sets is not Transformer.discover_sets

        if has_discover or has_discover_sets:
            return "dynamic"
        return "static"

    # --- Display ---

    def __repr__(self) -> str:
        """Return sklearn-style repr showing non-default parameters.

        Returns:
            String like ``ClassName(param=value, ...)``.
        """
        sig = inspect.signature(type(self).__init__)
        parts: list[str] = []
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            value = getattr(self, name)
            if value != param.default:
                parts.append(f"{name}={value!r}")
        return f"{type(self).__name__}({', '.join(parts)})"

    def _repr_html_(self) -> str:
        """Return HTML representation for Jupyter notebooks.

        Returns:
            HTML string with class name, parameters, and fitted status.
        """
        fitted_str = "fitted" if self._fitted else "not fitted"
        params = self.get_params()
        params_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"<div><strong>{type(self).__name__}</strong>({params_str}) [{fitted_str}]</div>"

    def get_feature_names_out(self) -> list[str]:
        """Return output feature names after fitting.

        Returns:
            List of column names from the output schema.

        Raises:
            NotFittedError: If transformer is not fitted.
        """
        if not self._fitted:
            msg = f"{type(self).__name__} is not fitted"
            raise NotFittedError(msg)
        if self.output_schema_ is None:
            msg = "output_schema_ is not set"
            raise NotFittedError(msg)
        return list(self.output_schema_.columns.keys())

    # --- sklearn introspection ---

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return __init__ parameters as dict. sklearn-compatible.

        Introspects the subclass __init__ signature. Parameters are
        retrieved via getattr, matching sklearn convention.

        Args:
            deep: If True, returns params for nested transformers
                using '__' separator. Not used until Pipeline lands.

        Returns:
            Dict of parameter names to current values.
        """
        sig = inspect.signature(type(self).__init__)
        params: dict[str, Any] = {}
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            params[name] = getattr(self, name)
        return params

    def set_params(self, **params: object) -> Transformer:
        """Set parameters. Returns self. sklearn-compatible.

        Args:
            **params: Parameter names and values to set.

        Returns:
            self (for method chaining).

        Raises:
            ValueError: If any parameter name is not a valid __init__ param.
        """
        valid_params = self.get_params()
        for key, value in params.items():
            if key not in valid_params:
                msg = f"Invalid parameter {key!r} for {type(self).__name__}"
                raise ValueError(msg)
            setattr(self, key, value)
        return self

    # --- Thread safety ---

    def _check_thread(self) -> None:
        """Guard against cross-thread/cross-process access.

        Stores _owner_thread and _owner_pid on first call. Raises
        RuntimeError on subsequent calls from different thread/process.

        Raises:
            RuntimeError: If accessed from different thread or process.
        """
        current_thread = threading.current_thread().ident
        current_pid = os.getpid()

        if self._owner_pid is None:
            self._owner_pid = current_pid
            self._owner_thread = current_thread
        elif self._owner_pid != current_pid:
            msg = (
                f"{type(self).__name__} accessed from a different process "
                f"(original pid={self._owner_pid}, current pid={current_pid}). "
                "DuckDB connections cannot be shared across processes."
            )
            raise RuntimeError(msg)
        elif self._owner_thread != current_thread:
            msg = (
                f"{type(self).__name__} accessed from a different thread. "
                "Pipelines are not thread-safe. Use .clone() to create "
                "a thread-safe copy with the same fitted parameters."
            )
            raise RuntimeError(msg)

    # --- Copying ---

    def clone(self) -> Transformer:
        """Create independent copy. Thread-safe (new connection).

        Deep copies params_, sets_, columns_. Resets thread ownership.
        Used by sq.Search for parallel training.

        Returns:
            New Transformer of the same type with same params and
            fitted state, but independent thread ownership.
        """
        params = self.get_params()
        new = type(self)(**params)
        new._fitted = self._fitted
        new.params_ = copy.deepcopy(self.params_)
        new.sets_ = copy.deepcopy(self.sets_)
        new.columns_ = copy.deepcopy(self.columns_)
        new.input_schema_ = self.input_schema_
        new.output_schema_ = self.output_schema_
        new._y_column = self._y_column
        new._owner_thread = None
        new._owner_pid = None
        new._connection = None
        return new

    def copy(self) -> Transformer:
        """Deep copy via copy.deepcopy().

        NOT thread-safe. Use clone() for cross-thread independence.

        Returns:
            Deep copy of this transformer.
        """
        return copy.deepcopy(self)

    # --- Serialization ---

    def __getstate__(self) -> dict[str, Any]:
        """Null out DuckDB connection before pickling.

        Returns:
            Instance state dict with _connection set to None.
        """
        state = self.__dict__.copy()
        state["_connection"] = None
        return state

    # --- Expression composition (internal) ---

    def _apply_expressions(
        self,
        exprs: dict[str, exp.Expression],
    ) -> dict[str, exp.Expression]:
        """Base class wrapper around expressions().

        Called by the compiler, not by users. Handles auto-passthrough
        of unmodified columns, detects undeclared new columns, and
        filters output to match output_schema().

        Args:
            exprs: Current expression dict for all columns.

        Returns:
            Expression dict after applying this transformer.

        Raises:
            NotFittedError: If columns_ is not set (not fitted).
        """
        if self.columns_ is None:
            msg = "columns_ not set — call fit() first"
            raise NotFittedError(msg)

        modified = self.expressions(self.columns_, exprs)
        result = dict(exprs)  # passthrough all input columns
        new_cols = set(modified.keys()) - set(exprs.keys())
        result.update(modified)  # overlay modifications and additions

        # Filter to output schema columns
        if self.input_schema_ is not None:
            output_cols = set(self.output_schema(self.input_schema_).columns.keys())
            undeclared = new_cols - output_cols
            if undeclared:
                warnings.warn(
                    f"{type(self).__name__}.expressions() created columns "
                    f"{undeclared} but output_schema() doesn't declare them. "
                    "Override output_schema() to include new columns.",
                    UserWarning,
                    stacklevel=2,
                )
            result = {k: v for k, v in result.items() if k in output_cols}

        return result

    # --- Operators ---

    def __add__(self, other: object) -> Any:
        """Sequential composition: a + b -> Pipeline([a, b]).

        Returns a new Pipeline containing both transformers.
        Flattens nested Pipeline operands.

        Args:
            other: Transformer or Pipeline to compose after self.

        Returns:
            A new Pipeline, or NotImplemented if type is unsupported.
        """
        from sqlearn.core.pipeline import (
            Pipeline,
            _auto_name,  # pyright: ignore[reportPrivateUsage]
        )

        if isinstance(other, Pipeline):
            name = _auto_name(other.steps, self)
            return Pipeline([(name, self), *other.steps])
        if isinstance(other, Transformer):
            return Pipeline([self, other])
        return NotImplemented

    def __iadd__(self, other: object) -> Any:
        """Incremental composition: pipe += step -> NEW Pipeline.

        Non-mutating — follows Python numeric convention.
        Returns a new Pipeline, does not modify self.

        Args:
            other: Transformer to append.

        Returns:
            A new Pipeline.
        """
        return self.__add__(other)

    # --- Stubs (implemented when Pipeline/Compiler/Backend land) ---

    def fit(
        self,
        data: Any,
        y: str | None = None,
        *,
        backend: Any = None,
    ) -> Transformer:
        """Learn parameters from data.

        Calls discover() and discover_sets() internally. Resolves
        column specifications against the data schema.

        Args:
            data: Input data (file path, table name, or DataFrame).
            y: Target column name, or None.
            backend: Backend override. Default uses DuckDB.

        Returns:
            self (for method chaining).

        Raises:
            NotImplementedError: Until Pipeline (issue #7) is implemented.
        """
        raise NotImplementedError

    def transform(
        self,
        data: Any,
        *,
        out: str = "numpy",
        backend: Any = None,
        batch_size: int | None = None,
        dtype: Any = None,
        exclude_target: bool = True,
    ) -> Any:
        """Apply transformation to data.

        Calls expressions() or query() internally. Compiles to SQL
        and executes via the backend.

        Args:
            data: Input data (file path, table name, or DataFrame).
            out: Output format (``'numpy'``, ``'pandas'``, ``'polars'``,
                ``'arrow'``).
            backend: Backend override.
            batch_size: Process data in batches of this size.
            dtype: NumPy dtype for output array.
            exclude_target: Exclude target column(s) from output.

        Returns:
            TransformResult (numpy-compatible).

        Raises:
            NotImplementedError: Until Pipeline (issue #7) is implemented.
        """
        raise NotImplementedError

    def fit_transform(
        self,
        data: Any,
        y: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Convenience: fit then transform.

        Args:
            data: Input data.
            y: Target column name.
            **kwargs: Passed to transform().

        Returns:
            TransformResult (numpy-compatible).

        Raises:
            NotImplementedError: Until Pipeline (issue #7) is implemented.
        """
        raise NotImplementedError

    def to_sql(
        self,
        *,
        dialect: str = "duckdb",
        table: str = "__input__",
    ) -> str:
        """Compile to SQL string without executing.

        Args:
            dialect: SQL dialect for output (``'duckdb'``, ``'postgres'``,
                ``'snowflake'``, etc.).
            table: Input table name placeholder.

        Returns:
            SQL query string.

        Raises:
            NotImplementedError: Until Compiler (issue #6) is implemented.
        """
        raise NotImplementedError

    def freeze(self) -> Any:
        """Return a FrozenPipeline: immutable, pre-compiled, deployment-ready.

        Returns:
            FrozenPipeline instance.

        Raises:
            NotImplementedError: Until FrozenPipeline (Milestone 7).
        """
        raise NotImplementedError
