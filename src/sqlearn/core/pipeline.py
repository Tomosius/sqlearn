"""Pipeline orchestrator for sqlearn.

Compiles ML preprocessing steps into SQL via the three-phase compiler.
Pipeline is NOT a Transformer subclass — it's a thin orchestrator that
delegates to plan_fit, build_fit_queries, and compose_transform.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import sqlglot.expressions as exp

from sqlearn.core.backend import DuckDBBackend
from sqlearn.core.compiler import build_fit_queries, compose_transform, plan_fit
from sqlearn.core.errors import InvalidStepError, NotFittedError
from sqlearn.core.io import resolve_input
from sqlearn.core.transformer import Transformer

if TYPE_CHECKING:
    from sqlearn.core.schema import Schema

# Steps input type: bare list, tuple list, or dict.
_StepsInput = list[Transformer] | list[tuple[str, Transformer]] | dict[str, Transformer]


def _auto_name(
    existing: list[tuple[str, Transformer]],
    step: Transformer,  # noqa: ARG001
) -> str:
    """Generate next step_NN name based on existing steps.

    Scans existing step names for the ``step_NN`` pattern and returns
    the next sequential name, zero-padded to at least width 2.

    Args:
        existing: Current list of (name, transformer) pairs.
        step: The transformer to name (unused, reserved for future
            subclass-based naming).

    Returns:
        Name like ``'step_00'``, ``'step_01'``, etc.
    """
    max_idx = -1
    for name, _ in existing:
        if name.startswith("step_") and name[5:].isdigit():
            max_idx = max(max_idx, int(name[5:]))
    idx = max_idx + 1
    width = max(2, len(str(idx)))
    return f"step_{idx:0{width}d}"


def _normalize_steps(steps: _StepsInput) -> list[tuple[str, Transformer]]:
    """Normalize all input formats to list of (name, transformer) tuples.

    Args:
        steps: One of three formats: bare list, tuple list, or dict.

    Returns:
        Normalized list of (name, transformer) pairs.

    Raises:
        InvalidStepError: If steps are empty, contain non-Transformers,
            or have duplicate names.
    """
    raw: list[tuple[str, Any]]
    if isinstance(steps, dict):
        raw = list(steps.items())
    else:
        # steps is a list — check whether first element is a tuple
        step_list: list[Any] = cast("list[Any]", steps)
        if step_list and isinstance(step_list[0], tuple):
            raw = list(step_list)
        else:
            # Bare list — auto-name
            width = max(2, len(str(len(step_list) - 1))) if step_list else 2
            raw = [(f"step_{i:0{width}d}", s) for i, s in enumerate(step_list)]

    # Validation
    if not raw:
        msg = "Pipeline requires at least one step"
        raise InvalidStepError(msg)

    seen_names: set[str] = set()
    result: list[tuple[str, Transformer]] = []
    for name, obj in raw:
        if not isinstance(obj, Transformer):
            msg = f"Step {name!r} is not a Transformer: {type(obj)}"
            raise InvalidStepError(msg)
        if name in seen_names:
            msg = f"Duplicate step name: {name!r}"
            raise InvalidStepError(msg)
        seen_names.add(name)
        result.append((name, obj))

    return result


class Pipeline:
    """Compile ML preprocessing steps to a single SQL query.

    Pipeline is a thin orchestrator that delegates to the compiler for
    all SQL work. It accepts three input formats for steps:

    - Bare list: ``[Imputer(), StandardScaler()]`` — auto-named
    - Tuple list: ``[("impute", Imputer()), ("scale", StandardScaler())]``
    - Dict: ``{"impute": Imputer(), "scale": StandardScaler()}``

    Generated SQL (3-step pipeline)::

        SELECT
          (COALESCE(price, 3.0) - 3.0) / NULLIF(1.58, 0) AS price,
          (COALESCE(quantity, 35.0) - 32.5) / NULLIF(14.79, 0) AS quantity,
          CASE WHEN COALESCE(city, 'London') = 'London' THEN 1 ELSE 0 END AS city_london,
          CASE WHEN COALESCE(city, 'London') = 'Paris' THEN 1 ELSE 0 END AS city_paris
        FROM __input__

    Args:
        steps: Pipeline steps in any of the three formats above.
        backend: Backend config — str (file path), DuckDBBackend instance,
            or None (auto-create in-memory).

    Raises:
        InvalidStepError: If steps are empty, contain non-Transformers,
            or have duplicate names.

    Examples:
        Basic pipeline with auto-named steps:

        >>> import sqlearn as sq
        >>> pipe = sq.Pipeline(
        ...     [
        ...         sq.Imputer(),
        ...         sq.StandardScaler(),
        ...         sq.OneHotEncoder(),
        ...     ]
        ... )
        >>> pipe.fit("train.parquet", y="target")
        >>> X = pipe.transform("test.parquet")  # numpy array
        >>> sql = pipe.to_sql()  # valid DuckDB SQL

        Named steps for easier access:

        >>> pipe = sq.Pipeline(
        ...     [
        ...         ("impute", sq.Imputer()),
        ...         ("scale", sq.StandardScaler()),
        ...     ]
        ... )
        >>> pipe.fit("data.parquet")
        >>> pipe.named_steps["scale"]  # access by name

        Compose pipelines with ``+``:

        >>> pipe1 = sq.Pipeline([sq.Imputer()])
        >>> pipe2 = sq.Pipeline([sq.StandardScaler()])
        >>> combined = pipe1 + pipe2  # new pipeline with both steps

        Get output column names:

        >>> pipe = sq.Pipeline([sq.OneHotEncoder()])
        >>> pipe.fit("data.parquet")
        >>> pipe.get_feature_names_out()
        ['city_london', 'city_paris', 'city_tokyo']

    See Also:
        :class:`~sqlearn.core.transformer.Transformer`: Base class for steps.
        :class:`~sqlearn.imputers.imputer.Imputer`: Fill missing values.
        :class:`~sqlearn.scalers.standard.StandardScaler`: Standardize numerics.
        :class:`~sqlearn.encoders.onehot.OneHotEncoder`: Encode categoricals.
    """

    def __init__(
        self,
        steps: _StepsInput,
        *,
        backend: str | DuckDBBackend | None = None,
    ) -> None:
        self._steps: list[tuple[str, Transformer]] = _normalize_steps(steps)
        self._backend: str | DuckDBBackend | None = backend
        self._backend_instance: DuckDBBackend | None = None
        self._fitted: bool = False
        self._schema_in: Schema | None = None
        self._schema_out: Schema | None = None
        self._owns_backend: bool = False

    # --- Properties ---

    @property
    def steps(self) -> list[tuple[str, Transformer]]:
        """Read-only defensive copy of pipeline steps.

        Returns:
            List of (name, transformer) tuples.
        """
        return list(self._steps)

    @property
    def named_steps(self) -> dict[str, Transformer]:
        """Dict access to steps by name.

        Returns:
            Dict mapping step names to transformers.
        """
        return dict(self._steps)

    @property
    def is_fitted(self) -> bool:
        """Whether fit() has been called successfully.

        Returns:
            True if fitted.
        """
        return self._fitted

    # --- Repr ---

    def __repr__(self) -> str:
        """Show pipeline steps with names and class names.

        Returns:
            Format: ``Pipeline(step_00=Imputer, step_01=StandardScaler)``.
        """
        parts = [f"{name}={type(step).__name__}" for name, step in self._steps]
        return f"Pipeline({', '.join(parts)})"

    # --- Backend resolution ---

    def _resolve_backend(self, backend: str | DuckDBBackend | None = None) -> DuckDBBackend:
        """Resolve backend with precedence: per-call > pipeline-level > auto-create.

        Args:
            backend: Per-call backend override.

        Returns:
            Resolved DuckDBBackend instance.
        """
        if backend is not None:
            if isinstance(backend, DuckDBBackend):
                return backend
            return DuckDBBackend(backend)
        if self._backend_instance is not None:
            return self._backend_instance
        if self._backend is not None:
            if isinstance(self._backend, DuckDBBackend):
                self._backend_instance = self._backend
                return self._backend
            self._backend_instance = DuckDBBackend(self._backend)
            self._owns_backend = True
            return self._backend_instance
        self._backend_instance = DuckDBBackend()
        self._owns_backend = True
        return self._backend_instance

    # --- fit / transform / to_sql ---

    def fit(
        self,
        data: str | object,
        y: str | None = None,
        *,
        backend: str | DuckDBBackend | None = None,
    ) -> Pipeline:
        """Learn parameters from data via three-phase compiler.

        Args:
            data: Input data (file path, table name, or DataFrame).
            y: Target column name, or None.
            backend: Per-call backend override.

        Returns:
            self (for method chaining).
        """
        resolved_backend = self._resolve_backend(backend)
        source = resolve_input(data, resolved_backend)
        self._schema_in = resolved_backend.describe(source)

        transformers = [step for _, step in self._steps]
        plan = plan_fit(transformers, self._schema_in, y)

        current_exprs: dict[str, exp.Expression] = {
            col: exp.Column(this=col) for col in self._schema_in.columns
        }

        for i, layer in enumerate(plan.layers):
            layer_source: str = source if i == 0 else f"__sq_layer_{i - 1}__"

            fit_queries = build_fit_queries(layer, layer_source, current_exprs)

            # Execute aggregate query
            if fit_queries.aggregate_query is not None:
                row = resolved_backend.fetch_one(fit_queries.aggregate_query)
                for alias, (step_idx_str, param_name) in fit_queries.param_mapping.items():
                    step_idx = int(step_idx_str)
                    step = layer.steps[step_idx].step
                    if step.params_ is None:
                        step.params_ = {}
                    step.params_[param_name] = row[alias]

            # Execute set queries
            for key, query in fit_queries.set_queries.items():
                step_idx_str, set_name = key.split("_", 1)
                step_idx = int(step_idx_str)
                step = layer.steps[step_idx].step
                if step.sets_ is None:
                    step.sets_ = {}
                step.sets_[set_name] = resolved_backend.execute(query)

            # Mark steps as fitted
            for step_info in layer.steps:
                step_info.step.columns_ = step_info.columns
                step_info.step.input_schema_ = step_info.input_schema
                step_info.step._fitted = True  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                # Re-compute output schema now that params_/sets_ are populated
                step_info.step.output_schema_ = step_info.step.output_schema(
                    step_info.input_schema
                )

            # Materialize intermediate layers as temp views
            if i < len(plan.layers) - 1:
                layer_transformers = [si.step for si in layer.steps]
                select_ast = compose_transform(layer_transformers, layer_source)
                create_ast = exp.Create(
                    this=exp.to_table(f"__sq_layer_{i}__"),  # pyright: ignore[reportUnknownMemberType]
                    kind="VIEW",
                    expression=select_ast,
                    properties=exp.Properties(expressions=[exp.TemporaryProperty()]),
                )
                resolved_backend.execute(create_ast)

                last_step = layer.steps[-1].step
                out_schema = (
                    last_step.output_schema_
                    if last_step.output_schema_ is not None
                    else layer.output_schema
                )
                current_exprs = {col: exp.Column(this=col) for col in out_schema.columns}

        self._fitted = True
        last_layer = plan.layers[-1]
        last_fitted = last_layer.steps[-1].step
        self._schema_out = (
            last_fitted.output_schema_
            if last_fitted.output_schema_ is not None
            else last_layer.output_schema
        )
        return self

    def transform(
        self,
        data: str | object,
        *,
        backend: str | DuckDBBackend | None = None,
    ) -> np.ndarray[Any, Any]:
        """Apply fitted pipeline to data, returning numpy array.

        Args:
            data: Input data (file path, table name, or DataFrame).
            backend: Per-call backend override.

        Returns:
            2D numpy array (float64 if possible, object otherwise).

        Raises:
            NotFittedError: If fit() has not been called.
        """
        if not self._fitted:
            msg = "Pipeline is not fitted. Call fit() first."
            raise NotFittedError(msg)

        resolved_backend = self._resolve_backend(backend)
        source = resolve_input(data, resolved_backend)
        transformers = [step for _, step in self._steps]
        select_ast = compose_transform(transformers, source)
        rows = resolved_backend.execute(select_ast)

        columns = list(rows[0].keys()) if rows else self.get_feature_names_out()
        if not rows:
            return np.empty((0, len(columns)), dtype=np.float64)

        data_list = [[row[col] for col in columns] for row in rows]
        try:
            return np.array(data_list, dtype=np.float64)
        except (ValueError, TypeError):
            return np.array(data_list, dtype=object)

    def fit_transform(
        self,
        data: str | object,
        y: str | None = None,
        *,
        backend: str | DuckDBBackend | None = None,
    ) -> np.ndarray[Any, Any]:
        """Convenience: fit then transform.

        Args:
            data: Input data.
            y: Target column name.
            backend: Per-call backend override.

        Returns:
            2D numpy array.
        """
        return self.fit(data, y, backend=backend).transform(data, backend=backend)

    def to_sql(
        self,
        *,
        dialect: str = "duckdb",
        table: str = "__input__",
    ) -> str:
        """Compile fitted pipeline to SQL string.

        Args:
            dialect: SQL dialect for output.
            table: Source table name in generated SQL.

        Returns:
            SQL query string.

        Raises:
            NotFittedError: If fit() has not been called.
        """
        if not self._fitted:
            msg = "Pipeline is not fitted. Call fit() first."
            raise NotFittedError(msg)

        transformers = [step for _, step in self._steps]
        query = compose_transform(transformers, table)
        result: str = query.sql(dialect=dialect)  # pyright: ignore[reportUnknownMemberType]
        return result

    def get_feature_names_out(self) -> list[str]:
        """Return output column names after fitting.

        Returns:
            List of output column names.

        Raises:
            NotFittedError: If fit() has not been called.
        """
        if not self._fitted or self._schema_out is None:
            msg = "Pipeline is not fitted. Call fit() first."
            raise NotFittedError(msg)

        return list(self._schema_out.columns.keys())

    # --- Operators ---

    def __add__(self, other: object) -> Pipeline:
        """Compose pipelines: Pipeline + Transformer or Pipeline + Pipeline.

        Args:
            other: Transformer or Pipeline to append.

        Returns:
            New Pipeline with combined steps.

        Raises:
            InvalidStepError: If step name collision detected.
        """
        if isinstance(other, Pipeline):
            combined = [*self._steps, *other._steps]
            return Pipeline(combined)
        if isinstance(other, Transformer):
            name = _auto_name(self._steps, other)
            combined = [*self._steps, (name, other)]
            return Pipeline(combined)
        return NotImplemented

    def __radd__(self, other: object) -> Pipeline:
        """Handle Transformer + Pipeline.

        Args:
            other: Transformer to prepend.

        Returns:
            New Pipeline with other prepended.
        """
        if isinstance(other, Transformer):
            name = _auto_name(self._steps, other)
            combined = [(name, other), *self._steps]
            return Pipeline(combined)
        return NotImplemented

    def __iadd__(self, other: object) -> Pipeline:
        """Non-mutating +=. Returns new Pipeline.

        Args:
            other: Transformer or Pipeline to append.

        Returns:
            New Pipeline.
        """
        return self.__add__(other)

    # --- Clone ---

    def clone(self) -> Pipeline:
        """Create independent copy of pipeline.

        Clones each step via step.clone(). Preserves step names and
        fitted state. Does NOT copy backend (lazy reconnect).

        Returns:
            New Pipeline with cloned steps.
        """
        cloned_steps = [(name, step.clone()) for name, step in self._steps]
        new_pipe = Pipeline(cloned_steps)
        new_pipe._fitted = self._fitted
        new_pipe._schema_in = self._schema_in
        new_pipe._schema_out = self._schema_out
        return new_pipe

    # --- Context Manager ---

    def __enter__(self) -> Pipeline:
        """Enter context manager.

        Returns:
            This pipeline instance.
        """
        return self

    def __exit__(self, *_: object) -> None:
        """Exit context manager, closing owned backend.

        Only closes the backend if Pipeline created it. User-provided
        backends are never closed.
        """
        if self._owns_backend and isinstance(self._backend_instance, DuckDBBackend):
            self._backend_instance.close()
