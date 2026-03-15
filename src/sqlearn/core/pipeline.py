"""Pipeline orchestrator for sqlearn.

Compiles ML preprocessing steps into SQL via the three-phase compiler.
Pipeline is NOT a Transformer subclass — it's a thin orchestrator that
delegates to plan_fit, build_fit_queries, and compose_transform.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from sqlearn.core.errors import InvalidStepError
from sqlearn.core.transformer import Transformer

if TYPE_CHECKING:
    from sqlearn.core.backend import DuckDBBackend
    from sqlearn.core.schema import Schema

# Steps input type: bare list, tuple list, or dict.
_StepsInput = list[Transformer] | list[tuple[str, Transformer]] | dict[str, Transformer]


def _auto_name(  # pyright: ignore[reportUnusedFunction]  # used by operators (Task 6)
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

    Args:
        steps: Pipeline steps in any of the three formats above.
        backend: Backend config — str (file path), DuckDBBackend instance,
            or None (auto-create in-memory).

    Raises:
        InvalidStepError: If steps are empty, contain non-Transformers,
            or have duplicate names.
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
