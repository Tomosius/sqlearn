"""Compiler for sqlearn pipelines.

Three-phase compiler that classifies pipeline steps, groups them into
layers, batches fit queries, and composes transform expressions into
a single SQL SELECT.

Phase 1: plan_fit() — classify steps, detect schema changes, group layers
Phase 2: build_fit_queries() — batch aggregations per layer
Phase 3: compose_transform() — compose expressions into one SELECT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlearn.core.errors import ClassificationError, CompilationError
from sqlearn.core.schema import Schema, resolve_columns
from sqlearn.core.transformer import Transformer

if TYPE_CHECKING:
    import sqlglot.expressions as exp


# ── Dataclasses ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class StepClassification:
    """Classification result for a pipeline step.

    Attributes:
        kind: ``'static'`` or ``'dynamic'``.
        tier: 1 (built-in), 2 (custom declared), or 3 (custom undeclared).
        reason: Human-readable explanation for the audit trail.
        warnings: UserWarning messages for ambiguous cases.
    """

    kind: str
    tier: int
    reason: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SchemaChangeResult:
    """Result of schema change detection.

    Attributes:
        changes: True if output schema differs from input.
        reason: Human-readable explanation for the audit trail.
    """

    changes: bool
    reason: str


@dataclass
class StepInfo:
    """Compiled information about one pipeline step.

    Attributes:
        step: The transformer instance.
        classification: Static/dynamic classification result.
        schema_change: Whether this step changes the schema.
        columns: Resolved target columns for this step.
        input_schema: Schema before this step.
        step_output_schema: Schema after this step.
    """

    step: Transformer
    classification: StepClassification
    schema_change: SchemaChangeResult
    columns: list[str]
    input_schema: Schema
    step_output_schema: Schema


@dataclass
class Layer:
    """A group of pipeline steps that share the same fit context.

    Attributes:
        steps: Ordered list of step info in this layer.
        input_schema: Schema at the start of this layer.
        output_schema: Schema at the end of this layer.
    """

    steps: list[StepInfo]
    input_schema: Schema
    output_schema: Schema


@dataclass
class FitPlan:
    """Complete fit plan for a pipeline.

    Attributes:
        layers: Ordered list of layers.
    """

    layers: list[Layer]


@dataclass
class FitQueries:
    """SQL queries needed to fit one layer.

    Attributes:
        aggregate_query: Batched scalar aggregates, or None if all static.
        set_queries: Per-step set discovery queries.
        param_mapping: Maps aggregate alias to (step_name, param_name).
    """

    aggregate_query: exp.Expression | None
    set_queries: dict[str, exp.Expression]
    param_mapping: dict[str, tuple[str, str]]


# ── Phase 1: Classification ─────────────────────────────────────────


def _is_builtin(step: Transformer) -> bool:
    """Check if a step is a built-in sqlearn transformer.

    Args:
        step: Transformer to check.

    Returns:
        True if the step's module starts with ``'sqlearn.'``.
    """
    return type(step).__module__.startswith("sqlearn.")


def classify_step(
    step: Transformer,
    columns: list[str],
    schema: Schema,
    y_column: str | None = None,
) -> StepClassification:
    """Classify a pipeline step as static or dynamic.

    Uses the three-tier model:
    - Tier 1 (built-in, declared): trusted immediately.
    - Tier 2 (custom, declared): verified once, then cached.
    - Tier 3 (custom, undeclared): full conservative inspection.

    Args:
        step: Transformer to classify.
        columns: Resolved target columns for this step.
        schema: Current schema.
        y_column: Target column name, if any.

    Returns:
        StepClassification with kind, tier, reason, and warnings.

    Raises:
        ClassificationError: If a Tier 2 declaration is inconsistent.
    """
    declared = step._classification  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    builtin = _is_builtin(step)

    # Tier 1: built-in with declaration → trust
    if builtin and declared is not None:
        return StepClassification(
            kind=declared,
            tier=1,
            reason=(
                f"Built-in {type(step).__name__} declares "
                f"_classification='{declared}'. Trusted (validated by CI)."
            ),
        )

    # Tier 2: custom with declaration → verify once
    if not builtin and declared is not None:
        if getattr(step, "_classification_verified", False):
            return StepClassification(
                kind=declared,
                tier=2,
                reason=(
                    f"Custom {type(step).__name__} declares "
                    f"_classification='{declared}'. Previously verified."
                ),
            )
        return _verify_custom_declaration(step, declared, columns, schema, y_column)

    # Tier 3: no declaration → full inspection
    return _inspect_undeclared_step(step, columns, schema, y_column)


def _verify_custom_declaration(
    step: Transformer,
    declared: str,
    columns: list[str],
    schema: Schema,
    y_column: str | None,
) -> StepClassification:
    """Verify a Tier 2 custom transformer's classification declaration.

    Args:
        step: Transformer to verify.
        declared: The declared classification (``'static'`` or ``'dynamic'``).
        columns: Resolved target columns.
        schema: Current schema.
        y_column: Target column name.

    Returns:
        StepClassification after verification.

    Raises:
        ClassificationError: If declaration is inconsistent with discover().
    """
    name = type(step).__name__
    step_warnings: list[str] = []

    # Check discover()
    # Cast to Any: defensive check for user-defined transformers that may
    # violate type contracts at runtime.
    try:
        discover_result: Any = step.discover(columns, schema, y_column)
    except Exception as e:
        return StepClassification(
            kind="dynamic",
            tier=2,
            reason=(
                f"{name}.discover() raised {type(e).__name__}. Cannot verify. Treating as dynamic."
            ),
        )

    if not isinstance(discover_result, dict):
        return StepClassification(
            kind="dynamic",
            tier=2,
            reason=(
                f"{name}.discover() returned {type(discover_result).__name__}, "
                "not dict. Treating as dynamic."
            ),
        )
    # Re-cast after isinstance check to resolve dict[Unknown, Unknown] narrowing.
    discover_dict: dict[str, Any] = discover_result  # pyright: ignore[reportUnknownVariableType]

    # Check discover_sets()
    # Cast to Any: same defensive rationale as discover().
    try:
        sets_result: Any = step.discover_sets(columns, schema, y_column)
    except Exception:
        sets_result = {}

    sets_dict: dict[str, Any]
    if isinstance(sets_result, dict):  # noqa: SIM108
        sets_dict = sets_result  # pyright: ignore[reportUnknownVariableType]
    else:
        sets_dict = {}
    has_sets = len(sets_dict) > 0

    # Verify consistency
    if declared == "static" and len(discover_dict) > 0:
        msg = (
            f"{name} declares _classification='static' but discover() returned "
            f"{len(discover_dict)} aggregation(s): {list(discover_dict.keys())}. "
            "Either fix discover() to return {} or change _classification to 'dynamic'."
        )
        raise ClassificationError(msg)

    if declared == "static" and has_sets:
        msg = (
            f"{name} declares _classification='static' but discover_sets() returned "
            f"{len(sets_dict)} set query(ies): {list(sets_dict.keys())}. "
            "Either fix discover_sets() to return {} or change _classification to 'dynamic'."
        )
        raise ClassificationError(msg)

    if declared == "dynamic" and len(discover_dict) == 0 and not has_sets:
        step_warnings.append(
            f"{name} declares _classification='dynamic' but discover() returned {{}}. "
            "This wastes one fit query per call. Consider _classification='static'."
        )

    # Cache verification
    step._classification_verified = True  # type: ignore[attr-defined]  # noqa: SLF001

    return StepClassification(
        kind=declared,
        tier=2,
        reason=(
            f"Custom {name} declares _classification='{declared}'. "
            "Verified: discover() result is consistent."
        ),
        warnings=tuple(step_warnings),
    )


def _inspect_undeclared_step(  # noqa: PLR0911
    step: Transformer,
    columns: list[str],
    schema: Schema,
    y_column: str | None,
) -> StepClassification:
    """Full conservative inspection for Tier 3 steps.

    The hard safety rule: if we can't prove static with 100% certainty,
    fall back to dynamic.

    Args:
        step: Transformer to inspect.
        columns: Resolved target columns.
        schema: Current schema.
        y_column: Target column name.

    Returns:
        StepClassification (always tier 3).
    """
    name = type(step).__name__

    # Check 1: Can we call discover()?
    # Cast to Any: defensive check for user-defined transformers that may
    # violate type contracts at runtime.
    try:
        discover_result: Any = step.discover(columns, schema, y_column)
    except Exception as e:
        return StepClassification(
            kind="dynamic",
            tier=3,
            reason=(f"{name}.discover() raised {type(e).__name__}: {e}. Assuming dynamic."),
        )

    # Check 2: Is it a dict?
    if discover_result is None:
        return StepClassification(
            kind="dynamic",
            tier=3,
            reason=f"{name}.discover() returned None instead of {{}}. Assuming dynamic.",
            warnings=(f"{name}.discover() returned None. Return {{}} explicitly for static.",),
        )

    if not isinstance(discover_result, dict):
        return StepClassification(
            kind="dynamic",
            tier=3,
            reason=(
                f"{name}.discover() returned {type(discover_result).__name__}, "
                "not dict. Assuming dynamic."
            ),
        )
    # Re-cast after isinstance check to resolve dict[Unknown, Unknown] narrowing.
    discover_dict: dict[str, Any] = discover_result  # pyright: ignore[reportUnknownVariableType]

    # Check 3: Is the dict non-empty?
    if len(discover_dict) > 0:
        return StepClassification(
            kind="dynamic",
            tier=3,
            reason=(
                f"discover() returned {len(discover_dict)} aggregation(s): "
                f"{list(discover_dict.keys())}"
            ),
        )

    # Check 3b: discover_sets()?
    # Cast to Any: same defensive rationale as discover().
    try:
        sets_result: Any = step.discover_sets(columns, schema, y_column)
    except Exception:
        sets_result = {}

    sets_dict: dict[str, Any]
    if isinstance(sets_result, dict):  # noqa: SIM108
        sets_dict = sets_result  # pyright: ignore[reportUnknownVariableType]
    else:
        sets_dict = {}
    if len(sets_dict) > 0:
        return StepClassification(
            kind="dynamic",
            tier=3,
            reason=(
                f"discover_sets() returned {len(sets_dict)} set query(ies): "
                f"{list(sets_dict.keys())}"
            ),
        )

    # Check 4: Does the step override fit()?
    base_fit = Transformer.fit
    step_fit = type(step).fit
    if step_fit is not base_fit:
        return StepClassification(
            kind="dynamic",
            tier=3,
            reason=(
                f"{name} overrides fit() directly. "
                "Cannot guarantee no data-dependent side effects."
            ),
        )

    # All checks passed → static
    return StepClassification(
        kind="static",
        tier=3,
        reason="discover() returned {} and all safety checks passed.",
    )


# ── Phase 1: Schema Change Detection ────────────────────────────────


def detect_schema_change(
    step: Transformer,
    input_schema: Schema,
) -> SchemaChangeResult:
    """Detect if a step changes the output schema.

    Conservative: if we can't determine the answer, assume it changes.

    Args:
        step: Transformer to check.
        input_schema: Schema before this step.

    Returns:
        SchemaChangeResult with changes flag and reason.
    """
    # Cast to Any: defensive check for user-defined transformers that may
    # violate type contracts at runtime.
    try:
        output: Any = step.output_schema(input_schema)
    except Exception as e:
        return SchemaChangeResult(
            changes=True,
            reason=(f"output_schema() raised {type(e).__name__}: {e}. Assuming schema changes."),
        )

    if output is None:
        return SchemaChangeResult(
            changes=True,
            reason="output_schema() returned None. Assuming schema changes.",
        )

    if not isinstance(output, Schema):
        return SchemaChangeResult(
            changes=True,
            reason=(
                f"output_schema() returned {type(output).__name__}, not Schema. "
                "Assuming schema changes."
            ),
        )

    if output.columns != input_schema.columns:
        added = set(output.columns) - set(input_schema.columns)
        removed = set(input_schema.columns) - set(output.columns)
        retyped = {
            col
            for col in set(output.columns) & set(input_schema.columns)
            if output.columns[col] != input_schema.columns[col]
        }
        return SchemaChangeResult(
            changes=True,
            reason=(
                f"Schema differs. Added: {added or 'none'}, "
                f"Removed: {removed or 'none'}, Retyped: {retyped or 'none'}"
            ),
        )

    return SchemaChangeResult(
        changes=False,
        reason="output_schema matches input_schema.",
    )


# ── Phase 1: Layer Grouping ─────────────────────────────────────────


def plan_fit(
    steps: list[Transformer],
    schema: Schema,
    y_column: str | None = None,
) -> FitPlan:
    """Classify steps and group into layers for fit execution.

    Layer boundary is created after any dynamic + schema-changing step.
    Static schema-changing steps do NOT create boundaries.

    Args:
        steps: Ordered list of pipeline transformers.
        schema: Initial input schema.
        y_column: Target column name, if any.

    Returns:
        FitPlan with ordered layers.

    Raises:
        CompilationError: If steps list is empty.
    """
    if not steps:
        msg = "Cannot compile empty pipeline — no steps provided."
        raise CompilationError(msg)

    current_schema = schema
    current_layer_steps: list[StepInfo] = []
    layer_input_schema = schema
    layers: list[Layer] = []

    for step in steps:
        # Resolve columns for this step
        col_spec = step._resolve_columns_spec()  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        if col_spec is None:
            columns = list(current_schema.columns.keys())
        else:
            columns = resolve_columns(current_schema, col_spec)

        # Classify step
        classification = classify_step(step, columns, current_schema, y_column)

        # Detect schema change
        schema_change = detect_schema_change(step, current_schema)

        # Reuse output schema from detect_schema_change when possible,
        # avoiding a redundant second call to output_schema().
        # Conservative fallback: if output_schema() failed or returned
        # non-Schema, we keep current_schema. This means the step is flagged
        # as schema-changing but the propagated schema stays the same —
        # correct and conservative, may create an unnecessary layer boundary.
        if schema_change.changes:
            # Cast to Any: defensive check for user-defined transformers that
            # may violate type contracts at runtime.
            try:
                raw_output: Any = step.output_schema(current_schema)
                step_output = raw_output if isinstance(raw_output, Schema) else current_schema
            except Exception:
                step_output = current_schema
        else:
            step_output = current_schema

        step_info = StepInfo(
            step=step,
            classification=classification,
            schema_change=schema_change,
            columns=columns,
            input_schema=current_schema,
            step_output_schema=step_output,
        )
        current_layer_steps.append(step_info)

        # Layer boundary: dynamic + schema-changing
        is_dynamic = classification.kind == "dynamic"
        if is_dynamic and schema_change.changes:
            layers.append(
                Layer(
                    steps=list(current_layer_steps),
                    input_schema=layer_input_schema,
                    output_schema=step_output,
                )
            )
            current_layer_steps = []
            layer_input_schema = step_output

        current_schema = step_output

    # Flush remaining steps as final layer
    if current_layer_steps:
        layers.append(
            Layer(
                steps=current_layer_steps,
                input_schema=layer_input_schema,
                output_schema=current_schema,
            )
        )

    return FitPlan(layers=layers)
