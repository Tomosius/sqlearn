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
from typing import Any

import sqlglot.expressions as exp

from sqlearn.core.errors import ClassificationError, CompilationError, SchemaError
from sqlearn.core.schema import Schema, resolve_columns
from sqlearn.core.transformer import Transformer

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


# ── Phase 2: Fit Query Batching ──────────────────────────────────────


def _substitute_columns(
    expression: exp.Expression,
    current_exprs: dict[str, exp.Expression],
) -> exp.Expression:
    """Replace column references in an expression with current composed expressions.

    When a dynamic step's discover() produces ``AVG(Column('price'))``,
    and a preceding static step already transformed price to ``price * 2``,
    this substitutes ``Column('price')`` → ``price * 2`` so the aggregation
    becomes ``AVG(price * 2)``.

    Args:
        expression: The aggregation expression to transform.
        current_exprs: Current column expressions (from static steps).

    Returns:
        New expression with column references substituted.
    """
    expression = expression.copy()
    for node in expression.walk():
        if isinstance(node, exp.Column) and not node.table:
            col_name = node.name
            if col_name in current_exprs:
                replacement = current_exprs[col_name].copy()
                node.replace(replacement)
    return expression


def _collect_aggregations(
    step_info: StepInfo,
    step_index: int,
    running_exprs: dict[str, exp.Expression],
    aggregate_selects: list[exp.Expression],
    param_mapping: dict[str, tuple[str, str]],
) -> None:
    """Collect scalar aggregations from one dynamic step into shared lists.

    Substitutes current composed expressions into each aggregation expression
    before appending so that preceding static transforms are inlined.

    Args:
        step_info: Compiled info for this step.
        step_index: Position of this step in the layer (for alias naming).
        running_exprs: Current column expressions (modified in-place by caller).
        aggregate_selects: Accumulator for aliased aggregate expressions.
        param_mapping: Accumulator for alias-to-(step, param) mapping.
    """
    # Cast to Any: defensive check for user-defined transformers that may
    # violate type contracts at runtime.
    try:
        raw_discover: Any = step_info.step.discover(
            step_info.columns, step_info.input_schema, None
        )
    except Exception:
        raw_discover = {}

    if not isinstance(raw_discover, dict):
        return
    discover_result: dict[str, Any] = raw_discover  # pyright: ignore[reportUnknownVariableType]

    for param_name, agg_expr in discover_result.items():
        if not isinstance(agg_expr, exp.Expression):
            continue
        substituted = _substitute_columns(agg_expr, running_exprs)
        alias = f"__agg_{step_index}_{param_name}"
        aggregate_selects.append(substituted.as_(alias))  # pyright: ignore[reportUnknownMemberType]
        param_mapping[alias] = (str(step_index), param_name)


def _collect_set_queries(
    step_info: StepInfo,
    step_index: int,
    source: str | exp.Expression,
    set_queries: dict[str, exp.Expression],
) -> None:
    """Collect set-valued queries from one dynamic step into a shared dict.

    Adds a FROM clause to any set query that doesn't already have one,
    since discover_sets() doesn't know the source table.

    Args:
        step_info: Compiled info for this step.
        step_index: Position of this step in the layer (for key naming).
        source: Source table/view name or expression to add as FROM clause.
        set_queries: Accumulator dict keyed by ``"{step_index}_{set_name}"``.
    """
    # Cast to Any: defensive check for user-defined transformers that may
    # violate type contracts at runtime.
    try:
        raw_sets: Any = step_info.step.discover_sets(
            step_info.columns, step_info.input_schema, None
        )
    except Exception:
        raw_sets = {}

    if not isinstance(raw_sets, dict):
        return
    sets_result: dict[str, Any] = raw_sets  # pyright: ignore[reportUnknownVariableType]

    for set_name, set_query in sets_result.items():
        if isinstance(set_query, exp.Expression):
            # Add FROM source to set queries that don't have one
            resolved_query = set_query
            if isinstance(set_query, exp.Select) and not set_query.find(exp.From):
                source_expr = (
                    exp.to_table(source)  # pyright: ignore[reportUnknownMemberType]
                    if isinstance(source, str)
                    else source
                )
                resolved_query = set_query.from_(source_expr)  # pyright: ignore[reportUnknownMemberType]
            set_queries[f"{step_index}_{set_name}"] = resolved_query


def build_fit_queries(
    layer: Layer,
    source: str | exp.Expression,
    current_exprs: dict[str, exp.Expression],
) -> FitQueries:
    """Build minimal SQL queries to fit one layer.

    Batches all scalar aggregations from dynamic steps into one SELECT.
    Set queries (from discover_sets) are kept separate.

    Static steps' expressions are inlined into dynamic steps' aggregation
    queries via column substitution.

    Args:
        layer: The layer to build queries for.
        source: Source table/view name or expression.
        current_exprs: Current column expressions (accumulated from
            prior steps, used for expression inlining).

    Returns:
        FitQueries with aggregate query, set queries, and param mapping.
    """
    aggregate_selects: list[exp.Expression] = []
    param_mapping: dict[str, tuple[str, str]] = {}
    set_queries: dict[str, exp.Expression] = {}
    running_exprs = dict(current_exprs)

    for i, step_info in enumerate(layer.steps):
        if step_info.classification.kind == "static":
            # Static step: compose expressions forward for inlining
            try:
                modified = step_info.step.expressions(step_info.columns, running_exprs)
                running_exprs = dict(running_exprs)
                running_exprs.update(modified)
            except NotImplementedError:
                pass
            continue

        # Dynamic step: collect aggregations and set queries
        _collect_aggregations(step_info, i, running_exprs, aggregate_selects, param_mapping)
        _collect_set_queries(step_info, i, source, set_queries)

    # Build aggregate query
    aggregate_query: exp.Expression | None = None
    if aggregate_selects:
        source_expr = exp.to_table(source) if isinstance(source, str) else source  # pyright: ignore[reportUnknownMemberType]
        aggregate_query = exp.select(*aggregate_selects).from_(source_expr)  # pyright: ignore[reportUnknownMemberType]

    return FitQueries(
        aggregate_query=aggregate_query,
        set_queries=set_queries,
        param_mapping=param_mapping,
    )


# ── Phase 3: Transform Composition ──────────────────────────────────


_DEFAULT_CTE_DEPTH = 8


def _expression_depth(expr: exp.Expression) -> int:
    """Calculate the nesting depth of an expression.

    Args:
        expr: sqlglot expression to measure.

    Returns:
        Maximum nesting depth.
    """
    if not isinstance(expr, exp.Expression):  # pyright: ignore[reportUnnecessaryIsInstance]
        return 0
    child_depths = (_expression_depth(child) for child in expr.iter_expressions())
    return max(child_depths, default=0) + 1


def _max_depth(exprs: dict[str, exp.Expression]) -> int:
    """Find maximum depth across all expressions.

    Args:
        exprs: Column expression dict.

    Returns:
        Maximum depth found.
    """
    if not exprs:
        return 0
    return max(_expression_depth(e) for e in exprs.values())


def _exprs_to_cte(
    exprs: dict[str, exp.Expression],
    source: str | exp.Expression,
) -> tuple[exp.Expression, dict[str, exp.Expression]]:
    """Wrap current expressions into a CTE SELECT and reset to bare column refs.

    Args:
        exprs: Current column expressions.
        source: Current FROM source.

    Returns:
        Tuple of (CTE body SELECT expression, new bare exprs referencing CTE).
    """
    # Build SELECT for the CTE
    selects = [v.as_(k) for k, v in exprs.items()]  # pyright: ignore[reportUnknownMemberType]
    source_expr = exp.to_table(source) if isinstance(source, str) else source  # pyright: ignore[reportUnknownMemberType]
    cte_query = exp.select(*selects).from_(source_expr)  # pyright: ignore[reportUnknownMemberType]

    # New bare column references against the CTE
    new_exprs: dict[str, exp.Expression] = {col: exp.Column(this=col) for col in exprs}

    return cte_query, new_exprs


def compose_transform(  # noqa: C901, PLR0912, PLR0915
    steps: list[Transformer],
    source: str,
    *,
    cte_depth: int = _DEFAULT_CTE_DEPTH,
) -> exp.Select:
    """Compose fitted transformer steps into a single SELECT.

    Walks through all steps, composing their expressions inline.
    Steps using query() get promoted to CTEs. Auto-promotes to CTE
    when expression depth exceeds the threshold.

    Args:
        steps: Ordered list of fitted transformers.
        source: Source table/view name.
        cte_depth: Maximum expression depth before auto-CTE promotion.

    Returns:
        sqlglot SELECT expression (possibly with CTEs).

    Raises:
        CompilationError: If steps list is empty, a step is unfitted, or
            expressions return non-AST values.
        SchemaError: If column name collision detected between steps.
    """
    if not steps:
        msg = "Cannot compile empty pipeline — no steps provided."
        raise CompilationError(msg)

    # Check that all steps have been fitted
    for step in steps:
        if step.columns_ is None:
            msg = f"{type(step).__name__} has no columns_ — call fit() first."
            raise CompilationError(msg)

    # Get initial columns from the first step's input schema
    first_step = steps[0]
    if first_step.input_schema_ is not None:
        initial_cols = list(first_step.input_schema_.columns.keys())
    else:
        initial_cols = list(first_step.columns_)  # type: ignore[arg-type]

    exprs: dict[str, exp.Expression] = {col: exp.Column(this=col) for col in initial_cols}
    current_source: str | exp.Expression = source
    ctes: list[tuple[str, exp.Expression]] = []
    cte_counter = 0
    original_cols = set(initial_cols)
    # Track which step added each non-original column (for collision detection)
    added_by: dict[str, int] = {}  # col_name -> step_index

    for step_idx, step in enumerate(steps):
        # Try query() first — if it returns something, promote to CTE.
        # Base class query() returns None (does not raise NotImplementedError).
        selects = [v.as_(k) for k, v in exprs.items()]  # pyright: ignore[reportUnknownMemberType]
        source_expr: exp.Expression = (
            exp.to_table(current_source)  # pyright: ignore[reportUnknownMemberType]
            if isinstance(current_source, str)  # pyright: ignore[reportUnnecessaryIsInstance]
            else current_source
        )
        input_query = exp.select(*selects).from_(source_expr)  # pyright: ignore[reportUnknownMemberType]

        query_result = step.query(input_query)

        if query_result is not None and isinstance(query_result, exp.Expression):  # pyright: ignore[reportUnnecessaryIsInstance]
            # CTE promotion for query() steps
            cte_name = f"__cte_{cte_counter}"
            cte_counter += 1
            ctes.append((cte_name, query_result))
            current_source = cte_name

            # Reset expressions to bare column references from output schema
            if step.output_schema_ is not None:
                out_cols = list(step.output_schema_.columns.keys())
            elif step.columns_ is not None:
                out_cols = list(step.columns_)
            else:
                out_cols = list(exprs.keys())

            exprs = {col: exp.Column(this=col) for col in out_cols}
            continue

        # Check for column name collisions BEFORE merging.
        # Inspect expressions() output to see what non-original columns
        # this step produces. If another step already added that column,
        # it's a collision.
        if step.columns_ is not None:
            try:
                modified = step.expressions(step.columns_, dict(exprs))
            except (NotImplementedError, Exception):
                modified = {}
            if isinstance(modified, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
                new_from_step = set(modified.keys()) - original_cols
                for col in new_from_step:
                    if col in added_by and added_by[col] != step_idx:
                        msg = (
                            f"Column '{col}' is produced by both step "
                            f"{added_by[col]} and step {step_idx} "
                            f"({type(step).__name__}). "
                            "Use Rename to resolve collisions."
                        )
                        raise SchemaError(msg)
                    added_by[col] = step_idx

        # Expression-level composition via _apply_expressions.
        exprs = step._apply_expressions(exprs)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

        # Validate output: all values must be sqlglot AST nodes.
        for col_name, col_expr in exprs.items():
            if not isinstance(col_expr, exp.Expression):  # pyright: ignore[reportUnnecessaryIsInstance]
                msg = (
                    f"{type(step).__name__}.expressions() returned "
                    f"{type(col_expr).__name__} for column '{col_name}', "
                    "expected sqlglot Expression."
                )
                raise CompilationError(msg)

        # Auto-CTE if depth exceeds threshold
        if _max_depth(exprs) > cte_depth:
            cte_name = f"__cte_{cte_counter}"
            cte_counter += 1
            cte_query, exprs = _exprs_to_cte(exprs, current_source)
            ctes.append((cte_name, cte_query))
            current_source = cte_name

    # Build final SELECT
    final_selects = [v.as_(k) for k, v in exprs.items()]  # pyright: ignore[reportUnknownMemberType]
    final_source_expr: exp.Expression = (
        exp.to_table(current_source)  # pyright: ignore[reportUnknownMemberType]
        if isinstance(current_source, str)  # pyright: ignore[reportUnnecessaryIsInstance]
        else current_source
    )
    final_query: exp.Select = exp.select(*final_selects).from_(final_source_expr)  # pyright: ignore[reportUnknownMemberType]

    # Attach CTEs using with_() so they are rendered by .sql()
    for cte_name, cte_query in ctes:
        final_query = final_query.with_(cte_name, as_=cte_query)  # pyright: ignore[reportUnknownMemberType]

    return final_query
