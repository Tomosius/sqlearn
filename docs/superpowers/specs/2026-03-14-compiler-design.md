# compiler.py — Design Spec

**Issue:** #6
**Milestone:** 2 — Core Compiler (v0.1.0)

## Overview

One module `src/sqlearn/core/compiler.py` with three phases implemented as public functions + dataclasses. Pipeline.py (#7) calls these functions — the compiler produces ASTs and plans, pipeline.py orchestrates execution.

The three phases:
1. **plan_fit()** — classify steps as static/dynamic, group into layers
2. **build_fit_queries()** — batch aggregations into minimal SQL queries per layer
3. **compose_transform()** — compose step expressions into a single SELECT with optional CTEs

## Phase 1: plan_fit() — Step Classification + Layer Grouping

### Public Function

```python
def plan_fit(
    steps: list[Transformer],
    schema: Schema,
    y_column: str | None = None,
) -> FitPlan
```

### Step Classification

Follows the three-tier model from docs/03-architecture.md Section 4.3:

| Tier | Condition | Behavior |
|---|---|---|
| 1 | Built-in, `_classification` set | Trust immediately, zero cost |
| 2 | Custom, `_classification` set | Verify once on first fit via `discover()` call |
| 3 | Custom, `_classification` is None | Full conservative inspection every fit |

Classification helper:

```python
def classify_step(
    step: Transformer,
    columns: list[str],
    schema: Schema,
    y_column: str | None = None,
) -> StepClassification
```

### Schema Change Detection

```python
def detect_schema_change(step: Transformer, input_schema: Schema) -> bool
```

Calls `output_schema()`, compares input vs output. Conservative — if it can't determine the answer, assumes the schema changes.

### Layer Grouping Rule

New layer boundary after any **dynamic + schema-changing** step. Downstream steps need the new schema to compute their aggregations.

Example:

```
Pipeline: [Imputer, StandardScaler, OneHotEncoder, MinMaxScaler]

Imputer        → dynamic, same schema      → Layer 0
StandardScaler → dynamic, same schema      → Layer 0
OneHotEncoder  → dynamic, CHANGES schema   → Layer 0 (boundary after)
MinMaxScaler   → dynamic, same schema      → Layer 1
```

Static steps never create boundaries. Static + schema-changing doesn't create a boundary either — the output schema is known at construction time.

### Dataclasses

```python
@dataclass(frozen=True)
class StepClassification:
    kind: str          # "static" | "dynamic"
    tier: int          # 1, 2, or 3
    reason: str        # human-readable for audit trail

@dataclass
class StepInfo:
    step: Transformer
    classification: StepClassification
    schema_changes: bool
    columns: list[str]          # resolved columns for this step
    input_schema: Schema
    output_schema: Schema

@dataclass
class Layer:
    steps: list[StepInfo]
    input_schema: Schema
    output_schema: Schema

@dataclass
class FitPlan:
    layers: list[Layer]
```

## Phase 2: build_fit_queries() — Aggregation Batching

### Public Function

```python
def build_fit_queries(
    layer: Layer,
    source: str | exp.Expression,
    current_exprs: dict[str, exp.Expression],
) -> FitQueries
```

Produces minimal SQL queries for one layer:

1. **Aggregate query** — batches all scalar aggregations from dynamic steps into one SELECT
2. **Set queries** — one per step that uses `discover_sets()` (different result shape)

### Expression Inlining

Static steps' expressions are composed into dynamic steps' aggregation queries. If `Log` (static) precedes `StandardScaler` (dynamic) in the same layer, the aggregation becomes `AVG(LN(price + 1))`, not `AVG(price)`.

The `current_exprs` parameter carries composed expressions accumulated from prior steps in the layer. When a dynamic step's `discover()` references columns, the compiler substitutes the current expression for that column.

### Dataclass

```python
@dataclass
class FitQueries:
    aggregate_query: exp.Expression | None   # batched scalar aggregates, None if all static
    set_queries: dict[str, exp.Expression]   # step_name -> set discovery query
    param_mapping: dict[str, str]            # maps aggregate alias -> (step_name, param_name)
```

## Phase 3: compose_transform() — Expression Composition

### Public Function

```python
def compose_transform(
    steps: list[Transformer],
    source: str,
    dialect: str = "duckdb",
) -> exp.Select
```

### Algorithm

1. Start with bare column references from the input schema
2. For each step:
   - If step uses `expressions()` → call `_apply_expressions(exprs)`, get updated dict
   - If step uses `query()` → promote to CTE, reset expressions to new column references
3. Build final `SELECT ... FROM source` (or from last CTE)

### Two Composition Modes

- **Inline** (expressions-only): expressions nest deeper with each step, no CTE
- **CTE promotion** (query() steps): step gets its own CTE, downstream references CTE columns

### Auto CTE at Depth > 8

Even for expression-only steps, if nesting exceeds 8 levels, the compiler extracts a CTE to keep SQL readable. The threshold is configurable.

### Depth Tracking

Each expression's nesting depth is tracked. When a step would exceed the threshold, the compiler wraps current expressions into a CTE and resets depth counters.

## Error Handling

| Situation | Error |
|---|---|
| `expressions()` returns non-AST | `CompilationError` |
| `query()` returns non-AST | `CompilationError` |
| Column name collision in merge | `SchemaError` |
| Empty pipeline (no steps) | `CompilationError` |
| Step produces no output columns | `SchemaError` |

## Edge Cases

1. **All-static pipeline** — `plan_fit` returns one layer with zero aggregation queries. `build_fit_queries` returns `None` for `aggregate_query`.
2. **Single step** — works fine, one layer, compose as normal.
3. **Schema-changing static step** (e.g. `DateParts`) — does NOT create layer boundary. Output schema known at construction time.
4. **Multiple query() steps** — each gets own CTE, chained: CTE1 from source, CTE2 from CTE1.
5. **discover() references nonexistent columns** — DuckDB raises error at execution time. Compiler doesn't validate column existence.

## Dependencies

- `compiler.py` imports: `sqlglot.expressions as exp`, `sqlearn.core.schema.Schema`, `sqlearn.core.transformer.Transformer`, `sqlearn.core.errors.CompilationError`, `sqlearn.core.errors.SchemaError`, `sqlearn.core.errors.ClassificationError`, `dataclasses`, `typing`
- No backend dependency — compiler produces ASTs, doesn't execute them

## Exports

Add to `sqlearn.core.__init__` and `sqlearn.__init__`:
- `compose_transform` (for users who want to inspect compiled SQL without a pipeline)

The dataclasses (`FitPlan`, `StepClassification`, `StepInfo`, `Layer`, `FitQueries`) and internal functions (`plan_fit`, `build_fit_queries`, `classify_step`, `detect_schema_change`) are primarily consumed by pipeline.py — don't export to top-level.

## Tests (test_compiler.py)

Using mock transformers with hardcoded `discover()`/`expressions()` returns. No DuckDB needed.

### classify_step tests (~12)

1. Tier 1 built-in static → trusted, returns `"static"`
2. Tier 1 built-in dynamic → trusted, returns `"dynamic"`
3. Tier 2 custom declared static, `discover()` returns `{}` → verified, cached
4. Tier 2 custom declared static, `discover()` returns non-empty → `ClassificationError`
5. Tier 2 custom declared dynamic, `discover()` returns `{}` → warning, honors declaration
6. Tier 2 verified flag skips re-verification
7. Tier 3 undeclared, `discover()` returns `{}` → static
8. Tier 3 undeclared, `discover()` returns non-empty → dynamic
9. Tier 3 `discover()` raises → dynamic fallback
10. Tier 3 `discover()` returns `None` → dynamic fallback
11. Tier 3 `discover_sets()` returns non-empty → dynamic
12. Tier 3 overrides `fit()` → dynamic

### detect_schema_change tests (~6)

1. Same schema → `False`
2. Added columns → `True`
3. Removed columns → `True`
4. Retyped columns → `True`
5. `output_schema()` raises → `True` (conservative)
6. `output_schema()` returns `None` → `True` (conservative)

### plan_fit tests (~8)

1. All static steps → one layer, no aggregations needed
2. All dynamic same-schema → one layer
3. Dynamic schema-changing → creates boundary, two layers
4. Static schema-changing → NO boundary (schema known at construction time)
5. Mixed: static + dynamic + schema-changing dynamic → correct layers
6. Single step → one layer
7. Empty steps list → `CompilationError`
8. Layer schemas propagate correctly through boundaries

### build_fit_queries tests (~10)

1. Single dynamic step → one aggregate query with correct expressions
2. Multiple dynamic steps → batched into one query
3. All static layer → `aggregate_query` is `None`
4. Step with `discover_sets()` → produces `set_queries` entry
5. Expression inlining: static before dynamic → composed into aggregation
6. Multiple set queries → one per step
7. `param_mapping` maps aliases back to step names
8. Empty `discover()` for a step → skipped in aggregation (no entries)
9. Mixed static+dynamic → only dynamic contribute to query
10. Source as string vs expression both work

### compose_transform tests (~15)

1. Single expression step → SELECT with composed expressions
2. Two expression steps → nested composition
3. `query()` step → CTE promotion
4. Expression then query then expression → CTE in middle
5. Multiple `query()` steps → chained CTEs
6. Passthrough: unmodified columns appear in output
7. Schema-changing step: new columns appear, old removed per `output_schema`
8. Auto CTE at depth > 8
9. Depth tracking across steps
10. Empty pipeline → `CompilationError`
11. All-static pipeline → valid SELECT, no CTEs
12. Column name collision → `SchemaError`
13. `expressions()` returns non-AST → `CompilationError`
14. Source name appears correctly in FROM clause
15. Final SELECT aliases match output column names

## Design Decisions

- **Functions + dataclasses, not planner classes:** Compilers are naturally functional — data flows through phases. No state to encapsulate. Less ceremony, same testability.
- **Frozen StepClassification:** Immutable, hashable — enables caching in M5.
- **Compiler produces ASTs, never executes:** Clean separation. Pipeline.py handles execution against backend. Compiler is pure logic, fully testable without DuckDB.
- **Depth threshold 8 for auto-CTE:** DuckDB handles deep nesting fine, but other databases may not. 8 is conservative. Configurable for users who know their target.
- **Expression inlining in fit queries:** Static steps' work composes into dynamic steps' aggregation queries. No materialization within a layer. This is the key performance optimization.
- **Layer boundary only on dynamic + schema-changing:** Static schema changes don't need materialization because the output schema is known at construction time. Only dynamic schema changes require downstream steps to see materialized results.

## Non-goals

- No Columns/Union parallel composition (M3)
- No CTE deduplication for shared subexpressions (M5)
- No FitPlan caching across CV folds (M5)
- No Tier 3 classification caching (M5)
- No multi-dialect transpilation (M7) — compiler produces sqlglot ASTs, dialect handling is in pipeline/backend
- No query execution — that's pipeline.py + backend.py
