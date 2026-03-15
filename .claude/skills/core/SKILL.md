---
name: core
description: Provides architecture guidance for sqlearn core modules — compiler, schema, backend, pipeline, IO, and errors. Trigger when working on expression composition, CTE promotion, layer resolution, schema propagation, the fit/transform lifecycle, or any file in src/sqlearn/core/. Also trigger when debugging pipeline behavior, investigating compilation issues, or understanding how transformers compose into SQL.
disable-model-invocation: false
user-invocable: true
---

# Core Architecture — sqlearn

Reference for working on `src/sqlearn/core/`. Read the relevant design doc for deep details:
- `docs/03-architecture.md` — Transformer base class, classification, discover/expressions
- `docs/04-compiler.md` — Expression composition, CTEs, layers

## Module Responsibilities

| Module | Purpose | Key exports |
|---|---|---|
| `schema.py` | Column types, type detection, selectors, column resolution | `Schema`, `resolve_columns()`, `ColumnSelector` |
| `transformer.py` | Base class all transformers extend | `Transformer` |
| `backend.py` | Database protocol + DuckDB implementation | `Backend`, `DuckDBBackend` |
| `io.py` | Input resolution (file/table/DataFrame → DuckDB) | `resolve_input()` |
| `compiler.py` | Three-phase compiler: classify, batch fit, compose transform | `plan_fit()`, `build_fit_queries()`, `compose_transform()` |
| `pipeline.py` | fit/transform/to_sql orchestration | `Pipeline` |
| `errors.py` | Error hierarchy | `SQLearnError` and 10 subclasses |

**Not yet implemented:** `output.py` (TransformResult — planned for M3+).

## The Transformer Lifecycle

```
fit(data, y="target")
  │
  ├── 1. IO: resolve_input() → DuckDB table/view name
  ├── 2. Schema: Schema.from_source() → detect column types
  ├── 3. Plan: plan_fit() → classify steps, resolve columns, group layers
  │     ├── classify_step() → StepClassification (kind, tier, reason)
  │     ├── detect_schema_change() → SchemaChangeResult
  │     └── Group into Layer objects (boundary = dynamic + schema-changing)
  ├── 4. Fit: build_fit_queries() per layer → batched SQL
  │     ├── discover() → scalar params → self.params_
  │     └── discover_sets() → set values → self.sets_
  └── 5. Store: params_, sets_, columns_, input_schema_, output_schema_ populated

transform(data)
  │
  ├── 1. IO: resolve_input()
  ├── 2. Compile: compose_transform() → one SELECT with optional CTEs
  │     ├── Start with bare column refs: {"price": Column("price")}
  │     ├── Each step: _apply_expressions() merges into expression dict
  │     ├── Steps using query() get CTE promotion
  │     └── Auto-CTE at depth > 8 (_DEFAULT_CTE_DEPTH)
  └── 3. Execute: backend.execute() → numpy array
```

## Compiler Data Structures

These are the actual dataclasses used by the compiler (in `compiler.py`):

```python
@dataclass(frozen=True)
class StepClassification:
    kind: str        # "static" or "dynamic"
    tier: int        # 1 (built-in), 2 (custom declared), 3 (custom undeclared)
    reason: str      # human-readable audit trail
    warnings: tuple[str, ...] = ()

@dataclass(frozen=True)
class SchemaChangeResult:
    changes: bool    # True if output differs from input
    reason: str

@dataclass
class StepInfo:
    step: Transformer
    classification: StepClassification
    schema_change: SchemaChangeResult
    columns: list[str]
    input_schema: Schema
    step_output_schema: Schema

@dataclass
class Layer:
    steps: list[StepInfo]
    input_schema: Schema
    output_schema: Schema

@dataclass
class FitPlan:
    layers: list[Layer]

@dataclass
class FitQueries:
    aggregate_query: exp.Expression | None   # batched scalars, or None if all static
    set_queries: dict[str, exp.Expression]   # per-step set discovery
    param_mapping: dict[str, tuple[str, str]]  # alias → (step_index, param_name)
```

## Three-Phase Compiler

### Phase 1: `plan_fit(steps, schema, y_column)` → `FitPlan`

Classifies steps and groups into layers:

1. For each step: resolve columns via `_resolve_columns_spec()` + `resolve_columns()`
2. Classify via `classify_step()` (three-tier model)
3. Detect schema change via `detect_schema_change()`
4. Create `StepInfo`
5. Layer boundary after any **dynamic + schema-changing** step

### Phase 2: `build_fit_queries(layer, source, current_exprs)` → `FitQueries`

Builds minimal SQL to fit one layer:

- Static steps: compose expressions forward via `step.expressions()` (for inlining into dynamic aggregations via `_substitute_columns()`)
- Dynamic steps: collect aggregations into one batched SELECT + separate set queries
- Result: typically 1-2 queries per layer (one aggregate + N set queries)

### Phase 3: `compose_transform(steps, source)` → `exp.Select`

Composes all steps into one SELECT:

1. Start with bare column refs from first step's `input_schema_`
2. For each step: try `query()` first (CTE if returned), else `_apply_expressions()`
3. Validate all outputs are sqlglot AST nodes
4. Auto-CTE when depth exceeds `_DEFAULT_CTE_DEPTH` (8)
5. Column collision detection: raises `SchemaError` if two steps produce same new column
6. Final: build SELECT with all CTEs attached via `.with_()`

## Expression Composition

Each transformer's `expressions()` receives the current expression dict and returns modified entries.
The compiler chains them — output of step N becomes input to step N+1:

```
Step 0 (bare):     {"price": Column("price")}
Step 1 (Imputer):  {"price": Coalesce(Column("price"), 42.5)}
Step 2 (Scaler):   {"price": (Coalesce(Column("price"), 42.5) - 42.5) / NULLIF(12.3, 0)}
```

Final SQL: `SELECT (COALESCE(price, 42.5) - 42.5) / NULLIF(12.3, 0) AS price FROM data`

Three Python steps → one SQL query. No intermediate materialization.

**Key: `_apply_expressions()`** — the base class wrapper that:
- Calls `self.expressions(self.columns_, exprs)`
- Merges modified columns into full expression dict (passthrough for unmodified)
- Detects undeclared new columns (warns if `output_schema()` doesn't declare them)
- Filters output to match `output_schema()` columns

## When to Use expressions() vs query()

| Situation | Method | Why |
|---|---|---|
| Column-level math (scale, encode, clip) | `expressions()` | Composes inline, no CTE needed |
| Window functions (LAG, RANK, rolling) | `query()` | Windows can't nest inside expressions |
| JOINs (Lookup, merge) | `query()` | Needs its own FROM clause |
| Row-level filtering (Filter, Sample) | `query()` | Needs WHERE/LIMIT clause |

If both `query()` and `expressions()` exist on a transformer, `query()` wins.
Base class `query()` returns `None` (not NotImplementedError) — this is the fallback signal.

## CTE Promotion Rules

**No CTE (default):** Column-level expressions compose inline.

**CTE required when:**
1. A step uses `query()` — always promoted (`__cte_0`, `__cte_1`, ...)
2. Expression depth > `_DEFAULT_CTE_DEPTH` (8) — auto-promoted for readability
3. Layer boundary during fit — new source materialized

**CTE naming:** `__cte_0`, `__cte_1`, etc. (sequential counter in `compose_transform`)

## Layer Resolution (Fit Phase Only)

Layers only matter during fit. Transform is always one SQL query.

**Layer boundary** = after any **dynamic + schema-changing** step.

```
Pipeline:
  1. Imputer          (dynamic, same schema)    ← Layer 0
  2. StandardScaler   (dynamic, same schema)    ← Layer 0
  3. OneHotEncoder    (dynamic, CHANGES schema) ← Layer 0 (boundary after)
  4. MinMaxScaler     (dynamic, same schema)    ← Layer 1
  5. Log              (static, same schema)     ← Layer 1 (free — no fit cost)
```

**Layer 0 fit:** One batched query for Imputer + StandardScaler + OneHotEncoder aggregates.
**Layer 1 fit:** One query against Layer 0's materialized output for MinMaxScaler.
Static steps contribute zero cost to the fit query.

## Schema Propagation

Schema flows through the pipeline. Most transformers don't change it.

**Schema-changing steps** must override `output_schema()`:
- OneHotEncoder: removes original, adds N binary columns
- StringSplit: adds N part columns
- Drop/Select: removes columns
- Rename: renames columns

The compiler uses `output_schema()` to detect schema changes and create layer boundaries.

## Three-Tier Classification

| Tier | What | Trust | Verification |
|---|---|---|---|
| 1 | Built-in with `_classification` set | Trusted immediately | CI-validated |
| 2 | Custom with `_classification` set | Verify once, then cache | `_verify_custom_declaration()` |
| 3 | Custom without `_classification` | Full conservative inspection | `_inspect_undeclared_step()` |

**Safety rule:** If in doubt, classify as dynamic (never miss a fit query).

Tier 2 verification: calls `discover()` and `discover_sets()`, checks consistency with declared classification. Raises `ClassificationError` on mismatch. Caches with `_classification_verified = True`.

Tier 3 inspection: calls `discover()`, `discover_sets()`, checks `fit()` override, returns static only if all checks pass.

## Error Hierarchy

```
SQLearnError (base)
├── SchemaError         — column not found, type mismatch, collision
├── FitError            — missing target, invalid data, empty table
├── CompilationError    — SQL generation failed, empty pipeline
├── ClassificationError — discover/expressions inconsistency
├── NotFittedError      — transform before fit
├── FrozenError         — modify frozen pipeline
├── InvalidStepError    — invalid step in pipeline
├── MissingColumnError  — referenced column doesn't exist
├── StaticViolationError — static step tried to learn
├── UnseenCategoryError — unseen category at transform time
└── ProFeatureError     — Studio Pro feature without license
```

## Transformer State Attributes

After `fit()`, these attributes are populated:

| Attribute | Type | Set by | Purpose |
|---|---|---|---|
| `_fitted` | `bool` | Pipeline | True after successful fit |
| `params_` | `dict[str, Any] \| None` | Pipeline (from discover) | Scalar stats |
| `sets_` | `dict[str, list[dict]] \| None` | Pipeline (from discover_sets) | Category lists |
| `columns_` | `list[str] \| None` | Pipeline | Resolved target columns |
| `input_schema_` | `Schema \| None` | Pipeline | Schema before this step |
| `output_schema_` | `Schema \| None` | Pipeline | Schema after this step |
| `_y_column` | `str \| None` | Pipeline | Target column name |
| `_owner_thread` | `int \| None` | `_check_thread()` | Thread safety guard |
| `_owner_pid` | `int \| None` | `_check_thread()` | Process safety guard |

## Pipeline API

```python
class Pipeline:
    steps: list[tuple[str, Transformer]]  # (name, transformer) pairs

    def fit(self, data, y=None, *, backend=None) -> Pipeline
    def transform(self, data, *, backend=None) -> np.ndarray
    def fit_transform(self, data, y=None, *, backend=None) -> np.ndarray
    def to_sql(self, *, dialect="duckdb") -> str
    def clone(self) -> Pipeline
```

Steps input accepts: bare list, tuple list, or dict. Normalized to `(name, Transformer)` pairs.
Auto-naming: `step_00`, `step_01`, etc. when names not provided.

## Key Design Decisions

These are settled — don't revisit without good reason:

- One `Transformer` base class (not sklearn's 8+ mixins)
- All SQL via sqlglot ASTs, never raw strings
- `y` is a column name string, not a numpy array
- `+` operator for pipeline composition (non-mutating `+=`)
- Pipelines are NOT thread-safe — use `clone()` for concurrent access
- Auto column routing via `_default_columns` attribute
- `discover()` for scalars, `discover_sets()` for category lists
- `_apply_expressions()` handles auto-passthrough of unmodified columns
- Column collision detection in `compose_transform()` (raises `SchemaError`)
- Three-tier classification with safety fallback to dynamic
