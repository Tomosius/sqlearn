---
name: core
description: Provides architecture guidance for sqlearn core modules — compiler, schema, backend, pipeline, IO, output, and errors. Trigger when working on expression composition, CTE promotion, layer resolution, schema propagation, the fit/transform lifecycle, or any file in src/sqlearn/core/. Essential for milestone 2 and all compiler-related work.
disable-model-invocation: false
user-invocable: true
---

# Core Architecture — sqlearn

Reference for working on `src/sqlearn/core/`. Read the relevant design doc for deep details:
- `docs/03-architecture.md` — Transformer base class, classification, discover/expressions
- `docs/04-compiler.md` — Expression composition, CTEs, layers

## Module Responsibilities

| Module | Purpose | Key class/function |
|---|---|---|
| `schema.py` | Column types, type detection, schema diffing | `Schema` dataclass |
| `transformer.py` | Base class all transformers extend | `Transformer` |
| `backend.py` | Database protocol + DuckDB implementation | `Backend` protocol |
| `io.py` | Input resolution (file/table/DataFrame → DuckDB) | `resolve_input()` |
| `compiler.py` | Expression composition, CTE promotion, layers | `Compiler` |
| `pipeline.py` | fit/transform/to_sql orchestration | `Pipeline` |
| `output.py` | DuckDB result → numpy/arrow/pandas/polars | `TransformResult` |
| `errors.py` | Error hierarchy | `SQLearnError` and subclasses |

## The Transformer Lifecycle

```
fit(data, y="target")
  │
  ├── 1. IO: resolve input → DuckDB table/view
  ├── 2. Schema: detect column types from source
  ├── 3. Classify: static or dynamic per step (Section 4.3 of docs/03)
  ├── 4. Layer: group steps into layers (boundary = schema-changing dynamic step)
  ├── 5. Discover: run batched aggregate queries per layer
  │     ├── discover() → scalar params → self.params_
  │     └── discover_sets() → set values → self.sets_
  └── 6. Store: params_ and sets_ populated, ready for transform

transform(data)
  │
  ├── 1. IO: resolve input
  ├── 2. Compile: chain all expressions() into one SELECT
  │     ├── Start with bare column refs: {"price": Column("price")}
  │     ├── Each step transforms the expression dict
  │     └── Steps using query() get CTE treatment
  └── 3. Execute: run compiled SQL, wrap result as TransformResult
```

## Expression Composition (The Core Innovation)

Each transformer's `expressions()` receives the current expression dict and returns modified entries.
The compiler chains them — output of step N becomes input to step N+1:

```
Step 0 (bare):     {"price": Column("price")}
Step 1 (Imputer):  {"price": Coalesce(Column("price"), 42.5)}
Step 2 (Scaler):   {"price": (Coalesce(Column("price"), 42.5) - 42.5) / NULLIF(12.3, 0)}
```

Final SQL: `SELECT (COALESCE(price, 42.5) - 42.5) / NULLIF(12.3, 0) AS price FROM data`

Three Python steps → one SQL query. No intermediate materialization.

## When to Use expressions() vs query()

| Situation | Method | Why |
|---|---|---|
| Column-level math (scale, encode, clip) | `expressions()` | Composes inline, no CTE needed |
| Window functions (LAG, RANK, rolling) | `query()` | Windows can't nest inside expressions |
| JOINs (Lookup, merge) | `query()` | Needs its own FROM clause |
| Row-level filtering (Filter, Sample) | `query()` | Needs WHERE/LIMIT clause |
| Cross-column operations (Normalizer L2) | `query()` | References multiple columns in one expression |

If both `query()` and `expressions()` exist on a transformer, `query()` wins.

## CTE Promotion Rules

CTEs aren't bad, but fewer = better DuckDB optimization.

**No CTE (default):** Column-level expressions compose inline.

**CTE required when:**
1. A step uses `query()` (window functions, JOINs, filtering)
2. Expression nesting depth > 8 (auto-promoted for readability)
3. Complex expressions referenced by multiple downstream branches (deduplication)

**Never promote:** Bare column refs (`Column("price")`) — they're free to duplicate.

## Layer Resolution (Fit Phase Only)

Layers only matter during fit. Transform is always one SQL query.

**Layer boundary** = after any schema-changing dynamic step.

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

**Total fit: 2-4 queries for the entire pipeline** (vs sklearn's N full data passes).

## Schema Propagation

Schema flows through the pipeline. Most transformers don't change it.

**Schema-changing steps** must override `output_schema()`:
- OneHotEncoder: removes original, adds N binary columns
- StringSplit: adds N part columns
- Drop/Select: removes columns
- Rename: renames columns

The compiler uses `output_schema()` to know what columns exist for the next step.

## Static vs Dynamic Classification

| Classification | `discover()` returns | Fit cost | Example |
|---|---|---|---|
| Static | `{}` (empty) | Zero | Log, Rename, Cast, Filter |
| Dynamic | `{name: aggregate}` | One query per layer | StandardScaler, Imputer, OneHotEncoder |

Built-in transformers declare `_classification` as a class variable.
Custom transformers leave it as `None` → auto-detected at first `fit()`.

Some transformers are **conditionally dynamic** — classification depends on constructor args:
- `Clip(lower=0, upper=100)` → static (literal values)
- `Clip(lower="p01", upper="p99")` → dynamic (needs percentiles from data)

## Columns & Union Compilation

**`Columns`** (parallel column routing) → compiles to one SELECT, no CTEs:
- Each branch sees only its target columns
- Branch expressions merged into single SELECT
- Column name collisions impossible (disjoint columns enforced)
- Exception: if a branch uses `query()`, that branch gets its own CTE

**`Union`** (parallel feature combination) → CTEs per branch:
- Each branch may produce different columns
- Name collision → prefix with branch name: `base__price`, `poly__price`
- Optimization: if ALL branches are expression-only, merge into one SELECT

## Two-Phase Discovery (for CV)

1. **Phase 1:** Schema from full data (one DESCRIBE query)
2. **Phase 2:** Values per fold (N fold-specific discover queries)

This ensures rare categories seen in training folds are preserved across all folds.

## Error Hierarchy

```
SQLearnError (base)
├── SchemaError      — column not found, type mismatch
├── FitError         — missing target, invalid data, empty table
├── CompilationError — SQL generation failed
├── ClassificationError — discover/expressions inconsistency
└── ProFeatureError  — Studio Pro feature without license
```

All errors should include helpful context and suggest solutions.

## Key Design Decisions

These are settled — don't revisit without good reason:

- One `Transformer` base class (not sklearn's 8+ mixins)
- All SQL via sqlglot ASTs, never raw strings
- `y` is a column name string, not a numpy array
- `+` operator for pipeline composition (non-mutating `+=`)
- `TransformResult` uses `__array__` protocol (not numpy subclass)
- Pipelines are NOT thread-safe — use `clone()` for concurrent access
- Auto column routing via `_default_columns` attribute
- `discover()` for scalars, `discover_sets()` for category lists
- FILTER clause aggregation for fold-specific CV discovery
