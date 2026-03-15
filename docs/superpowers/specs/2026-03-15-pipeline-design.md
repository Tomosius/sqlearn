# Pipeline Design Spec — M2 Minimal Viable Pipeline

## Goal

Implement `Pipeline` as a thin orchestrator that wires together the existing compiler
phases (`plan_fit`, `build_fit_queries`, `compose_transform`) with the backend and IO
layers. Ships: `fit()`, `transform()`, `fit_transform()`, `to_sql()`, `+`/`+=` operators,
`clone()`, `get_feature_names_out()`.

## Scope

**In M2:** Constructor (3 input formats), fit workflow (3-phase compiler), transform
(numpy output), fit_transform (simple delegation), to_sql (dialect parameter), `+`/`+=`
with flattening, clone, repr, context manager, get_feature_names_out.

**Deferred:** TransformResult wrapper, streaming/batching, output format options
(`out="pandas"`), freeze/FrozenPipeline, detect_drift, validate, audit, null_policy,
`__getitem__`/`__setitem__` step access, nested Pipeline-as-Transformer.

## Architecture

Pipeline is NOT a Transformer subclass in M2. It is a standalone class that delegates
to the compiler for all SQL work. This keeps it thin (~250-350 lines) and focused on
orchestration.

```
User API          Pipeline.fit() / .transform() / .to_sql()
                       │
Orchestration     resolve backend → resolve input → describe schema
                       │
Compiler          plan_fit() → build_fit_queries() → compose_transform()
                       │
Execution         backend.fetch_one() / backend.execute()
                       │
Output            numpy array (M2) / SQL string
```

## Constructor

```python
class Pipeline:
    """Compile ML preprocessing steps to a single SQL query."""

    def __init__(
        self,
        steps: list[Transformer] | list[tuple[str, Transformer]] | dict[str, Transformer],
        *,
        backend: str | DuckDBBackend | None = None,
    ) -> None:
```

### Step normalization

All three input formats normalize to `list[tuple[str, Transformer]]`:

- **Bare list:** Auto-name as `step_00`, `step_01`, ..., `step_NN` (zero-padded to
  max index width).
- **Tuple list:** Use as-is. sklearn-compatible `(name, estimator)` pairs.
- **Dict:** Convert `{k: v}` to `[(k, v), ...]`, preserving insertion order.

### Validation (on init)

| Check | Error |
|---|---|
| Empty steps list | `InvalidStepError("Pipeline requires at least one step")` |
| Non-Transformer element | `InvalidStepError(f"Step {name!r} is not a Transformer: {type(obj)}")` |
| Duplicate names | `InvalidStepError(f"Duplicate step name: {name!r}")` |

### Properties

- `steps: list[tuple[str, Transformer]]` — read-only, returns defensive copy
- `named_steps: dict[str, Transformer]` — dict access to steps by name
- `is_fitted: bool` — whether fit() has been called successfully

### Internal state

```python
self._steps: list[tuple[str, Transformer]]       # normalized steps
self._backend: str | DuckDBBackend | None         # user-provided backend config
self._backend_instance: DuckDBBackend | None = None  # resolved backend (created on first use)
self._fitted: bool = False
self._schema_in: Schema | None = None             # input schema at fit time
self._schema_out: Schema | None = None            # output schema after all steps
self._owns_backend: bool = False                  # True if Pipeline created the backend
```

## fit()

```python
def fit(
    self,
    data: str | object,
    y: str | None = None,
    *,
    backend: str | DuckDBBackend | None = None,
) -> Pipeline:
```

### Workflow

1. **Resolve backend:** per-call `backend` > `self._backend` > auto-create in-memory
   DuckDBBackend. Resolution logic in a private `_resolve_backend()` method:
   ```python
   def _resolve_backend(self, backend: str | DuckDBBackend | None = None) -> DuckDBBackend:
       if backend is not None:
           if isinstance(backend, DuckDBBackend):
               return backend
           return DuckDBBackend(backend)  # str → file path
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
   ```
   String values are always file paths for DuckDBBackend, not backend type names.

2. **Resolve input:** `resolve_input(data, backend)` → source name (string).

3. **Describe schema:** `backend.describe(source)` → `Schema`. Store as `_schema_in`.

4. **Phase 1 — Plan:** `plan_fit(transformers, schema, y)` → `FitPlan`.
   `transformers` is `[step for _, step in self._steps]`.

5. **Phase 2+Execute — Per layer:**

   a. `build_fit_queries(layer, source_expr, current_exprs)` → `FitQueries`.
      - First layer: `source_expr = source` (string).
      - Subsequent layers: `source_expr = "__sq_layer_{i-1}__"` (temp view name).
      - `current_exprs`: starts as `{col: exp.Column(this=col) for col in schema}`,
        updated per step via `_apply_expressions()`.

   b. Execute `FitQueries.aggregate_query` via `backend.fetch_one()` if not None.
      Returns a single row dict of all aggregated stats for the layer.

   c. Distribute aggregate results to steps using `FitQueries.param_mapping`:
      ```python
      for alias, (step_idx_str, param_name) in param_mapping.items():
          step_idx = int(step_idx_str)
          step = layer.steps[step_idx].step
          if step.params_ is None:
              step.params_ = {}
          step.params_[param_name] = row[alias]
      ```

   d. Execute each entry in `FitQueries.set_queries` via `backend.execute()`.
      Distribute to `step.sets_`:
      ```python
      for key, query in set_queries.items():
          step_idx_str, set_name = key.split("_", 1)
          step_idx = int(step_idx_str)
          step = layer.steps[step_idx].step
          if step.sets_ is None:
              step.sets_ = {}
          step.sets_[set_name] = backend.execute(query)
      ```

   e. Mark each step as fitted:
      ```python
      for step_info in layer.steps:
          step_info.step.columns_ = step_info.columns
          step_info.step.input_schema_ = step_info.input_schema
          step_info.step.output_schema_ = step_info.step_output_schema
          step_info.step._fitted = True
      ```
      Note: `StepInfo` uses `step_output_schema` (not `output_schema`).
      `Layer` has `output_schema`. Different field names.

   f. If not the last layer, materialize layer output as temp view for next layer.
      Build a SELECT using `compose_transform()` over the layer's steps only, then
      wrap in a `CREATE TEMP VIEW` via sqlglot AST:
      ```python
      select_ast = compose_transform(layer_transformers, layer_source)
      create_ast = exp.Create(
          this=exp.Table(this=f"__sq_layer_{i}__"),
          kind="VIEW",
          expression=select_ast,
          properties=exp.Properties(expressions=[exp.TemporaryProperty()]),
      )
      backend.execute(create_ast)
      ```
      This keeps all SQL as sqlglot ASTs — no raw SQL strings. The `execute()` method
      transpiles to DuckDB SQL before execution, same as all other queries.

      After materialization, reset `current_exprs` to bare column references for the
      next layer:
      ```python
      current_exprs = {
          col: exp.Column(this=col)
          for col in layer.output_schema.columns
      }
      ```

6. **Store fitted state:**
   ```python
   self._fitted = True
   self._schema_out = final_layer.output_schema
   ```

7. **Return self** for chaining.

### Source expression for layers

- Layer 0: reads from the original source (string from resolve_input)
- Layer N>0: reads from `__sq_layer_{N-1}__` (temp view materialized after previous layer)

### Temp view naming

Pattern: `__sq_layer_{index}__` (e.g., `__sq_layer_0__`, `__sq_layer_1__`).
Session-scoped, auto-cleaned when connection closes. Follows data safety guarantee
(Section 4.6b of architecture doc).

## transform()

```python
def transform(
    self,
    data: str | object,
    *,
    backend: str | DuckDBBackend | None = None,
) -> Any:
```

### Workflow

1. Check `self._fitted` — raise `NotFittedError` if False.
2. Resolve backend (same precedence as fit).
3. Resolve input → source name.
4. `compose_transform([step for _, step in self._steps], source)` → `exp.Select`.
5. Execute via `backend.execute(ast)` → `list[dict[str, Any]]`.
6. Convert to numpy array:
   - All values to a 2D array via `numpy.array([[row[col] for col in columns] for row in rows])`
   - Attempt `float64` dtype; fall back to `object` if mixed types.
7. Return numpy array.

### Numpy conversion

```python
import numpy as np

columns = list(rows[0].keys()) if rows else self.get_feature_names_out()
data = [[row[col] for col in columns] for row in rows]
try:
    return np.array(data, dtype=np.float64)
except (ValueError, TypeError):
    return np.array(data, dtype=object)
```

Empty result (no rows) returns `np.empty((0, len(columns)), dtype=np.float64)`.

## fit_transform()

```python
def fit_transform(
    self,
    data: str | object,
    y: str | None = None,
    *,
    backend: str | DuckDBBackend | None = None,
) -> Any:
```

M2 implementation: `return self.fit(data, y, backend=backend).transform(data, backend=backend)`.

Single-pass optimization deferred to later milestone.

## to_sql()

```python
def to_sql(
    self,
    *,
    dialect: str = "duckdb",
    table: str = "__input__",
) -> str:
```

### Workflow

1. Check `self._fitted` — raise `NotFittedError`.
2. `compose_transform([step for _, step in self._steps], table)` → AST.
3. Return `ast.sql(dialect=dialect)`.

No backend needed — pure AST compilation and transpilation.

`table` parameter lets user control the source reference in generated SQL:
- `pipe.to_sql(table="my_table")` → `SELECT ... FROM my_table`
- Default `__input__` is a placeholder.

## Operators

### __add__

```python
def __add__(self, other: object) -> Pipeline:
```

Flattening rules:
- `Pipeline + Transformer` → `Pipeline([*self._steps, ("step_N", other)])`
- `Pipeline + Pipeline` → `Pipeline([*self._steps, *other._steps])`
- Name collision → `InvalidStepError`
- `other` not Transformer/Pipeline → `NotImplemented`

Auto-naming for bare transformers appended: `step_NN` where NN continues from
the highest existing index. Uses the `_auto_name(steps, transformer)` helper:

```python
def _auto_name(existing: list[tuple[str, Transformer]], step: Transformer) -> str:
    """Generate next step_NN name based on existing steps."""
    max_idx = -1
    for name, _ in existing:
        if name.startswith("step_") and name[5:].isdigit():
            max_idx = max(max_idx, int(name[5:]))
    idx = max_idx + 1
    width = max(2, len(str(idx)))
    return f"step_{idx:0{width}d}"
```

### __radd__

```python
def __radd__(self, other: object) -> Pipeline:
```

Handles `Transformer + Pipeline` case. Returns `Pipeline([(_auto_name(self._steps, other), other), *self._steps])`.

### Transformer.__add__ and __iadd__

Update the existing stubs in transformer.py:

```python
def __add__(self, other: object) -> Pipeline | NotImplementedType:
    from sqlearn.core.pipeline import Pipeline
    if isinstance(other, Pipeline):
        return Pipeline([(_auto_name(other.steps, self), self), *other.steps])  # use public property
    if isinstance(other, Transformer):
        return Pipeline([self, other])  # both auto-named
    return NotImplemented

def __iadd__(self, other: object) -> Pipeline | NotImplementedType:
    return self.__add__(other)
```

Note: uses `other.steps` (public property, returns defensive copy) not `other._steps`
(private attribute). Avoids base class coupling to Pipeline internals.

### Pipeline.__iadd__

```python
def __iadd__(self, other: object) -> Pipeline:
    return self.__add__(other)
```

Non-mutating. Returns new Pipeline.

## clone()

```python
def clone(self) -> Pipeline:
```

Creates independent copy:
- Each step cloned via `step.clone()`
- Preserves step names
- Preserves fitted state (`_fitted`, `_schema_in`, `_schema_out`)
- Does NOT copy backend — new Pipeline gets `backend=None` (lazy reconnect)
- `_owns_backend = False` on clone

## get_feature_names_out()

```python
def get_feature_names_out(self) -> list[str]:
```

Returns `list(self._schema_out.columns.keys())`. Raises `NotFittedError` if not fitted.

## __repr__

```python
def __repr__(self) -> str:
```

Format: `Pipeline(step_00=Imputer, step_01=StandardScaler, step_02=OneHotEncoder)`

Uses `type(step).__name__` for class names.

## Context Manager

```python
def __enter__(self) -> Pipeline:
    return self

def __exit__(self, *exc: object) -> None:
    if self._owns_backend and isinstance(self._backend_instance, DuckDBBackend):
        self._backend_instance.close()
```

Only closes backend if Pipeline created it. User-provided backends are never closed.

`_backend_instance` is the resolved DuckDBBackend (created during first fit/transform).
Distinct from `_backend` (the user-provided config: str, DuckDBBackend, or None).

## Error Summary

| Situation | Error | Method |
|---|---|---|
| Empty steps | `InvalidStepError` | `__init__` |
| Non-Transformer step | `InvalidStepError` | `__init__` |
| Duplicate names | `InvalidStepError` | `__init__` |
| Name collision on `+` | `InvalidStepError` | `__add__` |
| transform before fit | `NotFittedError` | `transform()` |
| to_sql before fit | `NotFittedError` | `to_sql()` |
| get_feature_names_out before fit | `NotFittedError` | `get_feature_names_out()` |
| Empty query result | `FitError` | `fit()` (via backend) |
| Column collision | `SchemaError` | `fit()` (via compiler) |
| Classification mismatch | `ClassificationError` | `fit()` (via compiler) |
| Unsupported input type | `TypeError` | `fit()`/`transform()` (via `resolve_input`) |

No new error classes needed.

### Temp view cleanup on fit() failure

If `fit()` fails partway through multi-layer execution (e.g., layer 1 temp view created,
layer 2 fails), temp views may be left behind. For auto-created backends
(`_owns_backend = True`), this is harmless — connection close cleans everything. For
user-provided backends, temp views accumulate but are session-scoped and do not affect
source data. Explicit cleanup via `try/finally` is not needed in M2 — the data safety
guarantee holds regardless.

## File Structure

- **Create:** `src/sqlearn/core/pipeline.py` (~250-350 lines)
- **Modify:** `src/sqlearn/core/__init__.py` — export Pipeline
- **Modify:** `src/sqlearn/__init__.py` — export Pipeline
- **Modify:** `src/sqlearn/core/transformer.py` — update `__add__`/`__iadd__` stubs
- **Create:** `tests/core/test_pipeline.py`

## Dependencies

Pipeline imports from existing modules only:
- `sqlearn.core.compiler` — `plan_fit`, `build_fit_queries`, `compose_transform`
- `sqlearn.core.backend` — `DuckDBBackend`
- `sqlearn.core.io` — `resolve_input`
- `sqlearn.core.schema` — `Schema`
- `sqlearn.core.transformer` — `Transformer`
- `sqlearn.core.errors` — `NotFittedError`, `InvalidStepError`, `FitError`
- `numpy`

## Extension Points (Not Implemented in M2)

These are designed for but not shipped:

- **TransformResult:** `transform()` return type will change from `np.ndarray` to
  `TransformResult` (wraps numpy with `.columns`, `.sql`, `.dtypes` metadata).
- **Output formats:** `out=` parameter on `transform()` for pandas/polars/arrow/relation.
- **Streaming:** `batch_size=` parameter on `transform()` for chunked iteration.
- **freeze():** Returns `FrozenPipeline` (immutable, pre-compiled).
- **null_policy:** Pipeline-level NULL handling strategy.
- **Nesting:** Pipeline implements Transformer protocol for nested pipelines.
- **Step access:** `__getitem__`/`__setitem__` for `pipe["scale"]` access.
