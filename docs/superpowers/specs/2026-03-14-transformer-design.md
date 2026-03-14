# Transformer Base Class Design

**Module:** `src/sqlearn/core/transformer.py`
**Milestone:** 2 — Core Compiler (v0.1.0)
**Issue:** #5
**Dependencies:** `sqlearn.core.schema` (Schema, ColumnSelector, resolve_columns)

## Purpose

The Transformer base class is the single class all sqlearn transformers extend. It replaces
sklearn's 8+ mixins with one unified interface. Provides method signatures for subclass
override (`discover`, `expressions`, etc.), classification logic, sklearn-compatible
introspection, pipeline composition operators, and thread safety guards.

For Milestone 2, `fit()`/`transform()`/`to_sql()` are stubs — they become real when
Pipeline (#7) and Compiler (#6) land. The base class defines the interface and
implements everything that doesn't require those systems.

## Class Definition

```python
class Transformer:
    """Base class for all sqlearn transformers.

    Subclasses override discover(), expressions(), and optionally
    discover_sets(), query(), and output_schema() to define behavior.
    """

    # --- Class attributes (set by subclasses) ---
    _default_columns: str | None = None       # "all", "numeric", "categorical", "temporal", or None
    _classification: str | None = None        # "static", "dynamic", or None (auto-detect)

    def __init__(self, *, columns: str | list[str] | ColumnSelector | None = None) -> None:
        """Initialize transformer.

        Args:
            columns: Column specification override. If provided, takes precedence
                over _default_columns. Accepts column names, type strings,
                or ColumnSelector objects. Resolved against schema at fit time.
        """
```

### Class attributes

| Attribute | Type | Default | Purpose |
|-----------|------|---------|---------|
| `_default_columns` | `str \| None` | `None` | Auto column routing. Subclasses set to `"numeric"`, `"categorical"`, `"temporal"`, `"all"`, or leave `None` (explicit only). |
| `_classification` | `str \| None` | `None` | `"static"`, `"dynamic"`, or `None` (auto-detect at fit time). Built-in transformers must declare. |

### Instance attributes (set by `__init__`)

| Attribute | Type | Purpose |
|-----------|------|---------|
| `_columns_spec` | `str \| list[str] \| ColumnSelector \| None` | User's `columns=` override, stored until fit resolves it. |
| `_fitted` | `bool` | Whether `fit()` has been called. Starts `False`. |
| `_owner_thread` | `int \| None` | Thread ID from first `fit()`/`transform()` call. |
| `_owner_pid` | `int \| None` | Process ID from first `fit()`/`transform()` call. |
| `_connection` | `duckdb.DuckDBPyConnection \| None` | DuckDB connection. Lazily created, nulled on pickle. |

### Fitted attributes (set by Pipeline.fit, initially None)

| Attribute | Type | Purpose |
|-----------|------|---------|
| `params_` | `dict[str, Any] \| None` | Scalar learned values from `discover()`. |
| `sets_` | `dict[str, list[dict[str, Any]]] \| None` | Set-valued learned values from `discover_sets()`. Each entry is a list of row dicts with column names as keys. |
| `columns_` | `list[str] \| None` | Resolved target columns for this transformer. |
| `input_schema_` | `Schema \| None` | Input table schema. |
| `output_schema_` | `Schema \| None` | Output schema after this step. |
| `_y_column` | `str \| None` | Target column name if provided to fit. |

## Methods — Subclass Override

Subclasses override these to define behavior. `discover()` and `discover_sets()` default
to empty (static). `expressions()` raises `NotImplementedError` — subclasses must implement
either `expressions()` or `query()` (or both).

### Subclass `__init__` pattern

Subclasses add their own keyword parameters and call `super().__init__()`:

```python
class StandardScaler(Transformer):
    _default_columns = "numeric"
    _classification = "dynamic"

    def __init__(self, *, with_mean: bool = True, columns: str | list[str] | ColumnSelector | None = None) -> None:
        super().__init__(columns=columns)
        self.with_mean = with_mean
```

`get_params()` introspects the subclass `__init__` signature, not just the base class.

### discover()

```python
def discover(
    self,
    columns: list[str],
    schema: Schema,
    y_column: str | None = None,
) -> dict[str, exp.Expression]:
    """Learn scalar statistics from data via SQL aggregates.

    Override to return {param_name: sqlglot_aggregate} mappings.
    Results are executed as SQL and stored in self.params_.

    Default returns {} (static — no learning).

    Param naming convention: '{col}__{stat}' (e.g. 'price__mean').
    Must return sqlglot AST nodes, never raw strings or Python values.
    """
    return {}
```

### discover_sets()

```python
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
    """
    return {}
```

### expressions()

```python
def expressions(
    self,
    columns: list[str],
    exprs: dict[str, exp.Expression],
) -> dict[str, exp.Expression]:
    """Generate inline SQL column expressions.

    Args:
        columns: Target columns this transformer operates on.
        exprs: Current expression dict for ALL columns. Start as bare
            Column nodes, wrapped by each preceding step.

    Returns:
        Dict of ONLY modified/new columns. Unmentioned columns pass
        through automatically via base class.

    Default raises NotImplementedError. Subclasses that use only query()
    should override to return {}.
    Must return sqlglot AST nodes, never raw strings.
    """
    raise NotImplementedError
```

### query()

```python
def query(
    self,
    input_query: exp.Expression,
) -> exp.Expression | None:
    """Generate a full query wrapping input (window functions, joins, CTEs).

    Alternative to expressions() for transforms needing query-level control.
    Returns None to fall back to expressions().

    Default returns None.
    """
    return None
```

### output_schema()

```python
def output_schema(self, schema: Schema) -> Schema:
    """Declare output schema after this step.

    Override when adding, removing, renaming, or retyping columns.
    Default returns input schema unchanged.
    """
    return schema
```

## Methods — Base Class Provides

### is_fitted (property)

```python
@property
def is_fitted(self) -> bool:
    """Clean boolean. No trailing-underscore scanning."""
    return self._fitted
```

### _apply_expressions()

```python
def _apply_expressions(self, exprs: dict[str, exp.Expression]) -> dict[str, exp.Expression]:
    """Base class wrapper around expressions(). Called by the compiler, not by users.

    1. Calls self.expressions(self.columns_, exprs)
    2. Merges result with untouched columns from exprs (auto-passthrough)
    3. Detects column name collisions (error) and undeclared new columns (warn)
    4. Removes columns that output_schema() says are dropped
    """
```

### _resolve_columns_spec()

```python
def _resolve_columns_spec(self) -> str | list[str] | ColumnSelector | None:
    """Return the effective column spec (user override or class default).

    Returns _columns_spec if user passed columns=, else _default_columns.
    Actual resolution against schema happens at fit time via resolve_columns().
    """
```

### _classify() — Static/Dynamic Detection

Three-tier classification:

```python
def _classify(self) -> str:
    """Classify this transformer as 'static' or 'dynamic'.

    Tier 1 (built-in, _classification set): trust declaration.
    Tier 2 (custom, _classification set): verify on first fit.
    Tier 3 (custom, _classification=None): inspect discover()/discover_sets().

    Safety rule: if in doubt, classify as dynamic. Static is an optimization;
    false static = data corruption. False dynamic = one extra cheap query.
    """
```

For Milestone 2, only Tier 1 and Tier 3 are needed. `_classify()` lives on
Transformer and only checks `self._classification` (Tier 1) or inspects
`discover()`/`discover_sets()` return values (Tier 3). It does not need pipeline
knowledge. Tier 2 verification (custom with declaration) moves to the pipeline
module when custom transformer validation lands.

### get_params() / set_params()

```python
def get_params(self, deep: bool = True) -> dict[str, Any]:
    """Return __init__ parameters as dict. sklearn-compatible.

    Introspects __init__ signature. For nested transformers (Pipeline),
    uses '__' separator: 'scaler__with_mean'.
    """

def set_params(self, **params: Any) -> Transformer:
    """Set parameters. Returns self. sklearn-compatible."""
```

### get_feature_names_out()

```python
def get_feature_names_out(self) -> list[str]:
    """Return output column names. Requires fitted state.

    Returns list(self.output_schema_.columns.keys()).
    Raises ValueError if not fitted.
    """
```

### clone() / copy()

```python
def clone(self) -> Transformer:
    """Create independent copy. Thread-safe (new connection).

    Deep copies: params_, sets_, columns_, input_schema_, output_schema_.
    Resets _owner_thread and _owner_pid to None.
    Creates new DuckDB connection (fully independent).
    Used by sq.Search for parallel training.
    """

def copy(self) -> Transformer:
    """Deep copy via copy.deepcopy(). Shares connection reference.
    NOT thread-safe — use clone() for cross-thread independence."""
```

### _check_thread()

```python
def _check_thread(self) -> None:
    """Guard against cross-thread/cross-process access.

    Stores _owner_thread + _owner_pid on first call. Raises on
    subsequent call from different thread/process.
    Suggests clone() in error message.
    """
```

### __add__ / __iadd__

```python
def __add__(self, other: Transformer) -> Pipeline:
    """Sequential composition: a + b -> Pipeline([a, b])."""

def __iadd__(self, other: Transformer) -> Pipeline:
    """Incremental composition: pipe += step -> NEW Pipeline.
    Non-mutating (follows Python numeric convention)."""
```

Import of Pipeline is deferred (inside method body) to avoid circular imports.

### __repr__

```python
def __repr__(self) -> str:
    """Readable repr: StandardScaler(columns='numeric')."""
```

Shows class name + non-default `__init__` params.

### _repr_html_

```python
def _repr_html_(self) -> str:
    """Rich HTML repr for Jupyter notebooks.
    Shows transformer name, params, fitted state, and column routing."""
```

### __sklearn_is_fitted__

```python
def __sklearn_is_fitted__(self) -> bool:
    """sklearn compatibility: returns self._fitted."""
```

## Stubs (not implemented until dependencies land)

These methods exist with correct signatures but raise `NotImplementedError` until
Pipeline (#7), Compiler (#6), and Backend (#4) are implemented:

- `fit(data, y=None, *, backend=None)` → returns `self`
- `transform(data, *, out="numpy", backend=None, batch_size=None, dtype=None, exclude_target=True)` → returns `TransformResult`
- `fit_transform(data, y=None, **kwargs)` → convenience
- `to_sql(*, dialect="duckdb", table="__input__")` → returns `str`
- `freeze()` → returns `FrozenPipeline`

Each stub has the full docstring with Args/Returns so the API is documented even
before implementation.

## Serialization

```python
def __getstate__(self) -> dict:
    """Null out DuckDB connection before pickling."""

def __setstate__(self, state: dict) -> None:
    """Restore from pickle. Connection lazily recreated."""
```

## What is NOT in transformer.py

- **Pipeline** — separate module (`pipeline.py`, issue #7)
- **Compiler / expression composition** — separate module (`compiler.py`, issue #6)
- **Backend / IO** — separate modules (issue #4)
- **TransformResult** — separate module (`output.py`)
- **Custom transformer validation** — added when `custom.py` lands
- **FrozenPipeline** — Milestone 7
- **Fit warnings** (high cardinality, zero variance, etc.) — implemented in Pipeline.fit,
  not in the base class. The base class just defines the interface.

`transformer.py` depends only on `schema.py` and stdlib (`inspect`, `threading`, `os`, `copy`).

## Public API Surface

Exported from `sqlearn/__init__.py`:

- `Transformer` — the base class

Not exported (internal):
- `_classify()`, `_check_thread()`, `_resolve_columns_spec()` — implementation details
