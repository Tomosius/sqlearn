# Schema Module Design

**Module:** `src/sqlearn/core/schema.py`
**Milestone:** 2 — Core Compiler (v0.1.0)
**Dependencies:** stdlib only (`dataclasses`, `fnmatch`)

## Purpose

Schema is the foundation of sqlearn's type-aware column routing. It maps column names to
DuckDB SQL type strings and provides immutable mutation methods for schema propagation
through pipeline steps. Column selectors resolve against Schema to determine which columns
each transformer operates on.

## Schema Dataclass

Frozen dataclass. All mutation methods return new instances.

```python
@dataclass(frozen=True)
class Schema:
    """Immutable table schema mapping column names to SQL types."""

    columns: dict[str, str]  # {"price": "DOUBLE", "city": "VARCHAR"}

    # --- Mutation (returns new Schema) ---

    def add(self, new: dict[str, str]) -> Schema:
        """Add columns. Raises ValueError if any already exist."""

    def drop(self, cols: list[str]) -> Schema:
        """Remove columns. Raises KeyError if any don't exist."""

    def rename(self, mapping: dict[str, str]) -> Schema:
        """Rename columns. Raises KeyError if old name doesn't exist."""

    def cast(self, col: str | dict[str, str], new_type: str | None = None) -> Schema:
        """Change column type(s). Single: cast("col", "INT"). Batch: cast({"a": "INT", "b": "VARCHAR"})."""

    def select(self, cols: list[str]) -> Schema:
        """Keep only these columns, preserving order. Raises KeyError if missing."""

    # --- Query ---

    def column_category(self, col: str) -> str:
        """Return category: 'numeric', 'categorical', 'temporal', 'boolean', or 'other'.

        Uses _normalize_type() internally — 'DECIMAL(18,3)' matches 'numeric'.
        Raises KeyError if col doesn't exist.
        """

    def numeric(self) -> list[str]:
        """Return all numeric column names."""

    def categorical(self) -> list[str]:
        """Return all categorical column names."""

    def temporal(self) -> list[str]:
        """Return all temporal column names."""

    def boolean(self) -> list[str]:
        """Return all boolean column names."""

    # --- Dunder ---

    def __len__(self) -> int: ...
    def __contains__(self, col: str) -> bool: ...
    def __getitem__(self, col: str) -> str: ...   # schema["price"] -> "DOUBLE"
    def __iter__(self) -> Iterator[str]: ...       # iterate column names
    def __repr__(self) -> str: ...                 # Schema(price=DOUBLE, city=VARCHAR)
```

### Immutability

`frozen=True` on the dataclass. The `columns` dict is shallow-copied on construction via
`__post_init__` to prevent external mutation. Mutation methods construct new `Schema`
instances — just `Schema(new_dict)`.

### Column ordering

Column order is preserved (Python dict insertion order). This matters for output — the
final SELECT must emit columns in a predictable order.

## Type Categories

Module-level frozensets. Source of truth for auto column routing.

```python
NUMERIC_TYPES: frozenset[str] = frozenset({
    "TINYINT", "SMALLINT", "INTEGER", "INT", "INT4", "INT2", "INT1",
    "BIGINT", "INT8", "LONG", "HUGEINT", "SIGNED",
    "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT", "UHUGEINT",
    "FLOAT", "FLOAT4", "DOUBLE", "FLOAT8",
    "DECIMAL", "NUMERIC", "REAL",
})

CATEGORICAL_TYPES: frozenset[str] = frozenset({
    "VARCHAR", "TEXT", "STRING", "CHAR", "BPCHAR", "ENUM",
})

TEMPORAL_TYPES: frozenset[str] = frozenset({
    "DATE", "TIME", "TIMETZ", "TIME WITH TIME ZONE",
    "TIMESTAMP", "TIMESTAMPTZ", "TIMESTAMP WITH TIME ZONE",
    "TIMESTAMP_S", "TIMESTAMP_MS", "TIMESTAMP_NS",
    "INTERVAL",
})

BOOLEAN_TYPES: frozenset[str] = frozenset({
    "BOOLEAN", "BOOL", "LOGICAL",
})
```

### Type normalization

DuckDB returns parameterized types like `DECIMAL(18,3)`. Before matching against category
sets, strip parameters:

```python
def _normalize_type(sql_type: str) -> str:
    """Strip parameters from SQL type: 'DECIMAL(18,3)' -> 'DECIMAL'."""
    paren = sql_type.find("(")
    base = sql_type[:paren] if paren != -1 else sql_type
    return base.strip().upper()
```

This function is module-private. `Schema.__post_init__` does NOT normalize stored types —
it preserves the original DuckDB type string (e.g. `"DECIMAL(18,3)"`). Normalization
happens only during category matching in `column_category()` and selector resolution.

## Column Selectors

### Class hierarchy

```python
class ColumnSelector:
    """Base class for column selection criteria."""

    def resolve(self, schema: Schema) -> list[str]:
        """Resolve to concrete column names from schema."""
        ...

class TypeSelector(ColumnSelector):
    """Selects columns matching a type category."""

    def __init__(self, types: frozenset[str], name: str) -> None: ...

class PatternSelector(ColumnSelector):
    """Selects columns matching a glob pattern (fnmatch, not regex).

    Regex matching (e.g. Drop(pattern="^id_")) is handled by individual transformers,
    not by the selector system.
    """

    def __init__(self, pattern: str) -> None: ...

class DTypeSelector(ColumnSelector):
    """Selects columns matching a SQL type (both sides normalized).

    sq.dtype("DECIMAL") matches DECIMAL, DECIMAL(18,3), DECIMAL(10,2).
    sq.dtype("DECIMAL(18,3)") also normalizes to DECIMAL, matching all DECIMAL variants.
    """

    def __init__(self, sql_type: str) -> None: ...
```

### Factory functions

```python
def numeric() -> TypeSelector: ...
def categorical() -> TypeSelector: ...
def temporal() -> TypeSelector: ...
def boolean() -> TypeSelector: ...
def matching(pattern: str) -> PatternSelector: ...
def dtype(sql_type: str) -> DTypeSelector: ...
```

These are exported at `sq.numeric()`, `sq.categorical()`, etc. via `sqlearn/__init__.py`.

### Selector repr

Each selector has a readable repr for debugging: `numeric()`, `matching('price_*')`,
`dtype('DOUBLE')`.

## resolve_columns()

Unified resolution function handling all the ways columns can be specified:

```python
def resolve_columns(
    schema: Schema,
    columns: str | list[str] | ColumnSelector | None,
) -> list[str]:
    """Resolve column specification to concrete column names.

    Args:
        schema: Current table schema.
        columns: Column specification:
            - str literal ("numeric", "categorical", "temporal", "boolean"):
              type-based filtering, used by _default_columns on transformers.
            - "all": returns every column regardless of type (list(schema)).
            - list[str]: explicit column names, validated against schema.
            - ColumnSelector: calls .resolve(schema).
            - None: raises ValueError. Transformers with _default_columns = None
              require explicit columns from the user. The caller (Pipeline) should
              check for None before calling and raise a user-facing error.

    Returns:
        Ordered list of column names.

    Raises:
        ValueError: If columns is None or string literal is not recognized.
        KeyError: If explicit column names don't exist in schema.
    """
```

This means `_default_columns = "numeric"` on a transformer and `columns=sq.numeric()` from
user code both flow through `resolve_columns()` — string vs object, same result.

## What is NOT in schema.py

- **`read_schema()`** — lives in `io.py` (depends on backend/DuckDB connection)
- **Schema errors** — lives in `errors.py` (ValueError/KeyError suffice for now)
- **`TransformResult.dtypes`** — lives in `output.py` (consumes Schema)

`schema.py` is a pure leaf module with zero sqlearn dependencies.

## Public API Surface

Exported from `sqlearn/__init__.py`:

- `Schema` — the dataclass
- `ColumnSelector` — base class (for type hints in user code)
- `numeric()`, `categorical()`, `temporal()`, `boolean()` — type selectors
- `matching()`, `dtype()` — pattern/type selectors
- `resolve_columns()` — not exported (internal, used by Pipeline/Transformer)

Module-level constants (`NUMERIC_TYPES`, etc.) are accessible via
`sqlearn.core.schema.NUMERIC_TYPES` but not re-exported at package level.
