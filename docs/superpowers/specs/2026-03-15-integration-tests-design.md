# Integration Tests Design — Issue #9

## Goal

Comprehensive integration tests validating sqlearn transformers produce correct output via four test layers: sklearn numeric equivalence, SQL golden file snapshots, inline SQL pattern assertions, and AST structure verification.

## Scope

Tests cover the three M2 transformers: StandardScaler, Imputer, OneHotEncoder — individually and combined in pipelines.

## Test Layers

### 1. sklearn Equivalence (`test_sklearn_equivalence.py`)

Parameterized tests comparing sqlearn output to scikit-learn with `np.testing.assert_allclose(rtol=1e-6)`.

| sqlearn | sklearn | Data |
|---|---|---|
| `StandardScaler()` | `StandardScaler()` | Pure numeric, 7 rows |
| `StandardScaler(with_mean=False)` | `StandardScaler(with_mean=False)` | Pure numeric |
| `StandardScaler(with_std=False)` | `StandardScaler(with_std=False)` | Pure numeric |
| `Imputer(strategy="mean")` | `SimpleImputer(strategy="mean")` | Numeric with NULLs |
| `Imputer(strategy="median")` | `SimpleImputer(strategy="median")` | Numeric, odd row count |
| `Imputer(strategy="most_frequent")` | `SimpleImputer(strategy="most_frequent")` | Categorical with NULLs |
| `OneHotEncoder()` | `OneHotEncoder(sparse_output=False)` | Pure categorical |

Edge cases: constant columns (std=0), single row, all NULLs in one column, single unique category.

**Dependency:** `scikit-learn>=1.4` added to dev dependencies.

**Notes:**
- Both sqlearn and sklearn StandardScaler use population std (ddof=0) — should match exactly.
- Use odd row counts for median tests to avoid interpolation differences.
- OneHotEncoder comparison on pure categorical data to avoid column-routing differences. Both sort categories alphabetically.
- For Imputer most_frequent, use data with a clear mode (no ties).

### 2. SQL Golden File Snapshots (`test_sql_snapshots.py`)

Store full `to_sql()` output in `tests/integration/snapshots/*.sql`. Tests fail if SQL changes.

**Snapshots:**
- `standard_scaler.sql` — solo StandardScaler on 2-column numeric data
- `imputer_mean.sql` — Imputer(strategy="mean") on numeric data with NULLs
- `imputer_auto.sql` — Imputer() on mixed numeric+categorical data
- `onehot_encoder.sql` — solo OneHotEncoder on categorical data
- `full_pipeline.sql` — Pipeline([Imputer(), StandardScaler(), OneHotEncoder()])
- `composition.sql` — Pipeline([Imputer(strategy="mean"), StandardScaler()]) showing expression nesting

**Update mechanism:** `--update-snapshots` pytest CLI flag via conftest. When passed, tests regenerate snapshot files instead of comparing. Without it, tests compare and fail on mismatch.

**Snapshot format:** Raw SQL string from `to_sql(dialect="duckdb")`, stripped and normalized.

### 3. Inline SQL Pattern Assertions (in `test_sql_snapshots.py`)

Alongside golden files, assert key SQL patterns are present:

- StandardScaler SQL: `-`, `/`, `NULLIF`
- Imputer SQL: `COALESCE`
- OneHotEncoder SQL: `CASE`, `WHEN`, `THEN 1`, `ELSE 0`
- Full pipeline: all patterns present, `FROM __input__`

### 4. AST Structure Tests (`test_sql_ast.py`)

Verify sqlglot expression tree shape via `compose_transform()` output.

**Assertions:**
- StandardScaler: output expressions are `Div(Paren(Sub(...)), Nullif(...))`
- Imputer: output expressions are `Coalesce(Column, Literal)`
- OneHotEncoder: output expressions are `Case(If(EQ(...), Literal(1)), default=Literal(0))`
- Imputer+Scaler composition: `Div(Paren(Sub(Coalesce(...),...)), Nullif(...))` — nested, not separate
- No CTE nodes for expression-only pipelines (Imputer+Scaler)
- Verify expression depth doesn't exceed expected bounds

### 5. Shared Fixtures (`tests/conftest.py`)

Project-level conftest with reusable fixtures:

- `sample_numeric` — DuckDB table, 7 rows, 3 numeric columns, some NULLs (odd count for median)
- `sample_mixed` — numeric + categorical + NULLs
- `sample_categorical` — pure categorical columns (3+ categories per column)
- `backend` — DuckDBBackend from in-memory connection
- `snapshot_dir` — path to `tests/integration/snapshots/`
- `update_snapshots` — bool from `--update-snapshots` CLI flag

## File Structure

```
tests/
├── conftest.py                              # shared fixtures + --update-snapshots flag
└── integration/
    ├── test_pipeline_transformers.py        # existing (unchanged)
    ├── test_sklearn_equivalence.py          # sklearn comparison
    ├── test_sql_snapshots.py                # golden files + inline patterns
    ├── test_sql_ast.py                      # AST structure verification
    └── snapshots/                           # golden SQL files
        ├── standard_scaler.sql
        ├── imputer_mean.sql
        ├── imputer_auto.sql
        ├── onehot_encoder.sql
        ├── full_pipeline.sql
        └── composition.sql
```

## Dependencies

Add to `pyproject.toml` dev dependencies:
- `scikit-learn>=1.4`
