---
name: test
description: Use when writing tests, adding test cases, fixing failing tests, or when the user mentions testing. Also trigger when verifying sklearn equivalence, checking coverage, running mutation tests, validating transformer behavior, or discussing test strategy. If the user asks about testing patterns for any sqlearn component, use this skill.
disable-model-invocation: false
user-invocable: true
---

# Testing Rules — sqlearn

## Three-Tier Strategy

| Tier | Command | When | What |
|---|---|---|---|
| 1 | `make test` | During dev | `pytest --failed-first -x` — fast, stops on first failure |
| 2 | `make cov` | Before commit | `pytest --cov --cov-report=term-missing` — full coverage |
| 3 | `make test-full` | Before release | Coverage + `mutmut` mutation testing |

**Always run Tier 1 after writing tests. Run Tier 2 before committing.**

## Test Structure

Tests mirror source: `src/sqlearn/scalers/standard.py` → `tests/scalers/test_standard.py`

```
tests/
├── conftest.py              # shared fixtures (standard_dataset, schema, backend)
├── core/
│   ├── test_transformer.py
│   ├── test_pipeline.py
│   ├── test_compiler.py
│   ├── test_custom.py
│   └── test_schema.py
├── scalers/
│   ├── test_standard.py
│   └── test_minmax.py
├── encoders/
│   ├── test_onehot.py
│   └── test_ordinal.py
├── integration/
│   ├── test_sklearn_equivalence.py
│   ├── test_sql_snapshots.py
│   ├── test_sql_ast.py
│   └── test_pipeline_transformers.py
└── ...
```

## Every Transformer MUST Have These Tests

### 1. sklearn Equivalence (parameterized)

The output MUST match sklearn within floating-point tolerance:

```python
@pytest.mark.parametrize("sqlearn_cls,sklearn_cls,kwargs", [
    (sq.StandardScaler, sklearn.StandardScaler, {}),
    (sq.StandardScaler, sklearn.StandardScaler, {"with_mean": False}),
    (sq.StandardScaler, sklearn.StandardScaler, {"with_std": False}),
])
def test_sklearn_equivalence(sqlearn_cls, sklearn_cls, kwargs, standard_dataset):
    sq_result = sqlearn_cls(**kwargs).fit_transform(standard_dataset)
    sk_result = sklearn_cls(**kwargs).fit_transform(load_as_numpy(standard_dataset))
    np.testing.assert_allclose(sq_result, sk_result, atol=1e-10)
```

**This is non-negotiable.** If sqlearn produces different numbers than sklearn, it's a bug.

### 2. SQL Snapshot Test

```python
def test_standard_scaler_sql_snapshot(standard_dataset):
    pipe = sq.Pipeline([sq.StandardScaler(columns=["price"])])
    pipe.fit(standard_dataset)
    sql = pipe.to_sql()
    assert "price" in sql
    assert "42.5" in sql or "mean" in sql.lower()
```

### 3. Null Handling

```python
def test_standard_scaler_with_nulls(dataset_with_nulls):
    """NULL propagation: SQL NULL semantics must be preserved."""
    pipe = sq.Pipeline([sq.StandardScaler()])
    pipe.fit(dataset_with_nulls)
    result = pipe.transform(dataset_with_nulls)
    # NULLs in input → NaN in output (SQL NULL → numpy NaN)
    assert np.isnan(result[null_row_index, price_col_index])
```

### 4. Classification Test

```python
def test_standard_scaler_classification():
    scaler = sq.StandardScaler()
    assert scaler._classification == "dynamic"
    columns = ["price", "score"]
    schema = Schema({"price": "DOUBLE", "score": "DOUBLE"})
    result = scaler.discover(columns, schema)
    assert len(result) > 0  # dynamic must return non-empty
```

### 5. Roundtrip Test

```python
def test_standard_scaler_roundtrip(standard_dataset):
    """fit → to_sql → execute manually → same result as transform."""
    pipe = sq.Pipeline([sq.StandardScaler()])
    pipe.fit(standard_dataset)
    transform_result = pipe.transform(standard_dataset)
    sql_result = execute_raw(pipe.to_sql())
    np.testing.assert_allclose(transform_result, sql_result, atol=1e-10)
```

### 6. Clone Test

```python
def test_standard_scaler_clone(standard_dataset):
    """clone() produces independent copy with identical output."""
    pipe = sq.Pipeline([sq.StandardScaler()])
    pipe.fit(standard_dataset)
    cloned = pipe.clone()
    cloned.fit(standard_dataset)
    np.testing.assert_allclose(
        pipe.transform(standard_dataset),
        cloned.transform(standard_dataset),
    )
```

### 7. Edge Cases — EXTREME

Every transformer must be tested against these boundary conditions:

```python
def test_single_row():
    """Single row: std=0, count=1. Must not crash or produce Inf."""

def test_constant_column():
    """All values identical: std=0, variance=0. NULLIF must prevent division by zero."""

def test_empty_table():
    """Zero rows. Must raise FitError, not crash."""

def test_all_nulls_column():
    """Column where every value is NULL. Aggregates return NULL.
    Must handle gracefully — fill with NULL, not crash."""

def test_two_rows():
    """Minimal dataset: only 2 rows. Tests variance with n=2."""

def test_large_values():
    """Values near float64 limits (1e308, -1e308). Must not overflow."""

def test_tiny_values():
    """Values near zero (1e-300). Must not underflow to zero."""

def test_mixed_positive_negative():
    """Mix of large positive and large negative values."""

def test_nan_in_source():
    """NaN values in DuckDB source (distinct from NULL).
    DuckDB treats NaN as a valid float — verify behavior."""

def test_unicode_column_names():
    """Columns named '价格', 'цена', 'Ñ'. SQL must handle properly."""

def test_column_name_with_spaces():
    """Column 'unit price'. Must be quoted in SQL."""

def test_column_name_sql_keyword():
    """Column named 'select', 'from', 'group'. Must be escaped."""
```

### 8. Pickle Roundtrip

```python
def test_standard_scaler_pickle_roundtrip(standard_dataset):
    """Pickle → unpickle must preserve params_ and sets_."""
    import pickle

    pipe = sq.Pipeline([sq.StandardScaler()])
    pipe.fit(standard_dataset)
    original = pipe.transform(standard_dataset)

    restored = pickle.loads(pickle.dumps(pipe))
    np.testing.assert_allclose(original, restored.transform(standard_dataset))
```

### 9. Deep Clone Independence

```python
def test_standard_scaler_clone_independence(standard_dataset, alt_dataset):
    """Cloned pipeline re-fit must not affect original."""
    pipe = sq.Pipeline([sq.StandardScaler()])
    pipe.fit(standard_dataset)
    original_result = pipe.transform(standard_dataset)

    cloned = pipe.clone()
    cloned.fit(alt_dataset)  # re-fit with different data

    # Original must be unchanged
    np.testing.assert_allclose(original_result, pipe.transform(standard_dataset))
```

### 10. Composition Correctness

```python
def test_composition_with_prior_step():
    """Verify expressions() uses exprs[col] (composed), not Column(col) (raw).
    The most common transformer bug: using bare column ref skips prior steps."""
    pipe = sq.Pipeline([sq.Imputer(), sq.StandardScaler()])
    pipe.fit(data_with_nulls)
    sql = pipe.to_sql()
    # SQL must show COALESCE nested inside the StandardScaler expression
    assert "COALESCE" in sql
    # The COALESCE must be inside the subtraction, not beside it
    ast = sqlglot.parse_one(sql)
    # ... verify nesting structure
```

### 11. Not Fitted Guard

```python
def test_transform_before_fit_raises():
    """transform() without fit() must raise NotFittedError."""
    scaler = sq.StandardScaler()
    with pytest.raises(sq.NotFittedError):
        scaler.transform("data.parquet")
```

## Integration Tests

Located in `tests/integration/`. Test cross-transformer behavior:

### Pipeline Composition

```python
def test_full_pipeline_output():
    """Imputer + Scaler + Encoder end-to-end: must produce valid output."""

def test_pipeline_operator():
    """+ operator produces flat pipeline."""
    pipe = sq.Imputer() + sq.StandardScaler()
    assert len(pipe.steps) == 2

def test_pipeline_immutability():
    """Pipeline += creates new pipeline, doesn't mutate."""
    base = sq.Pipeline([sq.Imputer()])
    extended = base
    extended += sq.StandardScaler()
    assert len(base.steps) == 1
    assert len(extended.steps) == 2
```

### AST Structure Tests

```python
def test_imputer_scaler_ast():
    """Verify AST shape: Div(Sub(Coalesce(...), mean), Nullif(std, 0))."""
    select = _fit_and_compose(conn, "data", [Imputer(), StandardScaler()])
    exprs = _get_output_exprs(select)
    price_expr = exprs["price"]
    assert isinstance(price_expr, exp.Div)
    assert isinstance(price_expr.this, exp.Sub)
    assert isinstance(price_expr.this.this, exp.Coalesce)
```

### SQL Snapshot Tests

Use `tests/integration/snapshots/` directory for snapshot files. Compare generated SQL
against stored snapshots to catch unintended query changes.

## Property-Based Tests (hypothesis)

Use `hypothesis` for fuzzing. Ensures transformers never crash on any valid input:

```python
from hypothesis import given, strategies as st

@given(st.lists(st.floats(allow_nan=True, allow_infinity=False), min_size=1, max_size=100))
def test_standard_scaler_never_crashes(values):
    """StandardScaler must handle any valid float list without crashing."""
    # Create table, fit, transform — no exceptions allowed
    ...
```

Key properties to test:
- `inverse_transform(transform(X)) ≈ X` for invertible transforms
- `transform(fit(X))` is deterministic (run twice, same result)
- Compiled SQL is valid (parse with sqlglot, no errors)
- Random pipeline of N steps → `to_sql()` produces valid SQL

## Exhaustive Combinatorial Testing

For classification/planner logic, test all feature combinations:

```python
from itertools import combinations

ALL_COMBOS = [set(c) for r in range(1, 6) for c in combinations(range(1, 6), r)]

@pytest.mark.parametrize("features", ALL_COMBOS)
def test_classification_all_combos(features):
    """Every feature combination must classify correctly."""
```

## Mutation Testing (mutmut) — Tier 3

`make test-full` runs mutmut. Focus on:
- `compiler.py` — expression composition logic
- `expressions()` methods — arithmetic operators, CASE expressions
- `discover()` methods — aggregate function selection
- Schema change detection logic

Surviving mutants = tests that didn't catch a real code change. Fix these first.

## Test Fixtures

Standard dataset at `tests/fixtures/standard.parquet`:
- 1000 rows, 20 columns
- 5 numeric (1 skewed, 1 with outliers, 1 constant)
- 3 categorical (low/medium/high cardinality)
- 2 datetime, 1 boolean, 1 target
- Known nulls at controlled positions

Additional: `second_source.parquet`, `lookup_table.parquet`, `large_sample.parquet` (100K).

## DuckDB CSV Loading Note

When loading CSV for category tests, force `all_varchar=TRUE`:

```python
# WRONG: DuckDB may infer "1","2","3" as INTEGER
duckdb.sql("SELECT * FROM 'data.csv'")

# RIGHT: Explicit types
duckdb.sql("SELECT * FROM read_csv('data.csv', all_varchar=TRUE)")
```

## Test Naming

- `test_<transformer>_matches_sklearn` — equivalence
- `test_<transformer>_sql_snapshot` — SQL output
- `test_<transformer>_with_nulls` — null handling
- `test_<transformer>_classification` — static/dynamic
- `test_<transformer>_roundtrip` — fit → to_sql → execute
- `test_<transformer>_clone` — independent clone
- `test_<transformer>_pickle_roundtrip` — serialization
- `test_<transformer>_clone_independence` — clone doesn't affect original
- `test_<transformer>_composition` — works correctly after prior steps
- `test_<transformer>_not_fitted` — raises before fit
- `test_<transformer>_<edge_case>` — specific edge case

## Running Tests

```bash
make test          # Tier 1: fast, failed-first, stops on first failure
make cov           # Tier 2: full coverage report
make test-full     # Tier 2+3: coverage + mutation testing
pytest -k "standard_scaler"   # by name
pytest tests/scalers/          # one module
pytest tests/integration/      # integration tests only
```

## Milestone Awareness

Before writing tests:

1. **Check what exists** — `ls src/sqlearn/` to see implemented modules
2. **Match test scope to implementation** — only write tests for code that exists
3. **Check `BACKLOG.md`** for current milestone to understand what's in scope

## Full Test Catalog

See `docs/13-testing.md` for the complete test catalog including:
thread safety, y-column propagation, feature selection, auto features,
string splitting, datetime/temporal, outlier handling, target transforms,
data operations, merge/concat, discover_sets(), cross-validation schema safety,
TransformResult, freeze, drift detection, auto-passthrough, SQL fuzz,
FILTER clause validation, performance benchmarks, search, analysis,
studio, sklearn compatibility, and version matrix CI tests.
