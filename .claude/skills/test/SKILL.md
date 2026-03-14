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
└── ...
```

## Every Transformer Test Must Include

### 1. sklearn Equivalence (parameterized)

The output MUST match sklearn within floating-point tolerance. Use parameterized tests:

```python
@pytest.mark.parametrize("sqlearn_cls,sklearn_cls,kwargs", [
    (sq.StandardScaler, sklearn.StandardScaler, {}),
    (sq.MinMaxScaler, sklearn.MinMaxScaler, {}),
    (sq.RobustScaler, sklearn.RobustScaler, {}),
    (sq.Imputer, sklearn.SimpleImputer, {"strategy": "mean"}),
])
def test_sklearn_equivalence(sqlearn_cls, sklearn_cls, kwargs, standard_dataset):
    """sqlearn output must match sklearn within floating-point tolerance."""
    sq_result = sqlearn_cls(**kwargs).fit_transform(standard_dataset)
    sk_result = sklearn_cls(**kwargs).fit_transform(load_as_numpy(standard_dataset))
    np.testing.assert_allclose(sq_result, sk_result, atol=1e-10)
```

**This is non-negotiable.** If sqlearn produces different numbers than sklearn, it's a bug.

### 2. SQL Snapshot Test

```python
def test_standard_scaler_sql_snapshot(standard_dataset):
    """Compiled SQL must match expected form."""
    pipe = sq.Pipeline([sq.StandardScaler(columns=["price"])])
    pipe.fit(standard_dataset)
    sql = pipe.to_sql()
    assert "price" in sql
    assert "42.5" in sql or "mean" in sql.lower()
```

### 3. Null Handling

```python
def test_standard_scaler_with_nulls(dataset_with_nulls):
    """Nulls must propagate correctly (SQL NULL semantics)."""
    pipe = sq.Pipeline([sq.StandardScaler()])
    pipe.fit(dataset_with_nulls)
    result = pipe.transform(dataset_with_nulls)
    assert np.isnan(result[null_row_index, price_col_index])
```

### 4. Classification Test

```python
def test_standard_scaler_classification():
    """Built-in _classification must match discover() reality."""
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
    """fit -> to_sql -> execute manually -> same result as transform."""
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

### 7. Edge Cases

```python
def test_standard_scaler_single_row():
    """Must handle single-row input without division by zero."""

def test_standard_scaler_constant_column():
    """Constant column has std=0 — must not produce Inf/NaN."""

def test_standard_scaler_empty_table():
    """Empty input must raise FitError, not crash."""
```

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
- `+` operator is associative: `(a + b) + c` same SQL as `a + (b + c)`
- Random pipeline of N steps → `to_sql()` produces valid SQL

## Exhaustive Combinatorial Testing

For classification/planner logic, test all feature combinations:

```python
from itertools import combinations

ALL_COMBOS = [set(c) for r in range(1, 6) for c in combinations(range(1, 6), r)]

@pytest.mark.parametrize("features", ALL_COMBOS)
def test_classification_all_combos(features):
    """Every feature combination must classify correctly."""
    step = FeatureComboEstimator(features)
    result = _classify_step(step, ...)
    expected = "dynamic" if features else "static"
    assert result.kind == expected
```

## Pipeline Tests

```python
def test_pipeline_composition():
    """Imputer + Scaler + Encoder compiles to one SQL query."""
    pipe = sq.Pipeline([sq.Imputer(), sq.StandardScaler(), sq.OneHotEncoder()])
    pipe.fit("tests/fixtures/standard.parquet", y="target")
    sql = pipe.to_sql()
    assert sql.count("SELECT") == 1
    assert sql.count("WITH") == 0

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

def test_pipeline_output_matches_sklearn():
    """Full pipeline output matches equivalent sklearn pipeline."""
```

## Mutation Testing (mutmut) — Tier 3

`make test-full` runs mutmut to verify test quality. Mutmut generates code mutations
(`>` → `>=`, `+` → `-`, `True` → `False`) and verifies tests catch every mutation.

```bash
uv run mutmut run --paths-to-mutate=src/sqlearn/
uv run mutmut results  # view surviving mutants
```

Focus mutation testing on:
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
- 2 datetime (1 DATE, 1 TIMESTAMP), 1 comma-separated, 1 JSON
- 1 email, 1 URL, 1 IP, 1 boolean, 1 ID, 1 target
- Known nulls at controlled positions, known distributions

Additional: `second_source.parquet` (merge), `lookup_table.parquet` (Lookup),
`2023.parquet`/`2024.parquet` (concat), `large_sample.parquet` (100K rows, benchmarks).

## DuckDB CSV Loading Note

When loading CSV for category tests, force `all_varchar=TRUE`:

```python
# WRONG: DuckDB may infer "1","2","3" as INTEGER, losing category semantics
duckdb.sql("SELECT * FROM 'data.csv'")

# RIGHT: Explicit types, use Cast() where needed
duckdb.sql("SELECT * FROM read_csv('data.csv', all_varchar=TRUE)")
```

## Test Naming

- `test_<transformer>_matches_sklearn` — equivalence
- `test_<transformer>_sql_snapshot` — SQL output
- `test_<transformer>_with_nulls` — null handling
- `test_<transformer>_classification` — static/dynamic
- `test_<transformer>_roundtrip` — fit → to_sql → execute
- `test_<transformer>_clone` — independent clone
- `test_<transformer>_<edge_case>` — specific edge case

## Running Tests

```bash
make test          # Tier 1: fast, failed-first, stops on first failure
make cov           # Tier 2: full coverage report
make test-full     # Tier 2+3: coverage + mutation testing
pytest -k "standard_scaler"   # by name
pytest tests/scalers/          # one module
```

## Milestone Awareness

Not all test patterns apply yet. Before writing tests:

1. **Check what exists** — `ls src/sqlearn/` to see implemented modules
2. **Match test scope to implementation** — only write tests for code that exists
3. **Skip fixture references** for unbuilt components (e.g., `standard.parquet` may not exist yet)
4. **Start simple** — for new modules, begin with basic unit tests before adding property-based or mutation tests
5. **Check `BACKLOG.md`** for current milestone to understand what's in scope

The patterns above are the target for each transformer once the core is built. Early milestones
focus on core infrastructure tests (schema, compiler, pipeline) which follow simpler patterns.

## Full Test Catalog

See `docs/13-testing.md` for the complete test catalog including:
thread safety, y-column propagation, feature selection, auto features,
string splitting, datetime/temporal, outlier handling, target transforms,
data operations, merge/concat, discover_sets(), cross-validation schema safety,
TransformResult, freeze, drift detection, auto-passthrough, SQL fuzz,
FILTER clause validation, performance benchmarks, search, analysis,
studio, sklearn compatibility, and version matrix CI tests.
