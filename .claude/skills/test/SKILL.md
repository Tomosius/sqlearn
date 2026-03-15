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

## Pipeline Composition Stress Tests

These tests verify the compiler and pipeline handle complex real-world scenarios.
Every scenario here has caused bugs in similar systems. Test ALL of them.

### 12. Long Pipeline Chains

```python
@pytest.mark.parametrize("n_steps", [3, 5, 10, 15, 20])
def test_long_pipeline_chain(n_steps, standard_dataset):
    """N sequential steps must compose correctly without stack overflow or AST corruption."""
    steps = [sq.StandardScaler(columns=["price"])] * n_steps
    pipe = sq.Pipeline(steps)
    pipe.fit(standard_dataset)
    result = pipe.transform(standard_dataset)
    sql = pipe.to_sql()
    # Must produce valid SQL
    sqlglot.parse_one(sql, dialect="duckdb")
    # Must produce finite values
    assert np.all(np.isfinite(result) | np.isnan(result))
```

### 13. Mixed Static + Dynamic Pipeline Ordering

The classifier must correctly handle interleaved static and dynamic steps:

```python
@pytest.mark.parametrize("pipeline_steps", [
    # All static
    [sq.Log(), sq.Rename({"price": "cost"})],
    # All dynamic
    [sq.Imputer(), sq.StandardScaler()],
    # Static → Dynamic
    [sq.Log(), sq.StandardScaler()],
    # Dynamic → Static
    [sq.StandardScaler(), sq.Log()],
    # Interleaved: S → D → S → D
    [sq.Log(), sq.Imputer(), sq.Rename({"price": "cost"}), sq.StandardScaler()],
    # Dynamic → Dynamic → Dynamic (chain of data-dependent steps)
    [sq.Imputer(), sq.StandardScaler(), sq.MinMaxScaler()],
])
def test_mixed_static_dynamic_ordering(pipeline_steps, standard_dataset):
    """Pipeline must handle any mix of static/dynamic steps in any order."""
    pipe = sq.Pipeline(pipeline_steps)
    pipe.fit(standard_dataset)
    result = pipe.transform(standard_dataset)
    assert result.shape[0] == expected_rows
    sql = pipe.to_sql()
    sqlglot.parse_one(sql, dialect="duckdb")
```

### 14. Dependent Dynamic Steps

When step B's expressions() depends on stats computed by step A, composition must be correct:

```python
def test_dependent_dynamic_chain():
    """Imputer fills NULLs → StandardScaler uses filled values for mean/std.
    The SQL must reflect that StandardScaler's input is COALESCE(...), not raw column."""
    pipe = sq.Pipeline([sq.Imputer(), sq.StandardScaler()])
    pipe.fit(data_with_nulls)
    sql = pipe.to_sql()
    ast = sqlglot.parse_one(sql, dialect="duckdb")
    # StandardScaler's expression must wrap Imputer's COALESCE
    # i.e., (COALESCE(price, fill) - mean) / NULLIF(std, 0)
    # NOT: (price - mean) / NULLIF(std, 0)  ← this is the bug
    assert "COALESCE" in sql

def test_three_dependent_dynamic_steps():
    """Imputer → StandardScaler → MinMaxScaler: triple nesting."""
    pipe = sq.Pipeline([sq.Imputer(), sq.StandardScaler(), sq.MinMaxScaler()])
    pipe.fit(standard_dataset)
    sql = pipe.to_sql()
    # Verify the MinMaxScaler wraps StandardScaler which wraps Imputer
    ast = sqlglot.parse_one(sql, dialect="duckdb")
    # The expression tree depth should be at least 3 levels

def test_encoder_after_imputer():
    """Imputer fills categorical NULLs → OneHotEncoder uses filled values.
    Categories discovered must include the fill value, not NULL."""
    pipe = sq.Pipeline([sq.Imputer(strategy="most_frequent"), sq.OneHotEncoder()])
    pipe.fit(data_with_null_categories)
    # Verify encoder sees filled values, not NULLs
```

### 15. Classification Edge Cases

The static/dynamic classification system must handle all these correctly:

```python
def test_custom_transformer_empty_discover():
    """Custom transformer with discover() returning empty dict.
    Should be classified as static even if discover() exists."""

def test_custom_transformer_classification_mismatch():
    """Custom transformer claiming static but implementing discover().
    Validation must catch this on first fit()."""
    class BadTransformer(sq.Transformer):
        _classification = "static"
        def discover(self, columns, schema, y_column=None):
            return {"mean": exp.Avg(this=exp.Column(this="price"))}
        def expressions(self, columns, exprs):
            return {}
    pipe = sq.Pipeline([BadTransformer()])
    with pytest.raises(sq.ValidationError):
        pipe.fit(standard_dataset)

def test_conditional_classification_static_path():
    """Transformer that is conditionally static (literal args)."""
    clip = sq.Clip(lower=0, upper=100)
    assert clip._classification == "static"

def test_conditional_classification_dynamic_path():
    """Same transformer that is conditionally dynamic (percentile args)."""
    clip = sq.Clip(lower="p5", upper="p95")
    assert clip._classification == "dynamic"

def test_both_discover_and_discover_sets():
    """Transformer implementing both discover() AND discover_sets().
    Must execute both and populate params_ and sets_."""

def test_pipeline_all_static_no_queries():
    """Pipeline of only static steps must not run any discovery queries."""
    pipe = sq.Pipeline([sq.Log(), sq.Rename({"a": "b"})])
    # fit() should be fast — no SQL queries executed for stats

def test_pipeline_single_dynamic_among_statics():
    """One dynamic step among 5 static steps.
    Only the dynamic step's discover() should run."""
    pipe = sq.Pipeline([sq.Log(), sq.Log(), sq.StandardScaler(), sq.Log(), sq.Log()])
```

### 16. Custom Transformer Stress Tests

Users will create custom transformers in unexpected ways. Test all combinations:

```python
def test_custom_sql_expression_static():
    """sq.Expression('price * qty AS revenue') — static, no learning."""
    expr = sq.Expression("price * qty AS revenue")
    pipe = sq.Pipeline([expr])
    pipe.fit(standard_dataset)
    assert "revenue" in pipe.get_feature_names_out()

def test_custom_with_discover():
    """sq.custom() with learn=True — must discover stats and use them."""
    scaler = sq.custom(
        "({col} - {mean}) / NULLIF({std}, 0)",
        columns="numeric",
        learn={"mean": "AVG({col})", "std": "STDDEV_POP({col})"},
    )
    pipe = sq.Pipeline([scaler])
    pipe.fit(standard_dataset)
    result = pipe.transform(standard_dataset)
    # Compare with sklearn StandardScaler
    sk = sklearn.StandardScaler().fit_transform(load_as_numpy(standard_dataset))
    np.testing.assert_allclose(result, sk, atol=1e-10)

def test_custom_after_builtin():
    """Custom transformer after built-in: must receive composed expressions."""
    pipe = sq.Pipeline([sq.Imputer(), sq.custom("{col} * 2", columns="numeric")])
    pipe.fit(data_with_nulls)
    sql = pipe.to_sql()
    # Custom must multiply COALESCE result, not raw column
    assert "COALESCE" in sql

def test_custom_between_builtins():
    """Built-in → custom → built-in: custom must compose in both directions."""
    pipe = sq.Pipeline([
        sq.Imputer(),
        sq.custom("{col} + 1", columns="numeric"),
        sq.StandardScaler(),
    ])
    pipe.fit(data_with_nulls)
    sql = pipe.to_sql()
    # StandardScaler must wrap (COALESCE(...) + 1), not just COALESCE(...)
```

### 17. Schema Propagation Stress Tests

```python
def test_schema_after_column_add():
    """Transformer adding columns: downstream steps must see new columns."""
    pipe = sq.Pipeline([
        sq.Expression("price * qty AS revenue"),
        sq.StandardScaler(),  # must auto-detect 'revenue' as numeric
    ])
    pipe.fit(standard_dataset)
    assert "revenue" in pipe.get_feature_names_out()

def test_schema_after_column_drop():
    """Transformer dropping columns: downstream steps must NOT see dropped columns."""
    pipe = sq.Pipeline([
        sq.Drop(columns=["city"]),
        sq.OneHotEncoder(),  # must not try to encode 'city'
    ])

def test_schema_column_rename_propagation():
    """Rename in step 1 → step 2 references new name, not old name."""
    pipe = sq.Pipeline([
        sq.Rename({"price": "cost"}),
        sq.StandardScaler(columns=["cost"]),  # uses new name
    ])
    pipe.fit(standard_dataset)
    result = pipe.transform(standard_dataset)
    assert "cost" in pipe.get_feature_names_out()
    assert "price" not in pipe.get_feature_names_out()
```

## Cross-Library Validation

Compare sqlearn output against established libraries to verify correctness.
Every sqlearn transformer with a sklearn equivalent MUST have a cross-validation test.

### sklearn Equivalence Matrix

```python
SKLEARN_EQUIVALENCE = [
    # (sqlearn_class, sklearn_class, kwargs, tolerance)
    (sq.StandardScaler, sklearn.StandardScaler, {}, 1e-10),
    (sq.StandardScaler, sklearn.StandardScaler, {"with_mean": False}, 1e-10),
    (sq.StandardScaler, sklearn.StandardScaler, {"with_std": False}, 1e-10),
    (sq.MinMaxScaler, sklearn.MinMaxScaler, {}, 1e-10),
    (sq.MinMaxScaler, sklearn.MinMaxScaler, {"feature_range": (-1, 1)}, 1e-10),
    (sq.RobustScaler, sklearn.RobustScaler, {}, 1e-10),
    (sq.MaxAbsScaler, sklearn.MaxAbsScaler, {}, 1e-10),
    (sq.Imputer, sklearn.SimpleImputer, {"strategy": "mean"}, 1e-10),
    (sq.Imputer, sklearn.SimpleImputer, {"strategy": "median"}, 1e-10),
    (sq.OneHotEncoder, sklearn.OneHotEncoder, {"sparse_output": False}, 0),
    (sq.OrdinalEncoder, sklearn.OrdinalEncoder, {}, 0),
]

@pytest.mark.parametrize("sq_cls,sk_cls,kwargs,atol", SKLEARN_EQUIVALENCE)
def test_sklearn_cross_validation(sq_cls, sk_cls, kwargs, atol, standard_dataset):
    """Every transformer MUST match sklearn within tolerance."""
    sq_result = sq_cls(**kwargs).fit_transform(standard_dataset)
    sk_result = sk_cls(**kwargs).fit_transform(load_as_numpy(standard_dataset))
    np.testing.assert_allclose(sq_result, sk_result, atol=atol)
```

### scipy Statistical Validation

For statistical correctness independent of sklearn:

```python
import scipy.stats

def test_standard_scaler_matches_scipy():
    """Verify mean/std against scipy.stats, not just sklearn."""
    pipe = sq.Pipeline([sq.StandardScaler(columns=["price"])])
    pipe.fit(standard_dataset)
    scipy_mean = scipy.stats.tmean(raw_prices)
    scipy_std = np.std(raw_prices, ddof=0)  # population std
    assert abs(pipe.steps[0].params_["price__mean"] - scipy_mean) < 1e-10
    assert abs(pipe.steps[0].params_["price__std"] - scipy_std) < 1e-10

def test_robust_scaler_matches_scipy():
    """Verify median/IQR against scipy.stats."""
    pipe = sq.Pipeline([sq.RobustScaler(columns=["price"])])
    pipe.fit(standard_dataset)
    scipy_median = np.median(raw_prices)
    scipy_iqr = scipy.stats.iqr(raw_prices)
    assert abs(pipe.steps[0].params_["price__median"] - scipy_median) < 1e-10
    assert abs(pipe.steps[0].params_["price__iqr"] - scipy_iqr) < 1e-10
```

### DuckDB Direct Comparison

Verify sqlearn's SQL produces same results as hand-written DuckDB SQL:

```python
def test_standard_scaler_matches_duckdb_direct():
    """sqlearn SQL output must match hand-written equivalent SQL."""
    pipe = sq.Pipeline([sq.StandardScaler(columns=["price"])])
    pipe.fit(standard_dataset)
    sqlearn_result = pipe.transform(standard_dataset)

    hand_sql = """
    SELECT (price - (SELECT AVG(price) FROM data))
           / NULLIF((SELECT STDDEV_POP(price) FROM data), 0) AS price
    FROM data
    """
    duckdb_result = duckdb.sql(hand_sql).fetchnumpy()["price"]
    np.testing.assert_allclose(sqlearn_result[:, 0], duckdb_result, atol=1e-10)
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

@given(st.lists(st.text(min_size=0, max_size=50), min_size=1, max_size=50))
def test_onehot_encoder_never_crashes(categories):
    """OneHotEncoder must handle any string list without crashing."""
    # Including empty strings, unicode, SQL keywords

@given(st.lists(
    st.one_of(st.floats(allow_nan=True), st.none()),
    min_size=1, max_size=100,
))
def test_imputer_never_crashes(values_with_nulls):
    """Imputer must handle any mix of floats and NULLs."""
```

Key properties to test:
- `inverse_transform(transform(X)) ≈ X` for invertible transforms
- `transform(fit(X))` is deterministic (run twice, same result)
- Compiled SQL is valid (parse with sqlglot, no errors)
- Random pipeline of N steps → `to_sql()` produces valid SQL

### Random Pipeline Fuzzing

```python
ALL_TRANSFORMERS = [sq.Imputer, sq.StandardScaler, sq.MinMaxScaler, sq.Log]

@given(st.lists(
    st.sampled_from(ALL_TRANSFORMERS), min_size=1, max_size=8,
))
def test_random_pipeline_produces_valid_sql(transformer_classes):
    """Any random combination of transformers must produce parseable SQL."""
    steps = [cls() for cls in transformer_classes]
    pipe = sq.Pipeline(steps)
    pipe.fit(standard_dataset)
    sql = pipe.to_sql()
    sqlglot.parse_one(sql, dialect="duckdb")  # must not raise

@given(st.lists(
    st.sampled_from(ALL_TRANSFORMERS), min_size=2, max_size=5,
))
def test_random_pipeline_fit_transform_twice_deterministic(transformer_classes):
    """Same pipeline fit+transform twice must produce identical output."""
    steps = [cls() for cls in transformer_classes]
    pipe = sq.Pipeline(steps)
    pipe.fit(standard_dataset)
    r1 = pipe.transform(standard_dataset)
    r2 = pipe.transform(standard_dataset)
    np.testing.assert_array_equal(r1, r2)
```

## Exhaustive Combinatorial Testing

For classification/planner logic, test all feature combinations:

```python
from itertools import combinations, permutations

ALL_COMBOS = [set(c) for r in range(1, 6) for c in combinations(range(1, 6), r)]

@pytest.mark.parametrize("features", ALL_COMBOS)
def test_classification_all_combos(features):
    """Every feature combination must classify correctly."""

# Test every permutation of 3 transformer types
PERM_STEPS = list(permutations([sq.Imputer, sq.StandardScaler, sq.OneHotEncoder], 3))

@pytest.mark.parametrize("step_classes", PERM_STEPS)
def test_pipeline_permutation_order(step_classes, mixed_dataset):
    """Every ordering of steps must produce valid SQL (even if semantically odd)."""
    pipe = sq.Pipeline([cls() for cls in step_classes])
    pipe.fit(mixed_dataset)
    sql = pipe.to_sql()
    sqlglot.parse_one(sql, dialect="duckdb")
```

## Mutation Testing — Tier 3

### mutmut (Primary)

`make test-full` runs mutmut. Focus areas where surviving mutants are most dangerous:

```bash
# Run against specific high-risk modules
mutmut run --paths-to-mutate=src/sqlearn/core/compiler.py
mutmut run --paths-to-mutate=src/sqlearn/core/transformer.py
mutmut run --paths-to-mutate=src/sqlearn/scalers/
mutmut run --paths-to-mutate=src/sqlearn/encoders/
```

**Critical mutation targets:**
- `compiler.py` — expression composition, CTE promotion, layer resolution
- `transformer.py` — classification logic, validation, _apply_expressions
- `expressions()` methods — arithmetic operators, CASE expressions, NULLIF
- `discover()` methods — aggregate function selection
- `output_schema()` — column add/drop logic
- Schema propagation between steps

**Interpreting results:**
- Surviving mutant = a code change that tests didn't catch
- Fix by adding a test that specifically catches that mutation
- Zero surviving mutants in compiler.py and transformer.py is the goal

### cosmic-ray (Alternative)

For broader mutation coverage when mutmut misses patterns:

```bash
# Install: pip install cosmic-ray
cosmic-ray init config.toml src/sqlearn/core/
cosmic-ray exec config.toml
cosmic-ray report config.toml
```

### When to Run Mutation Testing

| Situation | Action |
|---|---|
| New transformer added | Run mutmut on the transformer file |
| Compiler logic changed | Run mutmut on compiler.py — zero survivors required |
| Before release | Full `make test-full` — report surviving mutants |
| CI/CD | Tier 3 runs nightly, blocks release if critical mutations survive |

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
