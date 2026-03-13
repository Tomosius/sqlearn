---
name: test
description: Use when writing tests, adding test cases, fixing failing tests, or when the user mentions testing
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Write, Edit, Bash, Grep, Glob
---

# Testing Rules — sqlearn

## Test Structure

Tests mirror source: `src/sqlearn/scalers/standard.py` → `tests/scalers/test_standard.py`

```
tests/
├── conftest.py              # shared fixtures (standard_dataset, schema, backend)
├── core/
│   ├── test_transformer.py
│   ├── test_pipeline.py
│   ├── test_compiler.py
│   ├── test_custom.py       # sq.custom() and sq.Expression()
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

### 1. sklearn Equivalence Test

The output MUST match sklearn within floating-point tolerance:

```python
def test_standard_scaler_matches_sklearn(standard_dataset):
    """sqlearn output must match sklearn output exactly."""
    # sqlearn
    sq_pipe = sq.Pipeline([sq.StandardScaler()])
    sq_pipe.fit(standard_dataset)
    sq_result = sq_pipe.transform(standard_dataset)

    # sklearn
    from sklearn.preprocessing import StandardScaler as SkStandardScaler
    sk_scaler = SkStandardScaler()
    sk_result = sk_scaler.fit_transform(load_as_numpy(standard_dataset))

    np.testing.assert_allclose(sq_result, sk_result, atol=1e-10)
```

**This is non-negotiable.** If sqlearn produces different numbers than sklearn for the same operation, it's a bug.

### 2. SQL Snapshot Test

The compiled SQL must match expected output:

```python
def test_standard_scaler_sql_snapshot(standard_dataset):
    """Compiled SQL must match expected form."""
    pipe = sq.Pipeline([sq.StandardScaler(columns=["price"])])
    pipe.fit(standard_dataset)
    sql = pipe.to_sql()
    assert "price" in sql
    assert "42.5" in sql or "mean" in sql.lower()  # learned value appears as literal
    # OR use snapshot testing:
    # assert sql == snapshot  (with pytest-snapshot or inline)
```

### 3. Null Handling Test

```python
def test_standard_scaler_with_nulls(dataset_with_nulls):
    """Nulls must propagate correctly (SQL NULL semantics)."""
    pipe = sq.Pipeline([sq.StandardScaler()])
    pipe.fit(dataset_with_nulls)
    result = pipe.transform(dataset_with_nulls)
    # NULL input → NULL output (SQL semantics)
    assert np.isnan(result[null_row_index, price_col_index])
```

### 4. Classification Test (for built-in transformers)

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

### 5. Edge Cases

```python
def test_standard_scaler_single_row():
    """Must handle single-row input without division by zero."""

def test_standard_scaler_constant_column():
    """Constant column has std=0 — must not produce Inf/NaN."""

def test_standard_scaler_empty_table():
    """Empty input must raise FitError, not crash."""
```

## Pipeline Tests

```python
def test_pipeline_composition():
    """Imputer + Scaler + Encoder compiles to one SQL query."""
    pipe = sq.Pipeline([sq.Imputer(), sq.StandardScaler(), sq.OneHotEncoder()])
    pipe.fit("tests/fixtures/standard.parquet", y="target")
    sql = pipe.to_sql()
    assert sql.count("SELECT") == 1  # one query, no CTEs
    assert sql.count("WITH") == 0

def test_pipeline_output_matches_sklearn():
    """Full pipeline output matches equivalent sklearn pipeline."""

def test_pipeline_operator():
    """+ operator produces flat pipeline."""
    pipe = sq.Imputer() + sq.StandardScaler()
    assert len(pipe.steps) == 2

def test_pipeline_immutability():
    """Pipeline += creates new pipeline, doesn't mutate."""
    base = sq.Pipeline([sq.Imputer()])
    extended = base
    extended += sq.StandardScaler()
    assert len(base.steps) == 1      # unchanged
    assert len(extended.steps) == 2   # new pipeline
```

## Custom Transformer Tests

```python
def test_sq_custom_static():
    """sq.custom() without learn= is static."""
    log = sq.custom("LN({col} + 1)", columns="numeric")
    assert log._classification == "static"

def test_sq_custom_dynamic():
    """sq.custom() with learn= is dynamic."""
    center = sq.custom("{col} - {mean}", columns="numeric", learn={"mean": "AVG({col})"})
    assert center._classification == "dynamic"

def test_sq_custom_validation():
    """sq.custom() catches bad SQL at creation time."""
    with pytest.raises(sqlglot.errors.ParseError):
        sq.custom("INVALID SQL {{{{", columns="numeric")

def test_custom_transformer_type_validation():
    """Custom transformer returning strings instead of ASTs is caught."""
    class Bad(sq.Transformer):
        def expressions(self, columns, exprs):
            return {"price": "price - 42"}  # string, not AST
    pipe = sq.Pipeline([Bad()])
    with pytest.raises(TypeError, match="must return sqlglot expressions"):
        pipe.fit(standard_dataset)
```

## Test Fixtures

Use the standard fixture from `tests/fixtures/standard.parquet`:
- 1000 rows, 20 columns
- 5 numeric (1 skewed, 1 outliers, 1 constant)
- 3 categorical (low/medium/high cardinality)
- 2 datetime, 1 comma-separated, 1 JSON, 1 email, 1 URL, 1 IP
- 1 boolean, 1 ID, 1 target
- Known nulls at controlled positions
- Known distributions (verifiable means/stds)

## Running Tests

```bash
pytest                           # all tests
pytest tests/scalers/            # one module
pytest -x                        # stop on first failure
pytest -k "standard_scaler"      # by name
pytest --cov=sqlearn             # with coverage
```

## Test Naming

- `test_<transformer>_matches_sklearn` — equivalence
- `test_<transformer>_sql_snapshot` — SQL output
- `test_<transformer>_with_nulls` — null handling
- `test_<transformer>_classification` — static/dynamic
- `test_<transformer>_<edge_case>` — specific edge case
