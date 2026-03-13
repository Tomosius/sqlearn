---
name: transformer
description: Use when creating a new transformer, scaler, encoder, imputer, feature transform, or any pipeline step
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Write, Edit, Bash, Grep, Glob
---

# Writing a New Transformer — sqlearn

## Before You Start

1. Read `CLAUDE.md` for project overview
2. Decide which level (see below)
3. Decide which folder it belongs in

## Three Levels

| Need | Level | Where |
|---|---|---|
| One-liner static expression | `sq.Expression()` | No new file needed |
| Per-column, optionally learns stats | `sq.custom()` | No new file needed |
| Full control (sets, CTEs, joins, windows) | `Transformer` subclass | New file in correct folder |

**Only create a new file for Level 3.**

## Folder Mapping

| Type | Folder | Examples |
|---|---|---|
| Scaler | `src/sqlearn/scalers/` | StandardScaler, MinMaxScaler |
| Encoder | `src/sqlearn/encoders/` | OneHotEncoder, TargetEncoder |
| Imputer | `src/sqlearn/imputers/` | Imputer |
| Feature | `src/sqlearn/features/` | Log, DateParts, StringSplit |
| Selection | `src/sqlearn/feature_selection/` | Drop, SelectKBest |
| Data op | `src/sqlearn/ops/` | Rename, Cast, Filter |

## Checklist for New Transformer (Level 3)

### 1. Create the file

`src/sqlearn/<folder>/<name>.py`

```python
import sqlglot.expressions as exp
from sqlearn.core.transformer import Transformer

class MyTransformer(Transformer):
    _default_columns = "numeric"    # "numeric", "categorical", "temporal", "all", or None
    _classification = "dynamic"     # "static" or "dynamic" — REQUIRED for built-ins

    def __init__(self, param=default):
        self.param = param
        # If classification depends on args:
        # self._classification = "static" if isinstance(param, int) else "dynamic"
```

### 2. Implement required methods

**Static transformer (no learning):**

```python
class Log(Transformer):
    _default_columns = "numeric"
    _classification = "static"

    def expressions(self, columns, exprs):
        # Return ONLY modified columns. Untouched columns pass through automatically.
        return {col: exp.Ln(exp.Add(this=exprs[col], expression=exp.Literal.number(1)))
                for col in columns}
```

**Dynamic transformer (learns from data):**

```python
class StandardScaler(Transformer):
    _default_columns = "numeric"
    _classification = "dynamic"

    def discover(self, columns, schema, y_column=None):
        # Return {param_name: sqlglot_aggregate}
        result = {}
        for col in columns:
            result[f"{col}__mean"] = exp.Avg(this=exp.Column(this=col))
            result[f"{col}__std"] = exp.StddevPop(this=exp.Column(this=col))
        return result

    def expressions(self, columns, exprs):
        # Learned values are in self.params_
        result = {}
        for col in columns:
            mean = self.params_[f"{col}__mean"]
            std = self.params_[f"{col}__std"]
            result[col] = exp.Div(
                this=exp.Sub(this=exprs[col], expression=exp.Literal.number(mean)),
                expression=exp.Nullif(
                    this=exp.Literal.number(std),
                    expression=exp.Literal.number(0)
                )
            )
        return result
```

**Schema-changing transformer (adds/removes columns):**

```python
def output_schema(self, schema):
    # REQUIRED when expressions() adds new column names
    return schema.add({f"{col}_flag": "INTEGER" for col in self.columns_})
    # Or for dropping: return schema.remove(["col_to_drop"])
```

**Query-level transformer (window functions, joins):**

```python
def query(self, input_query):
    # Returns sqlglot SELECT wrapping the input
    # Used for: window functions, JOINs, CTEs
    # If both query() and expressions() exist, query() wins
    return exp.Select(
        expressions=[exp.Star(), ...window_exprs...]
    ).from_(exp.Subquery(this=input_query))
```

### 3. Export from `__init__.py`

```python
# src/sqlearn/<folder>/__init__.py
from .my_transformer import MyTransformer
```

### 4. Write tests

Create `tests/<folder>/test_my_transformer.py` with ALL of:

- [ ] sklearn equivalence test (`np.testing.assert_allclose`)
- [ ] SQL snapshot test
- [ ] Null handling test
- [ ] Classification test (`_classification` matches `discover()` reality)
- [ ] Edge cases (single row, constant column, empty table)

See `/test` skill for full test patterns.

## Key Rules

**All SQL via sqlglot ASTs.** Never raw strings. This enables multi-database support.

```python
# WRONG — raw string
return {"price": "price - 42.5"}

# RIGHT — sqlglot AST
return {"price": exp.Sub(this=exprs["price"], expression=exp.Literal.number(42.5))}
```

**Return only modified columns from `expressions()`.** Unmentioned columns pass through automatically via `_apply_expressions()`. Never manually pass through columns.

**Param naming: `{col}__{param}`** — e.g., `price__mean`, `city__categories`. This is the convention all built-in transformers use.

**Use `NULLIF` for division.** Prevent division by zero:
```python
exp.Nullif(this=exp.Literal.number(std), expression=exp.Literal.number(0))
```

**Use stats building blocks.** Compose from `sqlearn/stats/aggregates.py` when possible:
```python
from sqlearn.stats.aggregates import Mean, Std
# Instead of writing exp.Avg(...) directly
```

## Validation

The base class validates custom transformers automatically on first `fit()`:
- `discover()` returns sqlglot ASTs (not Python values)
- `expressions()` returns sqlglot ASTs (not strings)
- New columns have matching `output_schema()` override
- `_classification` matches `discover()` reality
- No param name mismatches between `discover()` and `expressions()`

Built-in transformers (Tier 1) skip runtime validation — they're tested in CI instead.
