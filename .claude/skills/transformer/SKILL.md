---
name: transformer
description: Use when creating a new transformer, scaler, encoder, imputer, feature transform, or any pipeline step. Also trigger when modifying existing transformers, adding discover/expressions/query methods, implementing output_schema, or working on any class that extends Transformer. If the user mentions a specific transformer name (StandardScaler, OneHotEncoder, etc.), use this skill.
disable-model-invocation: false
user-invocable: true
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

**Conditionally dynamic transformer:**

Classification can depend on constructor args. Set `_classification` in `__init__`:

```python
class Clip(Transformer):
    _default_columns = "numeric"

    def __init__(self, lower=0, upper=100):
        self.lower = lower
        self.upper = upper
        # Literal values = static. Percentile strings = needs data = dynamic.
        self._classification = "static" if isinstance(lower, (int, float)) else "dynamic"
```

Other examples: `StringSplit(max_parts=3)` → static, `StringSplit(max_parts="auto")` → dynamic.
`AutoDatetime(granularity="hour")` → static, `AutoDatetime(granularity="auto")` → dynamic.

**Schema-changing transformer (adds/removes columns):**

```python
def output_schema(self, schema):
    # REQUIRED when expressions() adds new column names
    return schema.add({f"{col}_flag": "INTEGER" for col in self.columns_})
    # Or for dropping: return schema.remove(["col_to_drop"])
```

**Set-learning transformer (encoders):**

Use `discover_sets()` when you need category lists, not scalar stats:

```python
class OneHotEncoder(Transformer):
    _default_columns = "categorical"
    _classification = "dynamic"

    def discover_sets(self, columns, schema, y_column=None):
        # Return {name: sqlglot_query} — each query returns a list of values
        return {f"{col}__categories": exp.Select(
            expressions=[exp.Distinct(expressions=[exp.Column(this=col)])]
        ).from_(exp.Table(this="__source__"))
        for col in columns}

    def expressions(self, columns, exprs):
        # self.sets_ has {name: list_of_dicts} from discover_sets()
        result = {}
        for col in columns:
            categories = self.sets_[f"{col}__categories"]
            for cat in categories:
                result[f"{col}_{cat}"] = exp.Case(
                    ifs=[exp.If(this=exp.EQ(this=exprs[col],
                         expression=exp.Literal.string(cat)),
                         true=exp.Literal.number(1))],
                    default=exp.Literal.number(0),
                )
        return result
```

- `discover()` → scalar stats (mean, std, min, max) → stored in `self.params_`
- `discover_sets()` → category lists, ordered values → stored in `self.sets_`
- A transformer can use both if it needs scalars AND sets.

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

Two exports needed — folder-level AND top-level:

```python
# src/sqlearn/<folder>/__init__.py
from .my_transformer import MyTransformer

# src/sqlearn/__init__.py — add import + __all__ entry
from sqlearn.<folder>.my_transformer import MyTransformer
```

### 4. Write tests

Create `tests/<folder>/test_my_transformer.py` with ALL of:

- [ ] sklearn equivalence test (`np.testing.assert_allclose`)
- [ ] SQL snapshot test
- [ ] Null handling test
- [ ] Classification test (`_classification` matches `discover()` reality)
- [ ] Roundtrip test (fit → to_sql → execute → same result)
- [ ] Clone test (independent copy, identical output)
- [ ] Edge cases (single row, constant column, empty table)

See `/test` skill for full test patterns and the 3-tier testing strategy.

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

## Thread Safety, Clone, Pickle

Transformers are **NOT thread-safe**. Use `clone()` for concurrent access:

```python
pipe_copy = pipe.clone()  # independent copy with own params_/sets_
```

All transformers support pickle serialization and `copy.deepcopy()`. Clone produces
an independent copy — modifying or re-fitting the clone does not affect the original.

When implementing a new transformer, no special work needed — `Transformer` base class
handles `__copy__`, `__deepcopy__`, `__getstate__`/`__setstate__` automatically.
Just avoid storing non-picklable objects (open connections, locks) in instance attributes.

## Validation

The base class validates custom transformers automatically on first `fit()`:
- `discover()` returns sqlglot ASTs (not Python values)
- `expressions()` returns sqlglot ASTs (not strings)
- New columns have matching `output_schema()` override
- `_classification` matches `discover()` reality
- No param name mismatches between `discover()` and `expressions()`

Built-in transformers (Tier 1) skip runtime validation — they're tested in CI instead.

## Quick Reference

| Method | Returns | Stored in | When to use |
|---|---|---|---|
| `discover()` | `{name: sqlglot_agg}` | `self.params_` | Scalar stats (mean, std, min, max) |
| `discover_sets()` | `{name: sqlglot_query}` | `self.sets_` | Category lists, ordered values |
| `expressions()` | `{col: sqlglot_expr}` | — | Inline column transforms |
| `query()` | `sqlglot.Select` | — | Window functions, JOINs, CTEs |
| `output_schema()` | `Schema` | — | When adding/removing columns |
