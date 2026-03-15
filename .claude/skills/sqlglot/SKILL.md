---
name: sqlglot
description: Provides sqlglot AST expression patterns for sqlearn development. Trigger when writing or modifying code that imports sqlglot.expressions, constructs AST nodes (exp.Column, exp.Avg, exp.Case, exp.Select), or implements discover/expressions/query methods. Nearly every sqlearn file uses sqlglot — this is the most frequently needed reference.
disable-model-invocation: false
user-invocable: true
---

# sqlglot AST Patterns — sqlearn

All SQL in sqlearn is built via sqlglot AST nodes. Never raw SQL strings.

```python
import sqlglot.expressions as exp
```

## Two Contexts: discover() vs expressions()

The most important distinction in sqlearn. Getting this wrong is the #1 source of bugs.

**In `discover()`** — reference raw source columns (aggregates run against the source table):

```python
def discover(self, columns, schema, y_column=None):
    return {
        f"{col}__mean": exp.Avg(this=exp.Column(this=col)),
        f"{col}__std": exp.StddevPop(this=exp.Column(this=col)),
    }
```

**In `expressions()`** — use the `exprs` dict (contains composed expressions from prior steps):

```python
def expressions(self, columns, exprs):
    return {
        col: exp.Sub(this=exprs[col], expression=exp.Literal.number(self.params_[f"{col}__mean"]))
        for col in columns
    }
```

Why: if Imputer runs before StandardScaler, `exprs["price"]` is `COALESCE(price, 42.5)`, not
bare `Column("price")`. Using `exp.Column(this=col)` in `expressions()` silently skips prior steps.

## Literals

```python
exp.Literal.number(42.5)        # numeric
exp.Literal.string("London")    # string
exp.Literal.number(0)           # zero
exp.NULL                         # NULL
```

## Arithmetic

```python
exp.Sub(this=exprs[col], expression=exp.Literal.number(mean))       # col - mean
exp.Div(this=numerator, expression=denominator)                      # a / b
exp.Add(this=a, expression=b)                                        # a + b
exp.Mul(this=a, expression=b)                                        # a * b
```

## Safe Division — NULLIF

Always wrap denominators that could be zero. This prevents `Inf`/`NaN` in output:

```python
exp.Div(
    this=exp.Sub(this=exprs[col], expression=exp.Literal.number(mean)),
    expression=exp.Nullif(
        this=exp.Literal.number(std),
        expression=exp.Literal.number(0),
    ),
)
```

## COALESCE (null imputation)

```python
exp.Coalesce(this=exprs[col], expressions=[exp.Literal.number(fill_value)])
```

## CASE Expressions (encoding)

```python
# Single branch: CASE WHEN city = 'London' THEN 1 ELSE 0 END
exp.Case(
    ifs=[exp.If(
        this=exp.EQ(this=exprs[col], expression=exp.Literal.string("London")),
        true=exp.Literal.number(1),
    )],
    default=exp.Literal.number(0),
)

# Multi-branch ordinal encoding
exp.Case(
    ifs=[
        exp.If(this=exp.EQ(this=exprs[col], expression=exp.Literal.string(cat)),
               true=exp.Literal.number(i))
        for i, cat in enumerate(categories)
    ],
    default=exp.Literal.number(-1),
)
```

## Comparisons

```python
exp.EQ(this=a, expression=b)      # =
exp.GT(this=a, expression=b)      # >
exp.GTE(this=a, expression=b)     # >=
exp.LT(this=a, expression=b)      # <
exp.Is(this=a, expression=exp.NULL)  # IS NULL
exp.Between(this=a, low=b, high=c)   # BETWEEN
exp.In(this=a, expressions=[...])    # IN (...)
```

## Common Functions

```python
# Math
exp.Ln(this=exprs[col])                    # LN(col)
exp.Sqrt(this=exprs[col])                  # SQRT(col)
exp.Abs(this=exprs[col])                   # ABS(col)

# Clipping: GREATEST(LEAST(col, upper), lower)
exp.Greatest(this=exp.Least(this=exprs[col],
    expressions=[exp.Literal.number(upper)]),
    expressions=[exp.Literal.number(lower)])

# String
exp.Length(this=exprs[col])                # LENGTH
exp.Lower(this=exprs[col])                 # LOWER
exp.Trim(this=exprs[col])                  # TRIM

# Type casting
exp.Cast(this=exprs[col], to=exp.DataType.build("DOUBLE"))

# EXTRACT (datetime)
exp.Extract(this=exp.Var(this="YEAR"), expression=exprs[col])
exp.Extract(this=exp.Var(this="DOW"), expression=exprs[col])
```

## Aggregate Functions (discover() only)

```python
exp.Avg(this=exp.Column(this=col))             # AVG
exp.StddevPop(this=exp.Column(this=col))       # STDDEV_POP
exp.Min(this=exp.Column(this=col))             # MIN
exp.Max(this=exp.Column(this=col))             # MAX
exp.Count(this=exp.Column(this=col))           # COUNT
exp.Variance(this=exp.Column(this=col))        # VAR_POP
```

## DISTINCT Queries (discover_sets() only)

```python
def discover_sets(self, columns, schema, y_column=None):
    return {
        f"{col}__categories": exp.Select(
            expressions=[exp.Distinct(expressions=[exp.Column(this=col)])]
        ).from_(exp.Table(this="__source__"))
        for col in columns
    }
```

## Window Functions (query() only)

Window functions cannot nest inside column expressions — they need their own query level:

```python
def query(self, input_query):
    window_expr = exp.Window(
        this=exp.Avg(this=exp.Column(this="price")),
        partition_by=[exp.Column(this="store")],
        order=exp.Order(expressions=[exp.Ordered(this=exp.Column(this="date"))]),
        spec=exp.WindowSpec(
            kind="ROWS",
            start="PRECEDING",
            start_side=exp.Literal.number(7),
            end="CURRENT ROW",
        ),
    )
    return exp.Select(
        expressions=[exp.Star(), window_expr.as_("price_7d")],
    ).from_(exp.Subquery(this=input_query, alias="__input__"))
```

## Aliasing

```python
expr.as_("new_name")
# Or explicitly:
exp.Alias(this=expr, alias=exp.to_identifier("new_name"))
```

## Cyclic Encoding (sin/cos)

```python
# SIN(2 * PI() * hour / 24)
exp.Sin(this=exp.Div(
    this=exp.Mul(
        this=exp.Mul(this=exp.Literal.number(2),
                     expression=exp.Anonymous(this="PI", expressions=[])),
        expression=exprs["hour"],
    ),
    expression=exp.Literal.number(24),
))
```

## Column Substitution (Compiler Internal)

The compiler uses `_substitute_columns()` to inline static expressions into dynamic
aggregations. This replaces `Column("price")` with the current composed expression:

```python
# If static step transforms price → price * 2:
# Before substitution: AVG(Column("price"))
# After substitution:  AVG(price * 2)

def _substitute_columns(expression, current_exprs):
    expression = expression.copy()
    for node in expression.walk():
        if isinstance(node, exp.Column) and not node.table:
            col_name = node.name
            if col_name in current_exprs:
                node.replace(current_exprs[col_name].copy())
    return expression
```

## CTE Construction

```python
# Wrap expressions into a CTE and reset to bare column refs
selects = [v.as_(k) for k, v in exprs.items()]
cte_query = exp.select(*selects).from_(exp.to_table(source))

# Attach CTE to final query
final_query = final_query.with_("__cte_0", as_=cte_query)
```

## Building a Full SELECT

```python
# Build SELECT from expression dict
selects = [v.as_(k) for k, v in exprs.items()]
source_expr = exp.to_table("my_table")
query = exp.select(*selects).from_(source_expr)

# Render to SQL
sql = query.sql(dialect="duckdb")
```

## Anonymous Functions (database-specific)

```python
# For functions not built into sqlglot:
exp.Anonymous(this="PI", expressions=[])           # PI()
exp.Anonymous(this="REGEXP_EXTRACT", expressions=[  # REGEXP_EXTRACT(col, pattern)
    exprs[col], exp.Literal.string(r"\d+")
])
```

## sqlearn Conventions

1. **Param naming:** `{col}__{stat}` — e.g., `price__mean`, `city__categories`
2. **NULLIF for division:** Always. No exceptions.
3. **Return only modified columns** from `expressions()` — untouched pass through automatically
4. **`exprs[col]` in expressions()**, `exp.Column(this=col)` in `discover()`
5. **Never raw SQL strings** — always sqlglot AST nodes
6. **Use stats building blocks** from `sqlearn/stats/aggregates.py` when available

## Common Mistakes

```python
# WRONG: raw string in expressions()
return {"price": "price - 42.5"}

# WRONG: bare column ref in expressions() — skips prior transformers
return {"price": exp.Sub(this=exp.Column(this="price"), expression=...)}

# WRONG: division without NULLIF — produces Inf when std=0
exp.Div(this=..., expression=exp.Literal.number(std))

# WRONG: string for data type
exp.Cast(this=expr, to="DOUBLE")

# RIGHT versions use: exprs[col], exp.Nullif, exp.DataType.build()
```

## Debugging SQL Output

```python
import sqlglot

# Render expression as SQL (for debugging)
sql_str = expr.sql(dialect="duckdb")
print(sql_str)

# Parse SQL string to AST (for testing)
ast = sqlglot.parse_one("SELECT price - 42.5 FROM data", dialect="duckdb")
```
