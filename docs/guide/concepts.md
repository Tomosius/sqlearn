# Core Concepts

sqlearn compiles ML preprocessing pipelines to SQL. You write Python, the system writes
SQL. This page explains the key ideas that make that work.

## The Pipeline model

A Pipeline is an ordered sequence of transformers. Data flows through each step in order,
and the output of one step becomes the input of the next. The entire pipeline compiles to
a single SQL query.

```python
import sqlearn as sq

pipe = sq.Pipeline([
    sq.Imputer(),           # Step 0: fill missing values
    sq.StandardScaler(),    # Step 1: standardize to zero mean, unit variance
    sq.OneHotEncoder(),     # Step 2: encode categoricals as binary columns
])
```

### fit / transform lifecycle

The two-phase lifecycle should feel familiar if you have used scikit-learn:

1. **`fit(data)`** --- Learn parameters from training data. The Imputer learns fill
   values, the StandardScaler learns means and standard deviations, the OneHotEncoder
   learns category sets. All of this happens in SQL --- sqlearn generates aggregate
   queries like `SELECT AVG(price), STDDEV_POP(price) FROM train` and executes them
   against your database.

2. **`transform(data)`** --- Apply the learned parameters to new data. sqlearn compiles
   the entire pipeline into a single SQL SELECT statement and executes it. No
   intermediate tables, no row-by-row processing.

```python
pipe.fit("train.parquet")                # learn from training data
X = pipe.transform("test.parquet")       # apply to test data → numpy array
sql = pipe.to_sql()                      # inspect the generated SQL
```

The `to_sql()` method is useful for debugging and deployment. It returns the compiled
SQL string that you can run in any DuckDB-compatible environment:

```sql
SELECT
  (COALESCE(price, 3.0) - 3.0) / NULLIF(1.58, 0) AS price,
  (COALESCE(quantity, 35.0) - 32.5) / NULLIF(14.79, 0) AS quantity,
  CASE WHEN COALESCE(city, 'London') = 'London' THEN 1 ELSE 0 END AS city_london,
  CASE WHEN COALESCE(city, 'London') = 'Paris' THEN 1 ELSE 0 END AS city_paris
FROM __input__
```

Notice how the Imputer's `COALESCE` is nested inside the StandardScaler's subtraction
and division. This is **expression composition** --- sqlearn merges all steps into one
query automatically.

!!! tip "Method chaining"
    `fit()` returns the pipeline itself, so you can chain:
    `pipe.fit("train.parquet").transform("test.parquet")`, or use the shortcut
    `pipe.fit_transform("data.parquet")`.

### Three ways to define steps

Pipelines accept steps in three formats:

```python
# Bare list (auto-named: step_00, step_01, ...)
pipe = sq.Pipeline([sq.Imputer(), sq.StandardScaler()])

# Named tuples (your names)
pipe = sq.Pipeline([
    ("impute", sq.Imputer()),
    ("scale", sq.StandardScaler()),
])

# Dict (also your names)
pipe = sq.Pipeline({
    "impute": sq.Imputer(),
    "scale": sq.StandardScaler(),
})
```

Named steps let you access individual transformers later:

```python
pipe.named_steps["scale"]  # → StandardScaler(...)
```

### Composing pipelines with `+`

You can combine pipelines and transformers using the `+` operator:

```python
clean = sq.Pipeline([sq.Imputer()])
scale = sq.Pipeline([sq.StandardScaler()])
combined = clean + scale  # new Pipeline with both steps

# Or add a single step
extended = combined + sq.OneHotEncoder()
```

The `+` operator always creates a new Pipeline --- it never mutates the originals.


## Static vs Dynamic transformers

sqlearn classifies every transformer as either **static** or **dynamic**.

**Static** transformers need no data to work. They define a fixed transformation that
does not depend on training data statistics. Examples:

- `Rename({"old": "new"})` --- renames a column
- `Cast({"price": "FLOAT"})` --- changes a column's SQL type
- `Filter("price > 0")` --- filters rows
- `sq.Expression("price * qty AS revenue")` --- computes a new column

**Dynamic** transformers learn parameters from data during `fit()`. They use SQL
aggregate queries to compute statistics, then embed those statistics into the transform
expression. Examples:

- `StandardScaler()` --- learns mean and std via `AVG()` and `STDDEV_POP()`
- `Imputer()` --- learns fill values via `AVG()`, `MEDIAN()`, or `MODE()`
- `OneHotEncoder()` --- learns the set of categories via `SELECT DISTINCT`
- `MinMaxScaler()` --- learns min and max via `MIN()` and `MAX()`

### Why it matters

The compiler uses this classification to optimize fit queries. Static steps require no
SQL execution during `fit()` --- their expressions are composed immediately. Dynamic
steps generate aggregate queries that get batched into a single SQL statement per
pipeline layer.

### How classification works

Built-in transformers declare their classification explicitly:

```python
class StandardScaler(Transformer):
    _classification = "dynamic"  # needs data stats
```

```python
class Rename(Transformer):
    _classification = "static"  # no data needed
```

For custom transformers, sqlearn uses a three-tier system:

1. **Tier 1 (built-in):** Trusted immediately. Validated by the test suite.
2. **Tier 2 (custom, declared):** You set `_classification`. Verified once at first fit.
3. **Tier 3 (custom, undeclared):** Auto-detected by inspecting `discover()` output.
   If in doubt, classified as dynamic (safe default).

!!! note "You rarely need to think about this"
    Classification is automatic. The only time it matters is when writing custom
    transformers (see [Custom Transformers](custom-transformers.md)), where declaring
    `_classification = "static"` lets the compiler skip unnecessary fit queries.


## Column routing

Every transformer has a default set of columns it operates on, determined by the
`_default_columns` class attribute:

| Transformer | Default columns | What it means |
|---|---|---|
| `StandardScaler` | `"numeric"` | Scales all numeric columns |
| `OneHotEncoder` | `"categorical"` | Encodes all categorical (string) columns |
| `Imputer` | `"all"` | Fills nulls in every column |
| `Rename` | `None` | Requires explicit `columns=` parameter |

This means a basic pipeline just works without specifying columns:

```python
pipe = sq.Pipeline([
    sq.Imputer(),          # fills nulls in ALL columns
    sq.StandardScaler(),   # scales only NUMERIC columns
    sq.OneHotEncoder(),    # encodes only CATEGORICAL columns
])
```

sqlearn reads the table schema (column names and SQL types) and automatically routes
each column to the right transformer. No `ColumnTransformer` needed for 80% of use cases.

You can always override the defaults with the `columns=` parameter:

```python
# Scale only these two columns
sq.StandardScaler(columns=["price", "score"])

# Scale using a selector
sq.StandardScaler(columns=sq.numeric() - sq.columns("id"))
```

For the full story on column routing --- selectors, composition operators, `Columns`,
and `Union` --- see [Column Routing](column-routing.md).


## Expression composition

This is the core of sqlearn's compiler. When multiple transformers are stacked in a
pipeline, their SQL expressions are **nested** into a single SELECT statement.

Consider this pipeline:

```python
pipe = sq.Pipeline([
    sq.Imputer(strategy="mean"),
    sq.StandardScaler(),
])
```

The Imputer generates `COALESCE(price, 42.5)`. The StandardScaler generates
`(price - 42.5) / NULLIF(10.2, 0)`. But the StandardScaler does not operate on the
raw `price` column --- it operates on the **output of the Imputer**. So the compiler
substitutes the Imputer's expression into the StandardScaler's:

```sql
(COALESCE(price, 42.5) - 42.5) / NULLIF(10.2, 0) AS price
```

This is a single SQL expression --- no intermediate tables, no subqueries. The compiler
tracks a dict of `{column_name: current_expression}` and passes it forward to each step.
Each transformer's `expressions()` method receives the current expressions and builds on
top of them.

### How it works under the hood

1. The compiler starts with bare column references: `{"price": Column("price"), "city": Column("city")}`
2. The Imputer's `expressions()` returns `{"price": Coalesce(Column("price"), 42.5)}`
3. The dict becomes: `{"price": Coalesce(Column("price"), 42.5), "city": Column("city")}`
4. The StandardScaler's `expressions()` receives this dict and builds on `exprs["price"]`
5. The result is `{"price": Div(Paren(Sub(Coalesce(...), 42.5)), Nullif(10.2, 0))}`
6. The compiler renders this as the final SELECT

All expressions are sqlglot AST nodes, never raw SQL strings. This means they can be
rendered for any SQL dialect --- DuckDB, Postgres, Snowflake, BigQuery.

### CTE promotion

When expressions get deeply nested (more than 8 levels by default), the compiler
automatically promotes them into a CTE (Common Table Expression) to keep the SQL
readable:

```sql
WITH __cte_0 AS (
  SELECT COALESCE(price, 42.5) AS price, ... FROM __input__
)
SELECT (price - 42.5) / NULLIF(10.2, 0) AS price FROM __cte_0
```

This is a readability optimization --- the SQL is semantically identical either way.

For a detailed walkthrough of how the compiler works, see
[How the Compiler Works](compiler.md).


## Schema tracking

sqlearn tracks the schema (column names and types) through every pipeline step. This
enables:

- **Auto column routing** --- transformers know which columns are numeric, categorical,
  temporal, or boolean without you specifying it.
- **Schema change detection** --- the compiler knows when a step adds, removes, or
  renames columns, and adjusts accordingly.
- **Output column names** --- `pipe.get_feature_names_out()` returns the exact list of
  columns the pipeline produces.

A Schema is an immutable mapping of column names to SQL type strings:

```python
from sqlearn.core.schema import Schema

schema = Schema({"price": "DOUBLE", "city": "VARCHAR", "active": "BOOLEAN"})
schema.numeric()       # ['price']
schema.categorical()   # ['city']
schema.boolean()       # ['active']
```

Each transformer declares an `output_schema()` method that says how it changes the
schema. For example, OneHotEncoder removes the original categorical column and adds
new binary columns (`city_london`, `city_paris`). The compiler chains these schema
transformations so that each step sees the correct schema from its predecessor.

!!! tip "Inspecting the pipeline"
    After fitting, you can inspect the schema at any point:

    ```python
    pipe.fit("data.parquet")
    pipe.get_feature_names_out()        # final output columns
    pipe.named_steps["scale"].columns_  # columns this step operates on
    ```


## Key takeaways

1. **Pipeline = ordered sequence of transformers**, compiled to one SQL query.
2. **fit() learns parameters via SQL aggregates**, transform() applies them.
3. **Static transformers** need no data; **dynamic transformers** learn statistics.
4. **Column routing** is automatic by type --- override with `columns=` when needed.
5. **Expression composition** nests SQL expressions from multiple steps into a single SELECT.
6. **Schema tracking** propagates column types through the pipeline for automatic routing.
