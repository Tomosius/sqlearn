# How the Compiler Works

sqlearn's compiler turns your Python pipeline into SQL. This page traces the full
journey from Python code to a final SQL query, using a concrete example.

## Overview: three phases

The compiler works in three phases:

1. **Plan** (`plan_fit`) --- Classify each step as static or dynamic, detect schema
   changes, group steps into layers.
2. **Fit** (`build_fit_queries`) --- Batch all dynamic steps' aggregate queries into
   minimal SQL, execute them, and populate learned parameters.
3. **Compose** (`compose_transform`) --- Walk through all fitted steps and compose
   their expressions into a single SELECT statement.

Let's trace each phase with a real pipeline.


## The example pipeline

```python
import sqlearn as sq

pipe = sq.Pipeline([
    sq.Imputer(strategy="mean"),
    sq.StandardScaler(),
])
```

We will fit this pipeline on a table with two columns:

| price (DOUBLE) | quantity (INTEGER) |
|---:|---:|
| 10.0 | 20 |
| NULL | 40 |
| 8.0 | 30 |

```python
pipe.fit("sales.parquet")
```

Here is what happens inside.


## Phase 1: Plan

The `plan_fit()` function receives the list of transformers and the input schema. It
does three things for each step:

### 1a. Resolve columns

Each transformer has a column specification --- either from `_default_columns` or from
an explicit `columns=` parameter. The compiler resolves this against the current schema.

| Step | `_default_columns` | Resolved columns |
|---|---|---|
| `Imputer(strategy="mean")` | `"all"` | `["price", "quantity"]` |
| `StandardScaler()` | `"numeric"` | `["price", "quantity"]` |

Both steps target the same columns here because all columns are numeric. If we had a
`VARCHAR` column, the Imputer would include it but the StandardScaler would skip it.

### 1b. Classify steps

Each step is classified as static or dynamic:

| Step | `_classification` | Reason |
|---|---|---|
| `Imputer` | `"dynamic"` | Needs data stats (AVG per column) |
| `StandardScaler` | `"dynamic"` | Needs data stats (AVG, STDDEV_POP per column) |

Both steps are built-in (Tier 1), so the compiler trusts their declarations without
further verification.

### 1c. Group into layers

Steps are grouped into **layers** for fit execution. A new layer boundary is created
when a step is both **dynamic** and **schema-changing** (adds, removes, or renames
columns).

The Imputer and StandardScaler both preserve the same columns --- they transform values
in place without changing column names or types. So both steps fit into a **single
layer**.

```
Layer 0: [Imputer, StandardScaler]
  Input schema:  {price: DOUBLE, quantity: INTEGER}
  Output schema: {price: DOUBLE, quantity: INTEGER}
```

!!! note "When layers split"
    Consider a pipeline with `Imputer() + OneHotEncoder() + StandardScaler()`.
    The OneHotEncoder is dynamic (learns categories) and schema-changing (replaces
    `city` with `city_london`, `city_paris`). This creates a layer boundary:

    - Layer 0: `[Imputer, OneHotEncoder]` --- fit together against the original table
    - Layer 1: `[StandardScaler]` --- fit against a temp view of Layer 0's output

    The compiler materializes Layer 0's output as a temporary view so Layer 1 can
    query the transformed schema.


## Phase 2: Fit

The `build_fit_queries()` function takes a layer and produces the minimal SQL needed
to learn all parameters.

### 2a. Collect aggregations

Each dynamic step's `discover()` method returns sqlglot aggregate expressions:

**Imputer.discover()** returns:

```python
{
    "price__value": Avg(Column("price")),
    "quantity__value": Avg(Column("quantity")),
}
```

**StandardScaler.discover()** returns:

```python
{
    "price__mean": Avg(Column("price")),
    "price__std": StddevPop(Column("price")),
    "quantity__mean": Avg(Column("quantity")),
    "quantity__std": StddevPop(Column("quantity")),
}
```

### 2b. Batch into one query

The compiler merges all aggregations from all dynamic steps in the layer into a
**single SELECT**:

```sql
SELECT
  AVG(price)         AS __agg_0_price__value,
  AVG(quantity)      AS __agg_0_quantity__value,
  AVG(price)         AS __agg_1_price__mean,
  STDDEV_POP(price)  AS __agg_1_price__std,
  AVG(quantity)      AS __agg_1_quantity__mean,
  STDDEV_POP(quantity) AS __agg_1_quantity__std
FROM sales
```

One query, one table scan. The `__agg_N_` prefix maps each result back to its step.

!!! tip "Static expression inlining"
    If a static step precedes a dynamic step in the same layer, the compiler inlines
    the static step's expressions into the dynamic step's aggregations. For example,
    if a `Cast` preceded the Imputer, the compiler would substitute the cast
    expression into the AVG, so `AVG(CAST(price AS FLOAT))` runs in one query.

### 2c. Execute and distribute

The compiler executes the batched query and gets back one row of results:

| Column | Value |
|---|---|
| `__agg_0_price__value` | 9.0 |
| `__agg_0_quantity__value` | 30.0 |
| `__agg_1_price__mean` | 9.0 |
| `__agg_1_price__std` | 0.8165 |
| `__agg_1_quantity__mean` | 30.0 |
| `__agg_1_quantity__std` | 8.1650 |

The param mapping routes each value back to its step:

- Imputer gets `params_ = {"price__value": 9.0, "quantity__value": 30.0}`
- StandardScaler gets `params_ = {"price__mean": 9.0, "price__std": 0.8165, "quantity__mean": 30.0, "quantity__std": 8.1650}`

### Set queries

Some transformers need multi-row results instead of scalar aggregates. For example,
OneHotEncoder's `discover_sets()` returns:

```python
{
    "city__categories": Select("DISTINCT city FROM ..."),
}
```

Set queries cannot be batched with scalar aggregates, so they execute as separate
queries. The results are stored in `sets_` as lists of dicts.


## Phase 3: Compose

Now both steps are fitted (they have their `params_` populated). The
`compose_transform()` function walks through all steps and builds the final SELECT.

### 3a. Initialize expressions

Start with bare column references from the input schema:

```python
exprs = {
    "price": Column("price"),
    "quantity": Column("quantity"),
}
```

### 3b. Apply Imputer

The Imputer's `expressions()` method receives the current `exprs` dict and returns
modifications:

```python
# Imputer.expressions() returns:
{
    "price": Coalesce(exprs["price"], Literal(9.0)),
    "quantity": Coalesce(exprs["quantity"], Literal(30.0)),
}
```

The compiler merges this into the expression dict:

```python
exprs = {
    "price": Coalesce(Column("price"), 9.0),
    "quantity": Coalesce(Column("quantity"), 30.0),
}
```

### 3c. Apply StandardScaler

The StandardScaler's `expressions()` method receives the **updated** `exprs` dict ---
which now contains the Imputer's COALESCE expressions:

```python
# StandardScaler.expressions() builds on exprs["price"],
# which is Coalesce(Column("price"), 9.0):
{
    "price": Div(
        Paren(Sub(exprs["price"], Literal(9.0))),
        Nullif(Literal(0.8165), Literal(0)),
    ),
    "quantity": Div(
        Paren(Sub(exprs["quantity"], Literal(30.0))),
        Nullif(Literal(8.1650), Literal(0)),
    ),
}
```

After merging, the expression dict contains the fully composed expressions:

```python
exprs = {
    "price": Div(
        Paren(Sub(Coalesce(Column("price"), 9.0), 9.0)),
        Nullif(0.8165, 0),
    ),
    "quantity": Div(
        Paren(Sub(Coalesce(Column("quantity"), 30.0), 30.0)),
        Nullif(8.1650, 0),
    ),
}
```

### 3d. Build final SELECT

The compiler renders each expression as an aliased SELECT column:

```sql
SELECT
  (COALESCE(price, 9.0) - 9.0) / NULLIF(0.8165, 0) AS price,
  (COALESCE(quantity, 30.0) - 30.0) / NULLIF(8.165, 0) AS quantity
FROM sales
```

One query. One table scan. Two steps composed into nested expressions.

```python
print(pipe.to_sql())
# SELECT
#   (COALESCE(price, 9.0) - 9.0) / NULLIF(0.8165, 0) AS price,
#   (COALESCE(quantity, 30.0) - 30.0) / NULLIF(8.165, 0) AS quantity
# FROM __input__
```


## CTE promotion

Expression nesting can get deep. Consider a pipeline with 10 steps that each wrap the
column in another function call. The resulting SQL would have 10 levels of nesting,
which is hard to read and may hit database expression depth limits.

The compiler tracks expression depth and automatically promotes to a CTE when the
nesting exceeds a threshold (default: 8 levels):

```sql
WITH __cte_0 AS (
  SELECT
    COALESCE(price, 9.0) AS price,
    COALESCE(quantity, 30.0) AS quantity
  FROM __input__
)
SELECT
  (price - 9.0) / NULLIF(0.8165, 0) AS price,
  (quantity - 30.0) / NULLIF(8.165, 0) AS quantity
FROM __cte_0
```

The SQL is semantically identical --- the CTE is just a readability optimization. The
database query optimizer will typically flatten it back.

### query() steps

Some transformers need query-level control that cannot be expressed as inline column
expressions. For example, a window function transformer needs to wrap the entire input
in a subquery. These transformers implement `query()` instead of (or in addition to)
`expressions()`:

```python
class MyWindowTransformer(Transformer):
    def query(self, input_query):
        # Wrap the input query with a window function
        return sqlglot.parse_one(f"""
            SELECT *, ROW_NUMBER() OVER (ORDER BY id) AS row_num
            FROM ({input_query.sql()})
        """)
```

When `query()` returns a non-None result, the compiler promotes it to a CTE
automatically, then subsequent steps compose against the CTE's output columns.


## Layer resolution: the full picture

Here is the complete algorithm for how the compiler groups steps into layers and
handles multi-layer pipelines:

```
For each step in the pipeline:
  1. Resolve columns against current schema
  2. Classify as static/dynamic
  3. Detect schema change (output_schema != input_schema)
  4. Add to current layer
  5. If DYNAMIC + SCHEMA-CHANGING:
     → Close current layer
     → Start new layer with updated schema
```

**Single-layer pipeline** (most common): All steps fit into one layer. One batched
aggregate query, one composed SELECT.

**Multi-layer pipeline**: When a dynamic, schema-changing step forces a layer boundary:

1. Fit Layer 0: batch and execute all aggregate queries against the source table.
2. Materialize Layer 0 output as a temporary view.
3. Fit Layer 1: batch and execute queries against the temp view.
4. Compose the final SELECT across all layers.

!!! note "Static schema-changing steps do NOT create boundaries"
    A `Rename` step changes the schema (renames a column) but is static (needs no data).
    The compiler handles this without creating a new layer --- it simply updates the
    expression dict with the new column name and continues.


## Summary

| Phase | Function | What it does |
|---|---|---|
| Plan | `plan_fit()` | Classify steps, detect schema changes, group into layers |
| Fit | `build_fit_queries()` | Batch aggregates into one SQL query per layer, execute |
| Compose | `compose_transform()` | Compose expressions into a single SELECT |

The key insight is that expression composition replaces what would normally be N
separate queries or intermediate tables with a single nested SQL expression. The
database executes one query and scans the data once.
