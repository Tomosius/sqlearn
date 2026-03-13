> **sqlearn docs** — [Index](../CLAUDE.md) | Prev: [Architecture](03-architecture.md) | Next: [Built-in Transformers](05-transformers.md)

## 5. Compiler Architecture

### 5.1 Expression Composition (The Core Innovation)

The compiler chains `expressions()` calls. Each step transforms the expression dict.
All values are **sqlglot AST nodes** — shown here as SQL for readability:

```python
# Start: bare column references (sqlglot Column nodes)
exprs = {"price": Column("price"), "city": Column("city")}

# After Imputer:
exprs = {"price": Coalesce(Column("price"), 42.5),
         "city":  Coalesce(Column("city"), "Unknown")}

# After StandardScaler (only numeric):
exprs = {"price": (Cast(Coalesce(Column("price"), 42.5), DOUBLE) - 42.5) / 12.3,
         "city":  Coalesce(Column("city"), "Unknown")}

# After OneHotEncoder (categorical → binary columns):
exprs = {
    "price":       (Cast(Coalesce(Column("price"), 42.5), DOUBLE) - 42.5) / 12.3,
    "city_london": Case().when(Coalesce(Column("city"), "Unknown").eq("London"), 1).else_(0),
    "city_paris":  Case().when(Coalesce(Column("city"), "Unknown").eq("Paris"), 1).else_(0),
}
```

Final SQL — **one SELECT, zero CTEs:**

```sql
SELECT
    (COALESCE(price, 42.5) - 42.5) / 12.3 AS price,
    CASE WHEN COALESCE(city, 'Unknown') = 'London' THEN 1 ELSE 0 END AS city_london,
    CASE WHEN COALESCE(city, 'Unknown') = 'Paris' THEN 1 ELSE 0 END AS city_paris
FROM data
```

Three Python steps → one SQL query. DuckDB processes all columns in one vectorized pass.

### 5.2 CTEs: Avoid When Possible, Use When Needed

CTEs aren't bad. But simpler SQL = better DuckDB optimization. Rules:

**No CTE (default):** Column-level expressions compose inline.

```sql
-- Imputer + Scaler = one expression, no CTE
SELECT (COALESCE(price, 42.5) - 42.5) / 12.3 AS price FROM data
```

**CTE required:** Window functions can't nest inside other expressions.

```sql
-- RollingMean needs its own query level
WITH windowed AS (
    SELECT *,
        AVG(price) OVER (PARTITION BY store ORDER BY date ROWS 7 PRECEDING) AS price_7d
    FROM data
)
SELECT
    (COALESCE(price, 42.5) - 42.5) / 12.3 AS price,
    (price_7d - 100.5) / 25.3 AS price_7d_scaled
FROM windowed
```

Two query levels. Minimum possible. The compiler auto-decides.

**CTE for deduplication:** Complex expressions referenced by multiple downstream steps.

```python
# If expression "COALESCE(price, 42.5)" is used by both StandardScaler AND RobustScaler
# in a Union, promote to CTE to avoid computing twice.
# BUT: bare column refs (just "price") are never promoted — they're free to duplicate.
```

### 5.2b Columns & Union Compilation

`Pipeline` composes expressions sequentially — each step transforms the expression dict.
`Columns` and `Union` compose expressions in PARALLEL — they split, transform independently,
and recombine. The compiler handles both.

**`Columns` compilation (parallel column routing):**

`Columns` routes different columns to different transformers. Each branch sees only its
target columns. The compiler resolves this to a single SELECT with no CTEs:

```python
routed = sq.Columns({
    "price":    sq.RobustScaler(),      # scales price
    "city":     sq.OrdinalEncoder(),    # encodes city
    "zipcode":  sq.HashEncoder(n=64),   # hashes zipcode
}, remainder="passthrough")
```

```sql
-- Columns compiles to ONE SELECT. Each branch contributes its column expressions.
SELECT
    (CAST(price AS DOUBLE) - 50000) / 25000 AS price,    -- RobustScaler
    CASE WHEN city='NY' THEN 0 WHEN city='LA' THEN 1 ... END AS city,  -- OrdinalEncoder
    HASH(zipcode) % 64 AS zipcode,                         -- HashEncoder
    age,                                                    -- remainder: passthrough
    income                                                  -- remainder: passthrough
FROM data
```

No CTEs needed — each branch's `expressions()` is called independently, and the results
are merged into one SELECT. Column name collisions are impossible because each branch
targets disjoint columns (enforced at validation time).

If a branch uses `query()` (e.g., contains a window function), that branch gets its own
CTE, and the other branches compose inline around it.

**`Union` compilation (parallel feature combination):**

`Union` runs multiple transformers/pipelines on the SAME input and combines their outputs
side by side. This DOES require CTEs — each branch may produce different columns:

```python
combined = sq.Union([
    ("base", sq.StandardScaler()),
    ("poly", sq.PolynomialFeatures(degree=2)),
])
```

```sql
-- Union: each branch gets a CTE, results merged in final SELECT
WITH branch_base AS (
    SELECT
        (CAST(price AS DOUBLE) - 42.5) / 12.3 AS price,
        (CAST(income AS DOUBLE) - 65000) / 28000 AS income
    FROM data
),
branch_poly AS (
    SELECT
        price * price AS price_price,
        price * income AS price_income,
        income * income AS income_income
    FROM data
)
SELECT
    branch_base.price,
    branch_base.income,
    branch_poly.price_price,
    branch_poly.price_income,
    branch_poly.income_income
FROM branch_base, branch_poly
-- (same row count guaranteed — both read from same source, no filtering)
```

**Column name collision resolution in Union:**

If two branches produce columns with the same name, the compiler prefixes with the
branch name: `base__price`, `poly__price`. This mirrors sklearn's FeatureUnion behavior.
Users can control this: `sq.Union([...], prefix=True/False/"custom_{branch}_")`.

**Optimization: Union without CTEs when possible:**

If all branches use `expressions()` only (no `query()` branches), the compiler can
merge everything into a single SELECT:

```sql
-- Optimized Union: all branches are expression-level → one SELECT
SELECT
    (CAST(price AS DOUBLE) - 42.5) / 12.3 AS base__price,
    (CAST(income AS DOUBLE) - 65000) / 28000 AS base__income,
    price * price AS poly__price_price,
    price * income AS poly__price_income,
    income * income AS poly__income_income
FROM data
```

One query. Zero CTEs. The compiler checks: if all Union branches are expression-level,
merge into one SELECT. If any branch uses `query()`, all branches get CTEs.

**`y` column handling in Columns and Union:**

The `y_column` is propagated to ALL branches. Each branch that contains a step needing
`y_column` (e.g., `TargetEncoder`) receives it. The target column is excluded from the
FINAL output of the Columns/Union, not from intermediate branch outputs — branches may
need it internally.

### 5.3 Fit Resolution: Layers

**The problem:** Some transforms learn from data (dynamic). Some change the schema.
If Step 2 creates new columns, Step 3 can't learn its params until Step 2's output
schema is known.

**The solution:** Group steps into layers. Layer boundary = after any schema-changing
dynamic step. Classification uses the conservative detection from Section 4.3 —
if a step can't be proven static, it's treated as dynamic and gets its own
aggregation query in the layer.

```
Pipeline:
  1. Imputer          (dynamic, same schema)       ← discover() returns aggregates
  2. StandardScaler   (dynamic, same schema)       ← discover() returns aggregates
  3. OneHotEncoder    (dynamic, CHANGES schema)    ← layer boundary
  4. MinMaxScaler     (dynamic, same schema)       ← discover() returns aggregates
  5. Log              (STATIC, same schema)         ← discover() returns {} (proven)

Layer 0: [Imputer, StandardScaler, OneHotEncoder]
  → fit: one aggregation query for Imputer + StandardScaler + OneHotEncoder
  → Log is static — no aggregation, but still in the layer (just skipped in fit query)
Layer 1: [MinMaxScaler, Log]
  → fit: one aggregation query for MinMaxScaler only (Log contributes nothing)
```

**Static steps within a layer are free — they contribute zero cost to the fit query.**
They're still part of the layer for schema tracking purposes, but their empty
`discover()` result means no aggregation expressions are added to the SQL.

**Layer 0 fit:** One query discovers all params for Imputer + StandardScaler + OneHotEncoder:

```sql
-- Aggregates (one query)
SELECT
    MEDIAN(price)      AS imputer__price__median,
    AVG(price)         AS scaler__price__mean,
    STDDEV_POP(price)  AS scaler__price__std
FROM data;

-- Distinct values (one query)
SELECT DISTINCT city FROM data;
```

Then materialize Layer 0's output as a temp view (so Layer 1 can see the new columns).

**Layer 1 fit:** One query against the materialized output:

```sql
SELECT MIN(price) AS minmax__price__min, MAX(price) AS minmax__price__max,
       MIN(city_london) AS minmax__city_london__min, ...
FROM __layer_0_output;
```

**Total fit: 3-4 queries for the entire pipeline.** Compare to sklearn: one full data
pass per step.

**Key insight: layers only matter during fit. Transform is always one SQL query.**

### 5.3b Three-Phase Planner Architecture

The compiler's fit-and-transform work can be conceptually split into three planning
phases, each with a distinct responsibility. This was prototyped in `ducklearn1` as
explicit planner classes:

**Phase 1 — FitInspection:** Walk pipeline steps, classify each as static/dynamic
(Section 4.3). Produce a `FitInspectionPlan`: an ordered list of `StepFitInfo` objects,
each containing step name, estimator reference, SQL templates, and `is_static` flag.

```python
@dataclass
class StepFitInfo:
    step_name: str
    estimator: Transformer
    sql_templates: dict[str, Expression]  # from discover()
    is_static: bool

@dataclass
class FitInspectionPlan:
    steps: list[StepFitInfo]
```

**Phase 2 — FitExecution:** Take inspection results and plan how to batch fit SQL
into minimal DuckDB queries. Key optimization: **combine multiple static fit queries
into one multi-CTE query** to minimize database round-trips:

```sql
-- Instead of 3 separate queries for 3 static steps:
-- SELECT MEDIAN(price) FROM data;
-- SELECT AVG(price), STDDEV_POP(price) FROM data;
-- SELECT DISTINCT city FROM data;

-- Batch static aggregates into ONE query:
WITH
    cte_imputer AS (SELECT MEDIAN(price) AS median_price FROM data),
    cte_scaler  AS (SELECT AVG(price) AS mean_price,
                           STDDEV_POP(price) AS std_price FROM data)
SELECT
    (SELECT median_price FROM cte_imputer) AS imputer__price__median,
    (SELECT mean_price FROM cte_scaler) AS scaler__price__mean,
    (SELECT std_price FROM cte_scaler) AS scaler__price__std
```

Each CTE output is mapped back to its originating step/attribute. This reduces
the total number of DuckDB queries during fit from N (one per dynamic step) to
the theoretical minimum (one per layer for aggregates, plus set queries).

**Phase 3 — TransformPlanner:** Compose all `expressions()` / `query()` outputs
into a single optimized query. Applies optimizations:
- Flatten trivial wrappers (`SELECT * FROM (SELECT * FROM (...))` → single SELECT)
- Extract common subexpressions into CTEs
- Dialect-aware cleanup before sqlglot translation

The current sqlearn compiler (Section 5.1-5.3) already handles Phase 3 via expression
composition. Phases 1-2 are implicit in the layer resolution logic. Making them
explicit planner objects could improve testability and enable per-phase optimization.
*(from ducklearn1)*

### 5.4 Why This Beats sklearn (and Polars)

**sklearn:**

```
fit:       N separate full data scans (one per dynamic step)
transform: N numpy array allocations (one per step, each a full copy)
Memory:    N × dataset_size
```

**Polars lazy:**

```
fit:       Still N data scans (sklearn forces materialization)
transform: Polars can optimize the DataFrame ops, but sklearn steps
           still force numpy conversion at each boundary
Memory:    Better than pandas, but still step-by-step materialization
```

**sqlearn:**

```
fit:       1-3 SQL queries total (batched aggregates)
transform: 1 SQL query, DuckDB streams rows through all transforms
Memory:    batch_size × n_columns (constant, regardless of dataset size)
```

### 5.5 Bigger Than RAM

DuckDB handles out-of-core natively. The pipeline doesn't need to know.

```python
# 100GB parquet on S3 — DuckDB streams, never loads fully into RAM
pipe.fit("s3://bucket/100gb_dataset.parquet")

# Transform directly to disk — peak memory ≈ 1GB for a 100GB dataset
pipe.transform(
    "s3://bucket/100gb_dataset.parquet",
    out="parquet",
    path="features.parquet"
)

# Or stream in batches — constant memory
for batch in pipe.transform("s3://bucket/100gb.parquet", batch_size=100_000):
    model.partial_fit(batch)
```

This is not a special mode. It's just how SQL works. The pipeline compiles to one
query, DuckDB executes it however it wants (in-memory, spill to disk, stream from S3).

---
