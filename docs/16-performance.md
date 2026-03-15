> **sqlearn docs** — [Index](../CLAUDE.md) | Prev: [Documentation](15-documentation.md)

## 16. Performance Strategy

### 16.1 Why sqlearn Is Fast

sqlearn's performance advantage comes from three architectural decisions:

1. **Expression composition** — transforms compose into nested sqlglot ASTs, compiling
   to a single SELECT. DuckDB processes all columns in one vectorized pass. No intermediate
   arrays, no per-step materialization.

2. **Aggregation batching** — during fit, all dynamic steps in a layer share one SQL query.
   Where sklearn runs N data scans (one per step), sqlearn runs 1-3 queries total.

3. **DuckDB's engine** — columnar storage, vectorized execution, out-of-core processing.
   The pipeline doesn't need to know about memory management — DuckDB handles it.

| Operation | sklearn | sqlearn |
|---|---|---|
| fit (5 steps) | 5 full data scans | 1-3 SQL queries |
| transform | 5 numpy array copies | 1 SQL query, streaming |
| Memory | N × dataset_size | batch_size × n_columns |
| 100GB dataset | Out of memory | Works (DuckDB streams) |

### 16.2 Literal Substitution in Transform

After fit, learned values become SQL literals — not recomputed aggregates:

```python
# During fit: query computes mode
#   SELECT MODE(city) AS imputer__city__fill FROM data
#   → result: "London"

# During transform: literal value, not mode() again
#   COALESCE(city, 'London')
```

This is automatic. Every dynamic transformer's `expressions()` reads `self.params_`
(populated during fit) and emits literal AST nodes. The transform query never contains
aggregate functions — only the expressions built from cached values.

**Why this matters:** The transform query is pure column-level operations. No aggregation,
no grouping, no window functions (unless a step explicitly uses `query()`). DuckDB can
stream this without buffering the entire dataset.

### 16.3 Expression Composition Across Steps

Static steps don't create intermediate tables — their expressions nest inside
downstream steps' aggregation queries:

```
Pipeline: [Log, StandardScaler]

Log (static):          LN(price + 1)
StandardScaler (dynamic): needs AVG and STDDEV of its input
```

The compiler composes Log's expression INTO StandardScaler's discover query:

```sql
-- ONE query. Log's expression is inlined, not materialized.
SELECT
    AVG(LN(price + 1))        AS scaler__price__mean,
    STDDEV_POP(LN(price + 1)) AS scaler__price__std
FROM data
```

DuckDB reads `price` once, applies `LN(price + 1)`, computes both aggregates — single
pass. No temp table for the log-transformed values.

**The rule:** Within a layer, all expressions compose inline. Materialization only
happens at layer boundaries (when a dynamic step changes the schema).

### 16.4 Aggregation Batching Within Layers

All dynamic steps in the same layer share one aggregation query:

```
Layer 0: [Imputer (dynamic), StandardScaler (dynamic), OneHotEncoder (dynamic)]
```

```sql
-- ONE query for all three steps' aggregations
SELECT
    MEDIAN(price)      AS imputer__price__fill,
    AVG(price)         AS scaler__price__mean,
    STDDEV_POP(price)  AS scaler__price__std
FROM data;

-- Separate query for set discovery (different result shape)
SELECT DISTINCT city FROM data;
```

sklearn would run three separate full data scans here. sqlearn runs one (plus one for
distinct values which has a different result shape).

### 16.5 Layer Materialization (Minimized)

Layers exist because schema-changing dynamic steps create new columns that downstream
steps need to reference. Materialization happens via temp views:

```sql
-- Layer 0 output materialized as temp view
CREATE TEMP VIEW __layer_0 AS
SELECT
    COALESCE(price, 42.5) AS price,
    CASE WHEN city = 'London' THEN 1 ELSE 0 END AS city_london,
    CASE WHEN city = 'Paris' THEN 1 ELSE 0 END AS city_paris
FROM data;

-- Layer 1 can now reference the new columns
SELECT MIN(city_london), MAX(city_london), ... FROM __layer_0;
```

**Minimization rules:**
- Static schema-changing steps (e.g., DateParts adds columns without learning) do NOT
  create layer boundaries. Their output schema is known at pipeline construction time.
- Only **dynamic + schema-changing** steps force a boundary.
- Adjacent same-schema dynamic steps share a layer (no extra materialization).

### 16.6 Planned Optimizations (M5 — Search)

These optimizations target cross-validation, where the same pipeline is fitted multiple
times on different data splits. The M2 compiler designs its data structures to enable
these — even though the optimizations ship in M5.

#### FitPlan Caching

The FitPlan (layer structure + step classifications) depends on pipeline structure and
schema, NOT on data values. Same pipeline + same schema = same plan.

```python
# First fit: full classification + layer grouping
plan = plan_fit(steps, schema)        # O(N) classification calls

# CV folds 2-5: reuse the cached plan
plan = plan_fit(steps, schema)        # cache hit → O(1)
```

**Implementation:** Cache keyed on `(pipeline_id, schema_hash)`. Invalidated when
pipeline steps change or schema columns/types differ.

**Savings:** Eliminates N-1 full classification rounds per K-fold CV. For a 10-step
pipeline with 5-fold CV, this saves 40 classification calls.

#### Tier 3 Classification Cache

Custom transformers without `_classification` are inspected every fit. The cache key
includes the schema to handle schema-dependent behavior:

```python
cache_key = (type(step), frozenset(schema.columns.items()))
```

If the same class + same schema was previously classified, reuse the result. This is
safe because `discover()` is deterministic for a given (class, schema) pair.

**Edge case:** A custom transformer that uses randomness in `discover()` would bypass
this cache. This is pathological and documented as unsupported.

#### Aggregation Query Deduplication

If two steps in the same layer produce identical aggregation expressions (e.g., two
custom transformers that both compute `AVG(price)`), deduplicate via AST hashing:

```python
# Before dedup: 2 identical expressions
SELECT AVG(price) AS step1__price__mean,
       AVG(price) AS step2__price__mean    -- duplicate!
FROM data;

# After dedup: compute once, map to both
SELECT AVG(price) AS __dedup_0 FROM data;
# Map: step1__price__mean → __dedup_0, step2__price__mean → __dedup_0
```

**Implementation:** Hash sqlglot AST nodes. Identical hashes → same computation.
Map multiple parameter names to the same result.

#### Fold-Aware Materialization

In K-fold CV, layer materialization creates temp views per fold. If the materialization
query is identical across folds (e.g., a static-only layer), share the view:

```python
# Static-only layer: same output regardless of fold
# Create view ONCE, reuse across all folds

# Dynamic layer: different params per fold
# Create separate view per fold (unavoidable)
```

**Savings:** For pipelines with static preprocessing stages before the dynamic stages
(common pattern: Log → Clip → Imputer → Scaler), the static portion is materialized
once instead of K times.

#### Compile Once, Execute Many (QueryTemplate)

The self-contained CTE strategy (Section 4.7 of architecture) produces fold-parameterized
SQL — the same query works for any fold by changing `:k`. This optimization makes the
compilation step explicit and separates the pipeline (compiler) from the CV system (executor).

**The three-step flow:**

1. **Phase 1 — Superset discovery:** Run `discover_sets()` on full training data once.
   Learn the universe of categories, column types, output schema. This determines the
   AST structure — identical for every fold.

2. **Compile — Build QueryTemplate:** Compile the self-contained CTE query once from the
   superset schema. The result is a `QueryTemplate` — a frozen AST with `:k` as the only
   parameter. The pipeline acts as a compiler, not a state holder.

3. **Phase 2 — Per-fold execution:** For each fold, bind `:k` and execute. DuckDB computes
   per-fold scalar stats inline via `FILTER (WHERE __sq_fold__ != :k)`. No separate stats
   storage needed — the SQL IS the computation.

```python
@dataclass(frozen=True)
class QueryTemplate:
    """Pre-compiled AST with fold parameter slot.

    The pipeline compiles this once. The CV system executes it N times.
    No per-fold state lives on the pipeline — the template is the only artifact.
    """

    ast: sg.Expression          # self-contained CTE query with :k placeholder
    superset_schema: Schema     # output schema (consistent across all folds)
    fold_param: str             # parameter name (default: "__sq_fold__")

    def bind(self, fold: int) -> sg.Expression:
        """Substitute fold value, return executable AST."""
        return self.ast.transform(
            lambda node: exp.Literal.number(fold)
            if isinstance(node, exp.Parameter) and node.name == self.fold_param
            else node
        )
```

**Why the pipeline must not store fold state:** During CV, `fit()` is called per fold.
If `params_` is overwritten each time, previous fold stats are lost. With QueryTemplate,
there are no stored stats — the self-contained CTE computes them inline. The pipeline
compiles the template and hands it to the CV system. The CV system owns execution.

**Savings:** For a 10-step pipeline with 5-fold CV:
- Without template: 5 compilations × O(N) classification + AST build each
- With template: 1 compilation + 5 bind-and-execute (AST node swap is O(nodes))

#### Zero-Variance Column Skipping at Output Boundary

When Phase 1 uses the superset of categories (e.g., OneHotEncoder sees `{A, B, C}`
globally but fold 3's training data only has `{A, B}`), some folds produce columns
with all identical values (e.g., `city_C` is always 0 in fold 3).

These columns are handled at the **output boundary**, not in SQL:

- **SQL schema stays consistent** — all folds produce the same columns (guaranteed by
  superset discovery). This is critical for fold-to-fold comparability.
- **Constant columns are skipped at numpy handoff** — before passing to the model,
  check each column for zero variance. Constant columns are excluded from the feature
  matrix but tracked in metadata for reconstruction.
- **Detection is O(1) per column in DuckDB** — `SELECT MIN(col) = MAX(col)` uses
  zone maps (column statistics), no full scan needed.
- **This is NOT VarianceThreshold** — VarianceThreshold is an explicit pipeline step
  the user adds. This is an implicit optimization at the output layer that preserves
  SQL schema integrity while giving the model only informative features.

```python
# At the output boundary (inside TransformResult or CV executor):
# 1. Execute fold query → full result with all superset columns
# 2. Detect constant columns (O(1) per column via zone maps)
# 3. Exclude from feature matrix, record in metadata
# 4. Model trains on informative features only
# 5. All folds have consistent feature indexing after exclusion
```

### 16.7 CTE Promotion Strategy

Deep expression nesting hurts SQL readability and can hit database limits. The compiler
auto-promotes to CTEs at configurable depth:

**Default threshold: depth > 8.** Below this, expressions compose inline. Above this,
the compiler extracts the deepest subexpression into a CTE and references it by name.

```python
# Depth 1: COALESCE(price, 42.5)
# Depth 2: (COALESCE(price, 42.5) - 42.5)
# Depth 3: (COALESCE(price, 42.5) - 42.5) / 12.3
# Depth 4: CAST((COALESCE(price, 42.5) - 42.5) / 12.3 AS DOUBLE)
# ...
# Depth 9+: auto-promote to CTE
```

**Why 8?** DuckDB handles deep nesting fine, but:
- SQL readability degrades after ~8 levels
- Other databases (Postgres, BigQuery) may have lower limits
- CTEs let the optimizer share common subexpressions

The threshold is configurable via `Pipeline(cte_depth=N)` for users who know their
target database's limits.

### 16.8 What We Don't Optimize (And Why)

| Optimization | Why we skip it |
|---|---|
| Parallel query execution | DuckDB already parallelizes internally. Running multiple queries adds connection overhead. |
| Query plan caching | DuckDB's planner is fast enough (<1ms). Caching adds complexity for negligible gain. |
| Expression simplification | sqlglot handles this during transpilation. Adding our own simplifier risks correctness. |
| Async execution | DuckDB is synchronous. Wrapping in async adds complexity for no throughput gain on a single connection. |
| Connection pooling | One connection per backend by design. Pooling adds complexity for a use case we don't support (multi-tenant). |

### 16.9 Performance Design Principles

1. **Fewer queries beats faster queries.** Batching 5 aggregations into 1 query is
   better than making each individual query 5x faster.

2. **Inline beats materialize.** Expression composition (zero materialization) beats
   temp tables. Only materialize at layer boundaries when unavoidable.

3. **Cache structure, recompute values.** The FitPlan (which steps, which layers) is
   cached. The actual parameter values are always fresh from the data.

4. **Let DuckDB optimize.** Don't second-guess the query engine. Emit clean SQL,
   let DuckDB handle vectorization, parallelism, and memory management.

5. **Design for cacheability.** Use frozen dataclasses, hashable keys, and pure functions.
   Even if caching ships later, the data structures support it from day one.
