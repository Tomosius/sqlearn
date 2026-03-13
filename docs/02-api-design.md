> **sqlearn docs** — [Index](../CLAUDE.md) | Prev: [What sqlearn Is](01-overview.md) | Next: [Architecture](03-architecture.md)

## 3. API Design

### 3.1 The Boilerplate Problem (sklearn vs sqlearn)

This is sklearn's #1 pain point — ColumnTransformer:

```python
# sklearn: 15 lines of boilerplate for basic preprocessing
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, make_column_selector(dtype_include="number")),
    ("cat", categorical_transformer, make_column_selector(dtype_include="object")),
])
pipe = Pipeline([("preprocessor", preprocessor)])
pipe.fit(X_train)
X = pipe.transform(X_test)
```

sqlearn:

```python
# sqlearn: 5 lines, same result
import sqlearn as sq

pipe = sq.Pipeline([
    sq.Imputer(),
    sq.StandardScaler(),
    sq.OneHotEncoder(),
])
pipe.fit("train.parquet")
X = pipe.transform("test.parquet")
```

**The key insight: each transformer knows which columns it applies to by default.**

- `StandardScaler` → defaults to numeric columns
- `OneHotEncoder` → defaults to categorical columns
- `Imputer` → defaults to all columns (strategy adapts per type)

No ColumnTransformer needed for 80% of use cases. Column types are resolved from the
database schema at fit time — one `DESCRIBE` query.

### 3.2 Auto Column Routing

Every transformer has a `columns` parameter:

```python
sq.StandardScaler()                         # default: numeric columns (auto-detected)
sq.StandardScaler(columns=["price", "qty"]) # explicit list
sq.StandardScaler(columns=sq.numeric())     # explicit selector (same as default)
sq.StandardScaler(columns=sq.matching("price_*"))  # glob pattern
sq.StandardScaler(columns=sq.dtype("DOUBLE"))      # by SQL type
```

Each transformer class declares its default:

```python
class StandardScaler(Transformer):
    _default_columns = "numeric"     # auto-selects numeric columns

class OneHotEncoder(Transformer):
    _default_columns = "categorical" # auto-selects categorical columns

class Imputer(Transformer):
    _default_columns = "all"         # applies to everything
```

During fit, the pipeline reads the schema once (`DESCRIBE table`), resolves each
transformer's target columns, and caches the result. No ColumnTransformer needed.

### 3.3 Pipeline Input Formats

All three are equivalent:

```python
# Just transformers — auto-named ("imputer", "standardscaler", "onehotencoder")
sq.Pipeline([
    sq.Imputer(),
    sq.StandardScaler(),
    sq.OneHotEncoder(),
])

# Named tuples — sklearn compatible
sq.Pipeline([
    ("impute", sq.Imputer()),
    ("scale", sq.StandardScaler()),
    ("encode", sq.OneHotEncoder()),
])

# Dict — clean naming, no tuples
sq.Pipeline({
    "impute": sq.Imputer(),
    "scale": sq.StandardScaler(),
    "encode": sq.OneHotEncoder(),
})
```

### 3.4 Pipeline Nesting

A Pipeline IS a Transformer. Nesting is natural:

```python
# Sub-pipelines
numeric_pipe = sq.Pipeline([
    sq.Imputer(strategy="median"),
    sq.RobustScaler(),
])

categorical_pipe = sq.Pipeline([
    sq.Imputer(strategy="most_frequent"),
    sq.OrdinalEncoder(),
])

# Nest inside a parent pipeline
full = sq.Pipeline([
    numeric_pipe,
    categorical_pipe,
    sq.MinMaxScaler(),
])
```

Since everything compiles to SQL expressions, nesting costs nothing — it's just
AST composition.

### 3.5 Combining Pipelines

**Sequential — `+` operator (creates flat Pipeline):**

```python
cleaning = sq.Pipeline([sq.Imputer()])
features = sq.Pipeline([sq.StandardScaler(), sq.OneHotEncoder()])

# Option 1: nest in a new pipeline
full = sq.Pipeline([cleaning, features])

# Option 2: + operator — flattens into one Pipeline
full = cleaning + features
# → Pipeline([Imputer, StandardScaler, OneHotEncoder])  (flat, not nested)

# Works with bare transformers too
full = sq.Imputer() + sq.StandardScaler() + sq.OneHotEncoder()
# → Pipeline([Imputer, StandardScaler, OneHotEncoder])

# += for incremental building (great in notebooks)
pipe = sq.Pipeline([sq.Imputer()])
pipe += sq.StandardScaler()
pipe += sq.OneHotEncoder()
# → Pipeline([Imputer, StandardScaler, OneHotEncoder])
```

**Flattening rules:**
- `Transformer + Transformer` → `Pipeline([a, b])`
- `Pipeline + Transformer` → `Pipeline([*a.steps, b])`  (flat)
- `Pipeline + Pipeline` → `Pipeline([*a.steps, *b.steps])`  (flat)
- `Pipeline += Transformer` → creates new Pipeline with step appended (safe — no mutation)

Flattening is safe because nesting produces identical SQL — it's just AST composition.
If you WANT grouping (e.g., for naming), use explicit nesting: `Pipeline([pipe_a, pipe_b])`.

**Pipeline is immutable (value-type semantics).** All operations create new Pipelines,
never mutating the original. This follows the same convention as Python's `frozenset`
and `tuple` — container types that return new instances on modification. This prevents
a common notebook pitfall where rerunning cells keeps appending steps:

```python
base = sq.Pipeline([sq.Imputer()])
v1 = base + sq.StandardScaler()   # base is NOT modified
v2 = base + sq.RobustScaler()     # base is still just [Imputer]

# += creates a new Pipeline (like int += or tuple +=):
pipe = sq.Pipeline([sq.Imputer()])
pipe += sq.StandardScaler()   # pipe is now a NEW Pipeline([Imputer, StandardScaler])
pipe += sq.OneHotEncoder()    # pipe is now a NEW Pipeline([Imputer, StandardScaler, OneHotEncoder])
# No mutation of earlier references
```

**Why non-mutating `+=`?** Pipeline is explicitly an immutable value type, like
`frozenset` or `tuple`. Python's `+=` convention is that mutable containers (list, dict,
set) mutate in place, while immutable types (int, str, tuple, frozenset) return new
objects. Pipeline follows the immutable convention. Document this prominently so
users expecting list-like behavior understand the design choice.

**No `|` operator.** Union (parallel) is conceptually different from sequential and rare
enough to deserve explicit `sq.Union()`. Overloading `|` saves 10 characters but costs
discoverability.

**Parallel (combine outputs side by side — replaces FeatureUnion):**

```python
base_features = sq.Pipeline([sq.StandardScaler()])
interaction_features = sq.PolynomialFeatures(degree=2)

combined = sq.Union([
    ("base", base_features),
    ("interactions", interaction_features),
])
```

**Column routing (different transforms for different columns — replaces ColumnTransformer):**

Only needed when auto column routing isn't enough (e.g., different scalers for
different numeric columns):

```python
routed = sq.Columns({
    "revenue":    sq.RobustScaler(),
    "categories": sq.OrdinalEncoder(order=["low", "mid", "high"]),
    "ids":        sq.HashEncoder(n_bins=64),
}, remainder="passthrough")   # keep unrouted columns as-is
```

**When to use what:**

| Need | Use | Auto routing? |
|---|---|---|
| Scale all numerics, encode all categoricals | `sq.Pipeline` | Yes — transformers auto-select |
| Different scalers for different numeric groups | `sq.Columns` | No — must be explicit |
| Combine features from parallel pipelines | `sq.Union` | N/A |
| Chain pipelines sequentially | `sq.Pipeline`, `+`, or `+=` | Yes |

### 3.6 Data Input

DuckDB handles dispatch. User never thinks about loading:

```python
pipe.fit("table_name")                              # table already in DuckDB
pipe.fit("data.parquet")                             # local file
pipe.fit("s3://bucket/raw_data.parquet")             # S3 (DuckDB reads natively)
pipe.fit("data/*.csv")                               # glob of files
pipe.fit(pandas_df)                                  # registers as temp table
pipe.fit(polars_df)                                  # Arrow transfer, zero-copy
pipe.fit(arrow_table)                                # native DuckDB ingestion
pipe.fit(numpy_array, columns=["a", "b", "c"])       # registers as temp table
```

When a DataFrame/array is passed, it gets a deterministic temp table name
(`__sqlearn_input_0`). User can override: `pipe.fit(df, table_name="training")`.

### 3.7 Data Output

Phase 1: numpy only. Later phases add more.

```python
# Phase 1 — numpy (default, sklearn compatible)
X = pipe.transform("data")

# Phase 2 — other formats
X = pipe.transform("data", out="arrow")       # zero-copy from DuckDB
X = pipe.transform("data", out="pandas")
X = pipe.transform("data", out="polars")
rel = pipe.transform("data", out="relation")  # DuckDB relation (lazy, chainable)

# Phase 3 — streaming (constant memory)
for batch in pipe.transform("data", batch_size=100_000):
    model.partial_fit(batch)

# Direct to disk — data never fully in RAM
pipe.transform("huge.parquet", out="parquet", path="features.parquet")

# Data precision — cast output for faster model training
X = pipe.transform("data", dtype="float32")    # CAST all numeric to FLOAT
X = pipe.transform("data", dtype="float64")    # CAST all numeric to DOUBLE (default)

# SQL — no execution, just the query
sql = pipe.to_sql()                            # DuckDB dialect
sql = pipe.to_sql(dialect="snowflake")         # any sqlglot dialect
```

**`dtype` parameter:** Appends `CAST(col AS FLOAT)` or `CAST(col AS DOUBLE)` to every
numeric output column. Useful in GridSearch where `float32` halves memory and speeds up
model training with negligible accuracy loss. The cast is added at the outermost SELECT
level — it does not interfere with expression composition.

```python
# Search with float32 — 2x faster model fits
search = sq.Search(
    preprocessor=pipe, model=XGBClassifier(),
    space={...},
    cv=5,
    explore_dtype="float32",   # exploration rounds use float32 output
)
```

**`out="relation"` — DuckDB relational API:**

Returns a DuckDB `Relation` object instead of materializing data. The relation is lazy —
no computation until the user calls `.fetchdf()`, `.fetchnumpy()`, etc. Useful for
chaining additional DuckDB operations without materializing intermediate results:

```python
rel = pipe.transform("data", out="relation")
# Chain DuckDB operations — still lazy
filtered = rel.filter("price_scaled > 0").limit(1000)
# Materialize when ready
df = filtered.fetchdf()
```

**Large result warning (automatic):**

By default, `transform()` returns eager numpy — immediate materialization. For large
datasets this can OOM. sqlearn estimates result size from the query plan and warns:

```python
X = pipe.transform("huge_dataset.parquet")
# UserWarning: Estimated result size is ~4.2GB (50M rows × 84 columns × float64).
# Consider one of:
#   pipe.transform("data", out="parquet", path="features.parquet")  # write to disk
#   pipe.transform("data", batch_size=100_000)                       # stream batches
#   pipe.transform("data", out="relation")                           # lazy DuckDB Relation
#   pipe.transform("data", out="arrow")                              # Arrow (zero-copy)

# Configurable threshold (default 1GB):
sq.set_option("warn_result_size", "2GB")    # raise threshold
sq.set_option("warn_result_size", None)     # disable warning
```

The warning is informational only — the query still executes. This catches accidental
OOM without changing the default behavior or breaking sklearn compatibility.

**`TransformResult` — smart output wrapper:**

`transform()` returns a `TransformResult`, not a raw numpy array. It behaves like
a numpy array (for sklearn compatibility) but carries metadata for debugging:

```python
result = pipe.transform("data.parquet")

# Works exactly like numpy — sklearn never notices:
model.fit(result, y)                # __array__ protocol
result.shape                        # (50000, 42)
result[0:10]                        # slicing works
np.mean(result, axis=0)             # numpy ops work

# But also carries metadata:
result.columns                      # ["price", "city_london", "city_paris", ...]
result.dtypes                       # {"price": "DOUBLE", "city_london": "INTEGER", ...}
result.sql                          # the SQL query that produced this
result.to_dataframe()               # quick pandas/polars conversion with column names

# Convert to plain numpy when needed:
X = np.asarray(result)              # strips metadata, pure numpy
```

`TransformResult` is a lightweight wrapper with `__array__` protocol — NOT a numpy
subclass. numpy subclassing is notoriously fragile (`__array_ufunc__`, `__array_wrap__`,
`__array_finalize__` have subtle interactions across numpy versions; slicing may lose
metadata; `np.concatenate` strips subclasses). Instead, `TransformResult` implements
the `__array__` protocol and delegates common numpy properties:

```python
class TransformResult:
    """Wrapper around numpy array with metadata. NOT a numpy subclass."""

    def __init__(self, array, columns, dtypes, sql):
        self._array = array
        self.columns = columns
        self.dtypes = dtypes
        self.sql = sql

    def __array__(self, dtype=None):
        return self._array if dtype is None else self._array.astype(dtype)

    @property
    def shape(self):
        return self._array.shape

    def __getitem__(self, key):
        return self._array[key]

    def __len__(self):
        return len(self._array)

    def to_dataframe(self):
        """Quick pandas/polars conversion with column names."""
        import pandas as pd
        return pd.DataFrame(self._array, columns=self.columns)
```

Every sklearn function, XGBoost, LightGBM — they all accept it via `np.asarray(result)`.
The metadata is bonus information that helps debugging without breaking any existing
workflow. When numpy semantics are needed, `np.asarray(result)` gives a plain array.

```python
# Debugging: "which column is index 7?"
result = pipe.transform("data.parquet")
print(result.columns[7])            # "city_london"
print(result.sql)                   # SELECT (COALESCE(price, ...) ...

# sklearn/XGBoost accept it transparently via __array__ protocol:
model.fit(result, y)                # np.asarray() called internally
np.mean(np.asarray(result), axis=0) # explicit conversion for numpy ops
```

### 3.8 Backend (DuckDB Connection)

```python
# Default: in-memory DuckDB, auto-created on first use. Zero config.
pipe.fit("data.parquet")

# Persistent file — survives restarts, caches data
pipe = sq.Pipeline([...], backend="my_data.duckdb")

# Or set globally (convenience for scripts/notebooks)
sq.set_backend("my_data.duckdb")

# Existing connection
import duckdb
conn = duckdb.connect("warehouse.duckdb")
pipe.fit("data", backend=conn)

# Per-call override (highest priority)
pipe.fit("data", backend="prod.duckdb")

# No backend needed — just compile
pipe.to_sql(dialect="snowflake")
```

**Precedence:** per-call `backend=` > pipeline `backend=` > global `set_backend()` > auto-create in-memory.

**Persistent vs in-memory:**

| Mode | When to use |
|---|---|
| In-memory (default) | Quick experiments, small-medium data, no setup |
| Persistent file | Large datasets (DuckDB caches to disk), share across sessions |

### 3.9 Multi-Source Data & Merging

DuckDB supports ATTACH — connect to external databases from within DuckDB:

```python
# Attach external sources
# WARNING: attach() connects to real databases. Queries run by fit() and
# transform() may execute expensive operations (full table scans for discover()).
# Always use read-only credentials. Never attach to production write replicas.
sq.attach("postgres://readonly-replica/analytics", name="prod")
sq.attach("sqlite:///local.db", name="local")

# Pipeline reads from Postgres, processes in DuckDB
pipe.fit("prod.training_data")

# Cross-source joins
pipe.fit("SELECT * FROM prod.users JOIN local.features USING (user_id)")
```

**Rule: one pipeline = one DuckDB engine.** DuckDB is the hub that connects to
everything. Multi-database means DuckDB attaches external sources, not that pipeline
steps run on different engines.

**`sq.merge()` — SQL JOIN wrapper:**

Pre-pipeline data merging. Combines multiple sources before transformation:

```python
# Inner join (default)
merged = sq.merge("users.parquet", "orders.parquet", on="user_id")
pipe.fit(merged)

# Left join with explicit columns
merged = sq.merge(
    "users.parquet",
    "orders.parquet",
    on="user_id",
    how="left",         # "inner", "left", "right", "outer", "cross"
)

# Multiple keys
merged = sq.merge("a.parquet", "b.parquet", on=["user_id", "date"])

# Different column names
merged = sq.merge("a.parquet", "b.parquet", left_on="uid", right_on="user_id")

# Chain multiple merges
data = sq.merge("users.parquet", "orders.parquet", on="user_id")
data = sq.merge(data, "products.parquet", on="product_id")
pipe.fit(data)
```

Under the hood, `sq.merge()` returns a SQL view name — no data copying.
View names use content-based hashing (not sequential counters) to avoid collisions
between unrelated pipelines, serialization roundtrips, and concurrent usage:

```python
# Deterministic view name from merge specification
view_name = f"__sq_merge_{hashlib.md5(repr((left, right, on, how)).encode()).hexdigest()[:8]}__"
```

```sql
CREATE TEMP VIEW __sq_merge_a1b2c3d4__ AS
SELECT * FROM 'users.parquet' a
LEFT JOIN 'orders.parquet' b ON a.user_id = b.user_id
```

**`sq.concat()` — UNION ALL wrapper:**

Stack datasets vertically (same schema):

```python
# Concatenate files
combined = sq.concat(["2023.parquet", "2024.parquet"])
pipe.fit(combined)

# With source column — track which file each row came from
combined = sq.concat(
    ["2023.parquet", "2024.parquet"],
    source_column="year_file",    # adds column with filename
)
```

```sql
SELECT *, '2023.parquet' AS year_file FROM '2023.parquet'
UNION ALL
SELECT *, '2024.parquet' AS year_file FROM '2024.parquet'
```

**`sq.Lookup` — join features from another table (as a pipeline step):**

Unlike `sq.merge()` (pre-pipeline), `Lookup` is a transformer that joins features
mid-pipeline. Useful for enrichment tables:

```python
pipe = sq.Pipeline([
    sq.Imputer(),
    sq.Lookup(                           # join features from another table
        table="city_stats.parquet",
        on="city",                       # join key
        features=["population", "median_income", "state"],
    ),
    sq.StandardScaler(),
    sq.OneHotEncoder(),
])
```

SQL: `LEFT JOIN city_stats ON data.city = city_stats.city` — added as a CTE via
the `query()` interface.

**Pre-pipeline vs mid-pipeline:**

| Need | Use | When |
|---|---|---|
| Combine raw data before any processing | `sq.merge()`, `sq.concat()` | Before `pipe.fit()` |
| Enrich with lookup table during pipeline | `sq.Lookup(table=, on=, features=)` | Inside pipeline |
| Complex multi-table ETL | Write SQL, pass as input | Before `pipe.fit()` |

Data preparation (complex joins, filtering, aggregation) is a pre-pipeline concern.
The pipeline transforms features, it doesn't do ETL. Pass SQL or a view name as input:

```python
pipe.fit("SELECT u.*, SUM(o.amount) as total FROM users u JOIN orders o ...")
```

### 3.10 Custom Transformers — Three Levels

Three ways to write custom transformers, from simple to full-power. All three
are validated, type-checked, and compose safely with built-in transformers.

Pick the simplest level that handles your need:

| Need | Level | API | Complexity |
|---|---|---|---|
| Add a calculated column | 1 | `sq.Expression()` | One line |
| Per-column transform, optionally learn stats | 2 | `sq.custom()` | 2-8 lines |
| Sets of values, CTEs, joins, full control | 3 | `sq.Transformer` subclass | Class (~20 lines) |

#### Level 1: `sq.Expression()` — Static One-Liners

For simple calculated columns that don't learn from data. Write SQL, get a column.

```python
# New column from existing columns
sq.Expression("price * quantity AS revenue")
sq.Expression("CASE WHEN price > 100 THEN 'high' ELSE 'low' END AS tier")
sq.Expression("COALESCE(nickname, first_name) AS display_name")

# Use in pipeline like any transformer
pipe = sq.Pipeline([
    sq.Imputer(),
    sq.Expression("price * quantity AS revenue"),
    sq.StandardScaler(),
])
```

**How it works:** The SQL string is parsed through sqlglot (validated, multi-database safe).
The `AS name` clause defines the output column. Original columns pass through unchanged.

**Limitations:** Static only (no learning from data). One expression per instance. For
per-column transforms or data-dependent logic, use Level 2.

#### Level 2: `sq.custom()` — Template-Based (Covers 90% of Custom Needs)

Write SQL templates with `{col}` and `{param}` placeholders. sqlearn handles column
resolution, parameter discovery, schema tracking, and validation automatically.

**Static per-column transform (no learning):**

```python
# Apply to each numeric column — {col} expands per column
log = sq.custom("LN({col} + 1)", columns="numeric")

# Explicit column list
clip = sq.custom("GREATEST(LEAST({col}, 100), 0)", columns=["price", "score"])

# Category mapping (static, no learning)
flag = sq.custom(
    "CASE WHEN {col} IN ('US', 'CA', 'MX') THEN 'north_america' ELSE 'other' END",
    columns=["country"],
)
```

By default, `sq.custom()` **replaces** the source column with the transformed version
(same behavior as `StandardScaler`). The column name stays the same.

**Adding new columns (keep original alongside):**

```python
# AS {col}_log → adds new column, keeps original
log_cols = sq.custom("LN({col} + 1) AS {col}_log", columns="numeric")

# Fixed name for cross-column expression
bmi = sq.custom(
    "weight / (height * height) * 703 AS bmi",
    columns=["weight", "height"],
    mode="combine",  # combine multiple columns into one, don't iterate
)
```

When the template contains `AS new_name`, a new column is added to the output.
When it doesn't, the source column is replaced in-place.

**Dynamic transform (learns from data):**

```python
# Center each numeric column by its mean
center = sq.custom(
    "{col} - {mean}",
    columns="numeric",
    learn={"mean": "AVG({col})"},
)

# Z-score normalization (multiple learned params)
z_score = sq.custom(
    "({col} - {mean}) / {std}",
    columns="numeric",
    learn={
        "mean": "AVG({col})",
        "std": "STDDEV_POP({col})",
    },
)

# Percentile clipping with learned bounds
winsorize = sq.custom(
    "GREATEST(LEAST({col}, {p95}), {p05})",
    columns="numeric",
    learn={
        "p05": "PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY {col})",
        "p95": "PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY {col})",
    },
)
```

**`learn=` makes the transform dynamic.** During `fit()`, each SQL template in `learn`
is executed per target column. Results are stored as `{col}__{param}` (e.g.,
`price__mean = 65420.0`). During `transform()`, `{param}` placeholders are replaced
with the learned values.

**Dynamic with new output columns:**

```python
# Flag outliers (new column per source column)
outlier_flag = sq.custom(
    "CASE WHEN ABS({col} - {mean}) / {std} > 3 THEN 1 ELSE 0 END AS {col}_outlier",
    columns="numeric",
    learn={
        "mean": "AVG({col})",
        "std": "STDDEV_POP({col})",
    },
)

# Percentile rank (new column alongside original)
pct_rank = sq.custom(
    "({col} - {min}) / NULLIF({max} - {min}, 0) AS {col}_pct",
    columns="numeric",
    learn={
        "min": "MIN({col})",
        "max": "MAX({col})",
    },
)
```

**Using in pipelines — works like any built-in transformer:**

```python
pipe = sq.Pipeline([
    sq.Imputer(),
    sq.custom("LN({col} + 1)", columns=["price", "income"]),
    sq.custom(
        "({col} - {mean}) / {std}",
        columns="numeric",
        learn={"mean": "AVG({col})", "std": "STDDEV_POP({col})"},
    ),
    sq.OneHotEncoder(),
])
pipe.fit("train.parquet", y="target")
X = pipe.transform("test.parquet")  # works like any pipeline
```

**What `sq.custom()` handles automatically:**

| Concern | How it's handled |
|---|---|
| SQL safety | All templates parsed through sqlglot at creation time |
| Multi-database | Templates compile to any sqlglot dialect |
| Column resolution | `columns="numeric"` resolved from schema at fit time |
| Parameter naming | `{col}__{param}` convention, same as built-in transforms |
| Passthrough | Unmentioned columns pass through (base class handles it) |
| Schema tracking | Output columns auto-inferred from `AS` clauses in template |
| Static/dynamic | Auto-detected: `learn=` present → dynamic, else static |
| Null handling | SQL NULL propagation by default (matches SQL semantics) |
| Validation | Templates validated at creation, params validated at fit |

**What `sq.custom()` does NOT handle (use Level 3):**

- Learning sets of values (distinct categories, per-category means)
- Window functions (RANK, LAG, LEAD, ROW_NUMBER)
- Joins (enrichment from external tables)
- Multi-step transforms that need CTEs
- Transforms where the SQL structure changes based on discovered data

#### Level 3: `sq.Transformer` Subclass — Full Power

For transforms that need sets of values, window functions, joins, or full control
over the SQL generation. Write a class with sqlglot ASTs.

**When to use Level 3:**
- You need `discover_sets()` (learning categories, mappings, per-group values)
- You need `query()` (window functions, joins, CTEs)
- Your SQL structure depends on what's discovered from data
- You need fine-grained control over schema changes

**Simple static example (1 method):**

```python
import sqlglot.expressions as exp

class SquareRoot(sq.Transformer):
    _default_columns = "numeric"
    _classification = "static"  # optional: declare for zero-cost classification

    def expressions(self, columns, exprs):
        return {col: exp.Sqrt(this=exprs[col]) for col in columns}
```

**Dynamic example (2-3 methods):**

```python
class OutlierFlagger(sq.Transformer):
    _default_columns = "numeric"

    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def discover(self, columns, schema, y_column=None):
        """What SCALAR values to learn from data during fit.
        Returns {param_name: sqlglot_aggregate_expression}."""
        return {
            f"{col}__p": exp.PercentileCont(
                this=exp.Literal.number(self.threshold),
                expression=exp.Order(expressions=[exp.Column(this=col)])
            )
            for col in columns
        }

    def expressions(self, columns, exprs):
        """Transform expressions. Return ONLY modified/new columns.
        Unmentioned columns pass through automatically (base class handles it)."""
        result = {}
        for col in columns:
            p = self.params_[f"{col}__p"]
            result[f"{col}_outlier"] = (
                exp.Case()
                .when(exprs[col].gt(exp.Literal.number(p)), exp.Literal.number(1))
                .else_(exp.Literal.number(0))
            )
        return result

    def output_schema(self, schema):
        """Declare new columns. Required when adding/removing columns."""
        return schema.add({f"{col}_outlier": "INTEGER" for col in self.columns_})
```

**Set-valued discovery (categories, mappings):**

```python
class CustomEncoder(sq.Transformer):
    _default_columns = "categorical"

    def discover_sets(self, columns, schema, y_column=None):
        """Learn sets of values (not single aggregates).
        Returns {param_name: sqlglot_query} → results stored in self.sets_"""
        return {
            f"{col}__categories": exp.Select(
                expressions=[exp.Column(this=col)]
            ).from_(exp.Table(this="__source__")).distinct()
            for col in columns
        }

    def expressions(self, columns, exprs):
        result = {}
        for col in columns:
            categories = self.sets_[f"{col}__categories"]
            # Build CASE WHEN city='London' THEN 0 WHEN city='Paris' THEN 1 ...
            case = exp.Case()
            for i, cat_row in enumerate(categories):
                val = cat_row[col]
                case = case.when(
                    exprs[col].eq(exp.Literal.string(val)),
                    exp.Literal.number(i)
                )
            result[col] = case.else_(exp.Literal.number(-1))
        return result
```

**Query-level transform (window functions, joins):**

```python
class PercentileRank(sq.Transformer):
    _default_columns = "numeric"
    _classification = "static"  # window doesn't need learned values

    def query(self, input_query):
        """Wrap the input query with window functions.
        Returns a new sqlglot SELECT wrapping the input."""
        cols = [exp.Column(this=c) for c in self.columns_]
        rank_exprs = [
            exp.Alias(
                this=exp.Window(
                    this=exp.Anonymous(this="PERCENT_RANK"),
                    partition_by=[],
                    order=exp.Order(expressions=[exp.Column(this=c)])
                ),
                alias=f"{c}_pctrank"
            )
            for c in self.columns_
        ]
        return exp.Select(
            expressions=[exp.Star()] + rank_exprs
        ).from_(exp.Subquery(this=input_query))
```

**Summary of the three-method interface:**

| Method | Required? | Returns | Purpose |
|---|---|---|---|
| `discover()` | No | `{name: sqlglot_aggregate}` | Scalar stats to learn (mean, std, min, max, percentile) |
| `discover_sets()` | No | `{name: sqlglot_query}` | Set-valued stats (categories, per-group values) |
| `expressions()` | Yes* | `{col: sqlglot_expr}` | Inline SQL expressions (most transforms) |
| `query()` | Yes* | `sqlglot_select` | Full query wrapping input (windows, joins, CTEs) |
| `output_schema()` | No | `Schema` | Override when adding/removing columns |

*One of `expressions()` or `query()` is required. If both are defined, `query()` takes
precedence. If `query()` returns `None`, falls back to `expressions()`.

### 3.10b Custom Transformer Validation

Every custom transformer — at every level — is validated automatically. You don't
opt in. Safety is the default. Errors are caught early with clear messages.

**At creation time (Level 2 `sq.custom()`):**
- SQL template is parsed through sqlglot — syntax errors caught immediately
- Placeholders (`{col}`, `{param}`) are validated against `learn=` keys
- Output column names inferred from `AS` clauses

**At fit time (first use in a pipeline):**

```python
# The base class validates custom transformers automatically on first fit:

# 1. Type checking — discover() must return sqlglot ASTs, not values
#    BAD:  return {"mean": 42.0}
#    GOOD: return {"mean": exp.Avg(this=exp.Column(this=col))}
#    Error: "discover() must return sqlglot expressions, got float for key 'mean'.
#            Return an aggregate expression like exp.Avg(this=exp.Column(this='col'))"

# 2. Type checking — expressions() must return sqlglot ASTs, not strings
#    BAD:  return {"price": "price - 42.0"}
#    GOOD: return {"price": exp.Sub(this=exprs["price"], expression=exp.Literal.number(42))}
#    Error: "expressions() must return sqlglot expressions, got str for key 'price'.
#            Return a sqlglot expression like exp.Sub(this=..., expression=...)"

# 3. Schema consistency — new columns require output_schema()
#    BAD:  expressions() returns {"price_log": ...} but output_schema() not overridden
#    Error: "expressions() adds new column 'price_log' but output_schema() returns
#            unchanged schema. Override output_schema() to declare new columns:
#              def output_schema(self, schema):
#                  return schema.add({'price_log': 'DOUBLE'})"

# 4. Param consistency — expressions() can't reference undiscovered params
#    BAD:  expressions() uses self.params_["mean"] but discover() didn't define "mean"
#    Error: "expressions() references param 'mean' but discover() didn't define it.
#            Add to discover(): {'mean': exp.Avg(this=exp.Column(this=col))}"

# 5. Column collision — new column names can't shadow existing columns
#    BAD:  expressions() returns {"price": new_expr, "price": other_expr}
#    Error: "Column name collision: 'price' produced by both Imputer (step 1) and
#            OutlierFlagger (step 3). Rename one output column."
```

**At transform time (optional strict mode):**

```python
# Pipeline-level validation
pipe = sq.Pipeline([...], validate="strict")

# Checks after each step:
# - Output column count matches declared schema
# - Output types match declared schema
# - No unexpected NULL introduction (warns if null count increases)
# - No NaN/Inf values in numeric output
```

**Classification validation (Tier 2 custom transformers):**

If a custom transformer declares `_classification = "static"` but `discover()`
returns non-empty params, the system catches it:

```python
class BadTransformer(sq.Transformer):
    _classification = "static"  # WRONG — discover returns params

    def discover(self, columns, schema, y_column=None):
        return {"mean": exp.Avg(this=exp.Column(this=columns[0]))}

# At first fit():
# ClassificationError: BadTransformer declares _classification='static' but
# discover() returned params {'mean': ...}. Either:
#   1. Change to _classification='dynamic' (it learns from data)
#   2. Remove the discover() override (make it truly static)
```

### 3.10c Common Custom Transformer Patterns

**Pattern A: Combine columns into a new feature**

```python
# Level 1 — simple expression
sq.Expression("sqft / bedrooms AS sqft_per_bedroom")

# Level 2 — parameterized
rooms_ratio = sq.custom(
    "sqft / NULLIF(bedrooms, 0) AS sqft_per_bedroom",
    columns=["sqft", "bedrooms"],
    mode="combine",
)
```

**Pattern B: Per-column transform with learned stats**

```python
# Level 2 — center and scale (what StandardScaler does)
my_scaler = sq.custom(
    "({col} - {mean}) / NULLIF({std}, 0)",
    columns="numeric",
    learn={
        "mean": "AVG({col})",
        "std": "STDDEV_POP({col})",
    },
)
```

**Pattern C: Flag rows based on learned thresholds**

```python
# Level 2 — flag values beyond 3 standard deviations
outlier_flag = sq.custom(
    "CASE WHEN ABS({col} - {mean}) / NULLIF({std}, 0) > 3 THEN 1 ELSE 0 END AS {col}_outlier",
    columns="numeric",
    learn={"mean": "AVG({col})", "std": "STDDEV_POP({col})"},
)
```

**Pattern D: Bin continuous values by learned quantiles**

```python
# Level 2 — tercile binning
bin_tercile = sq.custom(
    "CASE WHEN {col} <= {p33} THEN 0 WHEN {col} <= {p66} THEN 1 ELSE 2 END AS {col}_bin",
    columns="numeric",
    learn={
        "p33": "PERCENTILE_CONT(0.33) WITHIN GROUP (ORDER BY {col})",
        "p66": "PERCENTILE_CONT(0.66) WITHIN GROUP (ORDER BY {col})",
    },
)
```

**Pattern E: Boolean feature from string column**

```python
# Level 1 — simple check
sq.Expression("CASE WHEN description LIKE '%premium%' THEN 1 ELSE 0 END AS is_premium")

# Level 2 — multiple keywords
keyword_flag = sq.custom(
    "CASE WHEN LOWER({col}) LIKE '%urgent%' OR LOWER({col}) LIKE '%asap%' THEN 1 ELSE 0 END AS {col}_urgent",
    columns=["subject", "body"],
)
```

**Pattern F: Window function (Level 3 only)**

```python
class RollingZScore(sq.Transformer):
    """Z-score relative to a rolling window (requires CTE/window)."""
    _default_columns = "numeric"
    _classification = "static"

    def __init__(self, window=30):
        self.window = window

    def query(self, input_query):
        selects = [exp.Star()]
        for col in self.columns_:
            roll_mean = exp.Window(
                this=exp.Avg(this=exp.Column(this=col)),
                spec=exp.WindowSpec(
                    order=exp.Order(expressions=[exp.Column(this="__sq_rownum__")]),
                    kind="ROWS",
                    start=exp.Literal.number(self.window),
                    start_side="PRECEDING",
                )
            )
            roll_std = exp.Window(
                this=exp.StddevPop(this=exp.Column(this=col)),
                spec=exp.WindowSpec(
                    order=exp.Order(expressions=[exp.Column(this="__sq_rownum__")]),
                    kind="ROWS",
                    start=exp.Literal.number(self.window),
                    start_side="PRECEDING",
                )
            )
            selects.append(exp.Alias(
                this=exp.Div(
                    this=exp.Sub(this=exp.Column(this=col), expression=roll_mean),
                    expression=exp.Nullif(this=roll_std, expression=exp.Literal.number(0)),
                ),
                alias=f"{col}_rolling_zscore",
            ))
        return exp.Select(expressions=selects).from_(exp.Subquery(this=input_query))

    def output_schema(self, schema):
        return schema.add({f"{col}_rolling_zscore": "DOUBLE" for col in self.columns_})
```

### 3.10d What Custom Transformers Cannot Do (and Why)

| Limitation | Reason | Workaround |
|---|---|---|
| Can't use Python functions on data | SQL-only — Python would break the single-query model | Write the logic in SQL, or preprocess outside the pipeline |
| Can't train models (PCA, embeddings) | Not SQL-expressible (iterative algorithms) | Use sklearn for that step, sqlearn for the rest |
| Can't access other steps' params | Steps are independent by design (composability) | Use pipeline `params_` dict for inspection |
| Can't mutate pipeline state | Immutability prevents subtle bugs | Return new values from discover/expressions |
| Can't execute arbitrary SQL | All SQL goes through sqlglot for safety | Use `sq.Expression()` which parses through sqlglot |

### 3.11 Inspection & Debugging

**Pipeline inspection:**

```python
pipe.to_sql()                     # see the compiled SQL
pipe.describe()                   # human-readable summary of steps + learned params
pipe.explain()                    # DuckDB query plan

pipe.preview("data", n=5)         # first 5 rows of transformed output
pipe.diff("data", n=5)            # side-by-side: original → transformed

pipe.lineage("price_scaled")      # price_scaled ← StandardScaler ← Imputer ← raw price
pipe.get_feature_names_out()      # list of output column names (sklearn compatible)

pipe.validate("new_data")         # check schema compatibility, dry-run SQL
```

**Dataset analysis (see Section 8 for full details):**

```python
sq.profile("data.parquet")                         # quick overview: types, nulls, stats
sq.analyze("data.parquet", target="price")         # correlations, multicollinearity, suggestions
sq.recommend("data.parquet", target="price")       # model + pipeline recommendations
sq.autopipeline("data.parquet", target="price")    # generate a complete Pipeline
```

---
