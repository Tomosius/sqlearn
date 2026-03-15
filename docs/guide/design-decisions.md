# Design Decisions

This document explains the reasoning behind sqlearn's most consequential design
choices. Each section covers what was chosen, why, what alternatives were
considered, and what trade-offs were accepted.

If you have ever wondered "why does sqlearn do it this way?" -- this is the page
for you.

---

## Architecture

### Why sqlglot ASTs instead of raw SQL strings?

**What we chose:** Every SQL expression in sqlearn is a sqlglot abstract syntax
tree node. Transformers return `exp.Sub(...)`, `exp.Case(...)`, etc. -- never
`"price - 42.5"` or `f"CASE WHEN {col} = 'London' ..."`.

**Why:** Multi-database support from day one. A sqlglot AST can be transpiled to
DuckDB, Postgres, Snowflake, BigQuery, MySQL, or any other dialect with a single
call to `.sql(dialect="snowflake")`. Raw strings lock you to one dialect's syntax
and make injection bugs trivial.

**Alternatives considered:**

- **Raw SQL strings with dialect-specific escaping.** Simpler to write initially
  but creates a combinatorial explosion of edge cases (quoting rules, type
  names, function names). Every new dialect would require auditing every
  transformer.
- **A custom intermediate representation.** Would need its own parser,
  optimizer, and code generator. sqlglot already solves this problem well.

**Trade-offs:**

- Writing `exp.Sub(this=exprs["price"], expression=exp.Literal.number(42.5))`
  is more verbose than writing `"price - 42.5"`. This is real friction,
  especially for custom transformers.
- The `sq.custom()` function (Level 2 custom transformers) exists largely to
  soften this trade-off: you write SQL strings that are parsed through sqlglot
  at creation time, getting readability with AST safety.

```python
# Level 2: readable SQL, parsed through sqlglot at creation time
center = sq.custom(
    "{col} - {mean}",
    columns="numeric",
    learn={"mean": "AVG({col})"},
)

# Level 3: full AST control when you need it
class Centerer(sq.Transformer):
    def expressions(self, columns, exprs):
        return {
            col: exp.Sub(
                this=exprs[col],
                expression=exp.Literal.number(self.params_[f"{col}__mean"]),
            )
            for col in columns
        }
```

---

### Why one base class instead of sklearn's mixin hierarchy?

**What we chose:** All sqlearn transformers inherit from a single `Transformer`
class that provides `discover()`, `expressions()`, `query()`, `output_schema()`,
`get_params()`, `set_params()`, `get_feature_names_out()`, `clone()`, and
thread safety -- all in one place.

**Why:** sklearn's class hierarchy is a historical accident. To make a
`StandardScaler`, you need `BaseEstimator` (for `get_params`/`set_params`),
`TransformerMixin` (for `fit_transform`), and `OneToOneFeatureMixin` (for
`get_feature_names_out`). That is three base classes, plus a tags system,
parameter validation conventions, and the trailing-underscore fitted-check
convention. Eight or more interacting pieces.

This happened because each mixin was added to solve one problem without
rethinking the whole hierarchy. `ClassifierMixin` adds `score()`,
`RegressorMixin` adds a different `score()`, `TransformerMixin` adds
`fit_transform()`, `OneToOneFeatureMixin` adds `get_feature_names_out()` for
same-shape transforms, `ClassNamePrefixFeaturesOutMixin` adds it for
different-shape transforms...

sqlearn does not have this historical baggage. One class, five override points
(`discover`, `discover_sets`, `expressions`, `query`, `output_schema`), clear
contracts.

**Alternatives considered:**

- **Mimic sklearn's mixin pattern.** Would ease migration for sklearn
  contributors, but it imports the same complexity problems. No one has ever
  said "I wish sklearn had more mixins."
- **Protocol-based approach (structural typing).** Considered a `HasDiscover`
  protocol instead of inheritance. Rejected because base-class defaults (empty
  `discover()` for static transforms, auto-passthrough in `_apply_expressions()`)
  provide too much value to give up.

**Trade-offs:**

- One class means every transformer carries the full interface. A static
  transformer that only needs `expressions()` still has `discover()` and
  `query()` available. This is intentional -- it keeps the contract uniform and
  makes classification straightforward.
- No separate `Classifier` or `Regressor` base. sqlearn does not train models,
  so this is not a limitation.

---

### Why auto column routing instead of explicit ColumnTransformer?

**What we chose:** Each transformer declares which column types it targets by
default. `StandardScaler` targets numeric columns. `OneHotEncoder` targets
categorical columns. `Imputer` targets all columns. This is resolved
automatically from the database schema at fit time.

```python
# sqlearn: 5 lines
pipe = sq.Pipeline([
    sq.Imputer(),
    sq.StandardScaler(),
    sq.OneHotEncoder(),
])
pipe.fit("train.parquet")
```

```python
# sklearn: 15 lines for the same thing
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
```

**Why:** The ColumnTransformer is sklearn's single biggest source of
boilerplate. In 80% of cases, you want "scale the numbers, encode the
categories." Auto routing makes that the default instead of requiring 15 lines
of plumbing.

Column types come from the database schema (`DESCRIBE table`), which is
authoritative. There is no ambiguity about whether a column is numeric or
categorical -- the database already knows.

**Alternatives considered:**

- **Require explicit column lists everywhere.** Safer but tedious. The `columns=`
  parameter exists for the 20% of cases where auto routing is not enough.
- **Only support ColumnTransformer-style routing.** Familiar to sklearn users
  but inherits its verbosity problems.

**Trade-offs:**

- Auto routing means a transformer might target columns you did not expect.
  `StandardScaler()` on a table with an integer ID column will scale that ID.
  This is why sqlearn emits fit-time warnings for likely identifier columns.
- When auto routing is wrong, you need the `columns=` parameter or the
  `sq.Columns()` explicit router:

```python
sq.Columns({
    "revenue":    sq.RobustScaler(),
    "categories": sq.OrdinalEncoder(order=["low", "mid", "high"]),
}, remainder="passthrough")
```

---

### Why static vs. dynamic classification?

**What we chose:** Every transformer is classified as either *static* (needs no
data statistics -- e.g., `Log`, `Rename`) or *dynamic* (learns from data --
e.g., `StandardScaler`, `OneHotEncoder`). This classification determines
whether the transformer contributes to the fit query.

Classification uses a three-tier model:

1. **Tier 1 (built-in, declared):** Trusted immediately. Validated by CI, never
   at runtime.
2. **Tier 2 (custom, declared):** Verified once at first `fit()`, then cached.
3. **Tier 3 (custom, undeclared):** Full conservative inspection every `fit()`.

**Why:** The compiler needs to know which steps require data to build the fit
query plan. Static steps contribute zero cost to the fit query -- they compose
their expressions without touching the database. Misclassifying a dynamic step
as static would silently skip learning, corrupting the pipeline output.

**Alternatives considered:**

- **Always inspect discover() at runtime.** Safe but wasteful. Calling
  `Log().discover()` every fit just to confirm it returns `{}` is pointless
  overhead for built-in transformers that we already tested in CI.
- **Require every transformer to declare.** Simpler but hostile to custom
  transformers. A user writing a quick `sq.custom()` should not need to think
  about classification.

**Trade-offs:**

- Three tiers add complexity to the compiler. But the safety rule is simple:
  if in doubt, classify as dynamic. The cost of a false "dynamic" is one cheap
  aggregation query. The cost of a false "static" is data corruption.
- Tier 3 inspection runs `discover()` to check if it returns `{}`. This means
  a buggy custom transformer that raises in `discover()` gets classified as
  dynamic (safe fallback).

---

### Why expression composition into a single SELECT?

**What we chose:** The compiler chains `expressions()` calls from each step,
building nested sqlglot AST nodes. The final result is one SQL SELECT
statement. Three Python steps become one query:

```sql
SELECT
    (COALESCE(price, 42.5) - 42.5) / 12.3 AS price,
    CASE WHEN COALESCE(city, 'Unknown') = 'London' THEN 1 ELSE 0 END AS city_london,
    CASE WHEN COALESCE(city, 'Unknown') = 'Paris' THEN 1 ELSE 0 END AS city_paris
FROM data
```

**Why:** Performance. DuckDB processes all column transformations in one
vectorized pass over the data. Compare to sklearn, which allocates a new numpy
array per step (N allocations for N steps), and to Polars lazy mode, which
still forces materialization at sklearn step boundaries.

A single SELECT also means a single query plan that DuckDB can optimize as a
whole -- predicate pushdown, column pruning, and parallelization all happen
naturally.

**Alternatives considered:**

- **One CTE per step.** Simpler to implement but generates unnecessarily
  complex SQL. DuckDB handles it fine, but the generated SQL is harder for
  humans to read and for other databases to optimize.
- **Materialize intermediate results as temp tables.** Guaranteed isolation
  between steps but destroys the performance advantage. Each materialization
  is a full data copy.

**Trade-offs:**

- Deep nesting can produce very large SQL expressions. The compiler auto-promotes
  to CTEs when expression depth exceeds 8 (configurable), keeping the SQL
  manageable.
- Steps that need their own query level (window functions, joins) use `query()`
  instead of `expressions()` and are automatically promoted to CTEs.

---

## API Design

### Why is `y` a column name, not a numpy array?

**What we chose:** The target variable is passed as a string:
`pipe.fit("data.parquet", y="price")`. Multi-target is
`y=["target1", "target2"]`. The target is a column in the same table, not a
separate array.

**Why:** This is the SQL-native answer. The target column is already in the
data. Extracting it into a numpy array just to pass it back is a round-trip
that serves no purpose. More importantly, it means transformers that need the
target (like `TargetEncoder`) can reference it by name in their SQL aggregations
without ever materializing it in Python.

Target columns are automatically excluded from `transform()` output by default.
`pipe.transform("data", exclude_target=False)` includes them when needed (for
EDA, debugging, or chaining pipelines).

**Alternatives considered:**

- **Accept both string and array (sklearn API).** Would require detecting the
  type at runtime and two code paths. The string path is strictly superior for
  SQL-based processing.
- **No target support in the pipeline.** Would make `TargetEncoder`,
  `SelectKBest`, and other target-aware transformers impossible as pipeline
  steps.

**Trade-offs:**

- Users coming from sklearn must change their mental model. Instead of
  `pipe.fit(X_train, y_train)` with two arrays, it is
  `pipe.fit("train.parquet", y="price")` with one data source and a column
  name. This is a one-time learning cost.

---

### Why Pipeline accepts file paths, not just DataFrames?

**What we chose:** `pipe.fit("train.parquet")` just works. So does
`pipe.fit("s3://bucket/data.parquet")`, `pipe.fit(pandas_df)`, and
`pipe.fit("SELECT * FROM users WHERE active")`.

**Why:** DuckDB reads all major formats natively -- Parquet, CSV, JSON, Arrow,
Excel, and remote files on S3/GCS/Azure. There is no reason to make the user
load data into a DataFrame first, especially when the data might be larger
than RAM.

When a DataFrame or numpy array is passed, it is registered as a temporary
table (zero-copy for Arrow-backed DataFrames). The pipeline always operates
on a table name internally.

**Alternatives considered:**

- **Accept only DataFrames (sklearn API).** Forces the user to load data into
  memory first. Breaks the "bigger than RAM" story.
- **Accept only table names.** Too restrictive for quick experiments. Users
  want to pass a DataFrame in a notebook.

**Trade-offs:**

- The `resolve_input()` abstraction adds a layer of indirection. It has to
  detect whether the input is a path, URL, DataFrame, array, or raw SQL, and
  handle each case. This complexity is invisible to the user but real for
  maintainers.
- Path inputs depend on DuckDB's file reading capabilities. If DuckDB cannot
  read a format, sqlearn cannot either.

---

### Why `+` operator for pipeline composition?

**What we chose:** `a + b` creates `Pipeline([a, b])`. Pipelines flatten when
composed: `Pipeline([a, b]) + Pipeline([c, d])` produces `Pipeline([a, b, c, d])`.
`+=` creates a new Pipeline (non-mutating, like `int += 1`).

```python
pipe = sq.Imputer() + sq.StandardScaler() + sq.OneHotEncoder()
```

**Why:** `+` means "and then" -- sequential composition. This is the most
natural operator for chaining steps. Flattening is safe because nesting
produces identical SQL; it is just AST composition, and a nested Pipeline
compiles the same way as a flat one.

The `+=` operator is explicitly non-mutating. Pipeline follows the same
convention as Python's immutable types (`int`, `str`, `tuple`, `frozenset`):
`x += y` creates a new object rather than modifying `x` in place. This
prevents a common notebook pitfall where rerunning cells keeps appending
steps to the same pipeline:

```python
base = sq.Pipeline([sq.Imputer()])
v1 = base + sq.StandardScaler()   # base is NOT modified
v2 = base + sq.RobustScaler()     # base is still just [Imputer]
```

**Alternatives considered:**

- **`|` operator for Union (parallel composition).** Saves 10 characters
  compared to `sq.Union([a, b])` but costs discoverability. Union is
  conceptually different from sequential composition and rare enough to
  deserve an explicit call.
- **Mutable `+=`.** Matches Python `list` behavior but causes real bugs in
  notebooks. Immutable `+=` is a deliberate safety choice.

**Trade-offs:**

- Users who expect list-like `+=` mutation will be surprised. This is
  documented prominently.
- No `|` operator means Union requires more typing. This is accepted because
  Union is used far less often than sequential composition.

---

### Why Columns instead of sklearn's ColumnTransformer?

**What we chose:** `sq.Columns` is the explicit column router for cases where
auto routing is not enough. It uses a dict interface:

```python
sq.Columns({
    "revenue":    sq.RobustScaler(),
    "categories": sq.OrdinalEncoder(order=["low", "mid", "high"]),
    "ids":        sq.HashEncoder(n_bins=64),
}, remainder="passthrough")
```

**Why:** Shorter name, same concept. `Columns` compiles to a single SELECT
with no CTEs -- each branch contributes its column expressions independently,
and the results are merged. Column name collisions are impossible because each
branch targets disjoint columns (enforced at validation time).

Similarly, `sq.Union` replaces `FeatureUnion` with a shorter name and SQL-native
compilation.

**Trade-offs:**

- The name `Columns` might be confused with a column selection utility. But in
  practice, the import context (`sq.Columns({...})`) makes the intent clear.

---

## Compiler

### Why CTE promotion at depth > 8?

**What we chose:** When composed expressions exceed nesting depth 8, the
compiler automatically wraps them in a Common Table Expression (CTE) and resets
to bare column references. The threshold is configurable.

**Why:** Deeply nested SQL is hard for humans to read, hard for some database
optimizers to plan, and risks hitting parser limits in certain engines. Depth 8
was chosen empirically -- it corresponds roughly to a 4-5 step pipeline where
each step adds 1-2 levels of nesting. Below 8, the single-SELECT form is
always readable. Above 8, a CTE makes the SQL cleaner without changing
semantics.

```sql
-- Without CTE promotion (depth > 8, hard to read):
SELECT ((COALESCE(price, 42.5) - 42.5) / NULLIF(12.3, 0) - 0.0) / NULLIF(1.0, 0) ...

-- With CTE promotion (same result, readable):
WITH __cte_0 AS (
    SELECT (COALESCE(price, 42.5) - 42.5) / NULLIF(12.3, 0) AS price FROM data
)
SELECT (price - 0.0) / NULLIF(1.0, 0) AS price FROM __cte_0
```

**Alternatives considered:**

- **Never use CTEs.** Simpler compiler but generates unreadable SQL for long
  pipelines. Also risks hitting database-specific nesting limits.
- **Always use CTEs (one per step).** Simplest compiler but generates
  unnecessarily verbose SQL for short pipelines and prevents the database
  from optimizing across step boundaries.

**Trade-offs:**

- CTEs can sometimes prevent the database optimizer from pushing predicates
  through. DuckDB handles this well (it inlines CTEs), but other databases may
  not. The default threshold of 8 is conservative enough that most pipelines
  compile to a single SELECT.

---

### Why layer-based compilation for fit?

**What we chose:** The compiler groups pipeline steps into *layers*. A layer
boundary is created after any dynamic step that also changes the schema (adds,
removes, or renames columns). All scalar aggregations within a layer are batched
into a single SQL query.

```
Pipeline: Imputer → StandardScaler → OneHotEncoder → MinMaxScaler
                                     ^-- schema changes (adds columns)
Layer 0: [Imputer, StandardScaler, OneHotEncoder]
  → one aggregation query for all three
Layer 1: [MinMaxScaler]
  → one aggregation query against Layer 0's output
```

**Why:** The problem is sequential dependency. If Step 3 (OneHotEncoder) creates
new columns like `city_london` and `city_paris`, Step 4 (MinMaxScaler) needs to
know those columns exist to compute their min/max. But those columns do not
exist in the original data -- they are created by OneHotEncoder.

Layers solve this by materializing each layer's output as a temp view before
the next layer can discover its parameters. The total number of fit queries is
the number of layers (typically 1-3), not the number of steps (which could be
10+).

!!! note
    Static steps within a layer are free. They contribute zero aggregation
    expressions to the fit query. They are part of the layer for schema tracking
    only.

**Alternatives considered:**

- **One query per step.** Simple but wasteful. Imputer and StandardScaler can
  share a single aggregation query since neither changes the schema.
- **Dependency graph analysis.** More granular than layers but significantly
  more complex to implement. Layers are a good-enough heuristic that captures
  the key optimization (batching aggregations) with minimal complexity.

**Trade-offs:**

- Layer boundaries require materializing intermediate results as temp views.
  This is one SQL `CREATE TEMP VIEW` per boundary, which is cheap, but it means
  the fit is not a single query.
- A pipeline with many schema-changing steps gets many layers, reducing the
  batching benefit. In practice, most pipelines have 1-2 schema-changing steps.

---

### Why two discovery methods (discover vs. discover_sets)?

**What we chose:** `discover()` returns scalar aggregations (mean, std, min,
max) executed via `SELECT ... FROM data` and returned as a single row.
`discover_sets()` returns set-valued queries (distinct categories, per-group
means) that return multiple rows.

**Why:** The distinction is fundamental to how results are fetched and stored.
Scalar aggregations are batched into a single SELECT with one row of results.
Set queries return variable-length result sets that cannot be batched into the
same query.

```python
# discover() → scalar aggregates, one-row result
# All of these can be in one SELECT:
{"price__mean": exp.Avg(this=exp.Column(this="price")),
 "price__std":  exp.StddevPop(this=exp.Column(this="price"))}

# discover_sets() → multi-row results, separate queries
# OneHotEncoder needs all distinct categories:
{"city__categories": SELECT DISTINCT city FROM data}
```

Without this split, OneHotEncoder's `DISTINCT` query would need special-casing
inside the aggregation batching logic.

**Alternatives considered:**

- **Single discover() method returning both.** Would require complex
  return-type discrimination to decide whether to batch or run separately.
- **Multiple scalar queries (one per step).** Simpler but wastes database
  round-trips. Batching all scalar aggregations into one SELECT is a
  significant optimization.

**Trade-offs:**

- Two methods mean two concepts to learn when writing custom transformers.
  Most custom transformers only need `discover()`. `discover_sets()` is only
  needed for encoders and similar set-learning transforms.

---

## Data Handling

### Why DuckDB as the default backend?

**What we chose:** DuckDB is the only required database dependency. It runs
in-process (no server), reads all major file formats natively, handles
out-of-core processing for larger-than-RAM datasets, and its SQL dialect is
close to standard.

**Why:** DuckDB eliminates the "loading data" step entirely. Parquet, CSV, JSON,
Arrow, S3, GCS -- DuckDB reads them all without the user installing format-specific
libraries or writing loading code. This is not a convenience feature; it is a
fundamental enabler. Without it, sqlearn would need pandas or Polars as a
dependency and would be limited to in-memory data.

In-memory DuckDB with zero configuration is the default. No server to start,
no connection string to configure. `pipe.fit("data.parquet")` just works.

**Alternatives considered:**

- **SQLite.** Widely available but lacks analytical query performance, window
  functions in older versions, and native file format support.
- **Polars.** Excellent performance but not SQL-based. Would require a
  different IR and make the "compile to SQL" story impossible.
- **No default database (require user to provide connection).** Too much
  friction for getting started.

**Trade-offs:**

- DuckDB is a C++ library with platform-specific binaries. Installation can
  fail on uncommon platforms (though coverage is excellent).
- DuckDB's SQL dialect has minor differences from Postgres/MySQL. The sqlglot
  transpiler handles most of these, but edge cases exist.
- In-memory DuckDB means fitted parameters (temp tables, views) are lost when
  the process exits. Persistent mode (`backend="my_data.duckdb"`) solves this
  when needed.

---

### Why Arrow as the internal transfer format?

**What we chose:** Data moves between DuckDB and Python via Apache Arrow. The
default output of `transform()` is a numpy array (for sklearn compatibility),
but internally the path is DuckDB -> Arrow -> numpy. Users can also get Arrow
directly with `out="arrow"` (zero-copy).

**Why:** Arrow is DuckDB's native output format. The DuckDB -> Arrow path is
zero-copy, meaning no data duplication. Arrow -> numpy is a single
`to_numpy()` call. Arrow -> pandas and Arrow -> Polars are similarly efficient.

This gives sqlearn a single internal path that branches at the output boundary:

```
DuckDB result → Arrow table → numpy (default)
                            → pandas (out="pandas")
                            → polars (out="polars")
                            → Arrow  (out="arrow", zero-copy)
```

**Trade-offs:**

- numpy is the default for sklearn compatibility, even though Arrow is more
  efficient. `out="arrow"` is the escape hatch for users who want zero-copy.
- The Arrow -> numpy conversion copies the data. This is unavoidable when
  sklearn expects a contiguous C-order array.

---

### Why TransformResult wrapper instead of numpy subclass?

**What we chose:** `transform()` returns a `TransformResult` that wraps a numpy
array with metadata (column names, dtypes, SQL). It implements the `__array__`
protocol so sklearn, XGBoost, and LightGBM accept it transparently.

```python
result = pipe.transform("data.parquet")
result.shape          # (50000, 42) — numpy-like
result.columns[7]     # "city_london" — metadata
result.sql            # the SQL that produced this
model.fit(result, y)  # sklearn accepts it via np.asarray()
```

**Why:** numpy subclassing is notoriously fragile.
`__array_ufunc__`, `__array_wrap__`, `__array_finalize__` have subtle
interactions across numpy versions. Slicing can lose metadata.
`np.concatenate` strips subclasses. These issues surface unpredictably
depending on the numpy version and the specific operations used.

The `__array__` protocol is the standard, stable way to tell numpy "I can
give you an array." Every sklearn function and ML library calls `np.asarray()`
internally, which invokes `__array__`. The wrapper adds column names and SQL
for debugging without breaking any existing workflow.

**Alternatives considered:**

- **Raw numpy array.** Would lose the debugging metadata (`columns`, `sql`).
  When a model produces bad results and the user asks "what is column 7?",
  they would have no answer.
- **numpy subclass.** Fragile across numpy versions as described above.
- **Return DuckDB Relation (lazy by default).** Would break sklearn
  compatibility. `model.fit(lazy_relation, y)` does not work. Lazy output is
  available via `out="relation"` for power users.

**Trade-offs:**

- `TransformResult` is not a "real" numpy array. Operations like
  `result + 1` do not work directly; you need `np.asarray(result) + 1`.
  This is rare in practice because the result goes straight into
  `model.fit()`.

---

## Custom Transformers

### Why three levels of custom transformers?

**What we chose:** Three tiers of increasing power and complexity:

| Level | API | Complexity | Covers |
|-------|-----|-----------|--------|
| 1 | `sq.Expression("price * qty AS revenue")` | One line | Static calculated columns |
| 2 | `sq.custom("{col} - {mean}", learn={"mean": "AVG({col})"})` | 2-8 lines | Per-column, optionally learns stats |
| 3 | `class MyTransformer(sq.Transformer)` | ~20 lines | Sets, CTEs, joins, full control |

**Why:** The gap between "use built-in transformers" and "write a class with
sqlglot ASTs" is too wide. Level 2 (`sq.custom()`) bridges it with SQL template
strings that are parsed through sqlglot at creation time -- readable SQL with
AST safety.

Most custom needs fall into Level 2. You want to log-transform a column, center
by the mean, flag outliers -- these are per-column operations with optional
learned statistics. You should not need to write a class for that.

Level 1 is even simpler for calculated columns that need no learning. Level 3
exists for the remaining 10% that need set-valued discovery, window functions,
joins, or other query-level control.

**Alternatives considered:**

- **Only Level 3 (class-based).** Pure and uniform but hostile to casual users.
  Writing sqlglot ASTs for `"LN(price + 1)"` is overkill.
- **Only Level 2 (template-based).** Covers 90% of cases but cannot express
  window functions or set-valued discovery.

**Trade-offs:**

- Three levels means three sets of documentation and three code paths in the
  compiler. But each level is self-contained -- you learn only the level you need.
- Level 2 templates are parsed through sqlglot at creation time. This means
  bad SQL fails immediately (good), but the error messages come from sqlglot's
  parser rather than from sqlearn (sometimes confusing).

---

### Why validate custom transformers at first fit?

**What we chose:** Every custom transformer is automatically validated on its
first `fit()` call. Checks include:

1. `discover()` returns sqlglot AST nodes, not raw Python values
2. `expressions()` returns sqlglot ASTs, not strings
3. New columns from `expressions()` are declared in `output_schema()`
4. `expressions()` does not reference parameters not defined in `discover()`
5. Classification declaration matches actual `discover()` behavior

**Why:** These are exactly the bugs that cause silent data corruption. If
`discover()` returns `{"mean": 42.0}` instead of
`{"mean": exp.Avg(this=...)}`, the compiler would try to compose a Python
float into a SQL AST -- the result would be nonsensical SQL or a runtime crash
far from the source of the bug.

Catching these at fit time with clear error messages and fix suggestions
prevents hours of debugging:

```
expressions() adds new column 'price_log' but output_schema() returns
unchanged schema. Override output_schema() to declare new columns:
    def output_schema(self, schema):
        return schema.add({'price_log': 'DOUBLE'})
```

**Alternatives considered:**

- **Validate at creation time.** Cannot be done for dynamic checks (need
  actual columns and schema).
- **Validate at transform time.** Too late. A fitted pipeline with bad
  parameters would silently produce wrong results until the validation
  catches it.
- **No validation (trust the user).** sklearn's approach. Results in subtle
  bugs that surface as "my model always predicts the same thing" instead of
  a clear error message.

**Trade-offs:**

- Validation adds overhead to the first `fit()`. For built-in transformers
  (Tier 1), validation is skipped entirely -- they are trusted because CI
  validates them. Custom transformers pay the cost once, then the result is
  cached.

---

### Why auto-passthrough in expressions()?

**What we chose:** The `_apply_expressions()` base class method handles column
passthrough automatically. When a transformer's `expressions()` method returns
only modified or new columns, all unmentioned columns pass through unchanged.

```python
class StandardScaler(sq.Transformer):
    def expressions(self, columns, exprs):
        # Only return the columns we modify.
        # All other columns pass through automatically.
        return {
            col: exp.Div(
                this=exp.Sub(this=exprs[col], expression=...),
                expression=...,
            )
            for col in columns
        }
```

**Why:** This eliminates the single most common custom transformer bug. In a
design where `expressions()` must return ALL columns, every custom transformer
would need an `else: result[col] = expr` passthrough branch. Forget it once,
and columns silently disappear from the output. Data corruption, no error
message, discovered only when the model produces bad results.

By making passthrough automatic, `expressions()` only needs to describe what
it changes. This is both safer (impossible to drop columns by accident) and
more concise.

**Trade-offs:**

- Transformers that intentionally remove columns must override
  `output_schema()` to declare the removal. The base class uses
  `output_schema()` to filter the result of `_apply_expressions()`.
- The automatic passthrough adds a dict merge operation per step. The cost
  is negligible.

---

## Safety

### Why are pipelines not thread-safe?

**What we chose:** Pipelines are explicitly not thread-safe. A runtime guard
detects cross-thread access and raises immediately with a message pointing to
`clone()`:

```
RuntimeError: StandardScaler accessed from a different thread.
Pipelines are not thread-safe. Use .clone() to create
a thread-safe copy with the same fitted parameters.
```

**Why:** In-memory DuckDB requires a shared connection. Temp tables created
during `fit()` must be visible during `transform()`. This means the pipeline
and its connection are coupled. Sharing a DuckDB connection across threads
leads to undefined behavior.

The alternative would be to add locking, but locking gives false confidence.
Users would assume they can call `pipe.transform()` from multiple threads
simultaneously, but the underlying DuckDB connection is not designed for that.
Failing loudly with a clear solution (`clone()`) is safer than failing
silently with locks.

**Alternatives considered:**

- **Thread-local connections.** Would require re-creating temp tables per
  thread, duplicating fitted state, and managing connection lifecycle. Complex
  and fragile.
- **Connection pooling.** Adds complexity and still does not solve the shared
  state problem (temp tables are per-connection).
- **Locking.** Serializes all operations to one thread. Works but users expect
  parallelism from thread-safe APIs, so it would be misleading.

**Trade-offs:**

- Users must call `clone()` for multi-threaded use. `sq.Search(n_jobs=4)`
  handles this internally (SQL preprocessing is sequential, model fits are
  parallel via joblib).

---

### Why read-only data safety?

**What we chose:** The backend protocol contains ONLY read operations -- SELECT
and DESCRIBE. No INSERT, UPDATE, DELETE, ALTER TABLE, DROP. All intermediate
state (fold columns, layer materialization) uses session-scoped temp views that
disappear when the connection closes.

**Why:** Source data must never be modified. Not by accident, not by design.
Users trust sqlearn with production databases. One accidental `ALTER TABLE`
means lost trust forever.

Temp views give all the materialization benefits (layer boundaries, fold
columns for cross-validation) with zero mutation risk. They exist only for the
duration of the session and cannot affect persistent data.

**Trade-offs:**

- Temp views cannot be shared across connections. This reinforces the "one
  pipeline per connection" model, which is already required by DuckDB's
  in-memory architecture.
- No persistent caching of intermediate results. If the same pipeline is fitted
  twice, the computation repeats. Persistent backends
  (`backend="my_data.duckdb"`) can mitigate this.

---

## Design Philosophy

### Why safe defaults with power opt-in?

**What we chose:** The base experience is sklearn-equivalent -- `fit()`,
`transform()`, float64, no surprises. Advanced features exist but all require
explicit opt-in:

| Feature | Default (safe) | Opt-in (advanced) |
|---------|---------------|-------------------|
| Precision | float64 | `explore_dtype="float32"` |
| Auto features | Off | `sq.AutoFeatures()` explicit step |
| Auto pipeline | Off | `sq.autopipeline()` explicit call |
| Multi-fidelity | Off | `fast_explore=True` |
| NULL handling | SQL propagation | `null_policy="error"` |

**Why:** ML correctness requires predictability. Auto-detection thresholds may
change between sqlearn versions. Clever defaults silently change pipeline
behavior -- hard to debug, hard to reproduce. Users must understand their
preprocessing to trust their model.

The `Auto*` family (`AutoFeatures`, `AutoDatetime`, `AutoEncoder`) are power
tools, not defaults. Users add them deliberately. For production,
`AutoFeatures.to_explicit()` freezes all auto-decisions into a deterministic
pipeline with baked parameters.

**Alternatives considered:**

- **Smart defaults that auto-detect everything.** Impressive in demos but
  fragile in production. A threshold change between library versions would
  silently alter pipeline behavior.
- **No auto features at all.** Would miss the opportunity to reduce
  boilerplate for exploratory work.

**Trade-offs:**

- New users write more code than they might with a "magic" library. This is
  intentional -- every step in the pipeline is visible, auditable, and
  reproducible.
- The `Auto*` features exist for users who want convenience. They just
  have to ask for it explicitly.
