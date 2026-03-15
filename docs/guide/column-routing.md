# Column Routing

sqlearn automatically routes columns to the right transformers based on their SQL types.
This page covers every way to control that routing, from automatic defaults to
complex multi-branch configurations.


## Automatic defaults

Every transformer has a `_default_columns` class attribute that determines which columns
it targets when you do not specify `columns=`:

| Transformer | `_default_columns` | Targets |
|---|---|---|
| `StandardScaler` | `"numeric"` | INTEGER, DOUBLE, FLOAT, DECIMAL, ... |
| `MinMaxScaler` | `"numeric"` | INTEGER, DOUBLE, FLOAT, DECIMAL, ... |
| `RobustScaler` | `"numeric"` | INTEGER, DOUBLE, FLOAT, DECIMAL, ... |
| `OneHotEncoder` | `"categorical"` | VARCHAR, TEXT, STRING, ... |
| `OrdinalEncoder` | `"categorical"` | VARCHAR, TEXT, STRING, ... |
| `Imputer` | `"all"` | Every column regardless of type |
| `Rename` | `None` | Must specify columns explicitly |
| `Cast` | `None` | Must specify columns explicitly |

This means a simple pipeline handles mixed-type data correctly with no configuration:

```python
import sqlearn as sq

# Table: price (DOUBLE), quantity (INTEGER), city (VARCHAR)
pipe = sq.Pipeline([
    sq.Imputer(),          # targets: price, quantity, city
    sq.StandardScaler(),   # targets: price, quantity (numeric only)
    sq.OneHotEncoder(),    # targets: city (categorical only)
])
pipe.fit("data.parquet")
```

The generated SQL handles each column type appropriately:

```sql
SELECT
  (COALESCE(price, 42.5) - 42.5) / NULLIF(10.2, 0) AS price,
  (COALESCE(quantity, 30) - 30.0) / NULLIF(8.16, 0) AS quantity,
  CASE WHEN COALESCE(city, 'London') = 'London' THEN 1 ELSE 0 END AS city_london,
  CASE WHEN COALESCE(city, 'London') = 'Paris' THEN 1 ELSE 0 END AS city_paris
FROM __input__
```


## Explicit column selection

Override the default with the `columns=` parameter on any transformer.

### List of column names

```python
# Scale only these two columns, leave others untouched
sq.StandardScaler(columns=["price", "score"])
```

### Type string

```python
# Scale all numeric columns (same as default, but explicit)
sq.StandardScaler(columns="numeric")

# Impute only categorical columns
sq.Imputer(strategy="most_frequent", columns="categorical")
```

Valid type strings: `"numeric"`, `"categorical"`, `"temporal"`, `"boolean"`, `"all"`.


## Schema-based selectors

For more flexible column selection, sqlearn provides selector factory functions that
resolve against the schema at fit time.

### Basic selectors

```python
import sqlearn as sq

sq.numeric()        # all numeric columns
sq.categorical()    # all categorical (string) columns
sq.temporal()       # all date/time columns
sq.boolean()        # all boolean columns
sq.all_columns()    # every column
sq.columns("a", "b")  # specific named columns
sq.matching("price_*")  # glob pattern matching
sq.dtype("DECIMAL")     # specific SQL type
```

Use them with the `columns=` parameter:

```python
sq.StandardScaler(columns=sq.numeric())
sq.Imputer(columns=sq.matching("feature_*"))
```

### Composing selectors

Selectors support set operators for combining and filtering:

| Operator | Meaning | Example |
|---|---|---|
| `\|` | Union (either) | `sq.numeric() \| sq.boolean()` |
| `&` | Intersection (both) | `sq.numeric() & sq.matching("price_*")` |
| `~` | Negation (not) | `~sq.categorical()` |
| `-` | Difference (except) | `sq.numeric() - sq.columns("id")` |

```python
# Scale all numeric columns except id
sq.StandardScaler(columns=sq.numeric() - sq.columns("id"))

# Impute everything that's not boolean
sq.Imputer(columns=~sq.boolean())

# Scale columns that are both numeric AND match a pattern
sq.StandardScaler(columns=sq.numeric() & sq.matching("feature_*"))
```

!!! tip "Selectors are resolved at fit time"
    Selectors are lazy --- they do not resolve to column names until `fit()` is called
    and the actual schema is known. This means you can reuse the same pipeline
    definition on tables with different schemas.


## Columns: routing groups

When you need different transformers on different column subsets within a single
pipeline step, use `sq.Columns`. It replaces scikit-learn's `ColumnTransformer`.

```python
pipe = sq.Pipeline([
    sq.Columns([
        ("scale", sq.StandardScaler(), sq.numeric()),
        ("encode", sq.OneHotEncoder(), sq.categorical()),
    ]),
])
pipe.fit("data.parquet")
```

Each group is a `(name, transformer, columns)` triple. The column spec can be anything:
a list, a type string, or a selector.

### How it works

`Columns` collects the aggregate queries from all sub-transformers, batches them into
one fit query, then distributes the learned parameters back to each sub-transformer.
At transform time, each sub-transformer generates expressions for its column subset,
and the results are merged into a single SELECT.

### Remainder columns

By default, columns not assigned to any group are **dropped** from the output. Use
`remainder="passthrough"` to keep them:

```python
# Scale price and quantity, but keep city unchanged
cols = sq.Columns(
    [("scale", sq.StandardScaler(), ["price", "quantity"])],
    remainder="passthrough",  # city passes through as-is
)
pipe = sq.Pipeline([cols])
pipe.fit("data.parquet")
```

### Overlap detection

`Columns` validates that no column appears in more than one group. If groups overlap,
you get a clear error at fit time:

```python
# This raises SchemaError: price assigned to both groups
sq.Columns([
    ("a", sq.StandardScaler(), ["price"]),
    ("b", sq.MinMaxScaler(), ["price"]),  # overlap!
])
```

### When to use Columns vs default routing

Most of the time you do not need `Columns`. The automatic routing from
`_default_columns` handles the common case:

```python
# These two pipelines produce the same result:

# Without Columns (preferred when defaults suffice)
pipe = sq.Pipeline([sq.StandardScaler(), sq.OneHotEncoder()])

# With Columns (explicit but more verbose)
pipe = sq.Pipeline([
    sq.Columns([
        ("scale", sq.StandardScaler(), sq.numeric()),
        ("encode", sq.OneHotEncoder(), sq.categorical()),
    ]),
])
```

Use `Columns` when:

- You need different transformers on overlapping column types (e.g., two different scalers)
- You want to control remainder behavior explicitly
- You need to name groups for later inspection
- You want a single pipeline step that handles all column routing


## Union: parallel branches

`sq.Union` combines the **outputs** of multiple transformer branches horizontally. Each
branch processes the input independently, and their output columns are merged. This
replaces scikit-learn's `FeatureUnion`.

```python
pipe = sq.Pipeline([
    sq.Union([
        ("scaled", sq.StandardScaler()),
        ("encoded", sq.OneHotEncoder()),
    ]),
])
pipe.fit("data.parquet")
```

### Branch prefixing

To avoid column name collisions, Union prefixes each branch's output columns with the
branch name:

```sql
SELECT
  (price - 42.5) / NULLIF(10.2, 0) AS scaled_price,
  (quantity - 30.0) / NULLIF(8.16, 0) AS scaled_quantity,
  CASE WHEN city = 'London' THEN 1 ELSE 0 END AS encoded_city_london,
  CASE WHEN city = 'Paris' THEN 1 ELSE 0 END AS encoded_city_paris
FROM __input__
```

The output column names are `scaled_price`, `scaled_quantity`, `encoded_city_london`,
etc.

### Columns vs Union

These serve different purposes:

| Feature | `Columns` | `Union` |
|---|---|---|
| Purpose | Route column subsets to different transformers | Combine multiple views of the same data |
| Column overlap | Not allowed between groups | Each branch sees ALL input columns |
| Output columns | Original column names preserved | Prefixed with branch name |
| Remainder | Configurable (`"drop"` or `"passthrough"`) | No remainder concept |
| Analogy | scikit-learn `ColumnTransformer` | scikit-learn `FeatureUnion` |

**Use Columns** when you want each column handled by exactly one transformer:

```python
# price → StandardScaler, city → OneHotEncoder
sq.Columns([
    ("scale", sq.StandardScaler(), ["price"]),
    ("encode", sq.OneHotEncoder(), ["city"]),
])
```

**Use Union** when you want multiple transformations of the same columns:

```python
# Both branches see all columns, outputs are combined
sq.Union([
    ("raw", sq.Imputer()),             # imputed columns
    ("scaled", sq.StandardScaler()),   # scaled columns
])
# Output: raw_price, raw_quantity, raw_city, scaled_price, scaled_quantity
```


## Combining everything

These routing mechanisms compose freely. Here is a more complex example:

```python
pipe = sq.Pipeline([
    # Step 1: Fill missing values everywhere
    sq.Imputer(),

    # Step 2: Route columns to specialized transformers
    sq.Columns(
        [
            ("scale", sq.StandardScaler(), sq.numeric() - sq.columns("id")),
            ("encode", sq.OneHotEncoder(), sq.categorical()),
        ],
        remainder="passthrough",  # keep id column
    ),
])
pipe.fit("data.parquet")
```

This pipeline:

1. Imputes all columns (numeric gets median, categorical gets most_frequent)
2. Scales all numeric columns except `id`
3. One-hot encodes all categorical columns
4. Passes `id` through unchanged

The generated SQL is a single SELECT with nested expressions --- no intermediate
tables, no multiple passes over the data.


## Quick reference

| Need | Solution |
|---|---|
| Use transformer's default column types | Just use the transformer: `sq.StandardScaler()` |
| Target specific columns by name | `sq.StandardScaler(columns=["price", "score"])` |
| Target columns by type | `sq.StandardScaler(columns="numeric")` |
| Use selector composition | `sq.StandardScaler(columns=sq.numeric() - sq.columns("id"))` |
| Different transformers for different columns | `sq.Columns([...])` |
| Multiple views of the same data | `sq.Union([...])` |
| Keep unrouted columns | `sq.Columns([...], remainder="passthrough")` |
