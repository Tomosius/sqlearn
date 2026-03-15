# Columns

Apply different transformers to different column subsets, combining results into a
single SQL SELECT. Replaces scikit-learn's `ColumnTransformer` with a simpler,
SQL-native API.

sqlearn's auto column routing (`_default_columns`) means most pipelines don't need
`Columns` at all -- `StandardScaler` defaults to numeric columns and `OneHotEncoder`
defaults to categorical columns automatically. Use `Columns` when you need explicit
control over which transformer gets which columns.

```python
import sqlearn as sq

pipe = sq.Pipeline([
    sq.Columns([
        ("scale", sq.StandardScaler(), sq.numeric()),
        ("encode", sq.OneHotEncoder(), sq.categorical()),
    ]),
])
pipe.fit("data.parquet")
sql = pipe.to_sql()
```

Generated SQL:

```sql
SELECT
  (price - 3.0) / NULLIF(1.41, 0) AS price,
  (quantity - 30.0) / NULLIF(14.14, 0) AS quantity,
  CASE WHEN city = 'London' THEN 1 ELSE 0 END AS city_london,
  CASE WHEN city = 'Paris' THEN 1 ELSE 0 END AS city_paris
FROM __input__
```

## Column selectors

Each group's column specification can be:

| Selector type | Example | What it matches |
|---|---|---|
| String list | `["price", "quantity"]` | Exact column names |
| `sq.numeric()` | `sq.numeric()` | All numeric columns |
| `sq.categorical()` | `sq.categorical()` | All categorical columns |
| `sq.temporal()` | `sq.temporal()` | All temporal columns |
| `sq.matching("feat_*")` | `sq.matching("feat_*")` | Glob pattern match |

## Remainder

Columns not assigned to any group are handled by the `remainder` parameter:

- `"drop"` (default) -- unmatched columns are excluded from output
- `"passthrough"` -- unmatched columns are included unchanged

```python
# Keep unscaled columns in the output
cols = sq.Columns(
    [("scale", sq.StandardScaler(), ["price"])],
    remainder="passthrough",
)
```

## Overlap detection

Assigning the same column to multiple groups raises `SchemaError`:

```python
# This will raise SchemaError at fit time
sq.Columns([
    ("a", sq.StandardScaler(), ["price"]),
    ("b", sq.MinMaxScaler(), ["price"]),  # overlap!
])
```

## When to use Columns vs auto routing

| Scenario | Use |
|---|---|
| Scale numerics, encode categoricals | Just `Pipeline([StandardScaler(), OneHotEncoder()])` |
| Different scalers for different numeric columns | `Columns` |
| Mix transformers with explicit column control | `Columns` |
| Single transformer on all columns | Just that transformer |

---

::: sqlearn.core.columns.Columns
