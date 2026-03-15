# Imputer

Replaces `NULL` values with a fill value learned during `fit()`. The core operation
for all strategies is SQL `COALESCE`:

$$
x' = \text{COALESCE}(x,\ v)
$$

where \(v\) depends on the imputation strategy:

| Strategy | Fill value \(v\) | SQL aggregate |
|---|---|---|
| `"mean"` | Column mean | `AVG(x)` |
| `"median"` | Column median | `PERCENTILE_CONT(0.5)` |
| `"most_frequent"` | Mode | `MODE(x)` |
| `"constant"` | User-provided value | _(literal)_ |

```sql
SELECT COALESCE(price, 3.0) AS price
FROM __input__
```

!!! tip "Auto strategy"
    The default strategy `"auto"` picks `mean` for numeric columns and
    `most_frequent` for categorical columns.

---

::: sqlearn.imputers.imputer.Imputer
