# Window Transforms

SQL window function wrappers for feature engineering. All window transforms
use `query()` (not `expressions()`) because window functions require their
own `SELECT` level.

## Overview

| Transformer | SQL Function | Output Column | Use Case |
|---|---|---|---|
| `Lag` | `LAG(col, N)` | `{col}_lag{N}` | Previous row access |
| `Lead` | `LEAD(col, N)` | `{col}_lead{N}` | Next row access |
| `RollingMean` | `AVG(col) OVER (ROWS ...)` | `{col}_rmean{N}` | Moving average |
| `RollingStd` | `STDDEV_POP(col) OVER (ROWS ...)` | `{col}_rstd{N}` | Moving volatility |
| `Rank` | `RANK()` / `DENSE_RANK()` | `{col}_rank` or `rank` | Row ranking |
| `RowNumber` | `ROW_NUMBER()` | `row_number` | Sequential numbering |

All are **static** transformers -- no statistics are learned during `fit()`.
All accept `order_by` (required) and `partition_by` (optional) parameters.

## Common Parameters

- **`order_by`**: Column(s) defining row order. Required for all window transforms.
  Accepts a string or list of strings.
- **`partition_by`**: Column(s) to partition the window. Optional. When specified,
  window functions reset at partition boundaries.

```sql
-- Without partition_by: single window over all rows
LAG(value, 1) OVER (ORDER BY ts)

-- With partition_by: separate windows per group
LAG(value, 1) OVER (PARTITION BY category ORDER BY ts)
```

## Examples

```python
import sqlearn as sq

# Lag/Lead for time series features
pipe = sq.Pipeline([
    sq.Lag(order_by="timestamp", columns=["price"]),
    sq.Lead(order_by="timestamp", columns=["price"]),
])

# Rolling statistics
pipe = sq.Pipeline([
    sq.RollingMean(window=7, order_by="date", columns=["sales"]),
    sq.RollingStd(window=7, order_by="date", columns=["sales"]),
])

# Ranking with partitioning
pipe = sq.Pipeline([
    sq.Rank(order_by="score", partition_by="department", method="dense_rank"),
])

# Row numbering
pipe = sq.Pipeline([
    sq.RowNumber(order_by="created_at"),
])
```

---

::: sqlearn.features.window.Lag

---

::: sqlearn.features.window.Lead

---

::: sqlearn.features.window.RollingMean

---

::: sqlearn.features.window.RollingStd

---

::: sqlearn.features.window.Rank

---

::: sqlearn.features.window.RowNumber
