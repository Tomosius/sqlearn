# Cast

Casts columns to specified SQL data types. A **static** data operation --
no statistics are learned during `fit()`.

Each column listed in `mapping` is wrapped in a `CAST(col AS type)` expression.
Unmapped columns pass through unchanged.

```sql
CAST(price AS DOUBLE) AS price, CAST(qty AS INTEGER) AS qty
```

!!! info "Static transformer"
    Cast needs no data statistics. The mapping is known at construction time,
    so `discover()` returns nothing. This makes Cast fast and safe to use
    anywhere in a pipeline.

---

::: sqlearn.ops.cast.Cast
