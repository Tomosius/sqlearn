# Rename

Renames columns in the dataset. An explicit mapping of old column names
to new column names is applied; unmapped columns pass through unchanged.

This is a **static** transformer -- no statistics are learned during `fit()`.

```sql
SELECT price AS cost, qty AS quantity, city
FROM __input__
```

!!! info "query()-based transformer"
    Unlike most transformers that use `expressions()` for inline column
    transforms, Rename uses `query()` to wrap the input in a new SELECT.
    This ensures old column names are fully replaced, not duplicated.

---

::: sqlearn.ops.rename.Rename
