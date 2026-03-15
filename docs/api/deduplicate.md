# Deduplicate

Removes **duplicate rows** from a dataset. Operates at the query level using
SQL window functions or `DISTINCT`.

**Full-row dedup** (default):

```sql
SELECT DISTINCT * FROM __input__ AS __input__
```

**Subset dedup** (keep first occurrence by `city` and `name`):

```sql
SELECT city, name, price
FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY city, name ORDER BY city) AS __rn__
    FROM __input__ AS __input__
) AS __dedup__
WHERE __rn__ = 1
```

**Remove all duplicates** (`keep="none"`):

```sql
SELECT city, name, price
FROM (
    SELECT *, COUNT(*) OVER (PARTITION BY city, name) AS __cnt__
    FROM __input__ AS __input__
) AS __dedup__
WHERE __cnt__ = 1
```

This is a **static** transformer -- no statistics are learned during `fit()`.

!!! info "subset vs columns"
    The `subset` parameter specifies which columns define "duplicate",
    not which columns to transform. All columns are always preserved in
    the output. This differs from the standard `columns` parameter used
    by scalers and encoders.

---

::: sqlearn.ops.deduplicate.Deduplicate
