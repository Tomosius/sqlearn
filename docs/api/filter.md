# Filter

Filters **rows** based on a SQL WHERE condition. Unlike column-level transformers,
Filter operates at the query level --- it removes rows that don't match the condition
while passing all columns through unchanged.

This is a **static** transformer -- no statistics are learned during `fit()`.
The condition is parsed through sqlglot at construction time for fail-fast validation.

```sql
SELECT * FROM (__input__) AS __input__ WHERE price > 0
```

!!! info "Row-level vs column-level"
    Most transformers (StandardScaler, Imputer) operate **per column** via `expressions()`.
    Filter operates **per row** via `query()` -- it adds a WHERE clause to the SQL query.
    All columns pass through unchanged.

---

::: sqlearn.ops.filter.Filter
