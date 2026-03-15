# concat

Concatenates multiple data sources vertically via SQL `UNION ALL`.
Builds a sqlglot query and registers the result as a DuckDB view
that can be used as pipeline input.

```sql
SELECT customer_id, name, city
FROM 'train.parquet'
UNION ALL
SELECT customer_id, name, city
FROM 'test.parquet'
```

!!! info "Function, not transformer"
    `concat()` is a standalone function, not a pipeline step. It returns a view
    name that you pass to `Pipeline.fit()` or `Pipeline.transform()` as input.

---

::: sqlearn.data.concat.concat

---

::: sqlearn.data.concat.concat_query
