# merge

Joins two data sources via SQL. Builds a sqlglot `SELECT ... JOIN` query
and registers the result as a DuckDB view that can be used as pipeline input.

Supports all standard join types: **inner**, **left**, **right**, **outer**, and **cross**.

```sql
SELECT
  __left__.customer_id,
  __left__.name,
  __right__.order_date,
  __right__.amount
FROM customers AS __left__
INNER JOIN orders AS __right__
  ON __left__.customer_id = __right__.customer_id
```

!!! info "Function, not transformer"
    `merge()` is a standalone function, not a pipeline step. It returns a view
    name that you pass to `Pipeline.fit()` or `Pipeline.transform()` as input.

---

::: sqlearn.data.merge.merge

---

::: sqlearn.data.merge.merge_query
