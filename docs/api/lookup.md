# Lookup

Mid-pipeline JOIN transformer that enriches data with columns from a lookup
source. This is a **static** transformer -- no statistics are learned during
`fit()`.

```sql
SELECT
  __input__.product_id,
  __input__.category_id,
  __input__.price,
  __lookup__.category_name,
  __lookup__.department
FROM (
  SELECT * FROM __input__
) AS __input__
LEFT JOIN 'categories.parquet' AS __lookup__
  ON __input__.category_id = __lookup__.category_id
```

!!! info "query()-based transformer"
    Unlike most transformers that use `expressions()` for inline column
    transforms, Lookup uses `query()` to wrap the input query with a JOIN.
    This adds new columns from the lookup source to the pipeline.

---

::: sqlearn.data.lookup.Lookup
