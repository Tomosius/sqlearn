# String Transforms

String transformers for text preprocessing. All are **static** -- no statistics
are learned during `fit()`. They default to categorical (string) columns.

```sql
LENGTH(name) AS name
LOWER(city) AS city
UPPER(city) AS city
TRIM(name) AS name
REPLACE(name, 'foo', 'bar') AS name
SUBSTRING(name, 1, 3) AS name
```

!!! info "Static transformers"
    String transforms need no data statistics. The operations are known at
    construction time, so `discover()` returns nothing. This makes them fast
    and safe to use anywhere in a pipeline.

---

::: sqlearn.features.string.StringLength

---

::: sqlearn.features.string.Lower

---

::: sqlearn.features.string.Upper

---

::: sqlearn.features.string.Trim

---

::: sqlearn.features.string.Replace

---

::: sqlearn.features.string.Substring
