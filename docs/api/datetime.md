# Datetime Transforms

Extract date and time components from temporal columns as numeric features.
All datetime transformers are **static** -- no statistics are learned during
`fit()`. They default to all temporal columns (`DATE`, `TIMESTAMP`, etc.).

## DateParts

Extract multiple date parts into separate integer columns. Each temporal
column produces one new column per requested part.

```sql
EXTRACT(YEAR FROM ts) AS ts_year,
EXTRACT(MONTH FROM ts) AS ts_month,
EXTRACT(DOW FROM ts) AS ts_dayofweek
```

!!! info "Static transformer"
    DateParts needs no data statistics. The parts list is known at
    construction time, making it fast and safe to use anywhere in a pipeline.

---

::: sqlearn.features.datetime.DateParts

---

## DateDiff

Compute the difference between temporal columns and a reference date or
column, in a specified time unit.

```sql
DATEDIFF('DAY', '2020-01-01', ts) AS ts
```

---

::: sqlearn.features.datetime.DateDiff

---

## IsWeekend

Binary flag: 1 if Saturday or Sunday, 0 otherwise.

```sql
CASE WHEN EXTRACT(DOW FROM ts) IN (0, 6) THEN 1 ELSE 0 END AS ts
```

---

::: sqlearn.features.datetime.IsWeekend

---

## Quarter

Extract the quarter number (1-4) from temporal columns.

```sql
EXTRACT(QUARTER FROM ts) AS ts
```

---

::: sqlearn.features.datetime.Quarter
