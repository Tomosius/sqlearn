# Sample

Sample a random subset of **rows** from the dataset. A data operation
that reduces dataset size without altering columns or learning statistics.

Two modes:

- **Count-based** (`n`): Select exactly `n` rows.
- **Fraction-based** (`fraction`): Select approximately `fraction` of rows.

```sql
-- Count-based (n=100)
SELECT * FROM (__input__) AS __input__
ORDER BY RANDOM()
LIMIT 100

-- Fraction-based (fraction=0.5)
SELECT * FROM (__input__) AS __input__
WHERE RANDOM() < 0.5
```

This is a **static** transformer -- no statistics are learned during `fit()`.

!!! info "Approximate vs exact"
    Count-based sampling (`n`) returns exactly `n` rows.
    Fraction-based sampling (`fraction`) is approximate -- the actual count
    will vary around `fraction * total_rows` because each row is independently
    included with probability `fraction`.

!!! warning "Non-deterministic"
    Each execution may return different rows. Seed-based reproducibility
    is reserved for future implementation.

---

::: sqlearn.ops.sample.Sample
