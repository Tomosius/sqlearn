> **sqlearn docs** — [Index](../CLAUDE.md) | Prev: [Export & Deployment](08-export.md) | Next: [Project Structure](10-project-structure.md)

## 10. Error Handling

Clean error hierarchy, not cryptic sklearn tracebacks:

```python
class SQLearnError(Exception): ...
class NotFittedError(SQLearnError): ...       # transform before fit
class SchemaError(SQLearnError): ...          # column missing, type mismatch
class FitError(SQLearnError): ...             # all-NULL column, empty table
class CompilationError(SQLearnError): ...     # can't compile to target dialect
class UnseenCategoryError(SQLearnError): ...  # new category at transform time
```

**Helpful messages:**

```
SchemaError: Column 'prce' not found. Did you mean 'price'?
FitError: Column 'price' is all NULL — cannot compute mean/std for StandardScaler.
UnseenCategoryError: OneHotEncoder encountered unseen category 'Berlin' in column 'city'.
    Set handle_unknown='ignore' to map unseen categories to zero, or refit the pipeline.
```

**Unseen categories policy:** `handle_unknown="error"` (default), `"ignore"` (→ zero),
`"infrequent"` (→ OTHER bucket).

**NULL policy (pipeline-level):**

```python
sq.Pipeline([...], null_policy="propagate")  # default: NULLs pass through (SQL semantics)
sq.Pipeline([...], null_policy="error")      # check for NULLs, raise if found
sq.Pipeline([...], null_policy="warn")       # warn on NULL output columns
```

### 10.1 Pipeline-Integrated Drift Detection

After fitting, a pipeline knows what the training data looked like (schema, distributions,
ranges). It can detect when new data diverges:

```python
pipe.fit("train.parquet")

# Quick check: does the input data match expected schema?
pipe.validate("new_data.parquet")
# ✓ Schema matches. 15 columns, all types correct.
# ⚠ Warning: column 'age' has 23% nulls (training had 2.1%)
# ⚠ Warning: column 'city' has 12 unseen categories

# Detailed drift report:
drift = pipe.detect_drift("new_data.parquet")
print(drift)
# Column      Metric          Train    New       Status
# ──────────  ──────────────  ───────  ───────   ──────
# price       mean            42.5     38.2      ⚠ shifted (-10%)
# price       std             12.3     18.7      ⚠ shifted (+52%)
# price       null_pct        0.0%     0.3%      ✓ ok
# city        n_categories    147      159       ⚠ 12 new categories
# city        top_category    London   London    ✓ ok
# age         null_pct        2.1%     23.4%     ✗ ALERT (>10x increase)

# Programmatic access:
drift.alerts                          # list of Alert objects (threshold-breached)
drift.warnings                        # list of Warning objects (notable shifts)
drift.passed                          # True if no alerts
```

Drift detection is SQL-based: compares `sq.profile(new_data)` stats against stored
training stats. The thresholds are configurable:

```python
pipe.detect_drift("new_data.parquet", thresholds={
    "mean_shift_pct": 0.2,          # alert if mean shifts >20%
    "null_increase_factor": 5,       # alert if null% increases >5x
    "new_category_pct": 0.1,        # alert if >10% of values are unseen categories
})
```
---

