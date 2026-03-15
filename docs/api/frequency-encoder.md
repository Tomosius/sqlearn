# FrequencyEncoder

Encodes categorical columns by replacing each category with its observed frequency
(proportion) in the training data. For a column with categories
\(\{c_1, c_2, \ldots, c_k\}\) and row counts \(\{n_1, n_2, \ldots, n_k\}\):

$$
f(x) = \frac{n_i}{N} \quad \text{where } x = c_i \text{ and } N = \sum n_j
$$

When `normalize=False`, raw counts are used instead:

$$
f(x) = n_i \quad \text{where } x = c_i
$$

Frequencies are learned during `fit()`. Each column produces a `CASE WHEN` expression:

```sql
CASE
  WHEN city = 'London' THEN 0.50
  WHEN city = 'Paris'  THEN 0.25
  WHEN city = 'Tokyo'  THEN 0.25
  ELSE 0.0
END AS city
```

!!! info "In-place replacement"
    Unlike `OneHotEncoder`, `FrequencyEncoder` replaces columns in-place --
    the output has the same column names as the input. Categories become
    numeric values.

!!! tip "Unknown categories"
    Categories not seen during `fit()` are mapped to `fill_value` (default 0.0)
    via the `ELSE` clause. Set `fill_value=-1.0` to flag unknowns explicitly.

---

::: sqlearn.encoders.frequency.FrequencyEncoder
