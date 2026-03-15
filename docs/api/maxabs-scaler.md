# MaxAbsScaler

Scales each numeric column by its maximum absolute value, mapping to [-1, 1]:

$$
x' = \frac{x}{\max(|x|)}
$$

where \(\max(|x|)\) is the maximum absolute value in the column,
learned during `fit()`. This preserves the sign of the original values.

Division by zero is prevented with `NULLIF` -- if all values are zero,
the result is `NULL`:

```sql
price / NULLIF(10.5, 0) AS price
```

!!! tip "When to use"
    MaxAbsScaler is useful for data that is already centered at zero
    or for sparse data where centering would destroy sparsity.

---

::: sqlearn.scalers.maxabs.MaxAbsScaler
