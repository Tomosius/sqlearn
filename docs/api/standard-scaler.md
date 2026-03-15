# StandardScaler

Standardizes numeric columns to zero mean and unit variance using the z-score formula:

$$
x' = \frac{x - \mu}{\sigma}
$$

where \(\mu\) is the column mean and \(\sigma\) is the population standard deviation,
both learned during `fit()`.

Division by zero is prevented with `NULLIF` -- if \(\sigma = 0\) (constant column),
the result is `NULL` rather than an error:

```sql
(price - 3.0) / NULLIF(1.41, 0) AS price
```

!!! note "Population vs sample std"
    sqlearn uses **population** standard deviation (`STDDEV_POP`) to match
    scikit-learn's default behavior.

---

::: sqlearn.scalers.standard.StandardScaler
