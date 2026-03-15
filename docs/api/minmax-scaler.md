# MinMaxScaler

Scales numeric columns to a given range (default [0, 1]) using min-max normalization:

$$
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}} \cdot (b - a) + a
$$

where \(a\) and \(b\) are the target range bounds (default 0 and 1),
and \(x_{\min}\), \(x_{\max}\) are the column min/max learned during `fit()`.

Division by zero is prevented with `NULLIF` -- if \(x_{\max} = x_{\min}\) (constant column),
the result is `NULL`:

```sql
(price - 1.0) / NULLIF(5.0 - 1.0, 0) AS price
```

---

::: sqlearn.scalers.minmax.MinMaxScaler
