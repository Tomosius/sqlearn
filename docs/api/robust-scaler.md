# RobustScaler

Scales numeric columns using statistics robust to outliers:

$$
x' = \frac{x - \text{median}}{Q_3 - Q_1}
$$

where median, \(Q_1\), and \(Q_3\) are learned during `fit()`.
The interquartile range (IQR = \(Q_3 - Q_1\)) is less sensitive to
outliers than standard deviation.

Division by zero is prevented with `NULLIF` -- if IQR = 0,
the result is `NULL`:

```sql
(price - 3.0) / NULLIF(4.0 - 2.0, 0) AS price
```

!!! tip "When to use"
    Prefer RobustScaler over StandardScaler when your data contains
    significant outliers that would skew the mean and standard deviation.

---

::: sqlearn.scalers.robust.RobustScaler
