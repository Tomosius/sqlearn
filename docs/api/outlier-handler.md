# OutlierHandler

Clips or removes outliers from numeric columns using IQR or z-score detection.

## IQR method (default)

Outliers are values outside the **fences**:

$$
\text{lower} = Q_1 - k \cdot \text{IQR}, \quad \text{upper} = Q_3 + k \cdot \text{IQR}
$$

where \(k\) is the `threshold` (default 1.5) and IQR = \(Q_3 - Q_1\).

## Z-score method

Outliers are values more than `threshold` standard deviations from the mean:

$$
\text{lower} = \mu - k \cdot \sigma, \quad \text{upper} = \mu + k \cdot \sigma
$$

where \(k\) is the `threshold` (default 3.0).

## Actions

**Clip** (default) -- cap outlier values to the fence boundaries:

```sql
GREATEST(LEAST(price, 12.0), -2.0) AS price
```

**Remove** -- filter out rows containing outlier values:

```sql
SELECT * FROM (__input__) AS __input__
WHERE price BETWEEN -2.0 AND 12.0
```

!!! tip "When to use"
    Use `OutlierHandler` before scaling when your data contains extreme values
    that would distort learned statistics. Prefer `action="clip"` to keep all
    rows; use `action="remove"` when outlier rows should be excluded entirely.

---

::: sqlearn.features.outlier.OutlierHandler
