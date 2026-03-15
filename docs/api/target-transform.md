# TargetTransform

Apply mathematical transformations to target columns for regression.

## Methods

**Log** (default) -- natural log with +1 offset for zero safety:

```sql
LN(price + 1) AS price
```

**Square root**:

```sql
SQRT(price) AS price
```

**Box-Cox** -- power transform with learned or fixed lambda:

$$
x' = \begin{cases}
\frac{x^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\
\ln(x) & \text{if } \lambda = 0
\end{cases}
$$

```sql
(POW(price, 0.5) - 1) / 0.5 AS price
```

!!! tip "When to use"
    Use `TargetTransform` when your regression target is skewed.
    Log and sqrt are static (no fitting needed). Box-Cox with
    `lambda_="auto"` learns the optimal transformation from data.

!!! note "Classification"
    `TargetTransform` is **static** for `method="log"` and `method="sqrt"`,
    and **dynamic** for `method="boxcox"` with `lambda_="auto"` (needs to
    learn lambda from data).

---

::: sqlearn.features.target.TargetTransform
