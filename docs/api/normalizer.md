# Normalizer

Normalizes each **row** to unit norm across target columns. Unlike per-column
scalers, Normalizer operates across features within each sample.

**L2 norm** (default):

$$
x'_i = \frac{x_i}{\sqrt{x_1^2 + x_2^2 + \cdots + x_n^2}}
$$

**L1 norm:**

$$
x'_i = \frac{x_i}{|x_1| + |x_2| + \cdots + |x_n|}
$$

**Max norm:**

$$
x'_i = \frac{x_i}{\max(|x_1|, |x_2|, \ldots, |x_n|)}
$$

This is a **static** transformer -- no statistics are learned during `fit()`.
The norm is computed inline from row values.

```sql
price / NULLIF(SQRT(price * price + quantity * quantity), 0) AS price
```

!!! info "Row-wise vs column-wise"
    Most scalers (StandardScaler, MinMaxScaler) operate **per column**.
    Normalizer operates **per row** -- useful when relative proportions
    between features matter (e.g., TF-IDF vectors).

---

::: sqlearn.scalers.normalizer.Normalizer
