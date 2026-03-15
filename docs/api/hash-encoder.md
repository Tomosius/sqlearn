# HashEncoder

Maps categorical columns to fixed-size hash buckets. For a column with any number
of categories, each value is hashed into one of \(n\) buckets:

$$
x_{\text{hash\_}i} = \begin{cases} 1 & \text{if } |\text{HASH}(x)| \bmod n = i \\ 0 & \text{otherwise} \end{cases}
$$

Unlike [`OneHotEncoder`](onehot-encoder.md), HashEncoder is a **static** transformer
--- it does not need to learn categories from data. This makes it ideal for
high-cardinality columns or streaming scenarios where categories are unknown ahead
of time.

Each input column produces `n_features` binary output columns:

```sql
CASE WHEN ABS(HASH(city)) % 8 = 0 THEN 1 ELSE 0 END AS city_hash_0,
CASE WHEN ABS(HASH(city)) % 8 = 1 THEN 1 ELSE 0 END AS city_hash_1,
...
CASE WHEN ABS(HASH(city)) % 8 = 7 THEN 1 ELSE 0 END AS city_hash_7
```

!!! info "Hash collisions"
    Multiple category values may map to the same bucket. Increase `n_features`
    to reduce collision probability. With `n_features=8` and 3 categories, the
    expected collision rate is approximately 34%.

!!! info "Column naming"
    Output columns are named `{original}_hash_{i}` where `i` ranges from
    `0` to `n_features - 1`.

---

::: sqlearn.encoders.hash.HashEncoder
