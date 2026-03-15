# OrdinalEncoder

Encodes categorical columns as integer codes. For a column with categories
\(\{c_1, c_2, \ldots, c_k\}\) sorted alphabetically, each row value is mapped:

$$
x \mapsto \begin{cases} 0 & \text{if } x = c_1 \\ 1 & \text{if } x = c_2 \\ \vdots \\ k-1 & \text{if } x = c_k \end{cases}
$$

Categories are discovered during `fit()`. The column is replaced in-place with a
`CASE WHEN` expression:

```sql
CASE WHEN city = 'Berlin' THEN 0
     WHEN city = 'London' THEN 1
     WHEN city = 'Paris'  THEN 2
     ELSE NULL END AS city
```

!!! info "Differences from OneHotEncoder"
    OrdinalEncoder replaces the original column in-place with integer codes.
    OneHotEncoder creates one new binary column per category and drops the original.
    Use OrdinalEncoder when downstream models handle ordinal inputs (e.g., tree-based
    models). Use OneHotEncoder for linear models that need independent features.

!!! warning "Unknown categories"
    By default (`handle_unknown="error"`), unseen categories produce NULL.
    Set `handle_unknown="use_encoded_value"` with `unknown_value=-1` to map
    unseen categories to a specific integer instead.

---

::: sqlearn.encoders.ordinal.OrdinalEncoder
