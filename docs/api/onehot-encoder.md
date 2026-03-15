# OneHotEncoder

Encodes categorical columns into binary indicator columns. For a column with
categories \(\{c_1, c_2, \ldots, c_k\}\), each category becomes a new column:

$$
x_{c_i} = \begin{cases} 1 & \text{if } x = c_i \\ 0 & \text{otherwise} \end{cases}
$$

Categories are discovered during `fit()`. Each produces a `CASE WHEN` expression:

```sql
CASE WHEN city = 'London' THEN 1 ELSE 0 END AS city_london,
CASE WHEN city = 'Paris'  THEN 1 ELSE 0 END AS city_paris,
CASE WHEN city = 'Tokyo'  THEN 1 ELSE 0 END AS city_tokyo
```

!!! info "Column naming"
    Output columns are named `{original}_{category}` with category values
    lowercased and special characters replaced by underscores.

---

::: sqlearn.encoders.onehot.OneHotEncoder
