# Feature Selection

Transformers for selecting which features to keep in the pipeline.
All feature selection transformers use `query()` to control which
columns appear in the output SELECT statement.

---

## Drop

Remove specified columns from the dataset. A **static** transformer --
no statistics are learned during `fit()`.

```sql
-- Drop columns "id" and "timestamp"
SELECT price, quantity, city
FROM __input__
```

::: sqlearn.feature_selection.drop.Drop

---

## DropCorrelated

Drop one of each pair of highly correlated features. A **dynamic**
transformer -- pairwise Pearson correlations are computed during `fit()`.

Uses a greedy strategy: for each correlated pair above the threshold,
drop the column that appears in more correlated pairs.

```sql
-- After identifying col_b as correlated with col_a
SELECT col_a, col_c
FROM __input__
```

::: sqlearn.feature_selection.correlated.DropCorrelated

---

## VarianceThreshold

Drop low-variance features. A **dynamic** transformer -- population
variance is computed during `fit()`.

With the default `threshold=0.0`, only constant columns (zero variance)
are removed.

```sql
-- After identifying "constant_col" as zero-variance
SELECT price, quantity
FROM __input__
```

::: sqlearn.feature_selection.variance.VarianceThreshold

---

## SelectKBest

Select the top K features by score against the target column. A **dynamic**
transformer -- feature scores are computed during `fit()`. Requires `y`
to be specified.

Scoring functions: `"f_regression"` (default), `"mutual_info"`, `"f_classif"`.

```sql
-- After selecting top 2 features by correlation with target
SELECT price, quantity
FROM __input__
```

::: sqlearn.feature_selection.kbest.SelectKBest
