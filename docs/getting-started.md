# Getting Started

## Installation

```bash
pip install sqlearn
```

Requires Python 3.10+. Core dependencies: `duckdb`, `numpy`, `sqlglot`.

## Your First Pipeline

=== "Python"

    ```python
    import sqlearn as sq

    # Create a pipeline
    pipe = sq.Pipeline([
        sq.Imputer(),                    # fill NULLs with column means
        sq.StandardScaler(),             # zero mean, unit variance
        sq.OneHotEncoder(),              # categorical → binary columns
    ])

    # Fit on training data (learns statistics via SQL)
    pipe.fit("train.parquet", y="target")

    # Transform to numpy array
    X = pipe.transform("test.parquet")

    # Or get the compiled SQL
    print(pipe.to_sql())
    ```

=== "Generated SQL"

    ```sql
    SELECT
      (COALESCE(price, 3.0) - 3.0) / NULLIF(1.58, 0) AS price,
      (COALESCE(quantity, 35.0) - 32.5) / NULLIF(14.79, 0) AS quantity,
      CASE WHEN COALESCE(city, 'London') = 'London' THEN 1 ELSE 0 END AS city_london,
      CASE WHEN COALESCE(city, 'London') = 'Paris' THEN 1 ELSE 0 END AS city_paris,
      CASE WHEN COALESCE(city, 'London') = 'Tokyo' THEN 1 ELSE 0 END AS city_tokyo
    FROM __input__
    ```

One pipeline. One SQL query. No loops, no row-by-row processing.

## What Each Step Does

### Imputer — Fill Missing Values

=== "Python"

    ```python
    pipe = sq.Pipeline([sq.Imputer(strategy="mean")])
    pipe.fit("data.parquet")
    print(pipe.to_sql())
    ```

=== "Generated SQL"

    ```sql
    SELECT
      COALESCE(price, 3.0) AS price,
      COALESCE(quantity, 32.5) AS quantity
    FROM __input__
    ```

Learns the mean (or median, most_frequent) during `fit()`, then applies `COALESCE` in SQL.

!!! tip "Auto strategy"
    `sq.Imputer()` (default) picks `mean` for numeric columns and `most_frequent` for
    categorical columns automatically.

### StandardScaler — Zero Mean, Unit Variance

=== "Python"

    ```python
    pipe = sq.Pipeline([sq.StandardScaler()])
    pipe.fit("data.parquet")
    print(pipe.to_sql())
    ```

=== "Generated SQL"

    ```sql
    SELECT
      (price - 3.0) / NULLIF(1.41, 0) AS price,
      (quantity - 30.0) / NULLIF(14.14, 0) AS quantity
    FROM __input__
    ```

Learns mean and population std during `fit()`. Division by zero is prevented via `NULLIF`.

### OneHotEncoder — Categories to Binary Columns

=== "Python"

    ```python
    pipe = sq.Pipeline([sq.OneHotEncoder()])
    pipe.fit("data.parquet")
    print(pipe.to_sql())
    ```

=== "Generated SQL"

    ```sql
    SELECT
      CASE WHEN city = 'London' THEN 1 ELSE 0 END AS city_london,
      CASE WHEN city = 'Paris' THEN 1 ELSE 0 END AS city_paris,
      CASE WHEN city = 'Tokyo' THEN 1 ELSE 0 END AS city_tokyo
    FROM __input__
    ```

Discovers categories during `fit()`, generates one `CASE WHEN` per category.

## Expression Composition

When steps are chained, sqlearn **composes expressions** — it doesn't run sequential queries:

=== "Python"

    ```python
    pipe = sq.Pipeline([
        sq.Imputer(),          # fills NULLs
        sq.StandardScaler(),   # scales using filled values
    ])
    ```

=== "Generated SQL"

    ```sql
    SELECT
      (COALESCE(price, 3.0) - 3.0) / NULLIF(1.58, 0) AS price
    FROM __input__
    ```

The `COALESCE` from Imputer is **nested inside** StandardScaler's expression.
Still one query — no intermediate tables.

## Custom Transformers

sqlearn offers three levels for custom SQL transforms:

### Level 1: One-liner

```python
# Static expression — no data learning needed
sq.Expression("price * quantity AS revenue")
```

### Level 2: Template

```python
# Static per-column
sq.custom("LN({col} + 1)", columns="numeric")

# Dynamic — learns statistics from data
sq.custom(
    "({col} - {mean}) / NULLIF({std}, 0)",
    columns="numeric",
    learn={"mean": "AVG({col})", "std": "STDDEV_POP({col})"},
)
```

### Level 3: Full Subclass

```python
import sqlglot.expressions as exp

class MeanCenterer(sq.Transformer):
    _default_columns = "numeric"
    _classification = "dynamic"

    def discover(self, columns, schema, y_column=None):
        return {f"{col}__mean": exp.Avg(this=exp.Column(this=col))
                for col in columns}

    def expressions(self, columns, exprs):
        return {col: exp.Sub(
                    this=exprs[col],
                    expression=exp.Literal.number(self.params_[f"{col}__mean"]))
                for col in columns}
```

All three levels compose safely with built-in transformers in any pipeline.

## Column Routing

Transformers know which columns to target by default:

| Transformer | Default columns |
|---|---|
| `StandardScaler` | numeric |
| `OneHotEncoder` | categorical |
| `Imputer` | all (strategy adapts per type) |

Override with `columns=`:

```python
sq.StandardScaler(columns=["price", "score"])     # specific columns
sq.StandardScaler(columns=sq.numeric())            # explicit selector
sq.StandardScaler(columns=sq.matching("price_*"))  # pattern matching
```

## sklearn Compatibility

sqlearn matches sklearn output within floating-point tolerance:

```python
import numpy as np
from sklearn.preprocessing import StandardScaler as SklearnScaler

sq_result = sq.StandardScaler().fit_transform("data.parquet")
sk_result = SklearnScaler().fit_transform(numpy_data)

np.testing.assert_allclose(sq_result, sk_result, atol=1e-10)  # passes
```

## Next Steps

- [API Reference](api/index.md) — full documentation of all classes and functions
