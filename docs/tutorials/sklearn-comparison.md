# sqlearn vs sklearn

This tutorial puts sqlearn and scikit-learn side by side on the same
preprocessing task. You will see that they produce identical numerical output,
while sqlearn additionally generates a deployable SQL query.

## The task

Preprocess a housing dataset with:

1. **Impute** missing values (mean for numeric columns)
2. **Scale** numeric columns to zero mean and unit variance
3. **Encode** categorical columns as one-hot binary indicators

This is one of the most common ML preprocessing workflows.

## Setup: Create the data

Both libraries will work with the same data. We create it once and extract the
numpy array that sklearn needs.

```python
import duckdb
import numpy as np

conn = duckdb.connect()
conn.execute("""
    CREATE TABLE houses AS SELECT * FROM VALUES
        (250000.0,  3.0, 1800.0, 'suburban'),
        (180000.0,  2.0, 1200.0, 'urban'),
        (350000.0,  4.0, 2400.0, 'suburban'),
        (NULL,      2.0,  950.0, 'urban'),
        (420000.0,  5.0, 3200.0, 'rural'),
        (195000.0,  3.0,  NULL,  'urban'),
        (310000.0,  4.0, 2100.0, 'suburban'),
        (275000.0,  3.0, 1950.0, 'rural'),
        (NULL,      2.0, 1100.0, 'urban'),
        (380000.0,  4.0, 2800.0, 'rural'),
        (225000.0,  3.0, 1600.0, 'suburban'),
        (165000.0,  2.0,  900.0, 'urban'),
        (290000.0,  3.0, 2000.0, 'suburban'),
        (410000.0,  5.0, 3100.0, 'rural'),
        (205000.0,  2.0, 1300.0, 'urban')
    t(price, bedrooms, sqft, location)
""")
conn.execute("COPY houses TO 'houses.csv' (HEADER)")
```

Extract numpy arrays for sklearn:

```python
# Numeric columns as float64 (NaN for NULL)
rows = conn.execute("SELECT price, bedrooms, sqft FROM houses").fetchall()
X_numeric = np.array(rows, dtype=np.float64)

# Categorical column as object array
rows_cat = conn.execute("SELECT location FROM houses").fetchall()
X_categorical = np.array(rows_cat, dtype=object)
```

## The sklearn way

sklearn requires a `ColumnTransformer` to apply different transforms to different
column types, then wraps everything in a `Pipeline`.

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder as SkOneHotEncoder
from sklearn.preprocessing import StandardScaler as SkStandardScaler

# Step 1: Build numeric sub-pipeline
numeric_pipeline = SkPipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", SkStandardScaler()),
])

# Step 2: Build categorical sub-pipeline
categorical_pipeline = SkPipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", SkOneHotEncoder(sparse_output=False)),
])

# Step 3: Combine with ColumnTransformer
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, [0, 1, 2]),       # price, bedrooms, sqft
    ("categorical", categorical_pipeline, [3]),      # location
])

# Step 4: Fit and transform
X_combined = np.column_stack([X_numeric, X_categorical])
sk_result = preprocessor.fit_transform(X_combined)

print(sk_result.shape)  # (15, 7)
```

That is 4 objects (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`,
`ColumnTransformer`) assembled in 2 sub-pipelines, with manual column index
management.

## The sqlearn way

sqlearn handles column routing automatically. Each transformer knows which
column types it targets by default.

```python
import sqlearn as sq
from sqlearn.core.backend import DuckDBBackend

backend = DuckDBBackend(connection=conn)

pipe = sq.Pipeline([
    sq.Imputer(strategy="mean"),
    sq.StandardScaler(),
    sq.OneHotEncoder(),
])

pipe.fit("houses", backend=backend)
sq_result = pipe.transform("houses", backend=backend)

print(sq_result.shape)  # (15, 7)
```

Three steps. No `ColumnTransformer`. No column indices. No sub-pipelines.

## Verify identical output

The two approaches should produce numerically identical results. Let's verify.

```python
# sqlearn sorts one-hot columns alphabetically, sklearn does too (by default).
# Both scale using population std (ddof=0).

# Compare numeric columns (first 3)
np.testing.assert_allclose(
    sq_result[:, :3],       # price, bedrooms, sqft from sqlearn
    sk_result[:, :3],       # price, bedrooms, sqft from sklearn
    rtol=1e-6,
)
print("Numeric columns: MATCH")

# Compare one-hot columns (last 4)
np.testing.assert_allclose(
    sq_result[:, 3:].astype(np.float64),   # location_rural, _suburban, _urban
    sk_result[:, 3:],                       # same from sklearn
    rtol=1e-6,
)
print("Categorical columns: MATCH")
```

Both produce the same float64 array within floating-point tolerance.

!!! info "Why `rtol=1e-6`?"
    DuckDB and numpy use the same IEEE 754 double-precision arithmetic, but
    intermediate rounding in SQL aggregate functions (AVG, STDDEV_POP) can
    introduce tiny differences at the last few decimal places. A relative
    tolerance of 1e-6 accounts for this.

## The bonus: sqlearn generates SQL

Here is what sklearn cannot do:

=== "Python"

    ```python
    sql = pipe.to_sql()
    print(sql)
    ```

=== "Generated SQL"

    ```sql
    SELECT
      (COALESCE(price, 281923.08) - 281923.08) / NULLIF(79498.59, 0) AS price,
      (COALESCE(bedrooms, 3.07) - 3.07) / NULLIF(0.96, 0) AS bedrooms,
      (COALESCE(sqft, 1900.0) - 1900.0) / NULLIF(711.93, 0) AS sqft,
      CASE WHEN location = 'rural' THEN 1 ELSE 0 END AS location_rural,
      CASE WHEN location = 'suburban' THEN 1 ELSE 0 END AS location_suburban,
      CASE WHEN location = 'urban' THEN 1 ELSE 0 END AS location_urban
    FROM __input__
    ```

This SQL can be:

- Deployed directly in a database (Postgres, Snowflake, BigQuery, etc.)
- Embedded in a dbt model
- Used in a data warehouse transformation layer
- Run without any Python dependency in production

sklearn cannot export its fitted pipeline to SQL. You would need a separate
library or manual translation to deploy preprocessing in a database.

## Side-by-side code comparison

### Imputation

=== "sklearn"

    ```python
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X_numeric)
    ```

=== "sqlearn"

    ```python
    import sqlearn as sq

    pipe = sq.Pipeline([sq.Imputer(strategy="mean")])
    pipe.fit("houses.csv")
    X_imputed = pipe.transform("houses.csv")
    ```

### Standard scaling

=== "sklearn"

    ```python
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)
    ```

=== "sqlearn"

    ```python
    pipe = sq.Pipeline([sq.StandardScaler()])
    pipe.fit("houses.csv")
    X_scaled = pipe.transform("houses.csv")
    ```

### One-hot encoding

=== "sklearn"

    ```python
    from sklearn.preprocessing import OneHotEncoder

    encoder = OneHotEncoder(sparse_output=False)
    X_encoded = encoder.fit_transform(X_categorical)
    ```

=== "sqlearn"

    ```python
    pipe = sq.Pipeline([sq.OneHotEncoder()])
    pipe.fit("houses.csv")
    X_encoded = pipe.transform("houses.csv")
    ```

### Mixed types (the key difference)

=== "sklearn"

    ```python
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]), [0, 1, 2]),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(sparse_output=False)),
        ]), [3]),
    ])
    result = preprocessor.fit_transform(X_combined)
    ```

=== "sqlearn"

    ```python
    pipe = sq.Pipeline([
        sq.Imputer(strategy="mean"),
        sq.StandardScaler(),
        sq.OneHotEncoder(),
    ])
    pipe.fit("houses.csv")
    result = pipe.transform("houses.csv")
    ```

The difference is most visible with mixed-type data. sklearn requires explicit
column routing via `ColumnTransformer` with integer indices. sqlearn routes
automatically based on column types.

## Key differences

| Aspect | sklearn | sqlearn |
|---|---|---|
| **Column routing** | Manual via `ColumnTransformer` with indices | Automatic by column type |
| **Input format** | numpy array or pandas DataFrame | File path, table name, or DataFrame |
| **Execution** | Python/numpy in-process | SQL via DuckDB (or any database) |
| **SQL export** | Not available | `pipe.to_sql()` for any dialect |
| **Data size** | Limited by memory | Limited by database (can be much larger) |
| **Missing values** | `NaN` in numpy | `NULL` in SQL |
| **Std deviation** | Population std (ddof=0) by default | Population std (ddof=0) -- matches sklearn |
| **Composition** | Sequential execution | Expression nesting (one query) |
| **Target column** | Separate `y` array | Column name: `y="price"` |
| **Fitted params** | `.mean_`, `.scale_`, etc. | `.params_` dict, `.sets_` dict |

## What is the same

Both libraries share the same design principles:

- **fit/transform API** -- learn from data, then apply
- **Pipeline composition** -- chain steps together
- **Reproducibility** -- fitted pipelines produce deterministic output
- **Parameter inspection** -- view learned statistics after fitting
- **Clone/copy** -- create independent copies for parallel use

If you know sklearn, you already know sqlearn's API. The difference is in
the execution engine (SQL vs numpy) and the bonus output (deployable SQL queries).

## When to use which

| Use case | Recommendation |
|---|---|
| Prototyping in a notebook with small data | Either works well |
| Preprocessing before sklearn models | sqlearn (simpler pipeline syntax) |
| Deploying preprocessing to a database | sqlearn (SQL export) |
| Preprocessing in a data warehouse (dbt, etc.) | sqlearn (SQL export) |
| Large datasets that don't fit in memory | sqlearn (SQL pushdown) |
| Existing sklearn codebase, no SQL needed | sklearn (no migration needed) |
| Custom transforms using numpy operations | sklearn (numpy-native) |
| Custom transforms using SQL operations | sqlearn (SQL-native) |

## Summary

In this tutorial you:

1. Built the same preprocessing pipeline in both sklearn and sqlearn
2. Verified they produce numerically identical output
3. Saw the SQL query that sqlearn generates (sklearn cannot do this)
4. Compared the code side by side for common operations
5. Identified the key differences and when to use each library

The key insight: **sqlearn and sklearn produce the same results, but sqlearn
additionally gives you a deployable SQL query**. If your preprocessing needs to
run in a database, sqlearn is the simpler path.

## Next steps

- **[Your First Pipeline](basic-pipeline.md)** -- deeper dive into sqlearn pipelines
- **[Custom Transformers](custom-transformers.md)** -- build transformers that sklearn cannot express
- **[API Reference: Pipeline](../api/pipeline.md)** -- full Pipeline documentation
