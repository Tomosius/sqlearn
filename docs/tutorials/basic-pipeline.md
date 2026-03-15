# Your First Pipeline

This tutorial walks through building a complete preprocessing pipeline with
sqlearn. You will create sample data, fit a pipeline that handles missing values,
scales numeric columns, and encodes categorical columns -- then inspect the
learned parameters and the generated SQL.

## What you will build

A three-step pipeline:

1. **Imputer** -- fill missing values with learned statistics
2. **StandardScaler** -- standardize numeric columns to zero mean and unit variance
3. **OneHotEncoder** -- convert categorical columns to binary indicator columns

The entire pipeline compiles to a single SQL query.

## Step 1: Import sqlearn

```python
import sqlearn as sq
```

That single import gives you access to `Pipeline`, all built-in transformers,
column selectors, and custom transformer factories.

## Step 2: Create sample data

sqlearn works directly with files, table names, or DataFrames. For this tutorial,
let's create a CSV file with mixed data types -- numeric columns with some missing
values and a categorical column.

```python
import duckdb

conn = duckdb.connect()
conn.execute("""
    CREATE TABLE housing AS SELECT * FROM VALUES
        (250000,  3, 1800, 'suburban', 1),
        (180000,  2, 1200, 'urban',    0),
        (350000,  4, 2400, 'suburban', 1),
        (NULL,    2, 950,  'urban',    0),
        (420000,  5, 3200, 'rural',    1),
        (195000,  3, NULL, 'urban',    0),
        (310000,  4, 2100, 'suburban', 1),
        (275000,  3, 1950, NULL,       0),
        (NULL,    2, 1100, 'urban',    1),
        (380000,  4, 2800, 'rural',    0),
        (225000,  3, 1600, 'suburban', 1),
        (165000,  2, 900,  'urban',    0),
        (290000,  3, 2000, 'suburban', 1),
        (410000,  5, 3100, 'rural',    0),
        (205000,  2, 1300, 'urban',    1)
    t(price, bedrooms, sqft, location, has_garage)
""")
conn.execute("COPY housing TO 'housing.csv' (HEADER)")
```

This creates a dataset with 15 rows and 5 columns:

- `price` -- numeric, with 2 missing values
- `bedrooms` -- numeric, no missing values
- `sqft` -- numeric, with 1 missing value
- `location` -- categorical (suburban, urban, rural), with 1 missing value
- `has_garage` -- numeric indicator (0/1)

## Step 3: Build the pipeline

=== "Python"

    ```python
    pipe = sq.Pipeline([
        sq.Imputer(),           # Step 1: fill missing values
        sq.StandardScaler(),    # Step 2: standardize numeric columns
        sq.OneHotEncoder(),     # Step 3: encode categorical columns
    ])
    ```

=== "With named steps"

    ```python
    pipe = sq.Pipeline([
        ("impute", sq.Imputer()),
        ("scale", sq.StandardScaler()),
        ("encode", sq.OneHotEncoder()),
    ])
    ```

=== "Dict syntax"

    ```python
    pipe = sq.Pipeline({
        "impute": sq.Imputer(),
        "scale": sq.StandardScaler(),
        "encode": sq.OneHotEncoder(),
    })
    ```

All three forms are equivalent. Named steps make it easier to access individual
transformers later.

!!! note "Auto column routing"
    Each transformer knows which columns to target by default:

    - `Imputer()` targets **all** columns (adapts strategy per column type)
    - `StandardScaler()` targets **numeric** columns only
    - `OneHotEncoder()` targets **categorical** columns only

    No `ColumnTransformer` needed -- sqlearn handles routing automatically.

## Step 4: Fit the pipeline

```python
pipe.fit("housing.csv", y="has_garage")
```

During `fit()`, sqlearn executes SQL queries against your data to learn statistics.
Here is what happens at each step:

| Step | What it learns | SQL executed |
|---|---|---|
| Imputer | Fill values per column (median for numeric, mode for categorical) | `SELECT MEDIAN(price), MEDIAN(bedrooms), ..., MODE(location) FROM housing` |
| StandardScaler | Mean and population std per numeric column | `SELECT AVG(price), STDDEV_POP(price), AVG(bedrooms), ... FROM housing` |
| OneHotEncoder | Distinct categories per categorical column | `SELECT DISTINCT location FROM housing` |

The `y="has_garage"` argument tells the pipeline which column is the prediction
target. Target-aware transformers (like future TargetEncoder) can use this
information. For now, all columns pass through every step.

!!! tip "Method chaining"
    `fit()` returns the pipeline itself, so you can chain calls:
    ```python
    X = pipe.fit("housing.csv", y="has_garage").transform("housing.csv")
    ```

## Step 5: Transform the data

```python
import numpy as np

X = pipe.transform("housing.csv")

print(type(X))    # <class 'numpy.ndarray'>
print(X.shape)    # (15, 7)
print(X.dtype)    # float64
```

`transform()` compiles the entire pipeline to SQL, executes it, and returns a
numpy array. All values are float64 -- missing values have been filled, numeric
columns standardized, and categorical columns replaced with binary indicators.

!!! info "What the 7 columns are"
    The 5 original columns become 7 output columns:

    - `price` -- imputed and standardized
    - `bedrooms` -- imputed and standardized
    - `sqft` -- imputed and standardized
    - `has_garage` -- imputed and standardized
    - `location_rural` -- binary: 1 if rural, 0 otherwise
    - `location_suburban` -- binary: 1 if suburban, 0 otherwise
    - `location_urban` -- binary: 1 if urban, 0 otherwise

    The original `location` column is replaced by the three binary columns.

## Step 6: View the generated SQL

=== "Python"

    ```python
    sql = pipe.to_sql()
    print(sql)
    ```

=== "Generated SQL"

    ```sql
    SELECT
      (COALESCE(price, 275000.0) - 281923.08) / NULLIF(80609.87, 0) AS price,
      (COALESCE(bedrooms, 3.0) - 3.07) / NULLIF(1.03, 0) AS bedrooms,
      (COALESCE(sqft, 1800.0) - 1823.21) / NULLIF(713.46, 0) AS sqft,
      (has_garage - 0.53) / NULLIF(0.50, 0) AS has_garage,
      CASE WHEN COALESCE(location, 'urban') = 'rural' THEN 1 ELSE 0 END AS location_rural,
      CASE WHEN COALESCE(location, 'urban') = 'suburban' THEN 1 ELSE 0 END AS location_suburban,
      CASE WHEN COALESCE(location, 'urban') = 'urban' THEN 1 ELSE 0 END AS location_urban
    FROM __input__
    ```

Notice how the expressions **compose** -- the `COALESCE` from the Imputer is
nested inside the StandardScaler's `(col - mean) / NULLIF(std, 0)`. This is
not three sequential queries. It is one query with nested expressions.

!!! tip "Target different databases"
    `to_sql()` defaults to DuckDB dialect. For other databases:
    ```python
    pipe.to_sql(dialect="postgres")    # PostgreSQL-compatible SQL
    pipe.to_sql(dialect="snowflake")   # Snowflake-compatible SQL
    pipe.to_sql(dialect="bigquery")    # BigQuery-compatible SQL
    ```

    The SQL structure is the same -- only syntax details change.

## Step 7: Inspect the pipeline

### View output column names

```python
print(pipe.get_feature_names_out())
# ['price', 'bedrooms', 'sqft', 'has_garage',
#  'location_rural', 'location_suburban', 'location_urban']
```

### Access individual steps

```python
# By index
imputer = pipe.steps[0][1]    # (name, transformer) tuple
print(imputer)                # Imputer()

# By name (if using named steps)
pipe_named = sq.Pipeline([
    ("impute", sq.Imputer()),
    ("scale", sq.StandardScaler()),
    ("encode", sq.OneHotEncoder()),
])
pipe_named.fit("housing.csv", y="has_garage")

scaler = pipe_named.named_steps["scale"]
print(scaler)                 # StandardScaler()
```

### Inspect learned parameters

Every fitted transformer stores its learned statistics in `params_` (for scalar
values) and `sets_` (for multi-row values like category lists).

```python
# StandardScaler learns mean and std per column
scaler = pipe_named.named_steps["scale"]
print(scaler.params_)
# {'price__mean': 281923.08, 'price__std': 80609.87,
#  'bedrooms__mean': 3.07, 'bedrooms__std': 1.03,
#  'sqft__mean': 1823.21, 'sqft__std': 713.46,
#  'has_garage__mean': 0.53, 'has_garage__std': 0.50}

# OneHotEncoder learns categories per column
encoder = pipe_named.named_steps["encode"]
print(encoder.sets_)
# {'location__categories': [{'location': 'rural'},
#                            {'location': 'suburban'},
#                            {'location': 'urban'}]}
```

### Check fitted state

```python
print(pipe.is_fitted)          # True

# Individual steps
for name, step in pipe.steps:
    print(f"{name}: fitted={step.is_fitted}")
# step_00: fitted=True
# step_01: fitted=True
# step_02: fitted=True
```

## Going further

### Compose pipelines with `+`

```python
clean = sq.Pipeline([sq.Imputer()])
scale = sq.Pipeline([sq.StandardScaler()])
encode = sq.Pipeline([sq.OneHotEncoder()])

full = clean + scale + encode  # new pipeline with all three steps
```

You can also add individual transformers:

```python
pipe = sq.Pipeline([sq.Imputer()])
pipe = pipe + sq.StandardScaler()  # returns a new pipeline
```

### Override column selection

```python
# Scale only specific columns
sq.StandardScaler(columns=["price", "sqft"])

# Use selectors
sq.StandardScaler(columns=sq.numeric())       # all numeric columns
sq.StandardScaler(columns=sq.matching("*_score"))  # pattern matching
```

### Use `fit_transform()` shortcut

```python
# Equivalent to pipe.fit(data).transform(data)
X = pipe.fit_transform("housing.csv", y="has_garage")
```

## Summary

In this tutorial you:

1. Created a dataset with mixed types and missing values
2. Built a three-step pipeline: Imputer, StandardScaler, OneHotEncoder
3. Fitted the pipeline to learn statistics from data via SQL
4. Transformed data to a numpy array
5. Viewed the single SQL query that the pipeline compiles to
6. Inspected learned parameters and output column names

The key insight: **sqlearn compiles your entire pipeline to one SQL query with
nested expressions**. No intermediate tables, no row-by-row processing, no
Python-side data manipulation.

## Next steps

- **[Custom Transformers](custom-transformers.md)** -- build your own SQL transforms
- **[sqlearn vs sklearn](sklearn-comparison.md)** -- compare with scikit-learn side by side
- **[API Reference: Pipeline](../api/pipeline.md)** -- full Pipeline documentation
- **[API Reference: Imputer](../api/imputer.md)** -- all imputation strategies
- **[API Reference: StandardScaler](../api/standard-scaler.md)** -- scaling options
