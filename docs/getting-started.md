# Getting Started

## Installation

```bash
pip install sqlearn
```

## Your First Pipeline

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

## Custom Transformers

### Level 1: One-liner

```python
sq.Expression("price * quantity AS revenue")
```

### Level 2: Template

```python
# Static per-column
sq.custom("LN({col} + 1)", columns="numeric")

# Dynamic (learns from data)
sq.custom(
    "({col} - {mean}) / NULLIF({std}, 0)",
    columns="numeric",
    learn={"mean": "AVG({col})", "std": "STDDEV_POP({col})"},
)
```

### Level 3: Subclass

```python
class MyTransformer(sq.Transformer):
    _default_columns = "numeric"
    _classification = "dynamic"

    def discover(self, columns, schema, y_column=None):
        # Return aggregates to learn from data
        ...

    def expressions(self, columns, exprs):
        # Return column expressions using learned params
        ...
```
