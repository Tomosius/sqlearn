# sqlearn

**Compile ML preprocessing pipelines to SQL.**

sqlearn lets you write scikit-learn-style preprocessing pipelines that compile to SQL
queries via sqlglot ASTs. Every pipeline becomes one query. DuckDB is the default engine,
but any sqlglot-supported database is a valid target.

```python
import sqlearn as sq

pipe = sq.Pipeline([
    sq.Imputer(),
    sq.StandardScaler(),
    sq.OneHotEncoder(),
])
pipe.fit("train.parquet", y="target")
X = pipe.transform("test.parquet")    # numpy array
sql = pipe.to_sql()                    # valid DuckDB SQL
```

## Key Features

- **One base class** -- all transformers extend `Transformer` with three methods
- **Auto column routing** -- `StandardScaler` defaults to numeric, `OneHotEncoder` to categorical
- **Expression composition** -- transformers compose into nested sqlglot ASTs
- **Custom transformers** -- from one-liners (`sq.Expression`) to full subclasses

## Installation

```bash
pip install sqlearn
```

## Quick Links

- [Getting Started](getting-started.md) -- build your first pipeline
- [API Reference](api/index.md) -- full API documentation
