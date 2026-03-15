# sqlearn

**Compile ML preprocessing pipelines to SQL.**

sqlearn lets you write scikit-learn-style preprocessing pipelines that compile to SQL
queries via sqlglot ASTs. Every pipeline becomes one query. DuckDB is the default engine,
but any sqlglot-supported database is a valid target (Postgres, MySQL, Snowflake, BigQuery).

=== "Python"

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

## Why sqlearn?

| Problem | sklearn | sqlearn |
|---|---|---|
| Transform new data | Load into Python, run `.transform()` | Run one SQL query |
| Deploy preprocessing | Serialize pipeline, wrap in API | Copy SQL to your database |
| Handle large datasets | Load everything into memory | SQL engine handles it |
| Debug transforms | Print arrays, guess what happened | Read the SQL |

## Key Features

- **sklearn-compatible API** — `fit()`, `transform()`, `fit_transform()`, same interface you know
- **SQL output** — every pipeline compiles to a single SQL query via sqlglot ASTs
- **Expression composition** — transformers chain into nested expressions, not sequential queries
- **Auto column routing** — `StandardScaler` defaults to numeric, `OneHotEncoder` to categorical
- **Custom transformers** — from one-liners (`sq.Expression`) to full subclasses
- **Multi-database** — DuckDB, Postgres, MySQL, Snowflake, BigQuery via sqlglot dialects

## Installation

```bash
pip install sqlearn
```

Requires Python 3.10+.

## Quick Links

- [Getting Started](getting-started.md) — build your first pipeline, see the SQL output
- [API Reference](api/index.md) — full API documentation for all classes and functions
