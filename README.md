# sqlearn

**Compile ML preprocessing pipelines to SQL.**

Write Python pipelines (sklearn-style), get valid SQL. Every pipeline becomes one query.
DuckDB is the default engine, but any sqlglot-supported database works
(Postgres, MySQL, Snowflake, BigQuery).

[![CI](https://github.com/Tomosius/sqlearn/actions/workflows/ci.yml/badge.svg)](https://github.com/Tomosius/sqlearn/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://tomosius.github.io/sqlearn)

## Quick Example

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

The generated SQL:

```sql
SELECT
  (COALESCE(price, 3.0) - 3.0) / NULLIF(1.58, 0) AS price,
  (COALESCE(quantity, 35.0) - 32.5) / NULLIF(14.79, 0) AS quantity,
  CASE WHEN COALESCE(city, 'London') = 'London' THEN 1 ELSE 0 END AS city_london,
  CASE WHEN COALESCE(city, 'London') = 'Paris' THEN 1 ELSE 0 END AS city_paris,
  CASE WHEN COALESCE(city, 'London') = 'Tokyo' THEN 1 ELSE 0 END AS city_tokyo
FROM __input__
```

One query. No loops. No row-by-row processing.

## Why sqlearn?

| Problem | sklearn | sqlearn |
|---|---|---|
| Transform new data | Load into Python, run `.transform()` | Run one SQL query |
| Deploy preprocessing | Serialize pipeline, wrap in API | Copy SQL to your database |
| Handle large datasets | Load everything into memory | SQL engine handles it |
| Reproduce transforms | Hope pickle versions match | SQL is the artifact |
| Debug transforms | Print arrays, guess what happened | Read the SQL |

## Features

- **sklearn-compatible API** — `fit()`, `transform()`, `fit_transform()`, same interface
- **SQL output** — every pipeline compiles to a single SQL query via sqlglot ASTs
- **Expression composition** — transformers chain into nested expressions, not sequential queries
- **Auto column routing** — `StandardScaler` defaults to numeric, `OneHotEncoder` to categorical
- **Custom transformers** — three levels from one-liner SQL to full subclass

## Custom Transformers

```python
# Level 1: One-liner
sq.Expression("price * quantity AS revenue")

# Level 2: Per-column with stats
sq.custom(
    "({col} - {mean}) / NULLIF({std}, 0)",
    columns="numeric",
    learn={"mean": "AVG({col})", "std": "STDDEV_POP({col})"},
)

# Level 3: Full subclass
class MyTransformer(sq.Transformer):
    def discover(self, columns, schema, y_column=None):
        return {f"{col}__mean": exp.Avg(this=exp.Column(this=col)) for col in columns}
    def expressions(self, columns, exprs):
        return {col: exp.Sub(this=exprs[col], expression=exp.Literal.number(self.params_[f"{col}__mean"])) for col in columns}
```

## Installation

```bash
pip install sqlearn
```

Requires Python 3.10+. Core dependencies: `duckdb`, `numpy`, `sqlglot`.

## Documentation

- [Getting Started](https://tomosius.github.io/sqlearn/getting-started/)
- [API Reference](https://tomosius.github.io/sqlearn/api/)

## Status

**v0.1.0** — Core compiler complete. Pipeline, Transformer base, Imputer, StandardScaler,
OneHotEncoder, custom transformers (`sq.Expression`, `sq.custom`), and full test suite (570 tests).

See [milestones](docs/12-milestones.md) for the roadmap.

## License

MIT
