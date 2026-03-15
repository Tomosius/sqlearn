# API Reference

Auto-generated from source code docstrings.

## Core

| Module | Description |
|--------|-------------|
| [Pipeline](pipeline.md) | `Pipeline` class -- fit, transform, to_sql |
| [Transformer](transformer.md) | `Transformer` base class for all transformers |
| [Schema](schema.md) | `Schema` dataclass and column selectors |
| [Compiler](compiler.md) | Expression composition and query building |
| [Backend](backend.md) | `Backend` protocol and `DuckDBBackend` |
| [IO](io.md) | Input resolution (files, tables, DataFrames) |
| [Errors](errors.md) | Error hierarchy |

## Custom Transformers

| Module | Description |
|--------|-------------|
| [Custom](custom.md) | `Expression()` and `custom()` factories |

## Transformers

| Module | Description |
|--------|-------------|
| [Imputer](imputer.md) | NULL imputation (mean, median, most_frequent, constant) |
| [StandardScaler](standard-scaler.md) | Zero mean, unit variance scaling |
| [OneHotEncoder](onehot-encoder.md) | Categorical to binary encoding |

## See Also

- **[Getting Started](../getting-started.md)** -- end-to-end walkthrough with code examples and generated SQL
- **[Pipeline](pipeline.md)** -- how to compose transformers into a single SQL query
- **[Custom Transformers](custom.md)** -- three levels for writing your own SQL transforms
