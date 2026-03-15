# Selectors

Column selectors provide type-aware and composable column routing for
sqlearn pipelines. Instead of manually listing column names, selectors
resolve against the schema at fit time.

## Base Class

::: sqlearn.core.schema.ColumnSelector

## Type Selectors

::: sqlearn.core.schema.numeric

::: sqlearn.core.schema.categorical

::: sqlearn.core.schema.temporal

::: sqlearn.core.schema.boolean

## Pattern & Type Matching

::: sqlearn.core.schema.matching

::: sqlearn.core.schema.dtype

## Explicit Selection

::: sqlearn.core.schema.all_columns

::: sqlearn.core.schema.columns

## Composition Operators

Selectors support set-like composition via Python operators. All operators
return new `ColumnSelector` instances (the originals are unchanged).

| Operator | Meaning | Example |
|----------|---------|---------|
| `a \| b` | Union | `numeric() \| boolean()` |
| `a & b` | Intersection | `numeric() & matching('price_*')` |
| `~a` | Negation | `~categorical()` |
| `a - b` | Difference | `numeric() - matching('id_*')` |

### Usage Examples

```python
import sqlearn as sq

# Select numeric OR boolean columns
pipe = sq.Pipeline([sq.StandardScaler(columns=sq.numeric() | sq.boolean())])

# Select numeric columns except those starting with "id_"
pipe = sq.Pipeline([sq.StandardScaler(columns=sq.numeric() - sq.matching("id_*"))])

# Select everything except categoricals
pipe = sq.Pipeline([sq.StandardScaler(columns=sq.all_columns() - sq.categorical())])

# Select specific columns by name
pipe = sq.Pipeline([sq.StandardScaler(columns=sq.columns("price", "quantity"))])

# Chain multiple operators
sel = (sq.numeric() | sq.boolean()) - sq.matching("id_*")
pipe = sq.Pipeline([sq.StandardScaler(columns=sel)])
```
