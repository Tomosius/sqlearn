# Building Custom Transformers

sqlearn provides three levels of customization for creating your own SQL-based
transformers. This tutorial walks through all three, from simplest to most
powerful, and shows how they compose with built-in transformers in a single
pipeline.

## The three levels

| Level | API | Best for |
|---|---|---|
| 1 | `sq.Expression("SQL")` | Static one-liner -- add a computed column |
| 2 | `sq.custom(sql, learn=...)` | Per-column template, optionally learns stats from data |
| 3 | Subclass `sq.Transformer` | Full control -- CTEs, window functions, joins |

All three levels are validated at creation time, compose safely with built-in
transformers, and generate the same quality SQL.

## Setup

Let's create a dataset to use throughout this tutorial. It represents an
e-commerce order log with numeric, categorical, and derived features.

```python
import duckdb
import sqlearn as sq

conn = duckdb.connect()
conn.execute("""
    CREATE TABLE orders AS SELECT * FROM VALUES
        (101, 29.99,  2, 'electronics', '2024-01-15', 'standard'),
        (102, 149.50, 1, 'electronics', '2024-01-16', 'express'),
        (103, 8.99,   5, 'books',       '2024-01-16', 'standard'),
        (104, 89.00,  1, 'clothing',    '2024-01-17', 'standard'),
        (105, 12.50,  3, 'books',       '2024-01-17', 'express'),
        (106, 299.99, 1, 'electronics', '2024-01-18', 'express'),
        (107, 45.00,  2, 'clothing',    '2024-01-18', 'standard'),
        (108, 6.99,   8, 'books',       '2024-01-19', 'standard'),
        (109, 199.00, 1, 'electronics', '2024-01-19', 'express'),
        (110, 34.99,  4, 'clothing',    '2024-01-20', 'standard'),
        (111, 15.99,  2, 'books',       '2024-01-20', 'standard'),
        (112, 549.99, 1, 'electronics', '2024-01-21', 'express'),
        (113, 22.00,  3, 'clothing',    '2024-01-21', 'standard'),
        (114, 79.99,  1, 'books',       '2024-01-22', 'express'),
        (115, 159.00, 2, 'electronics', '2024-01-22', 'standard')
    t(order_id, unit_price, quantity, category, order_date, shipping)
""")
conn.execute("COPY orders TO 'orders.csv' (HEADER)")
```

## Level 1: `sq.Expression()` -- static one-liner

Use `sq.Expression()` when you need to add a single computed column from a SQL
expression. The expression is static -- it does not learn anything from data.

### Example: Add a revenue column

=== "Python"

    ```python
    revenue = sq.Expression("unit_price * quantity AS revenue")

    pipe = sq.Pipeline([revenue])
    pipe.fit("orders.csv")
    print(pipe.to_sql())
    ```

=== "Generated SQL"

    ```sql
    SELECT
      order_id,
      unit_price,
      quantity,
      category,
      order_date,
      shipping,
      unit_price * quantity AS revenue
    FROM __input__
    ```

The original columns pass through unchanged, and the new `revenue` column is
appended.

### Example: Log-transform a price column

```python
log_price = sq.Expression("LN(unit_price + 1) AS log_price")

pipe = sq.Pipeline([log_price])
pipe.fit("orders.csv")
print(pipe.to_sql())
```

### Rules for `sq.Expression()`

- The SQL must contain `AS <name>` to define the output column name
- Original columns are always preserved (pass-through)
- The expression is parsed through sqlglot at creation time -- invalid SQL
  fails immediately, not at fit time
- Any valid SQL expression works: functions, arithmetic, `CASE WHEN`, etc.

!!! warning "Validation is immediate"
    ```python
    # This fails at creation time, not at fit() time:
    sq.Expression("bad sql !!!")  # ValueError: Invalid SQL template
    ```

## Level 2: `sq.custom()` -- template-based

Use `sq.custom()` when you need per-column processing, or when the transformer
needs to learn statistics from data. This covers about 90% of custom transformer
needs.

### Static per-column template

The `{col}` placeholder is replaced with each target column name.

=== "Python"

    ```python
    log_transform = sq.custom("LN({col} + 1)", columns=["unit_price", "quantity"])

    pipe = sq.Pipeline([log_transform])
    pipe.fit("orders.csv")
    print(pipe.to_sql())
    ```

=== "Generated SQL"

    ```sql
    SELECT
      order_id,
      LN(unit_price + 1) AS unit_price,
      LN(quantity + 1) AS quantity,
      category,
      order_date,
      shipping
    FROM __input__
    ```

Notice that `{col}` was replaced with `unit_price` and `quantity` separately.
The transform replaces the original columns in-place (no `AS` alias means
in-place replacement).

### Dynamic template with `learn`

The `learn` parameter defines statistics to compute from data during `fit()`.
Each key becomes a `{placeholder}` in the template, and each value is a SQL
aggregate.

=== "Python"

    ```python
    # Build your own StandardScaler from scratch
    my_scaler = sq.custom(
        "({col} - {mean}) / NULLIF({std}, 0)",
        columns="numeric",
        learn={
            "mean": "AVG({col})",
            "std": "STDDEV_POP({col})",
        },
    )

    pipe = sq.Pipeline([my_scaler])
    pipe.fit("orders.csv")
    print(pipe.to_sql())
    ```

=== "Generated SQL"

    ```sql
    SELECT
      (order_id - 108.0) / NULLIF(4.32, 0) AS order_id,
      (unit_price - 113.53) / NULLIF(146.11, 0) AS unit_price,
      (quantity - 2.47) / NULLIF(1.92, 0) AS quantity,
      category,
      order_date,
      shipping
    FROM __input__
    ```

During `fit()`, sqlearn executes:

```sql
SELECT
  AVG(order_id)         AS order_id__mean,
  STDDEV_POP(order_id)  AS order_id__std,
  AVG(unit_price)       AS unit_price__mean,
  STDDEV_POP(unit_price) AS unit_price__std,
  AVG(quantity)         AS quantity__mean,
  STDDEV_POP(quantity)  AS quantity__std
FROM orders
```

The learned values are stored in `params_` and substituted into the template
at transform time.

### Template with custom alias

Add `AS <name>` to create a new column instead of replacing in-place:

```python
# Creates new columns: unit_price_log, quantity_log
log_features = sq.custom(
    "LN({col} + 1) AS {col}_log",
    columns=["unit_price", "quantity"],
)

pipe = sq.Pipeline([log_features])
pipe.fit("orders.csv")
print(pipe.get_feature_names_out())
# ['order_id', 'unit_price', 'quantity', 'category', 'order_date', 'shipping',
#  'unit_price_log', 'quantity_log']
```

The alias template `{col}_log` is expanded per column: `unit_price` becomes
`unit_price_log`, `quantity` becomes `quantity_log`.

### Combine mode

Use `mode="combine"` for cross-column expressions (referencing columns by name
directly, not via `{col}`):

```python
total_revenue = sq.custom(
    "unit_price * quantity AS total_revenue",
    columns=["unit_price", "quantity"],
    mode="combine",
)

pipe = sq.Pipeline([total_revenue])
pipe.fit("orders.csv")
print(pipe.to_sql())
# ... unit_price * quantity AS total_revenue ...
```

!!! note "combine vs Expression"
    `mode="combine"` is similar to `sq.Expression()` but allows you to specify
    which columns the transformer depends on via `columns=`. This is useful
    for documentation and for the compiler to track dependencies.

### `sq.custom()` reference

| Parameter | Description |
|---|---|
| `sql` | SQL template with `{col}` and `{param}` placeholders |
| `columns` | Column selector: `"numeric"`, `["a", "b"]`, etc. |
| `learn` | Dict of `{param: aggregate_sql}` -- makes the transformer dynamic |
| `mode` | `"per_column"` (default) or `"combine"` |

## Level 3: Subclass `sq.Transformer` -- full power

When you need full control -- custom parameter logic, multiple output columns per
input, window functions, or CTEs -- subclass `sq.Transformer` directly.

### Example: PercentileClip

Let's build a transformer that clips values to the 5th and 95th percentiles
(winsorization). It needs to learn the percentile values from data, then
generate a `LEAST(GREATEST(col, p5), p95)` expression.

```python
import sqlglot.expressions as exp

class PercentileClip(sq.Transformer):
    """Clip numeric columns to the 5th and 95th percentiles."""

    _default_columns = "numeric"
    _classification = "dynamic"

    def __init__(self, *, lower: float = 0.05, upper: float = 0.95, columns=None):
        super().__init__(columns=columns)
        self.lower = lower
        self.upper = upper

    def discover(self, columns, schema, y_column=None):
        """Learn percentile boundaries from data."""
        result = {}
        for col in columns:
            # PERCENTILE_CONT is a DuckDB aggregate function
            result[f"{col}__p_lower"] = exp.Anonymous(
                this="PERCENTILE_CONT",
                expressions=[
                    exp.Literal.number(self.lower),  # (1)!
                ],
            )
            result[f"{col}__p_upper"] = exp.Anonymous(
                this="PERCENTILE_CONT",
                expressions=[
                    exp.Literal.number(self.upper),
                ],
            )
        return result

    def expressions(self, columns, exprs):
        """Generate LEAST(GREATEST(col, lower), upper) per column."""
        params = self.params_ or {}
        result = {}
        for col in columns:
            p_lower = params.get(f"{col}__p_lower", 0)
            p_upper = params.get(f"{col}__p_upper", 0)

            # GREATEST(col, p_lower) -- clip from below
            clipped_low = exp.Greatest(
                this=exprs[col],
                expressions=[exp.Literal.number(p_lower)],
            )
            # LEAST(..., p_upper) -- clip from above
            result[col] = exp.Least(
                this=clipped_low,
                expressions=[exp.Literal.number(p_upper)],
            )
        return result
```

1. `exp.Literal.number()` creates a numeric literal in the sqlglot AST.

### Using it in a pipeline

=== "Python"

    ```python
    pipe = sq.Pipeline([
        PercentileClip(lower=0.05, upper=0.95),
        sq.StandardScaler(),
    ])
    pipe.fit("orders.csv")
    print(pipe.to_sql())
    ```

=== "Generated SQL"

    ```sql
    SELECT
      (LEAST(GREATEST(order_id, 101.7), 114.3) - 108.0) / NULLIF(4.32, 0) AS order_id,
      (LEAST(GREATEST(unit_price, 7.69), 487.49) - 113.53) / NULLIF(146.11, 0) AS unit_price,
      (LEAST(GREATEST(quantity, 1.0), 7.3) - 2.47) / NULLIF(1.92, 0) AS quantity,
      category,
      order_date,
      shipping
    FROM __input__
    ```

The PercentileClip and StandardScaler expressions compose into a single query.
The `GREATEST`/`LEAST` from PercentileClip is nested inside the StandardScaler's
`(col - mean) / NULLIF(std, 0)`.

### Anatomy of a Transformer subclass

| Attribute/Method | Purpose | Required? |
|---|---|---|
| `_default_columns` | Which columns to target by default (`"numeric"`, `"categorical"`, `"all"`) | Recommended |
| `_classification` | `"static"` or `"dynamic"` -- or omit for auto-detection | Recommended |
| `discover()` | Return `{param: aggregate_expr}` to learn scalar statistics | If dynamic |
| `discover_sets()` | Return `{param: select_query}` to learn multi-row data (categories, etc.) | If needs sets |
| `expressions()` | Return `{col: expr}` for inline SQL transforms | Yes (or `query()`) |
| `query()` | Return a full query wrapping input (for CTEs, windows) | Alternative to `expressions()` |
| `output_schema()` | Declare output schema (if adding/removing/renaming columns) | If schema changes |

### Key rules for subclasses

1. **Use `exprs[col]`**, not `exp.Column(this=col)` -- this is what enables
   expression composition with prior pipeline steps

2. **Return only modified columns** from `expressions()` -- unmodified columns
   pass through automatically

3. **Use sqlglot AST nodes**, never raw SQL strings -- this ensures
   multi-database compatibility

4. **Name parameters as `{col}__{stat}`** -- the double-underscore convention
   prevents naming collisions in batched aggregate queries

## All three levels in one pipeline

Let's build a complete pipeline that uses all three levels together.

=== "Python"

    ```python
    pipe = sq.Pipeline([
        # Level 1: static expression -- add a revenue column
        sq.Expression("unit_price * quantity AS revenue"),

        # Level 2: template -- log-transform numeric columns
        sq.custom("LN({col} + 1)", columns=["unit_price", "quantity"]),

        # Built-in: impute missing values
        sq.Imputer(),

        # Level 3: custom subclass -- clip outliers
        PercentileClip(columns=["unit_price", "quantity"]),

        # Built-in: standardize
        sq.StandardScaler(),

        # Built-in: encode categoricals
        sq.OneHotEncoder(),
    ])

    pipe.fit("orders.csv")

    # View output columns
    print(pipe.get_feature_names_out())

    # View the SQL
    print(pipe.to_sql())
    ```

=== "Output columns"

    ```
    ['order_id', 'unit_price', 'quantity', 'order_date', 'revenue',
     'category_books', 'category_clothing', 'category_electronics',
     'shipping_express', 'shipping_standard']
    ```

All six steps -- built-in and custom -- compose into a single SQL query. The
custom transformers are indistinguishable from built-in ones in the generated SQL.

## When to use which level

| Situation | Level |
|---|---|
| Add a computed column from existing columns | Level 1: `sq.Expression()` |
| Apply the same formula to multiple columns | Level 2: `sq.custom()` with `{col}` |
| Learn per-column statistics, then transform | Level 2: `sq.custom()` with `learn` |
| Need window functions, CTEs, or joins | Level 3: subclass `sq.Transformer` |
| Need custom parameter logic (e.g., conditional thresholds) | Level 3: subclass `sq.Transformer` |
| Need to add/remove/rename columns with custom logic | Level 3: subclass `sq.Transformer` |

!!! tip "Start at Level 2"
    Most custom transforms fit naturally as `sq.custom()` templates. Start there
    and only move to Level 3 if you need capabilities that templates cannot express.

## Summary

In this tutorial you:

1. Used `sq.Expression()` to add static computed columns
2. Used `sq.custom()` with `{col}` templates for per-column transforms
3. Used `sq.custom()` with `learn` for dynamic transforms that learn from data
4. Built a full `sq.Transformer` subclass with `discover()` and `expressions()`
5. Combined all three levels with built-in transformers in one pipeline

The key insight: **all three levels produce sqlglot AST nodes that compose
identically with built-in transformers**. The compiler does not distinguish
between custom and built-in -- they are all just transformers.

## Next steps

- **[sqlearn vs sklearn](sklearn-comparison.md)** -- compare sqlearn and scikit-learn
- **[API Reference: Expression and custom()](../api/custom.md)** -- full API docs
- **[API Reference: Transformer](../api/transformer.md)** -- base class details
