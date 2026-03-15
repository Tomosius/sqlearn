# Custom Transformers

sqlearn provides three levels of customization for writing your own transformers. Start
with the simplest level that meets your needs and move up only when necessary.

| Level | API | When to use |
|---|---|---|
| 1 | `sq.Expression("SQL")` | Static one-liner that adds a new column |
| 2 | `sq.custom(sql, ...)` | Per-column templates, optionally learning stats |
| 3 | Subclass `sq.Transformer` | Full power: sets, CTEs, joins, windows |


## Level 1: Expression

For a single new column derived from existing ones, `sq.Expression` is the simplest
option. Write the SQL expression with an `AS` clause, and sqlearn handles the rest.

```python
import sqlearn as sq

pipe = sq.Pipeline([
    sq.Imputer(),
    sq.Expression("price * quantity AS revenue"),
    sq.StandardScaler(),
])
pipe.fit("data.parquet")
```

The expression is parsed through sqlglot at creation time, so you get immediate
validation. At transform time, column references in the expression are automatically
substituted with composed expressions from prior pipeline steps.

=== "Python"

    ```python
    sq.Expression("price * quantity AS revenue")
    ```

=== "Generated SQL"

    ```sql
    SELECT
      ...,
      price * quantity AS revenue
    FROM __input__
    ```

If the expression follows an Imputer, the column references are composed:

=== "Python"

    ```python
    pipe = sq.Pipeline([
        sq.Imputer(strategy="mean"),
        sq.Expression("price * quantity AS revenue"),
    ])
    ```

=== "Generated SQL"

    ```sql
    SELECT
      ...,
      COALESCE(price, 42.5) * COALESCE(quantity, 30.0) AS revenue
    FROM __input__
    ```

### Rules for Expression

- Must contain `AS <name>` to define the output column name.
- Original columns pass through unchanged --- the expression adds a new column.
- Always static --- no data is learned during fit.
- Any valid SQL expression works: `COALESCE`, `CASE WHEN`, `CAST`, function calls, etc.

```python
# Various expression examples
sq.Expression("COALESCE(nickname, first_name) AS display_name")
sq.Expression("CASE WHEN age >= 18 THEN 'adult' ELSE 'minor' END AS age_group")
sq.Expression("EXTRACT(YEAR FROM created_at) AS created_year")
sq.Expression("LN(price + 1) AS log_price")
```

!!! warning "Expression adds a column, it does not replace one"
    `sq.Expression("LN(price) AS log_price")` creates a new `log_price` column
    while keeping the original `price`. If you want to transform a column in place,
    use Level 2 (`sq.custom`).


## Level 2: custom()

For per-column templates with optional learning, `sq.custom()` covers about 90% of
custom transformer needs without subclassing.

### Static per-column

Apply the same SQL template to each target column. Use `{col}` as a placeholder:

```python
# Apply log transform to all numeric columns
log = sq.custom("LN({col} + 1)", columns="numeric")
pipe = sq.Pipeline([log])
```

Without an `AS` clause, the template replaces the column in place --- `{col}` is
used as the output column name:

=== "Python"

    ```python
    sq.custom("LN({col} + 1)", columns="numeric")
    ```

=== "Generated SQL"

    ```sql
    SELECT
      LN(price + 1) AS price,
      LN(quantity + 1) AS quantity
    FROM __input__
    ```

With an `AS` clause, the template creates new columns:

=== "Python"

    ```python
    sq.custom("LN({col} + 1) AS {col}_log", columns="numeric")
    ```

=== "Generated SQL"

    ```sql
    SELECT
      price,
      quantity,
      LN(price + 1) AS price_log,
      LN(quantity + 1) AS quantity_log
    FROM __input__
    ```

### Dynamic per-column

Add the `learn=` parameter to learn statistics from data. Each key becomes a
placeholder in the template, and each value is a SQL aggregate:

```python
# Z-score normalization (equivalent to StandardScaler)
z_score = sq.custom(
    "({col} - {mean}) / NULLIF({std}, 0)",
    columns="numeric",
    learn={
        "mean": "AVG({col})",
        "std": "STDDEV_POP({col})",
    },
)
pipe = sq.Pipeline([z_score])
pipe.fit("data.parquet")
```

=== "Python"

    ```python
    sq.custom(
        "({col} - {mean}) / NULLIF({std}, 0)",
        columns="numeric",
        learn={"mean": "AVG({col})", "std": "STDDEV_POP({col})"},
    )
    ```

=== "Generated SQL (after fit)"

    ```sql
    SELECT
      (price - 42.5) / NULLIF(10.2, 0) AS price,
      (quantity - 30.0) / NULLIF(8.16, 0) AS quantity
    FROM __input__
    ```

During `fit()`, the compiler expands the `learn` templates per column, batches them
into a single aggregate query, and stores the results. During `transform()`, the
learned values replace the placeholders.

### Combine mode

For cross-column expressions that reference multiple columns by name instead of using
the `{col}` placeholder:

```python
bmi = sq.custom(
    "weight / (height * height) * 703 AS bmi",
    columns=["weight", "height"],
    mode="combine",
)
pipe = sq.Pipeline([bmi])
```

=== "Python"

    ```python
    sq.custom(
        "weight / (height * height) * 703 AS bmi",
        columns=["weight", "height"],
        mode="combine",
    )
    ```

=== "Generated SQL"

    ```sql
    SELECT
      ...,
      weight / (height * height) * 703 AS bmi
    FROM __input__
    ```

!!! note "`mode='combine'` cannot use `{col}`"
    In combine mode, reference columns by their actual names in the SQL template.
    The `{col}` placeholder is not available.

### custom() summary

| Feature | `per_column` (default) | `combine` |
|---|---|---|
| `{col}` placeholder | Yes, iterates per column | Not available |
| `learn=` | Yes, per column | Not yet supported |
| Creates new columns | With `AS` clause | With `AS` clause |
| Replaces columns | Without `AS` clause | N/A |


## Level 3: Subclass Transformer

For full control, subclass `sq.Transformer`. This gives you access to:

- Multi-row set learning (`discover_sets`)
- Query-level transformations (`query` for CTEs, window functions, joins)
- Custom schema changes (`output_schema`)
- Full access to the sqlglot AST

### The three key methods

Every transformer can override up to three methods:

| Method | Purpose | Called during |
|---|---|---|
| `discover()` | Return sqlglot aggregates to learn scalar stats | `fit()` |
| `discover_sets()` | Return sqlglot SELECT queries for multi-row data | `fit()` |
| `expressions()` | Return sqlglot AST nodes for inline SQL | `transform()` |
| `query()` | Return a full query wrapping the input | `transform()` |
| `output_schema()` | Declare how the schema changes | Both |

### Example: Mean centering

A minimal dynamic transformer that subtracts the mean from each column:

```python
import sqlglot.expressions as exp
import sqlearn as sq

class MeanCenterer(sq.Transformer):
    _default_columns = "numeric"
    _classification = "dynamic"

    def discover(self, columns, schema, y_column=None):
        """Learn the mean of each column via AVG()."""
        return {
            f"{col}__mean": exp.Avg(this=exp.Column(this=col))
            for col in columns
        }

    def expressions(self, columns, exprs):
        """Subtract the learned mean from each column."""
        result = {}
        for col in columns:
            mean = self.params_[f"{col}__mean"]
            result[col] = exp.Sub(
                this=exprs[col],
                expression=exp.Literal.number(mean),
            )
        return result
```

Use it like any built-in transformer:

```python
pipe = sq.Pipeline([MeanCenterer()])
pipe.fit("data.parquet")

print(pipe.to_sql())
# SELECT price - 42.5 AS price, quantity - 30.0 AS quantity FROM __input__
```

### Example: Static log transform

A static transformer needs no data:

```python
class LogTransform(sq.Transformer):
    _default_columns = "numeric"
    _classification = "static"

    def expressions(self, columns, exprs):
        """Apply LN(col + 1) to each column."""
        return {
            col: exp.Anonymous(
                this="LN",
                expressions=[exp.Add(this=exprs[col], expression=exp.Literal.number(1))],
            )
            for col in columns
        }
```

Since `_classification = "static"`, the compiler skips `discover()` during fit.
No SQL aggregate queries are generated for this step.

### Example: Adding columns with output_schema

When your transformer adds new columns to the output, you must override
`output_schema()` to declare them:

```python
class InteractionFeatures(sq.Transformer):
    _default_columns = "numeric"
    _classification = "static"

    def expressions(self, columns, exprs):
        """Create pairwise products of numeric columns."""
        result = {}
        for i, col_a in enumerate(columns):
            for col_b in columns[i + 1:]:
                name = f"{col_a}_x_{col_b}"
                result[name] = exp.Mul(this=exprs[col_a], expression=exprs[col_b])
        return result

    def output_schema(self, schema):
        """Declare the new interaction columns."""
        cols = schema.numeric()
        new_cols = {}
        for i, col_a in enumerate(cols):
            for col_b in cols[i + 1:]:
                new_cols[f"{col_a}_x_{col_b}"] = "DOUBLE"
        return schema.add(new_cols)
```

!!! warning "Always override output_schema when adding or removing columns"
    If `expressions()` returns columns that `output_schema()` does not declare,
    sqlearn will warn you and filter them out. This is a safety check to prevent
    schema drift.

### The discover / expressions contract

The contract between `discover()` and `expressions()` is straightforward:

1. `discover()` returns `{param_name: sqlglot_aggregate}`. The compiler executes
   these as SQL and stores the results in `self.params_`.
2. `expressions()` reads from `self.params_` to build the transform expressions.
3. Parameter naming convention: `"{col}__{stat}"` (e.g., `"price__mean"`).

```python
# discover() returns:
{"price__mean": exp.Avg(this=exp.Column(this="price"))}

# After fit, self.params_ contains:
{"price__mean": 42.5}

# expressions() reads params_ and builds:
{"price": exp.Sub(this=exprs["price"], expression=exp.Literal.number(42.5))}
```

### Using discover_sets for multi-row data

Some transformers need more than scalar statistics. For example, an encoder that needs
the full list of categories. Override `discover_sets()` to return SELECT queries:

```python
class MyEncoder(sq.Transformer):
    _default_columns = "categorical"
    _classification = "dynamic"

    def discover_sets(self, columns, schema, y_column=None):
        """Learn the distinct categories per column."""
        return {
            f"{col}__categories": exp.select(
                exp.Distinct(expressions=[exp.Column(this=col)])
            )
            for col in columns
        }

    def expressions(self, columns, exprs):
        """Generate CASE WHEN expressions for each category."""
        result = {}
        for col in columns:
            categories = self.sets_[f"{col}__categories"]
            for row in categories:
                cat_value = row[col]
                name = f"{col}_{cat_value}"
                result[name] = exp.Case(
                    ifs=[exp.If(
                        this=exp.EQ(this=exprs[col], expression=exp.Literal.string(cat_value)),
                        true=exp.Literal.number(1),
                    )],
                    default=exp.Literal.number(0),
                )
        return result
```

Results from `discover_sets()` are stored in `self.sets_` as lists of dicts
(one dict per row). This is different from `discover()` whose results go to
`self.params_` as scalar values.

### Using query() for CTE-level transforms

When inline expressions are not enough (window functions, JOINs, complex CTEs),
override `query()`:

```python
class RankTransformer(sq.Transformer):
    _default_columns = "numeric"
    _classification = "static"

    def query(self, input_query):
        """Wrap the input with a window function."""
        # input_query is a sqlglot SELECT expression
        # Return a new SELECT that wraps it
        return exp.select(
            exp.Star(),
            exp.Window(
                this=exp.RowNumber(),
                order=exp.Order(expressions=[exp.Column(this="price")]),
            ).as_("price_rank"),
        ).from_(input_query.subquery("__sub__"))
```

When `query()` returns a non-None expression, the compiler promotes it to a CTE.
Subsequent pipeline steps compose against the CTE's output columns.

!!! note "query() and expressions() are alternatives"
    The compiler tries `query()` first. If it returns `None` (the default), the
    compiler falls back to `expressions()`. You can implement both --- `query()` for
    the complex parts and `expressions()` for simple column transforms.


## Choosing the right level

| Need | Level | Example |
|---|---|---|
| Add a derived column | 1: `Expression` | `sq.Expression("price * qty AS revenue")` |
| Apply same formula per column | 2: `custom` | `sq.custom("LN({col} + 1)")` |
| Learn stats and apply per column | 2: `custom` with `learn=` | `sq.custom("({col} - {mean}) / {std}", learn=...)` |
| Cross-column formula | 2: `custom` with `mode="combine"` | `sq.custom("a / b AS ratio", mode="combine")` |
| Multi-row learning (categories) | 3: Subclass | Override `discover_sets()` |
| Window functions | 3: Subclass | Override `query()` |
| Complex schema changes | 3: Subclass | Override `output_schema()` |

Start at Level 1 or 2. You will know when you need Level 3 because the template
system will not be expressive enough for what you need.

!!! tip "All levels compose safely"
    Custom transformers at any level compose with built-in transformers in a pipeline.
    Expression composition, CTE promotion, and schema tracking all work the same way
    regardless of whether a transformer is built-in or custom.
