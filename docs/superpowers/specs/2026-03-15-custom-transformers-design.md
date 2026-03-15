# Custom Transformers Design — Issue #10

## Goal

Implement `sq.Expression` and `sq.custom()` — two convenience APIs for creating custom SQL transformers without subclassing `Transformer`. Both parse SQL through sqlglot at creation time for safety and multi-database support.

## Scope

One file: `src/sqlearn/custom.py`. Two public APIs:
- `sq.Expression(sql)` — static one-liner (Level 1)
- `sq.custom(sql, columns=, learn=, mode=)` — template-based factory (Level 2)

Both return `Transformer` instances that compose normally in pipelines.

## API

### `sq.Expression(sql: str) -> Transformer`

Static one-liner. The SQL string is parsed through sqlglot at creation time.

```python
sq.Expression("price * quantity AS revenue")
sq.Expression("CASE WHEN price > 100 THEN 'high' ELSE 'low' END AS tier")
sq.Expression("COALESCE(nickname, first_name) AS display_name")
```

**Rules:**
- Must contain `AS <name>` — the alias defines the new output column name
- Original columns pass through unchanged (new column is added alongside)
- Static only (no learning from data)
- Parsed through sqlglot at creation time — `ValueError` on invalid SQL
- The parsed expression is stored as a sqlglot AST node, not a raw string

**Implementation:** Returns an `_ExpressionTransformer` instance:
- `_classification = "static"`
- `_default_columns = None` (no column routing — expression references columns directly)
- `expressions()` returns `{alias: parsed_expr}` — adds the new column
- `output_schema()` adds the new column with type `"VARCHAR"` (conservative default)

### `sq.custom(sql, *, columns=, learn=, mode="per_column") -> Transformer`

Template-based factory. Returns a `Transformer` instance (static or dynamic).

```python
# Static per-column (default mode)
sq.custom("LN({col} + 1)", columns="numeric")
sq.custom("GREATEST(LEAST({col}, 100), 0)", columns=["price", "score"])

# Static per-column with new output column
sq.custom("LN({col} + 1) AS {col}_log", columns="numeric")

# Static combine mode (cross-column)
sq.custom("weight / (height * height) * 703 AS bmi", columns=["weight", "height"], mode="combine")

# Dynamic per-column (learns from data)
sq.custom(
    "({col} - {mean}) / NULLIF({std}, 0)",
    columns="numeric",
    learn={"mean": "AVG({col})", "std": "STDDEV_POP({col})"},
)

# Dynamic with new output column
sq.custom(
    "CASE WHEN ABS({col} - {mean}) / NULLIF({std}, 0) > 3 THEN 1 ELSE 0 END AS {col}_outlier",
    columns="numeric",
    learn={"mean": "AVG({col})", "std": "STDDEV_POP({col})"},
)
```

**Parameters:**

| Param | Type | Default | Description |
|---|---|---|---|
| `sql` | `str` | required | SQL template with `{col}` and `{param}` placeholders |
| `columns` | `str \| list[str] \| None` | `None` | Column selector (same as built-in transformers) |
| `learn` | `dict[str, str] \| None` | `None` | `{param: aggregate_sql}` — makes it dynamic |
| `mode` | `str` | `"per_column"` | `"per_column"` or `"combine"` |

**Template placeholders:**
- `{col}` — expands to the current column name (in `per_column` mode)
- `{param}` — expands to the learned value (from `learn=` dict)
- In `combine` mode: no `{col}` — reference columns by name directly

**Mode behavior:**
- `per_column` (default): template is applied once per target column, `{col}` iterates
- `combine`: template is applied once, target columns are referenced by name in the SQL

**Output column rules:**
- Template contains `AS <name>`: new column added, original preserved
- Template has no `AS`: column replaced in-place (only in `per_column` mode)
- `{col}` in `AS` clause expands: `AS {col}_log` → `price_log`, `income_log`

**Static vs dynamic:**
- `learn=None` → static (`_classification = "static"`)
- `learn={...}` → dynamic (`_classification = "dynamic"`)

**Validation at creation time:**
- SQL template parsed through sqlglot (with dummy column substitution)
- `learn=` values parsed through sqlglot
- `ValueError` on unparseable SQL
- `ValueError` if `mode="combine"` and template contains `{col}`
- `ValueError` if `mode="per_column"` and `columns` not specified and template uses `{col}`

**Implementation:** Returns a `_CustomTransformer` instance:
- `_classification` set based on `learn=`
- `_default_columns` set from `columns` parameter
- `discover()` expands `learn=` templates per column into sqlglot aggregates
- `expressions()` expands SQL template per column with learned param values substituted
- `output_schema()` adds new columns when `AS` clause detected in template

## Implementation Details

### Parsing strategy

Templates are validated at creation time by substituting `{col}` with `__col__` and `{param}` with `0` (a numeric literal), then parsing through `sqlglot.parse_one()`. This catches syntax errors early.

At `expressions()` time, the template is re-parsed with actual column names and learned values substituted. The resulting sqlglot AST nodes compose correctly with prior pipeline steps via `exprs[col]`.

**Critical:** In `per_column` mode, `expressions()` must use `exprs[col]` (the composed expression from prior steps), not bare `exp.Column(this=col)`. The template's `{col}` reference is replaced in the parsed AST by swapping `Column("__col__")` nodes with `exprs[col]`.

### AST substitution for expression composition

When the template is `"({col} - {mean}) / NULLIF({std}, 0)"` and this runs after an Imputer:
1. Parse template: `(__col__ - 0) / NULLIF(0, 0)` → AST
2. Find `Column("__col__")` nodes → replace with `exprs[col]` (which is `Coalesce(...)` from Imputer)
3. Replace `Literal(0)` placeholders for `{mean}`, `{std}` with actual learned values
4. Result: `(Coalesce(col, fill) - mean) / NULLIF(std, 0)` — correctly composed

### Learn template expansion

For `learn={"mean": "AVG({col})"}`:
1. At creation: parse `AVG(__col__)` through sqlglot to validate
2. At `discover()` time: for each target column, substitute `{col}` → actual column name
3. Return `{f"{col}__mean": exp.Avg(this=exp.Column(this=col))}` — standard discover() format

### Output schema inference

When the SQL template contains `AS <name>`:
- Extract the alias from the parsed AST
- If alias contains `{col}`, expand per target column
- Add new columns to output schema (type `"VARCHAR"` as conservative default)
- Do NOT drop original columns (new columns are added alongside)

When no `AS` clause:
- Column is replaced in-place
- No schema change

## Error Handling

| Error | When | Message |
|---|---|---|
| `ValueError` | Invalid SQL in template | `"Invalid SQL template: {parse_error}"` |
| `ValueError` | Invalid SQL in learn value | `"Invalid SQL in learn['{key}']: {parse_error}"` |
| `ValueError` | `mode="combine"` with `{col}` | `"mode='combine' cannot use {{col}} placeholder — reference columns by name"` |
| `ValueError` | `Expression` without `AS` | `"Expression must contain AS <name> to define output column"` |
| `ValueError` | Unknown mode | `"mode must be 'per_column' or 'combine', got '{mode}'"` |

## Exports

Add to `src/sqlearn/__init__.py` and `src/sqlearn/core/__init__.py`:
- `Expression` (the function, not the internal class)
- `custom` (the function)

## File Structure

```
src/sqlearn/
├── custom.py              # NEW — Expression() and custom() factories
├── __init__.py             # MODIFIED — add Expression, custom to exports
└── core/
    └── __init__.py         # MODIFIED — add Expression, custom to exports
```

## Testing

Tests in `tests/test_custom.py`:

**Expression tests:**
- Parse valid SQL with AS clause
- Reject SQL without AS clause
- Reject invalid SQL
- Use in pipeline — new column added, originals preserved
- Expression composes with prior steps (Imputer → Expression)
- `to_sql()` output contains the expression
- `get_feature_names_out()` includes new column
- Clone/pickle round-trip

**custom() static tests:**
- Per-column with `{col}` substitution
- Per-column with `AS {col}_suffix` (new column)
- Combine mode (cross-column expression)
- Reject `mode="combine"` with `{col}`
- Reject invalid SQL template
- Use in pipeline — composes correctly
- `to_sql()` output
- Column routing: `columns="numeric"`, `columns=["a", "b"]`

**custom() dynamic tests:**
- `learn=` with single param
- `learn=` with multiple params
- Learned values substituted correctly in expressions
- Dynamic custom composes with prior steps
- `to_sql()` output contains aggregates
- Classification: static when no learn, dynamic when learn present

**Edge cases:**
- Empty column list (no-op)
- Single column
- Template with no placeholders in combine mode
