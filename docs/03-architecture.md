> **sqlearn docs** — [Index](../CLAUDE.md) | Prev: [API Design](02-api-design.md) | Next: [Compiler](04-compiler.md)

## 4. Architecture: One Base Class, Not Eight

### 4.1 sklearn's Class Hierarchy (the mess)

To make a simple StandardScaler, sklearn needs:

```
object
  └── BaseEstimator                       # get_params, set_params
       └── TransformerMixin               # fit_transform, set_output
            └── OneToOneFeatureMixin      # get_feature_names_out
                 └── StandardScaler       # actual logic
```

Plus tags system (`__sklearn_tags__`), parameter validation (`_validate_params`),
fitted-check convention (trailing `_` attributes), MetaEstimatorMixin for Pipeline...

Total: 8+ base classes/mixins, naming conventions, private APIs.

**Why is it this way?** Historical accident. Each mixin was added to solve one problem
without rethinking the whole hierarchy. ClassifierMixin adds `score()`, RegressorMixin
adds a different `score()`, TransformerMixin adds `fit_transform()`,
OneToOneFeatureMixin adds `get_feature_names_out()` for same-shape transforms,
ClassNamePrefixFeaturesOutMixin adds it for different-shape transforms...

### 4.2 sqlearn's Approach: One Class

```python
class Transformer:
    """Base class for all sqlearn transformers."""

    _default_columns = "all"     # "all", "numeric", "categorical", "temporal", or None (explicit only)
    _classification = None       # "static", "dynamic", or None (must inspect at runtime)
                                 # Built-in transformers DECLARE this upfront.
                                 # Custom transformers leave as None → full inspection at fit().

    # ── User overrides these ─────────────────────────────

    def discover(self, columns, schema, y_column=None):
        """What SCALAR values to learn from data during fit.
        Returns {param_name: sqlglot_aggregate_expression}.
        If empty AND discover_sets() is empty, transformer is static.

        y_column: target column name (str or None). Only non-None
        when user passes y= to pipeline.fit(). Steps that need the
        target (e.g., TargetEncoder) use it; others ignore it.

        MUST return sqlglot AST nodes, not strings."""
        return {}

    def discover_sets(self, columns, schema, y_column=None):
        """What SET-VALUED things to learn from data during fit.
        Returns {param_name: sqlglot_query} where query returns multiple rows.

        Use this for transforms that learn SETS of values, not single aggregates:
        - OneHotEncoder: distinct categories per column
        - TargetEncoder: per-category target means
        - OrdinalEncoder: ordered distinct values per column

        Each entry becomes a separate query. Results stored in self.sets_
        as {param_name: list_of_dicts}.

        MUST return sqlglot AST nodes, not strings."""
        return {}

    def expressions(self, columns, exprs):
        """Transform column expressions.
        `columns` = the columns this transformer targets (resolved from _default_columns).
        `exprs` = full dict {col: sqlglot_expr} of ALL current columns.

        Return ONLY the columns you modify or add. Unmentioned columns
        pass through automatically (handled by the base class).
        This prevents the most common bug: forgetting to passthrough untouched columns.

        MUST return sqlglot AST nodes, not strings. This enables
        multi-database support via sqlglot dialect transpilation."""
        raise NotImplementedError

    def query(self, input_query):
        """Alternative to expressions() for transforms that need their own query level
        (window functions, aggregations, joins).
        Returns a new sqlglot SELECT wrapping the input.

        MUST return sqlglot AST node."""
        return None  # Default: use expressions() instead

    def output_schema(self, schema):
        """Declare output schema. Default: same as input.
        Override when adding/removing/renaming columns."""
        return schema

    # ── Provided by base class (user does NOT override) ──

    def fit(self, data, y=None, *, backend=None):
        """Learn parameters from data. Calls discover() and discover_sets() internally.

        y: target column name (str), list of column names, or None.
           Stored as self._y_column. Passed to discover()/discover_sets() for steps
           that need target info (e.g., TargetEncoder).
           Target column(s) auto-excluded from transform() output."""
        self._check_thread()
        ...
        return self

    def transform(self, data, *, out="numpy", backend=None, batch_size=None, dtype=None,
                  exclude_target=True):
        """Apply transformation. Calls expressions() or query() internally.
        Returns TransformResult (numpy-compatible with metadata).
        Target column(s) excluded from output by default (exclude_target=True).
        Set exclude_target=False to include target in output (useful for EDA,
        debugging, or chaining pipelines)."""
        self._check_thread()
        ...

    def fit_transform(self, data, y=None, **kwargs):
        """Convenience: fit then transform."""
        return self.fit(data, y=y, **kwargs).transform(data, **kwargs)

    def to_sql(self, *, dialect="duckdb", table="__input__"):
        """Compile to SQL string without executing."""
        ...

    def get_params(self, deep=True):
        """Return __init__ params as dict. Enables GridSearch + clone."""
        ...

    def set_params(self, **params):
        """Set params. Supports nested __ notation for pipelines."""
        ...

    def get_feature_names_out(self):
        """Output column names. sklearn compatible."""
        return list(self.output_schema_.columns.keys())

    def clone(self):
        """Create independent copy with same params + fitted state, new connection.
        Safe for use in another thread."""
        ...

    def freeze(self):
        """Return a FrozenPipeline: immutable, pre-compiled, deployment-ready.
        See Section 9 for full freeze semantics."""
        ...

    @property
    def is_fitted(self):
        """Clean boolean. No trailing-underscore scanning."""
        return self._fitted

    # ── Internal: expression composition (user does NOT call) ──

    def _apply_expressions(self, exprs):
        """Base class wrapper around expressions(). Handles passthrough automatically.

        1. Calls self.expressions(self.columns_, exprs) — user returns only modified/new cols
        2. Merges result with untouched columns from exprs
        3. Detects undeclared new columns (warn) and name collisions (error)
        4. Removes columns that output_schema() says are dropped

        User never calls this. The compiler calls it during composition."""
        modified = self.expressions(self.columns_, exprs)
        result = dict(exprs)       # start with all input columns (passthrough)

        # Detect column name collisions: new columns that clash with existing
        new_cols = set(modified.keys()) - set(exprs.keys())
        collisions = new_cols & set(exprs.keys())
        if collisions:
            raise SchemaError(
                f"Column name collision: {collisions} already exist in input. "
                f"Use sq.Rename() to resolve, or choose different output names."
            )

        result.update(modified)    # overlay modifications and additions

        # Remove columns that output_schema drops
        output_cols = set(self.output_schema(self.input_schema_).columns.keys())

        # Warn if expressions() created columns that output_schema() doesn't declare
        undeclared = new_cols - output_cols
        if undeclared:
            import warnings
            warnings.warn(
                f"{self.__class__.__name__}.expressions() created columns "
                f"{undeclared} but output_schema() doesn't declare them. "
                "Override output_schema() to include new columns. "
                "These columns will be silently dropped.",
                UserWarning,
            )

        return {k: v for k, v in result.items() if k in output_cols}

    # ── Thread safety ────────────────────────────────────

    def _check_thread(self):
        """Guard: detect cross-thread AND cross-process access. Pipelines are NOT
        thread-safe and DuckDB connections cannot be shared across processes.
        Use pipeline.clone() to create a thread-safe copy."""
        import threading, os
        current_thread = threading.current_thread().ident
        current_pid = os.getpid()

        if self._owner_pid is None:
            self._owner_pid = current_pid
            self._owner_thread = current_thread
        elif self._owner_pid != current_pid:
            raise SQLearnError(
                f"{self.__class__.__name__} accessed from a different process "
                f"(original pid={self._owner_pid}, current pid={current_pid}). "
                "DuckDB connections cannot be shared across processes. "
                "Create a new pipeline in each process."
            )
        elif self._owner_thread != current_thread:
            raise SQLearnError(
                f"{self.__class__.__name__} accessed from a different thread. "
                "Pipelines are not thread-safe. Use .clone() to create "
                "a thread-safe copy with the same fitted parameters."
            )

    # ── Operators ────────────────────────────────────────

    def __add__(self, other):
        """Sequential composition: a + b → Pipeline([a, b]), flattened."""
        from .pipeline import Pipeline
        left = list(self.steps) if isinstance(self, Pipeline) else [self]
        right = list(other.steps) if isinstance(other, Pipeline) else [other]
        return Pipeline([*left, *right])

    def __iadd__(self, other):
        """Incremental append: pipe += step → NEW Pipeline (non-mutating).
        Safe for notebooks — rerunning cells won't corrupt earlier references."""
        return self.__add__(other)
```

**What sklearn's base classes give you vs sqlearn:**

| sklearn needs | sqlearn equivalent | Difference |
|---|---|---|
| `BaseEstimator` for `get_params`/`set_params` | Built into `Transformer` | Same mechanism (inspect `__init__`), one class |
| `TransformerMixin` for `fit_transform` | Built into `Transformer` | One class |
| `OneToOneFeatureMixin` for `get_feature_names_out` | `output_schema()` returns schema | Handles both same-shape and different-shape |
| `check_is_fitted` convention | `self._fitted` boolean | No attribute scanning |
| `_validate_params` | Validation in `fit()` | Simpler, explicit |
| `__sklearn_tags__` | Not needed | We only have transformers, not classifiers/regressors |
| `MetaEstimatorMixin` | Pipeline just IS a Transformer | No marker needed |
| `clone()` | `Transformer.clone()` method | Same mechanism, cleaner API |
| N/A | `discover_sets()` for multi-row discovery | New — enables OneHotEncoder/TargetEncoder cleanly |
| N/A | `_apply_expressions()` auto-passthrough | New — user returns only modified cols, base handles passthrough |
| N/A | `freeze()` for deployment | New — immutable pre-compiled pipeline |

**Result:** Users implement 1-5 methods (`discover`, `discover_sets`, `expressions`,
`query`, `output_schema`). Simple transforms need 1 (`expressions` only). Most need 2-3
(e.g., `discover` + `expressions`, or `discover` + `expressions` + `output_schema` for
schema-changing transforms). The full 5-method surface is only needed for complex
transforms like TargetEncoder. Everything else is inherited.

**Three levels of custom transforms:**

| Level | API | When to use |
|---|---|---|
| 1 | `sq.Expression("SQL")` | Static one-liner, single new column |
| 2 | `sq.custom(template, columns=, learn=)` | Per-column transforms, optionally learn stats |
| 3 | `class MyTransformer(sq.Transformer)` | Full control: sets, CTEs, joins, windows |

See Section 3.10 in `02-api-design.md` for the full guide with examples.

**Shortcut for the trivial case — decorator API:**

```python
@sq.static_transform(columns="numeric")
def log_plus_one(col_expr):
    """One function. Takes sqlglot expr, returns sqlglot expr."""
    return exp.Ln(exp.Add(col_expr, exp.Literal.number(1)))

# Use in pipeline — no class needed:
pipe = sq.Pipeline([sq.Imputer(), log_plus_one, sq.StandardScaler()])
```

**`sq.custom()` — template-based, covers 90% of custom needs:**

```python
# Static: per-column SQL template
log = sq.custom("LN({col} + 1)", columns="numeric")

# Dynamic: learn stats, then use in template
center = sq.custom(
    "{col} - {mean}",
    columns="numeric",
    learn={"mean": "AVG({col})"},
)
```

Under the hood, `sq.custom()` returns a `_CustomTransformer` instance (subclass of
`Transformer`). Templates are parsed through sqlglot at creation time. The `{col}`
placeholder expands per target column. The `{param}` placeholders expand to learned
values from `self.params_[f"{col}__{param}"]`. All validation rules from Section 3.10b
apply.

**`_CustomTransformer` internals:**

```python
class _CustomTransformer(Transformer):
    """Returned by sq.custom(). Template-based transformer with full validation."""

    def __init__(self, template, columns, learn, mode, name):
        self._template = template          # SQL template string
        self._learn = learn                # {param: aggregate_template} or None
        self._mode = mode                  # "per_column" or "combine"
        self._parsed = _parse_template(template)  # sqlglot AST of template
        self._output_col = _extract_alias(self._parsed)  # from AS clause, or None

        # Auto-classify
        self._classification = "dynamic" if learn else "static"

        # Set column defaults
        if isinstance(columns, str) and columns in ("numeric", "categorical", "temporal", "all"):
            self._default_columns = columns
        else:
            self._default_columns = None
            self._explicit_columns = columns

    def discover(self, columns, schema, y_column=None):
        if not self._learn:
            return {}
        result = {}
        for col in columns:
            for param, agg_template in self._learn.items():
                sql = agg_template.replace("{col}", col)
                result[f"{col}__{param}"] = sqlglot.parse_one(sql)
        return result

    def expressions(self, columns, exprs):
        result = {}
        for col in columns:
            sql = self._template.replace("{col}", col)
            # Replace learned params
            if self._learn:
                for param in self._learn:
                    value = self.params_[f"{col}__{param}"]
                    sql = sql.replace(f"{{{param}}}", str(value))
            parsed = sqlglot.parse_one(sql)
            out_name = self._output_col.replace("{col}", col) if self._output_col else col
            result[out_name] = parsed
        return result

    def output_schema(self, schema):
        if self._output_col:
            new_cols = {
                self._output_col.replace("{col}", col): _infer_type(self._parsed)
                for col in self.columns_
            }
            return schema.add(new_cols)
        return schema
```

**Validation at every level — the `_validate_custom()` method:**

The base `Transformer` class runs `_validate_custom()` on the first `fit()` for all
non-built-in transformers. This catches bugs before they corrupt data:

```python
def _validate_custom(self, columns, schema):
    """Run once on first fit. Catches custom transformer bugs early."""

    # 1. discover() must return sqlglot ASTs, not Python values
    disc = self.discover(columns, schema)
    for k, v in disc.items():
        if not isinstance(v, exp.Expression):
            raise TypeError(
                f"discover() must return sqlglot expressions, got {type(v).__name__} "
                f"for key '{k}'.\n\n"
                f"  BAD:  return {{'{k}': 42.0}}\n"
                f"  GOOD: return {{'{k}': exp.Avg(this=exp.Column(this='{columns[0]}'))}}\n"
            )

    # 2. expressions() must return sqlglot ASTs, not strings
    test_exprs = {col: exp.Column(this=col) for col in columns}
    try:
        result = self.expressions(columns, test_exprs)
    except Exception as e:
        raise CompilationError(
            f"expressions() raised {type(e).__name__}: {e}\n"
            f"Make sure expressions() handles these columns: {columns}"
        ) from e

    if result is not None:
        for k, v in result.items():
            if isinstance(v, str):
                raise TypeError(
                    f"expressions() must return sqlglot expressions, got str for key '{k}'.\n\n"
                    f"  BAD:  return {{'{k}': '{v}'}}\n"
                    f"  GOOD: return {{'{k}': sqlglot.parse_one('{v}')}}\n"
                )
            if not isinstance(v, exp.Expression):
                raise TypeError(
                    f"expressions() must return sqlglot expressions, got "
                    f"{type(v).__name__} for key '{k}'."
                )

        # 3. New columns must have output_schema()
        new_cols = set(result.keys()) - set(columns) - set(test_exprs.keys())
        if new_cols:
            declared = self.output_schema(schema)
            if declared == schema:
                raise SchemaError(
                    f"expressions() adds new columns {new_cols} but output_schema() "
                    f"returns unchanged schema.\n\n"
                    f"Override output_schema() to declare new columns:\n"
                    f"  def output_schema(self, schema):\n"
                    f"      return schema.add({{{', '.join(repr(c) + ': \"DOUBLE\"' for c in new_cols)}}})"
                )

    # 4. Classification consistency
    if self._classification == "static" and disc:
        raise ClassificationError(
            f"{self.__class__.__name__} declares _classification='static' but "
            f"discover() returned params {set(disc.keys())}.\n\n"
            f"Either:\n"
            f"  1. Change to _classification='dynamic' (it learns from data)\n"
            f"  2. Remove the discover() override (make it truly static)"
        )
```

This validation runs once (first fit), results are cached. Built-in transformers
(Tier 1) skip validation entirely — they're tested in CI.

### 4.3 Static vs Dynamic — Definitive Classification Specification

This section is the **single source of truth** for how the pipeline classifies a step
as static or dynamic. Every implementation decision traces back to these rules.
If a rule isn't listed here, it doesn't exist.

#### 4.3.1 Definitions

**Static transformer:** A step whose `expressions()` or `query()` output is fully
determined by constructor arguments alone. It does NOT need to see any data to know
what SQL to generate. Examples: `Log()` always emits `LN(col + 1)`. `Clip(lower=0, upper=100)`
always emits `GREATEST(LEAST(col, 100), 0)`. The SQL is known at definition time.

**Dynamic transformer:** A step whose `expressions()` or `query()` output depends on
values learned from the data. It MUST run aggregate queries during `fit()` to learn
parameters before it can generate SQL. Examples: `StandardScaler` must learn `mean` and
`std` from data. `OneHotEncoder` must learn distinct category values from data.

**Schema-changing transformer:** A step whose output columns differ from its input
columns (added, removed, renamed, or retyped). Orthogonal to static/dynamic — a step
can be static AND schema-changing (e.g., `DateParts` adds columns without learning
from data) or dynamic AND schema-preserving (e.g., `StandardScaler` replaces values
but keeps same columns).

#### 4.3.2 Three-Tier Classification Model

Not all transformers need runtime inspection. We wrote the built-in ones — we *know*
what they are. Inspecting `Log()` every `fit()` to confirm it's static is wasted work.
Only unknown code (user-created transformers) needs the full inspection.

**The three tiers:**

| Tier | Who | `_classification` value | What happens at `fit()` | Cost |
|---|---|---|---|---|
| **Tier 1: Built-in, declared** | sqlearn's own transformers | `"static"` or `"dynamic"` (set on class or in `__init__`) | **Trusted. Zero inspection.** CI tests guarantee the declaration is correct. | Zero |
| **Tier 2: Custom, declared** | User set `_classification` explicitly | `"static"` or `"dynamic"` | **Verify once at first `fit()`**, cache result. Warn/error on mismatch. | One `discover()` call, first fit only |
| **Tier 3: Custom, undeclared** | User left `_classification = None` (default) | `None` | **Full inspection every `fit()`** using the conservative checklist (Section 4.3.3). | One `discover()` call per fit |

**How we detect which tier:**

```python
def _get_tier(step):
    """Determine classification tier for a pipeline step."""
    is_builtin = type(step).__module__.startswith("sqlearn.")

    if is_builtin and step._classification is not None:
        return 1   # Built-in, declared → trust it
    elif not is_builtin and step._classification is not None:
        return 2   # Custom, declared → verify once
    else:
        return 3   # No declaration → full inspection
```

**Tier 1 — Built-in, declared (zero cost):**

We wrote these. We test them. The declaration is gospel.

```python
class Log(Transformer):
    _classification = "static"          # DECLARED — never inspected at runtime
    _default_columns = None

class StandardScaler(Transformer):
    _classification = "dynamic"         # DECLARED — never inspected at runtime
    _default_columns = "numeric"
```

At `fit()` time, the pipeline reads `step._classification` and moves on. No `discover()`
call needed for classification. (`discover()` is still called for dynamic steps to get
their aggregation expressions — but the *decision* to call it is instant.)

**Tier 1 safety guarantee:** CI tests validate every built-in declaration:

```python
# tests/test_classification_declarations.py — runs in CI for every built-in transformer
@pytest.mark.parametrize("cls,args", ALL_BUILTIN_TRANSFORMERS)
def test_declaration_matches_reality(cls, args):
    """Built-in _classification MUST match what discover() actually returns."""
    step = cls(**args)
    columns = ["a", "b", "c"]
    schema = Schema({"a": "DOUBLE", "b": "VARCHAR", "c": "TIMESTAMP"})

    result = step.discover(columns, schema, y_column=None)

    if step._classification == "static":
        assert result == {}, (
            f"{cls.__name__} declares _classification='static' "
            f"but discover() returned {result!r}. Fix the declaration or discover()."
        )
    elif step._classification == "dynamic":
        assert isinstance(result, dict) and len(result) > 0, (
            f"{cls.__name__} declares _classification='dynamic' "
            f"but discover() returned empty dict. Fix the declaration or discover()."
        )
```

If a built-in declaration is wrong, **CI fails**. This is caught before release, never at
user runtime. The runtime can trust built-in declarations unconditionally because CI
has already proven them correct.

**Conditionally dynamic — set `_classification` in `__init__`:**

Some built-in transformers are static with one set of args and dynamic with another.
They set `_classification` in `__init__()` based on their constructor arguments:

```python
class StringSplit(Transformer):
    _default_columns = None

    def __init__(self, columns=None, by=",", max_parts=3, keep_original=True):
        self.by = by
        self.max_parts = max_parts
        self.keep_original = keep_original

        # DECLARE classification based on constructor args
        if isinstance(max_parts, int):
            self._classification = "static"     # max_parts known → no data needed
        else:
            self._classification = "dynamic"    # max_parts="auto" → must inspect data

class Clip(Transformer):
    def __init__(self, lower=None, upper=None):
        self.lower = lower
        self.upper = upper

        # Explicit bounds → static. Percentile bounds → dynamic.
        if isinstance(lower, (int, float, type(None))) and isinstance(upper, (int, float, type(None))):
            self._classification = "static"
        else:
            self._classification = "dynamic"    # lower="p01" → needs PERCENTILE_CONT

class AutoDatetime(Transformer):
    def __init__(self, columns=None, granularity="auto"):
        self.granularity = granularity

        # Explicit granularity → static. Auto → dynamic (inspects data range).
        if granularity != "auto":
            self._classification = "static"
        else:
            self._classification = "dynamic"
```

CI tests cover all argument combinations:

```python
def test_stringsplit_static():
    s = StringSplit(max_parts=3)
    assert s._classification == "static"
    assert s.discover(["tags"], schema, None) == {}

def test_stringsplit_dynamic():
    s = StringSplit(max_parts="auto")
    assert s._classification == "dynamic"
    assert len(s.discover(["tags"], schema, None)) > 0
```

**Tier 2 — Custom, declared (verify once):**

A user writes a custom transformer and declares `_classification = "static"`. We
respect their declaration but **verify it once** at the first `fit()`:

```python
class MyTransform(sq.Transformer):
    _classification = "static"    # user declares

    def discover(self, columns, schema, y_column=None):
        return {}                  # consistent with declaration ✓

    def expressions(self, exprs):
        ...
```

At first `fit()`, the pipeline calls `discover()` once to verify. If `discover()` returns
`{}` and the declaration says `"static"` → confirmed, cached, never checked again. If
there's a mismatch:

```python
# User declared "static" but discover() returned params → ERROR
if declared == "static" and len(discover_result) > 0:
    raise ClassificationError(
        f"{step.__class__.__name__} declares _classification='static' "
        f"but discover() returned {len(discover_result)} aggregation(s). "
        f"Either fix discover() to return {{}} or change _classification to 'dynamic'."
    )

# User declared "dynamic" but discover() returned {} → WARNING (not error)
if declared == "dynamic" and len(discover_result) == 0:
    warnings.warn(
        f"{step.__class__.__name__} declares _classification='dynamic' "
        f"but discover() returned {{}}. This wastes one fit query per call. "
        f"Consider changing _classification to 'static' for better performance.",
        UserWarning,
    )
    # Still honor the declaration — dynamic is always safe.
```

After verification, the result is cached: `step._classification_verified = True`.
Subsequent `fit()` calls skip verification.

**Tier 3 — Custom, undeclared (full inspection every time):**

A user writes a custom transformer and doesn't set `_classification`:

```python
class MyTransform(sq.Transformer):
    # _classification not set → defaults to None

    def discover(self, columns, schema, y_column=None):
        ...   # we don't know what this returns until we call it
```

This is the **only tier** that runs the full conservative inspection checklist
(Section 4.3.3 below). Every `fit()` call inspects the step from scratch.

**Why not cache Tier 3 results?** Because `discover()`'s return value could change
between calls. The same transformer instance might return `{}` for one schema and
`{"a": expr}` for a different schema (unlikely but possible with weird custom code).
Caching would be an assumption about user code we can't make. The inspection is cheap
(one `discover()` call), so running it every time is fine.

#### 4.3.3 The Hard Safety Rule

> **If the pipeline cannot prove with 100% certainty that a step is static,
> it MUST fall back to dynamic.** This rule applies to **Tier 2 and Tier 3 only**.
> Tier 1 (built-in) is proven correct by CI tests, not runtime inspection.

Static is an optimization: skip the aggregation query for that step during `fit()`.
Getting it wrong means `expressions()` runs without learned parameters →
**silently wrong output**. Dynamic is always safe — at worst one unnecessary query.

**Consequence:** False-positive "static" = data corruption. False-positive "dynamic" =
one extra cheap query. The cost is asymmetric. Always err toward dynamic.

#### 4.3.4 What We Inspect — Tier 3 Checklist (Custom Transformers Only)

This checklist runs during `pipeline.fit()` for Tier 3 steps only. Tier 1 steps skip
it entirely. Tier 2 steps use a lighter verification (Section 4.3.2 above).

Each check runs in sequence. **First failure stops and returns dynamic.**

```python
def _classify_step(step, columns, schema, y_column):
    """
    Master classification dispatcher. Routes to the correct tier.

    Returns: StepClassification with:
        .kind:     "static" | "dynamic"
        .tier:     1, 2, or 3
        .reason:   human-readable explanation (for pipe.describe() audit trail)
        .warnings: list of UserWarning messages (for ambiguous cases)
    """
    is_builtin = type(step).__module__.startswith("sqlearn.")
    declared = step._classification   # "static", "dynamic", or None

    # ── TIER 1: Built-in, declared → trust unconditionally ────────
    if is_builtin and declared is not None:
        return StepClassification(
            kind=declared,
            tier=1,
            reason=f"Built-in {step.__class__.__name__} declares "
                   f"_classification='{declared}'. Trusted (validated by CI).",
        )

    # ── TIER 2: Custom, declared → verify once, then trust ────────
    if not is_builtin and declared is not None:
        if getattr(step, "_classification_verified", False):
            # Already verified on a previous fit() → trust cached result
            return StepClassification(
                kind=declared,
                tier=2,
                reason=f"Custom {step.__class__.__name__} declares "
                       f"_classification='{declared}'. Verified on first fit.",
            )
        # First fit() → verify the declaration
        return _verify_custom_declaration(step, declared, columns, schema, y_column)

    # ── TIER 3: No declaration → full conservative inspection ─────
    return _inspect_unknown_step(step, columns, schema, y_column)


def _verify_custom_declaration(step, declared, columns, schema, y_column):
    """Tier 2: User declared a classification. Verify it matches discover()."""
    name = step.__class__.__name__
    warnings = []

    try:
        result = step.discover(columns, schema, y_column)
    except Exception as e:
        # discover() failed → can't verify. Fall back to dynamic regardless of declaration.
        return StepClassification(
            kind="dynamic", tier=2,
            reason=f"{name}.discover() raised {type(e).__name__}. "
                   "Cannot verify declaration. Treating as dynamic.",
        )

    if not isinstance(result, dict):
        return StepClassification(
            kind="dynamic", tier=2,
            reason=f"{name}.discover() returned {type(result).__name__}, not dict. "
                   "Treating as dynamic regardless of declaration.",
        )

    # Also check discover_sets()
    try:
        sets_result = step.discover_sets(columns, schema, y_column)
    except Exception:
        sets_result = {}

    has_sets = isinstance(sets_result, dict) and len(sets_result) > 0

    if declared == "static" and has_sets:
        raise ClassificationError(
            f"{name} declares _classification='static' but discover_sets() returned "
            f"{len(sets_result)} set query(ies): {list(sets_result.keys())}. "
            f"Either fix discover_sets() to return {{}} or change _classification to 'dynamic'."
        )

    # Check for mismatch
    if declared == "static" and len(result) > 0:
        raise ClassificationError(
            f"{name} declares _classification='static' but discover() returned "
            f"{len(result)} aggregation(s): {list(result.keys())}. "
            f"Either fix discover() to return {{}} or change _classification to 'dynamic'."
        )

    if declared == "dynamic" and len(result) == 0:
        warnings.append(
            f"{name} declares _classification='dynamic' but discover() returned {{}}. "
            "This wastes one fit query per call. Consider _classification='static'."
        )
        # Honor the declaration — dynamic is always safe.

    # Verification passed → cache so we don't re-verify
    step._classification_verified = True

    return StepClassification(
        kind=declared, tier=2,
        reason=f"Custom {name} declares _classification='{declared}'. "
               "Verified: discover() result is consistent.",
        warnings=warnings,
    )


def _inspect_unknown_step(step, columns, schema, y_column):
    """
    Tier 3: Full conservative inspection for custom transformers
    that did not declare _classification.

    THE RULE: static ONLY if ALL checks pass. ANY failure → dynamic.
    """
    name = step.__class__.__name__
    warnings = []

    # ──────────────────────────────────────────────────────────────
    # CHECK 1: Can we call discover() at all?
    # ──────────────────────────────────────────────────────────────
    try:
        discover_result = step.discover(columns, schema, y_column)
    except NotImplementedError:
        return StepClassification(
            kind="dynamic", tier=3,
            reason=f"{name}.discover() raised NotImplementedError. "
                   "Assuming step needs data (dynamic).",
        )
    except Exception as e:
        return StepClassification(
            kind="dynamic", tier=3,
            reason=f"{name}.discover() raised {type(e).__name__}: {e}. "
                   "Cannot determine data requirements. Assuming dynamic.",
        )

    # ──────────────────────────────────────────────────────────────
    # CHECK 2: Is the return value exactly a dict?
    # ──────────────────────────────────────────────────────────────
    if discover_result is None:
        warnings.append(
            f"{name}.discover() returned None instead of {{}}. "
            "Return an empty dict {{}} explicitly to mark as static. "
            "Treating as dynamic (safe fallback)."
        )
        return StepClassification(kind="dynamic", tier=3,
            reason="discover() returned None", warnings=warnings)

    if not isinstance(discover_result, dict):
        warnings.append(
            f"{name}.discover() returned {type(discover_result).__name__} "
            f"instead of dict. Return {{}} for static or {{name: expr}} for dynamic. "
            "Treating as dynamic (safe fallback)."
        )
        return StepClassification(kind="dynamic", tier=3,
            reason=f"discover() returned {type(discover_result).__name__}",
            warnings=warnings)

    # ──────────────────────────────────────────────────────────────
    # CHECK 3: Is the dict non-empty?
    # ──────────────────────────────────────────────────────────────
    if len(discover_result) > 0:
        for key, value in discover_result.items():
            if not isinstance(key, str):
                warnings.append(
                    f"{name}.discover() key {key!r} is not a string. "
                    "Keys must be param name strings."
                )
            if not isinstance(value, exp.Expression):
                warnings.append(
                    f"{name}.discover() value for '{key}' is "
                    f"{type(value).__name__}, not a sqlglot Expression."
                )
        return StepClassification(
            kind="dynamic", tier=3,
            reason=f"discover() returned {len(discover_result)} aggregation(s): "
                   f"{list(discover_result.keys())}",
            warnings=warnings,
        )

    # ──────────────────────────────────────────────────────────────
    # CHECK 3b: Does discover_sets() return anything?
    # ──────────────────────────────────────────────────────────────
    try:
        sets_result = step.discover_sets(columns, schema, y_column)
    except (NotImplementedError, Exception):
        sets_result = {}

    if isinstance(sets_result, dict) and len(sets_result) > 0:
        return StepClassification(
            kind="dynamic", tier=3,
            reason=f"discover_sets() returned {len(sets_result)} set query(ies): "
                   f"{list(sets_result.keys())}",
            warnings=warnings,
        )

    # ──────────────────────────────────────────────────────────────
    # CHECK 4: Dict is empty — additional safety checks
    # ──────────────────────────────────────────────────────────────

    # CHECK 4a: _default_columns resolution is a schema operation,
    # not a data operation. Static steps CAN have _default_columns.
    # (Not a disqualifier.)

    # CHECK 4b: We CANNOT detect if expressions() reads self.params_
    # via introspection. Instead, the _FrozenEmptyParams runtime guard
    # catches this at transform time (see Section 4.3.8).

    # CHECK 4c: Does the step override fit() directly?
    if _overrides_method(step, "fit"):
        return StepClassification(
            kind="dynamic", tier=3,
            reason=f"{name} overrides fit() directly. "
                   "Cannot guarantee no data-dependent side effects.",
            warnings=[
                f"{name} overrides fit() instead of discover(). "
                "sqlearn transformers should use discover() to declare data needs."
            ],
        )

    # CHECK 4d: query() interface is fine with static — CTE is for SQL
    # structure (window functions), not data learning. Static + query()
    # = CTE without aggregation.

    # ──────────────────────────────────────────────────────────────
    # ALL CHECKS PASSED → Static
    # ──────────────────────────────────────────────────────────────
    return StepClassification(
        kind="static", tier=3,
        reason="discover() returned {} and all safety checks passed. "
               "Tip: add _classification='static' to skip inspection next time.",
    )


def _overrides_method(instance, method_name):
    """Check if instance's class overrides a method from Transformer base class."""
    base_method = getattr(Transformer, method_name, None)
    instance_method = getattr(type(instance), method_name, None)
    return instance_method is not None and instance_method is not base_method
```

#### 4.3.5 Schema Change Detection (Same Conservative Approach)

```python
def _detect_schema_change(step, input_schema):
    """
    Determine if a step changes the output schema.
    Conservative: if we can't tell, assume YES (schema changes).

    Schema change matters for LAYER BOUNDARIES during fit (Section 5.3).
    If we wrongly say "no change" when there IS a change, downstream steps
    in the same layer will see the wrong columns → wrong aggregation queries.
    """

    # CHECK S1: Can we call output_schema()?
    try:
        output = step.output_schema(input_schema)
    except Exception as e:
        return SchemaChangeResult(
            changes=True,
            reason=f"output_schema() raised {type(e).__name__}: {e}. Assuming schema changes.",
        )

    # CHECK S2: Did it return a Schema object?
    if output is None:
        return SchemaChangeResult(
            changes=True,
            reason="output_schema() returned None. Assuming schema changes.",
        )
    if not isinstance(output, Schema):
        return SchemaChangeResult(
            changes=True,
            reason=f"output_schema() returned {type(output).__name__}, not Schema. "
                   "Assuming schema changes.",
        )

    # CHECK S3: Compare input and output schemas structurally.
    # Must compare BOTH column names AND column types.
    # Extra columns, missing columns, renamed columns, retyped columns
    # all count as schema changes.
    if output.columns != input_schema.columns:
        # Determine what changed for the audit trail
        added = set(output.columns) - set(input_schema.columns)
        removed = set(input_schema.columns) - set(output.columns)
        retyped = {
            col for col in set(output.columns) & set(input_schema.columns)
            if output.columns[col] != input_schema.columns[col]
        }
        return SchemaChangeResult(
            changes=True,
            reason=f"Schema differs. Added: {added or 'none'}, "
                   f"Removed: {removed or 'none'}, Retyped: {retyped or 'none'}",
        )

    # CHECK S4: Schemas are identical.
    return SchemaChangeResult(changes=False, reason="output_schema matches input_schema.")
```

#### 4.3.6 What Exactly Makes a Step Static — The Exhaustive Criteria

A step is **proven static** when ALL of the following are true:

| # | Criterion | What we check | Failure → |
|---|---|---|---|
| S1 | `discover()` is callable | Call it, catch exceptions | Dynamic |
| S2 | `discover()` returns a `dict` | `isinstance(result, dict)` | Dynamic |
| S3 | `discover()` returns an **empty** dict | `len(result) == 0` | Dynamic |
| S3b | `discover_sets()` returns an **empty** dict | `len(result) == 0` | Dynamic |
| S4 | The class does NOT override `fit()` | Compare method identity to `Transformer.fit` | Dynamic |
| S5 | No exception during any check | All checks complete without error | Dynamic |

A step is **proven dynamic** when ANY of the following are true:

| # | Criterion | What we check |
|---|---|---|
| D1 | `discover()` returns a non-empty dict | `len(result) > 0` — has aggregation expressions |
| D1b | `discover_sets()` returns a non-empty dict | `len(result) > 0` — has set queries |
| D2 | `discover()` returns `None` | Ambiguous — not the empty-dict contract |
| D3 | `discover()` returns a non-dict type | `list`, `tuple`, `int`, `str`, `bool`, `set` — contract violation |
| D4 | `discover()` raises any exception | `NotImplementedError`, `TypeError`, `ValueError`, etc. |
| D5 | The class overrides `fit()` directly | Can't guarantee no data-dependent side effects |

**There is no third state.** Every step is either static or dynamic. The classification
is binary, deterministic, and logged.

#### 4.3.7 What Exactly Makes a Step Schema-Changing

A step is **proven schema-preserving** when ALL of the following are true:

| # | Criterion | What we check | Failure → |
|---|---|---|---|
| SC1 | `output_schema()` is callable | Call it, catch exceptions | Schema-changing |
| SC2 | `output_schema()` returns a `Schema` | `isinstance(result, Schema)` | Schema-changing |
| SC3 | Output columns == input columns (names) | `set(output.columns.keys()) == set(input.columns.keys())` | Schema-changing |
| SC4 | Output types == input types | `output.columns[col] == input.columns[col]` for all `col` | Schema-changing |

Any single failure → treated as schema-changing → forces a layer boundary during fit.

#### 4.3.8 Validation of `discover()` Return Values

When `discover()` returns a non-empty dict (dynamic step), we validate the contents
before using them in the fit query. This catches bugs early instead of at SQL execution.

```python
def _validate_discover_result(step, result):
    """
    Validate the dict returned by discover(). Called only for dynamic steps.
    Raises CompilationError on invalid content. Warns on suspicious content.
    """
    import sqlglot.expressions as exp

    for key, value in result.items():
        # KEY CHECKS:
        # K1: Key must be a string (param name)
        if not isinstance(key, str):
            raise CompilationError(
                f"{step.__class__.__name__}.discover() returned non-string key {key!r}. "
                "Keys must be param name strings like 'price__mean'."
            )

        # K2: Key must not be empty
        if len(key.strip()) == 0:
            raise CompilationError(
                f"{step.__class__.__name__}.discover() returned empty string key. "
                "Keys must be descriptive param names like 'price__mean'."
            )

        # K3: Key must not collide with reserved prefixes
        if key.startswith("__sq_"):
            raise CompilationError(
                f"{step.__class__.__name__}.discover() key '{key}' uses reserved "
                "prefix '__sq_'. Choose a different param name."
            )

        # K4: Key should follow convention: colname__statname
        if "__" not in key:
            warnings.warn(
                f"{step.__class__.__name__}.discover() key '{key}' doesn't follow "
                "the 'column__stat' naming convention. This works but makes "
                "debugging harder.",
                UserWarning,
            )

        # VALUE CHECKS:
        # V1: Value must be a sqlglot Expression (aggregate)
        if not isinstance(value, exp.Expression):
            raise CompilationError(
                f"{step.__class__.__name__}.discover() value for '{key}' is "
                f"{type(value).__name__}, expected sqlglot Expression. "
                "discover() must return sqlglot AST nodes, not raw strings or numbers. "
                f"Got: {value!r}"
            )

        # V2: Value should be an aggregate expression (AVG, SUM, COUNT, etc.)
        # We can't enforce this perfectly (custom functions exist), but we
        # can warn if the expression doesn't contain any known aggregate.
        known_aggregates = (
            exp.Avg, exp.Sum, exp.Count, exp.Min, exp.Max,
            exp.StddevPop, exp.StddevSamp, exp.Variance, exp.VariancePop,
            exp.ApproxQuantile, exp.PercentileCont, exp.PercentileDisc,
            exp.ArrayAgg, exp.GroupConcat,
        )
        has_aggregate = any(
            isinstance(node, known_aggregates)
            for node in value.walk()
        )
        if not has_aggregate:
            warnings.warn(
                f"{step.__class__.__name__}.discover() value for '{key}' does not "
                "contain a recognized aggregate function (AVG, SUM, COUNT, etc.). "
                "discover() expressions are executed as SELECT aggregates — "
                "non-aggregate expressions may produce unexpected results. "
                f"Expression: {value.sql()}",
                UserWarning,
            )
```

#### 4.3.9 Runtime Guard — Catching Static Steps That Lie

Even after classification, a static step might have a bug: `discover()` returns `{}`
but `expressions()` secretly reads `self.params_`. We catch this at runtime:

```python
class _FrozenEmptyParams(dict):
    """
    Assigned to self.params_ for steps classified as static.
    Looks like an empty dict, but raises on any key access
    with a diagnostic message.
    """
    def __init__(self, step_name):
        super().__init__()
        self._step_name = step_name

    def __getitem__(self, key):
        raise StaticViolationError(
            f"{self._step_name}.expressions() tried to read self.params_['{key}'], "
            f"but this step was classified as STATIC (discover() returned {{}}). "
            f"This means discover() promised it doesn't need data, but expressions() "
            f"disagrees. Fix one of:\n"
            f"  1. discover() should return {{'{key}': <sqlglot aggregate>}} to learn this param\n"
            f"  2. expressions() should not read self.params_ (use constructor args instead)"
        )

    def __contains__(self, key):
        return False  # no key exists

    def get(self, key, default=None):
        # get() ALSO raises — a static step has no business reading params_ at all.
        # If discover() returned {}, then expressions() should use constructor args,
        # not params_. Even .get() with a default would silently produce wrong results
        # when the classification is incorrect (e.g., .get("mean", 0) returns 0
        # instead of the actual learned mean).
        raise StaticViolationError(
            f"{self._step_name}.expressions() tried to read self.params_.get('{key}'), "
            f"but this step was classified as STATIC (discover() returned {{}}). "
            f"This means discover() promised it doesn't need data, but expressions() "
            f"disagrees. Fix one of:\n"
            f"  1. discover() should return {{'{key}': <sqlglot aggregate>}} to learn this param\n"
            f"  2. expressions() should not read self.params_ (use constructor args instead)"
        )


# In Transformer.fit():
def fit(self, data, y=None, *, backend=None):
    ...
    classification = _classify_step(self, columns, schema, y_column)

    if classification.kind == "static":
        # Proven static: no aggregation query needed.
        self.params_ = _FrozenEmptyParams(self.__class__.__name__)
    else:
        # Dynamic: run aggregation query, store learned values.
        raw = backend.fetch_one(fit_query)
        self.params_ = {k: raw[k] for k in discover_result}

    self._classification = classification  # store for audit trail
    ...
```

`StaticViolationError` is a subclass of `SQLearnError` — it's a hard error, not a
warning. If a static step reads `self.params_`, that's a bug that MUST be fixed.

#### 4.3.10 Audit Trail — Every Decision Is Logged

Every classification decision is stored and inspectable. No silent decisions.

```python
pipe.fit("data.parquet")
pipe.describe()
```

```
Pipeline(6 steps):
  1. Filter("price > 0")
     → STATIC  | tier 1 (built-in) | _classification='static' trusted
     → schema: PRESERVING

  2. Imputer()
     → DYNAMIC | tier 1 (built-in) | _classification='dynamic' trusted
     → schema: PRESERVING

  3. StandardScaler()
     → DYNAMIC | tier 1 (built-in) | _classification='dynamic' trusted
     → schema: PRESERVING

  4. OneHotEncoder()
     → DYNAMIC | tier 1 (built-in) | _classification='dynamic' trusted
     → schema: CHANGING  | Added: {'city_london', 'city_paris'}, Removed: {'city'}
     → LAYER BOUNDARY

  5. DropConstant()
     → DYNAMIC | tier 1 (built-in) | _classification='dynamic' trusted
     → schema: CHANGING  | Removed: {'country'}

  6. Log(columns=["income"])
     → STATIC  | tier 1 (built-in) | _classification='static' trusted
     → schema: PRESERVING

  7. MyCustomStep()                                    ← user code
     → DYNAMIC | tier 3 (inspected) | discover() returned 2 aggregation(s)
     → schema: PRESERVING

Layers:
  Layer 0: [Filter, Imputer, StandardScaler, OneHotEncoder]
    → Fit: 1 query (12 aggregates from Imputer + StandardScaler)
           + 1 query (DISTINCT for OneHotEncoder)
           Filter contributes 0 aggregates (static — tier 1, zero-cost classification)

  Layer 1: [DropConstant, Log, MyCustomStep]
    → Fit: 1 query (12 + 2 aggregates from DropConstant + MyCustomStep)
           Log contributes 0 aggregates (static — tier 1, zero-cost classification)

Transform: 1 SQL query (all 7 steps composed)
Classification cost: 0 discover() calls for tiers 1-2, 1 discover() call for tier 3
```

Every classification includes:
- **kind**: `"static"` or `"dynamic"`
- **reason**: exactly WHY this classification was made (which check determined it)
- **warnings**: any ambiguities encountered (returned None, wrong type, etc.)
- **schema_change**: `True` or `False` with diff details
- **layer**: which fit layer this step belongs to

#### 4.3.11 Built-in Transformer Classification Reference

Every built-in transformer has `_classification` set on the class or in `__init__`.
These are **Tier 1** — trusted at runtime, validated by CI tests. The tables below
are the single source of truth for what each built-in declares.

**Declared Static (`_classification = "static"`):**

| Transformer | `_classification` | SQL is fully determined by | Uses `query()`? |
|---|---|---|---|
| `Log(base=)` | `"static"` | `base` arg | No |
| `Sqrt()` | `"static"` | nothing | No |
| `Power(n=)` | `"static"` | `n` arg | No |
| `Abs()` | `"static"` | nothing | No |
| `Round(decimals=)` | `"static"` | `decimals` arg | No |
| `Sign()` | `"static"` | nothing | No |
| `Add(a, b)` | `"static"` | `a`, `b` args | No |
| `Multiply(a, b)` | `"static"` | `a`, `b` args | No |
| `Ratio(a, b)` | `"static"` | `a`, `b` args | No |
| `Diff(a, b)` | `"static"` | `a`, `b` args | No |
| `Modulo(a, b)` | `"static"` | `a`, `b` args | No |
| `Binarizer(threshold=)` | `"static"` | `threshold` arg | No |
| `Expression(sql)` | `"static"` | `sql` arg | No |
| `StringLength()` | `"static"` | nothing | No |
| `StringLower()` | `"static"` | nothing | No |
| `StringUpper()` | `"static"` | nothing | No |
| `StringTrim()` | `"static"` | nothing | No |
| `StringContains(pattern=)` | `"static"` | `pattern` arg | No |
| `StringReplace(old, new)` | `"static"` | `old`, `new` args | No |
| `StringExtract(regex=)` | `"static"` | `regex` arg | No |
| `DateParts(parts=)` | `"static"` | `parts` arg | No |
| `DateDiff(start, end, unit=)` | `"static"` | `start`, `end`, `unit` args | No |
| `IsWeekend()` | `"static"` | nothing | No |
| `Rename(mapping=)` | `"static"` | `mapping` arg | No |
| `Cast(mapping=)` | `"static"` | `mapping` arg | No |
| `Reorder(columns=)` | `"static"` | `columns` arg | No |
| `Drop(columns=)` | `"static"` | `columns` arg | No |
| `Drop(pattern=)` | `"static"` | `pattern` (resolved at schema time, not data time) | No |
| `PolynomialFeatures(degree=)` | `"static"` | `degree` arg + column names | No |
| `HashEncoder(n_bins=)` | `"static"` | `n_bins` arg | No |
| `Filter(condition=)` | `"static"` | `condition` arg | Yes |
| `Sample(n=, frac=, seed=)` | `"static"` | `n`, `frac`, `seed` args | Yes |
| `Normalizer(norm=)` | `"static"` | `norm` arg + column names | Yes (cross-column) |
| `Lag(n=)` | `"static"` | `n` + partition/order args | Yes (window) |
| `Lead(n=)` | `"static"` | `n` + partition/order args | Yes (window) |
| `RollingMean(window=)` | `"static"` | `window` arg | Yes (window) |
| `RollingStd(window=)` | `"static"` | `window` arg | Yes (window) |
| `RollingMin(window=)` | `"static"` | `window` arg | Yes (window) |
| `RollingMax(window=)` | `"static"` | `window` arg | Yes (window) |
| `RollingSum(window=)` | `"static"` | `window` arg | Yes (window) |
| `Rank(by=, order=)` | `"static"` | `by`, `order` args | Yes (window) |
| `PercentRank()` | `"static"` | nothing | Yes (window) |
| `CumSum(order=)` | `"static"` | `order` arg | Yes (window) |
| `CumMax(order=)` | `"static"` | `order` arg | Yes (window) |
| `GroupFeatures(by=, aggs=)` | `"static"` | `by`, `aggs` args | Yes (window) |
| `EWM(alpha=, span=)` | `"static"` | `alpha`/`span` args | Yes (recursive CTE) |
| `Deduplicate(subset=, keep=)` | `"static"` | `subset`, `keep` args | Yes (window) |

**Key insight:** Static ≠ inline. Static means "no fit query needed." The `query()`
column shows whether the step gets a CTE (for SQL structure reasons like window
functions), but that's a **separate decision** from static/dynamic. A step can be
static + CTE (e.g., `Lag`) or dynamic + inline (e.g., `StandardScaler`).

**Declared Dynamic (`_classification = "dynamic"`):**

| Transformer | `_classification` | What it learns from data |
|---|---|---|
| `Imputer(strategy=)` | `"dynamic"` | `MEDIAN(col)`, `AVG(col)`, or `MODE(col)` per column |
| `StandardScaler()` | `"dynamic"` | `AVG(col)`, `STDDEV_POP(col)` per column |
| `MinMaxScaler()` | `"dynamic"` | `MIN(col)`, `MAX(col)` per column |
| `RobustScaler()` | `"dynamic"` | `PERCENTILE_CONT(0.25/0.5/0.75)` per column |
| `MaxAbsScaler()` | `"dynamic"` | `MAX(ABS(col))` per column |
| `KBinsDiscretizer(n=)` | `"dynamic"` | `MIN`, `MAX` or `PERCENTILE_CONT` per column |
| `OneHotEncoder()` | `"dynamic"` | `DISTINCT col` per column |
| `OrdinalEncoder()` | `"dynamic"` | `DISTINCT col` (ordered) per column |
| `TargetEncoder()` | `"dynamic"` | `AVG(y)` per category, global mean |
| `FrequencyEncoder()` | `"dynamic"` | `COUNT(*)` per category |
| `BinaryEncoder()` | `"dynamic"` | `DISTINCT col` per column |
| `AutoEncoder()` | `"dynamic"` | `COUNT(DISTINCT col)` + strategy-specific stats |
| `DropConstant()` | `"dynamic"` | `COUNT(DISTINCT col)` per column |
| `DropCorrelated(threshold=)` | `"dynamic"` | `CORR(a, b)` for all numeric pairs |
| `DropLowVariance(threshold=)` | `"dynamic"` | `VAR_POP(col)` per column |
| `DropHighNull(threshold=)` | `"dynamic"` | `SUM(CASE WHEN col IS NULL ...)` per column |
| `DropHighCardinality(threshold=)` | `"dynamic"` | `COUNT(DISTINCT col)` per column |
| `SelectKBest(k=, method=)` | `"dynamic"` | `CORR(col, target)` or MI approximation |
| `VarianceThreshold(threshold=)` | `"dynamic"` | `VAR_POP(col)` per column |
| `OutlierHandler(method=)` | `"dynamic"` | `PERCENTILE_CONT(0.25/0.75)` or `AVG`/`STDDEV_POP` |
| `EWM(alpha=)` | `"static"` | Recursive CTE structure determined by `alpha`/`span` args |
| `Deduplicate(subset=)` | `"static"` | `ROW_NUMBER()` OVER — window structure determined by `subset`/`keep` args |
| `AutoFeatures()` | `"dynamic"` | All — always inspects data |

**Conditionally Declared (set `_classification` in `__init__` based on args):**

| Transformer | When `_classification = "static"` | When `_classification = "dynamic"` |
|---|---|---|
| `StringSplit(max_parts=3)` | `max_parts` is an integer | `max_parts="auto"` |
| `AutoDatetime(granularity="hour")` | Explicit granularity string | `granularity="auto"` |
| `AutoSplit(by=",")` | Explicit `by` string | `by=None` or `by="auto"` |
| `AutoNumeric({"price": "log"})` | All columns have explicit transforms | No args or any column = "auto" |
| `CyclicEncode(period=24)` | Explicit integer period | `period="auto"` (still static — infers from name, not data) |
| `IsHoliday(country=)` | Always — lookup table is static | Never dynamic |
| `TimeSinceEvent(reference=date)` | Explicit date reference | `reference="min"` → needs `MIN(col)` |
| `TargetTransform(func="log")` | `func` in `("log", "sqrt")` — fixed math | `func` in `("standard", "minmax", "quantile")` — needs stats |
| `Clip(lower=0, upper=100)` | `lower`/`upper` are numbers or None | `lower="p01"` or `upper="p99"` — needs percentiles |

These transformers set `_classification` in their `__init__()` based on the arguments
they receive. CI tests cover BOTH branches:

```python
# CI validates both branches for every conditionally dynamic transformer
def test_stringsplit_static():
    s = StringSplit(max_parts=3)
    assert s._classification == "static"
    assert s.discover(["tags"], schema, None) == {}

def test_stringsplit_dynamic():
    s = StringSplit(max_parts="auto")
    assert s._classification == "dynamic"
    assert len(s.discover(["tags"], schema, None)) > 0

def test_clip_static():
    c = Clip(lower=0, upper=100)
    assert c._classification == "static"
    assert c.discover(["price"], schema, None) == {}

def test_clip_dynamic():
    c = Clip(lower="p01", upper="p99")
    assert c._classification == "dynamic"
    assert len(c.discover(["price"], schema, None)) > 0
```

**The full implementation of a conditionally dynamic transformer:**

```python
class StringSplit(Transformer):
    _default_columns = None

    def __init__(self, columns=None, by=",", max_parts=3, keep_original=True):
        self.by = by
        self.max_parts = max_parts
        self.keep_original = keep_original

        # ── DECLARE classification based on constructor args ──
        if isinstance(max_parts, int):
            self._classification = "static"
        else:
            self._classification = "dynamic"

    def discover(self, columns, schema, y_column=None):
        # MUST be consistent with _classification declaration above
        if isinstance(self.max_parts, int):
            return {}   # static — max_parts known, no data needed
        # max_parts="auto" — learn from data
        return {
            f"{col}__max_parts": exp.Max(
                this=exp.Sub(
                    this=exp.Length(this=exp.Column(this=col)),
                    expression=exp.Length(
                        this=exp.Replace(
                            this=exp.Column(this=col),
                            expression=exp.Literal.string(self.by),
                            replacement=exp.Literal.string(""),
                        )
                    )
                )
            )
            for col in columns
        }
```

**The `_classification` in `__init__` and the `discover()` return value MUST agree.**
If they don't, CI tests catch the mismatch. The `__init__` declaration is what the
pipeline reads at runtime (Tier 1, zero cost). The `discover()` return value is what
CI tests verify against the declaration.

#### 4.3.12 Summary — The Three Tiers

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   TIER 1 — Built-in transformer?                                │
│   (_classification set, module is sqlearn.*)                    │
│   → TRUST the declaration. Zero inspection. CI guarantees it.   │
│   → Cost: zero discover() calls.                                │
│                                                                 │
│   TIER 2 — Custom transformer WITH _classification set?         │
│   → VERIFY once at first fit(). Cache result. Error on mismatch.│
│   → Cost: one discover() call total (first fit only).           │
│                                                                 │
│   TIER 3 — Custom transformer WITHOUT _classification?          │
│   → FULL INSPECTION every fit(). Conservative: any doubt = dynamic.│
│   → Cost: one discover() call per fit.                          │
│                                                                 │
│   In ALL tiers:                                                 │
│     Static means: skip aggregation query. Zero fit cost.        │
│     Dynamic means: run aggregation query. Always safe.          │
│     If ANY doubt → dynamic. The cost of being wrong is          │
│     asymmetric: false "static" = data corruption.               │
│     False "dynamic" = one cheap extra query.                    │
│                                                                 │
│   For custom transformers (Tier 2 & 3):                         │
│     discover() returned EXACTLY {} (empty dict)?                │
│     AND class does NOT override fit()?                          │
│     AND no exception was raised?                                │
│       YES to ALL  →  STATIC                                    │
│       NO to ANY   →  DYNAMIC                                   │
│                                                                 │
│   There is no "maybe." There is no heuristic.                   │
│   The check is mechanical and deterministic.                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3.13 `discover()` Contract

`discover()` **MUST be idempotent** — calling it multiple times with the same arguments
must produce the same result. The pipeline may call `discover()` multiple times for
Tier 3 steps (once for classification, once for collecting aggregation expressions).
Side effects (logging, incrementing counters, modifying state) are forbidden.

```python
# GOOD — pure function:
def discover(self, columns, schema, y_column=None):
    return {f"{col}__mean": exp.Avg(this=exp.Column(this=col)) for col in columns}

# BAD — has side effects:
def discover(self, columns, schema, y_column=None):
    self._call_count += 1  # NO — breaks idempotency
    return {f"{col}__mean": exp.Avg(this=exp.Column(this=col)) for col in columns}
```

### 4.3.14 Error Propagation From `expressions()`

If a custom transformer's `expressions()` raises during `transform()`, the base class
catches the exception and wraps it with context:

```python
try:
    modified = step.expressions(step.columns_, exprs)
except Exception as e:
    raise CompilationError(
        f"{step.__class__.__name__}.expressions() raised {type(e).__name__}: {e}\n"
        f"Columns: {step.columns_}\n"
        f"Params: {list(step.params_.keys()) if hasattr(step, 'params_') else 'N/A'}"
    ) from e
```

Users see a diagnostic message with the step name, columns, and available params —
not a raw sqlglot traceback.

### 4.3.15 Fit Execution Flow — The Complete Pipeline

This is the **most critical flow in the system** — the bridge between SQL aggregation
and Python-side parameter storage. Shown end-to-end with a concrete example:

```
pipeline.fit("train.parquet", y="price")

1. Input Resolution
   io.py resolves "train.parquet" → DuckDB can read it directly

2. Schema Discovery
   backend.describe("train.parquet")
   → Schema({price: DOUBLE, city: VARCHAR, age: INTEGER})

3. Column Resolution (per step)
   Imputer._default_columns = "all"         → [price, city, age]
   StandardScaler._default_columns = "numeric" → [price, age]
   OneHotEncoder._default_columns = "categorical" → [city]

4. Classification (per step, using Section 4.3.2 tiers)
   Imputer:        Tier 1, declared "dynamic" → trust, will call discover()
   StandardScaler: Tier 1, declared "dynamic" → trust, will call discover()
   OneHotEncoder:  Tier 1, declared "dynamic" → trust, will call discover()+discover_sets()

5. Layer Resolution (Section 5.3)
   OneHotEncoder is schema-changing → layer boundary after it.
   Layer 0: [Imputer, StandardScaler, OneHotEncoder]

6. discover() Collection — THE KEY STEP
   Imputer.discover(["price","city","age"], schema, y_column="price")
     → {"price__median": Avg(Column("price")), "city__mode": Mode(Column("city")), ...}
   StandardScaler.discover(["price","age"], schema, y_column="price")
     → {"price__mean": Avg(Column("price")), "price__std": StddevPop(Column("price")), ...}

7. discover_sets() Collection
   OneHotEncoder.discover_sets(["city"], schema, y_column="price")
     → {"city__categories": Select(Distinct(Column("city"))).from_("train.parquet")}

8. Fit Query Generation — Aggregates batched into ONE query
   Query 1 (scalars): SELECT
       MEDIAN(price) AS imputer__price__median,
       MODE(city) AS imputer__city__mode,
       AVG(price) AS scaler__price__mean,
       STDDEV_POP(price) AS scaler__price__std,
       AVG(age) AS scaler__age__mean,
       STDDEV_POP(age) AS scaler__age__std
     FROM 'train.parquet'

   Query 2 (sets): SELECT DISTINCT city FROM 'train.parquet'

9. Parameter Storage — Results flow back into each step
   backend.fetch_one(query_1) → {"imputer__price__median": 42.5, ...}
   backend.execute(query_2) → [{"city": "London"}, {"city": "Paris"}, {"city": "Berlin"}]

   Imputer.params_        = {"price__median": 42.5, "city__mode": "London", ...}
   StandardScaler.params_  = {"price__mean": 42.5, "price__std": 12.3, ...}
   OneHotEncoder.sets_     = {"city__categories": ["London", "Paris", "Berlin"]}

10. Schema Tracking
    Output schema: {price: DOUBLE, age: DOUBLE, city_london: INTEGER,
                    city_paris: INTEGER, city_berlin: INTEGER}
    (price column excluded from transform output because y="price")

11. Done. pipeline._fitted = True
```

**Total: 2 SQL queries for the entire pipeline.** Compare to sklearn: one full data
pass per step (3 passes for this pipeline).

### 4.4 Two Interfaces: `expressions()` vs `query()`

Most transforms are column-level. They take expression in, return expression out.
**All expressions are sqlglot AST nodes** — enabling transpilation to any SQL dialect:

```python
import sqlglot.expressions as exp

class StandardScaler(Transformer):
    _default_columns = "numeric"

    def discover(self, columns, schema, y_column=None):
        # Returns sqlglot AST nodes — these get batched into one query
        aggs = {}
        for col in columns:
            aggs[f"{col}__mean"] = exp.Avg(this=exp.Column(this=col))
            aggs[f"{col}__std"] = exp.StddevPop(this=exp.Column(this=col))
        return aggs

    def expressions(self, columns, exprs):
        # Only return columns we modify. Untouched columns pass through automatically.
        result = {}
        for col in columns:
            mean = self.params_[f"{col}__mean"]
            std = self.params_[f"{col}__std"]
            # sqlglot AST — NOT a string. Transpiles to any dialect.
            cast = exp.Cast(this=exprs[col], to=exp.DataType(this=exp.DataType.Type.DOUBLE))
            result[col] = (cast - exp.Literal.number(mean)) / exp.Literal.number(std)
        return result
```

Note: `expressions()` takes `columns` (this step's target columns) and `exprs` (the full
expression dict). It returns ONLY the columns it modifies — the base class merges the
result with all untouched columns via `_apply_expressions()`. No manual passthrough loop.

These compose inline — no CTE needed:

```sql
-- Imputer + StandardScaler composed into one expression:
(CAST(COALESCE(price, 42.5) AS DOUBLE) - 42.5) / 12.3 AS price
-- Same AST transpiled to Postgres, Snowflake, BigQuery — sqlglot handles differences
```

Some transforms need their own query level (window functions, aggregations):

```python
class RollingMean(Transformer):
    _default_columns = None  # must be explicit

    def query(self, input_query):
        # Returns sqlglot AST — NOT a string
        window = exp.Window(
            this=exp.Avg(this=exp.Column(this=self.column)),
            partition_by=[exp.Column(this=self.partition_by)],
            order=exp.Order(expressions=[exp.Column(this=self.order_by)]),
            spec=exp.WindowSpec(
                kind="ROWS",
                start="PRECEDING",
                start_side=exp.Literal.number(self.window),
                end="CURRENT ROW",
            ),
        )
        return exp.Select(
            expressions=[exp.Star(), window.as_(self.name)]
        ).from_(input_query)
```

The compiler checks which interface a transformer uses:
- `expressions()` → compose inline (no CTE)
- `query()` → wrap in CTE/subquery

**Interface detection is also conservative.** If a transformer overrides both
`expressions()` and `query()`, the compiler uses `query()` (the safer, more
isolated path). If `query()` returns `None`, fall back to `expressions()`.
If `expressions()` raises `NotImplementedError` and `query()` returns `None`,
that's an error — the transformer doesn't implement either interface.

```python
def _resolve_interface(step, input_query, exprs):
    """Determine which interface to use. Conservative: prefer query() if both exist."""
    query_result = step.query(input_query)
    if query_result is not None:
        return "query", query_result   # query() wins — gets its own CTE

    try:
        expr_result = step._apply_expressions(exprs)
    except NotImplementedError:
        raise CompilationError(
            f"{step.__class__.__name__} implements neither expressions() nor query(). "
            "Override at least one."
        )
    return "expressions", expr_result  # compose inline
```

**Decision tree for CTE creation:**

```
1. All transforms use expressions() only?       → one SELECT, zero CTEs
2. A transform uses query()?                     → CTE for that transform
3. A transform overrides both?                   → query() wins (conservative)
4. A complex expression referenced 2+ times?     → CTE to avoid duplication
5. Bare column ref referenced 2+ times?          → NO CTE (column refs are free)
```

### 4.5 Schema Tracking

A `Schema` object flows through the pipeline, tracking column names and types:

```python
@dataclass
class Schema:
    columns: dict[str, str]     # {name: sql_type}

    def numeric(self) -> list[str]: ...
    def categorical(self) -> list[str]: ...
    def temporal(self) -> list[str]: ...
    def select(self, spec) -> list[str]: ...
    def add(self, cols: dict) -> Schema: ...
    def drop(self, names: list) -> Schema: ...
    def rename(self, mapping: dict) -> Schema: ...
```

Read once from the database at fit time. After that, each step
receives input schema and returns output schema — no more database queries needed
for schema resolution. This is how auto column routing works without scanning the data.

### 4.6 Backend Protocol (Multi-Database from Day One)

Design for multiple databases now, implement DuckDB first:

```python
class Backend(Protocol):
    """Interface for database backends. Phase 1: DuckDB only.
    Phase 5+: Postgres, MySQL, Snowflake, BigQuery."""

    @property
    def dialect(self) -> str: ...                              # "duckdb", "postgres", etc.

    def execute(self, ast: sg.Expression) -> Result: ...       # run query, return result
    def fetch_one(self, ast: sg.Expression) -> dict: ...       # run query, return one row
    def describe(self, source: str) -> Schema: ...             # inspect table/file schema
    def register(self, data, name: str) -> str: ...            # register DataFrame/array as table
    def supports(self, feature: str) -> bool: ...              # "filter_clause", "median", etc.
```

The compiler generates dialect-neutral sqlglot ASTs. The backend transpiles to its
dialect and executes. Adding a new database = implementing this protocol.

**Dialect compatibility (handled by sqlglot + thin adapter):**

| Feature | DuckDB | Postgres | MySQL | Snowflake | BigQuery |
|---|---|---|---|---|---|
| `FILTER` clause | Yes | Yes | No (CASE) | No (CASE) | No (CASE) |
| `MEDIAN` | Native | `PERCENTILE_CONT` | Workaround | `MEDIAN` | `APPROX_QUANTILES` |
| `HASH(x)` | Native | `MD5(x)` | `CRC32(x)` | `HASH(x)` | `FARM_FINGERPRINT` |
| `STDDEV_POP` | Yes | Yes | Yes | Yes | Yes |
| `NTILE` | Yes | Yes | Yes | Yes | Yes |
| Read Parquet | Native | No | No | Stage | External table |
| Schema inspect | `DESCRIBE` | `information_schema` | `DESCRIBE` | `DESCRIBE` | `INFORMATION_SCHEMA` |

Where a feature isn't supported, the compiler generates equivalent SQL:

```sql
-- FILTER supported (DuckDB, Postgres)
AVG(price) FILTER (WHERE __sq_fold__ != 1)

-- FILTER not supported (MySQL, Snowflake) — equivalent CASE
AVG(CASE WHEN __sq_fold__ != 1 THEN price END)
```

### 4.7 Compilation Strategies: Baked vs Self-Contained

Two strategies for how learned parameters appear in compiled SQL.
**The user never chooses.** The pipeline picks automatically based on context.

**Baked (default for fit+transform and export):**

```python
pipe.fit("train.parquet")           # computes AVG(price)=42.5, stores it
pipe.transform("test.parquet")      # uses literal 42.5 in SQL
```

```sql
SELECT (CAST(price AS DOUBLE) - 42.5) / 12.3 AS price FROM test
```

Simple, portable, no data dependency. Used for normal fit+transform and all exports.

**Self-contained (internal optimization for cross-validation):**

```sql
-- Stats computed inline via CTE — same SQL works for any fold
WITH stats AS (
    SELECT AVG(price) AS mean_price, STDDEV_POP(price) AS std_price
    FROM data WHERE __sq_fold__ != :k
)
SELECT (CAST(price AS DOUBLE) - stats.mean_price) / stats.std_price AS price
FROM data CROSS JOIN stats
WHERE __sq_fold__ = :k
```

No separate fit needed. Change `:k` and the same SQL works for any fold.

**Why both exist:**

| Context | Strategy | Why |
|---|---|---|
| `fit()` + `transform()` | Baked | Simple SQL, inspectable, deterministic |
| `to_sql()` / `export()` / `save()` | Baked | Portable, no data dependency |
| `pipe.describe()` | Baked | User sees "learned mean=42.5" |
| `GridSearchCV` / cross-validation | Self-contained | One scan for all folds at any scale |

**At scale this matters:**

```
100M rows, 5-fold CV, 10 columns:
  Baked:          5 fit scans × ~3s each = ~15s just for stats
  Self-contained: 1 scan with FILTER    = ~3s total (5x faster)
```

### 4.7b Cross-Validation Schema Safety — The Fold Problem

**The problem:** In cross-validation, different folds may have different data distributions.
If fold 1's training data contains categories `{London, Paris, Berlin}` but fold 2's
training data only contains `{London, Paris}`, a per-fold OneHotEncoder would produce
**different output schemas per fold** — fold 1 has `city_berlin`, fold 2 doesn't.
This corrupts everything: models expect different features, scores aren't comparable,
ensembling fails.

This isn't theoretical. It happens with any rare category, and it happens in sklearn too
(sklearn just silently handles it per fold, producing inconsistent feature sets that
the model has to absorb).

**sqlearn's solution: Two-Phase Discovery**

Schema-affecting discovery and value-affecting discovery are separated:

| Phase | What it learns | From what data | When | Example |
|---|---|---|---|---|
| **Phase 1: Schema discovery** | The UNIVERSE of possible outputs | **Full training data** (all folds) | Once, before CV starts | OneHotEncoder: all distinct categories |
| **Phase 2: Value discovery** | Per-fold statistics | **Per-fold training data** (fold-excluded) | Per fold, via self-contained CTE | StandardScaler: per-fold mean/std |

**Phase 1 runs ONCE, Phase 2 runs per fold.** The output schema is identical for all folds.

This is implemented via the `discover()` / `discover_sets()` split:

- `discover()` returns scalar aggregates (mean, std, variance) → **these change per fold**
  and use the self-contained CTE strategy (inline `WHERE __sq_fold__ != :k`).
- `discover_sets()` returns set-valued queries (DISTINCT categories, per-category stats)
  → **these are learned from full training data** and baked into every fold.

```sql
-- Phase 1 (once): learn ALL categories from full training data
-- This determines the output schema — identical for every fold.
SELECT DISTINCT city FROM data;
-- Result: {London, Paris, Berlin} → always 3 one-hot columns

-- Phase 2 (per fold, self-contained CTE):
-- Scalar stats use FILTER to exclude the test fold.
WITH stats AS (
    SELECT
        AVG(price) FILTER (WHERE __sq_fold__ != :k)   AS mean_price,
        STDDEV_POP(price) FILTER (WHERE __sq_fold__ != :k) AS std_price
    FROM data
)
SELECT
    (CAST(price AS DOUBLE) - stats.mean_price) / stats.std_price AS price,
    CASE WHEN city = 'London' THEN 1 ELSE 0 END AS city_london,
    CASE WHEN city = 'Paris'  THEN 1 ELSE 0 END AS city_paris,
    CASE WHEN city = 'Berlin' THEN 1 ELSE 0 END AS city_berlin
FROM data CROSS JOIN stats
WHERE __sq_fold__ = :k
```

**Is Phase 1 data leakage?** No. Phase 1 learns only the SCHEMA (what categories exist),
not the VALUES (encoding weights, target means, frequencies). Knowing that "Berlin" is
a possible city is not leakage — it's structural information. The actual encoding of
Berlin in each fold depends only on that fold's training data. This is the same as
sklearn's common practice of fitting `ColumnTransformer` on all training data before CV.

**For TargetEncoder (the hardest case):**

TargetEncoder needs per-category target means — that's a set-valued result that IS
value-dependent. How we handle it:

```python
class TargetEncoder(Transformer):
    def discover_sets(self, columns, schema, y_column=None):
        # Phase 1: learn category UNIVERSE (schema)
        return {
            f"{col}__categories": exp.Select(
                expressions=[exp.Distinct(expressions=[exp.Column(this=col)])]
            ).from_(exp.Table(this="__input__"))
            for col in columns
        }

    def discover(self, columns, schema, y_column=None):
        # Phase 2: per-fold target means use SELF-CONTAINED CTE with FILTER
        # Global mean + per-category mean, all fold-aware
        return {
            f"__global_mean": exp.Avg(this=exp.Column(this=y_column)),
            # Per-category means are computed in the self-contained CTE
            # using GROUP BY with FILTER — see compiler for details
        }
```

The compiler generates:

```sql
-- Phase 1 (once): categories = {A, B, C} from full training data

-- Phase 2 (per fold k):
WITH cat_means AS (
    SELECT
        city,
        AVG(price) FILTER (WHERE __sq_fold__ != :k) AS target_mean,
        COUNT(*) FILTER (WHERE __sq_fold__ != :k) AS cat_count
    FROM data
    GROUP BY city
),
global AS (
    SELECT AVG(price) FILTER (WHERE __sq_fold__ != :k) AS global_mean
    FROM data
)
SELECT
    -- Smoothed target encoding: (cat_count * cat_mean + smooth * global) / (cat_count + smooth)
    ...
FROM data
LEFT JOIN cat_means ON data.city = cat_means.city
CROSS JOIN global
WHERE __sq_fold__ = :k
```

**The guarantee:** Every fold produces the SAME columns. Different folds may have
different VALUES for target-encoded columns (because per-fold statistics differ),
but the SCHEMA is always identical. No model ever receives an unexpected feature set.

**Why this matters at scale:**

| Dataset | Categories | Rare category in <5% of data |
|---|---|---|
| 1K rows, 5-fold | ~200 per fold | ~50 missing from random fold — **schema corruption** |
| 100K rows, 5-fold | ~20K per fold | Rare categories routinely absent — **schema corruption** |
| 1M rows, 10-fold | ~100K per fold | Long-tail categories guaranteed absent — **schema corruption** |

Without two-phase discovery, cross-validation on real data is **silently broken** for
any schema-changing step with cardinality-dependent output. sqlearn is the only library
that handles this correctly by design.

### 4.8 Fold Column Convention

Cross-validation fold assignments stored as a column in the data:

```
Column name:  __sq_fold__
Values:       1, 2, 3, ..., k
Assignment:   NTILE(k) OVER (ORDER BY HASH(rowid || CAST(seed AS VARCHAR)))
```

**Reserved prefix:** All internal columns use `__sq_*__` prefix. Validated on schema read:

```python
# During schema.describe():
for col in columns:
    if col.startswith("__sq_") and col.endswith("__"):
        raise SchemaError(
            f"Column '{col}' uses reserved sqlearn prefix '__sq_*__'. "
            "Please rename this column."
        )
```

**Fold column variants:**

| Split type | SQL |
|---|---|
| `KFold` | `NTILE(k) OVER (ORDER BY HASH(rowid \|\| seed))` |
| `StratifiedKFold` | `NTILE(k) OVER (PARTITION BY target ORDER BY HASH(rowid \|\| seed))` |
| `GroupKFold` | `NTILE(k) OVER (ORDER BY HASH(group_col \|\| seed))` |
| `TimeSeriesSplit` | `NTILE(k) OVER (ORDER BY timestamp)` (no shuffle) |
| `RepeatedKFold` | Multiple columns: `__sq_fold_r1__`, `__sq_fold_r2__`, ... |

Fold column excluded from feature output automatically.

---

