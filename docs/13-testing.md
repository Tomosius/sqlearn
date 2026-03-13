> **sqlearn docs** — [Index](../CLAUDE.md) | Prev: [Implementation Milestones](12-milestones.md) | Next: [Decisions & Risks](14-decisions.md)

## 14. Testing Strategy

### Per-Transformer Tests

Every transformer gets:
- **Correctness:** Compare output to sklearn on identical data (where sklearn equivalent exists)
- **NULL handling:** Columns with NULLs, all-NULL columns, no NULLs
- **Edge cases:** Single row, constant column, empty table, single unique value
- **SQL snapshot:** Verify compiled SQL matches expected output
- **Schema tracking:** `output_schema()` matches actual output columns and types
- **Roundtrip:** `fit → to_sql → execute manually → same result as transform`
- **Clone:** `clone()` produces independent copy with identical transform output

### Parameterized sklearn Equivalence Tests

Use `@pytest.mark.parametrize` to systematically test sqlearn transformers against their
sklearn equivalents. For each transformer: fit+transform with both libraries on identical
data, compare outputs within tolerance (`np.allclose`).

```python
@pytest.mark.parametrize("sqlearn_cls,sklearn_cls,kwargs", [
    (sq.StandardScaler, sklearn.StandardScaler, {}),
    (sq.MinMaxScaler, sklearn.MinMaxScaler, {}),
    (sq.RobustScaler, sklearn.RobustScaler, {}),
    (sq.Imputer, sklearn.SimpleImputer, {"strategy": "mean"}),
    (sq.Imputer, sklearn.SimpleImputer, {"strategy": "median"}),
])
def test_sklearn_equivalence(sqlearn_cls, sklearn_cls, kwargs, sample_data):
    """sqlearn output must match sklearn within floating-point tolerance."""
    sq_result = sqlearn_cls(**kwargs).fit_transform(sample_data)
    sk_result = sklearn_cls(**kwargs).fit_transform(sample_data_numpy)
    np.testing.assert_allclose(sq_result, sk_result, rtol=1e-6)
```

Cover edge cases: NaN columns, constant columns, single-row input.

### Pipeline Tests

- Full pipeline output matches sklearn
- Layer resolution with multi-layer pipelines
- Expression composition (verify no unnecessary CTEs)
- All input formats (table, file, DataFrame, Arrow)
- Roundtrip: `fit → to_sql → execute manually → same result as transform`
- `+` operator: `a + b` produces flat `Pipeline([a, b])`, not nested
- `Pipeline + Pipeline` flattens: `Pipeline([a,b]) + Pipeline([c,d])` → `Pipeline([a,b,c,d])`
- `+=` operator: `pipe += step` creates NEW Pipeline with step appended (non-mutating)
- `+=` non-mutation: `base = pipe; pipe += step; assert base is not pipe` (base unchanged)
- `Transformer + Transformer` creates Pipeline (neither is already a Pipeline)
- `y="target"` propagation: target column excluded from transform output
- `y=["t1","t2"]` multi-target: both columns excluded from output
- `y=None` default: all columns included in output
- Large result warning: mock query plan with >1GB result → `UserWarning` emitted
- `sq.set_option("warn_result_size", None)` → warning suppressed

### Thread Safety Tests

- `_check_thread()`: access pipeline from a different thread → `SQLearnError` raised
- `_check_thread()`: access pipeline from a different process (fork) → `SQLearnError` raised
  with message about DuckDB connections not being shareable across processes
- `_check_thread()`: access pipeline from a different process (spawn) → pickle error before
  thread check (DuckDB connection doesn't survive pickling)
- Error message for thread access mentions `.clone()` as the solution
- Error message for process access says "Create a new pipeline in each process"
- `clone()`: cloned pipeline produces identical output on a different thread
- `clone()` has independent connection (mutating clone doesn't affect original)
- Single-threaded usage: no performance penalty, no spurious errors
- `sq.Search(n_jobs=4)`: parallel model training works without thread errors
  (SQL runs on main thread, model fits run on worker threads via joblib)
- `concurrent.futures.ProcessPoolExecutor`: pipeline raises appropriate error

### `y` Column Propagation Tests

- `pipe.fit("data", y="price")` → `TargetEncoder` receives `y_column="price"` in `discover()`
- `pipe.fit("data")` → `TargetEncoder` raises `FitError` with helpful message
- `pipe.fit("data", y="price")` → `StandardScaler` ignores `y_column` (works fine)
- `pipe.transform("data")` → output does NOT contain `price` column (exclude_target=True default)
- `pipe.transform("data", exclude_target=False)` → output DOES contain `price` column
- `pipe.transform("data")` without prior `y=` → output contains all columns
- `y=["target1", "target2"]` → both columns excluded from transform output
- `y="nonexistent_column"` → `SchemaError` raised at fit time
- `sq.read_column("data", "price")` returns correct numpy array matching DuckDB result

### Feature Selection & Dropping Tests

- `Drop(columns=["a","b"])` → output has all columns except a, b
- `Drop(columns=["nonexistent"])` → `SchemaError` raised
- `Drop(pattern="^id_")` → drops columns starting with "id_", keeps others
- `Drop(dtype="VARCHAR")` → drops all string columns
- `DropConstant()` on column with 1 unique value → column removed
- `DropConstant()` on all-NULL column → column removed
- `DropConstant()` on column with 2+ unique values → column kept
- `DropCorrelated(threshold=0.95)` on synthetic columns with r=0.98 → one dropped
- `DropCorrelated(threshold=0.95)` on columns with r=0.90 → both kept
- `DropCorrelated` with `y=` → drops the column with lower target correlation
- `DropCorrelated` without `y=` → drops alphabetically second column
- `DropLowVariance(threshold=0.01)` on constant column → dropped
- `DropHighNull(threshold=0.5)` on column with 60% nulls → dropped
- `DropHighCardinality(threshold=100)` on column with 200 unique → dropped
- `SelectKBest(k=3, method="correlation")` → keeps 3 columns with highest |CORR(col, target)|
- `SelectKBest(k=3)` without `y=` → `FitError` raised
- `SelectByName(pattern="feature_*")` → keeps only matching columns
- `VarianceThreshold(threshold=0.0)` → drops zero-variance columns only

### Auto Feature Engineering Tests

- `AutoDatetime()` on TIMESTAMP column → generates year, month, day, hour, dayofweek, is_weekend
- `AutoDatetime()` on DATE column → generates year, month, day, dayofweek, is_weekend (no hour)
- `AutoDatetime(granularity="month")` → generates only year, quarter, month, dayofweek, is_weekend
- `AutoDatetime({"col_a": "hour", "col_b": "month"})` → per-column granularity
- `AutoDatetime()` auto-detection: DATE with >365-day range → "day" granularity
- `AutoSplit()` on "a,b,c" → detects comma, splits into 3 columns
- `AutoSplit()` on "a|b|c" → detects pipe separator
- `AutoSplit(columns=["tags"], by=",")` → explicit delimiter
- `AutoSplit` with `max_parts="auto"` → learns correct max from data
- `AutoSplit` on column without delimiters → no split (passthrough)
- `AutoNumeric()` on right-skewed column (skew>2) → applies log transform
- `AutoNumeric()` on normal column → passthrough
- `AutoNumeric({"income": "log"})` → forces log regardless of distribution
- `AutoNumeric({"score": "none"})` → skips column
- `AutoFeatures()` → correctly chains AutoDatetime + AutoSplit + AutoNumeric + AutoEncoder
- `AutoFeatures(split_strings=False)` → skips string splitting
- `AutoFeatures(cyclic=False)` → skips CyclicEncode
- `AutoFeatures(drop_original=True)` → original columns removed after expansion
- `AutoFeatures` on fixture dataset → no errors, output has more columns than input

### String Splitting & Structured Data Tests

- `StringSplit(columns=["tags"], by=",", max_parts=3)` → 3 new columns
- `StringSplit` on "a,b" with max_parts=3 → third column is NULL
- `StringSplit` with `keep_original=False` → original column removed
- `StringSplit` with `names=["first", "last"]` → custom output column names
- `StringSplit(max_parts="auto")` → discover() learns correct max from data
- `JsonExtract(columns=["meta"], fields=["source","score"])` → 2 new columns
- `JsonExtract` on malformed JSON → NULL values (not error)
- `URLParts(columns=["url"])` → scheme, domain, path, query columns
- `URLParts` on non-URL string → NULL values
- `EmailParts(columns=["email"])` → local_part, domain columns
- `EmailParts` on string without @ → NULL values
- `IPParts(columns=["ip"])` → octet_1..4, is_private columns
- All extractors with `keep_original=True` (default) → original column preserved
- All extractors with `keep_original=False` → original column removed

### Datetime & Temporal Tests

- `DateParts(parts=["year","month"])` → 2 new columns with correct values
- `DateDiff("start", "end", unit="day")` → correct day differences
- `IsWeekend()` on known dates → correct 0/1 flags
- `CyclicEncode(columns=["hour"], period=24)` → sin and cos columns
- `CyclicEncode` hour=0 and hour=23 → close in sin/cos space (verify distance)
- `CyclicEncode(period="auto")` on column named "hour" → infers period=24
- `CyclicEncode(period="auto")` on column named "month" → infers period=12
- `CyclicEncode` drops original integer column by default
- `CyclicEncode(keep_original=True)` → keeps original + adds sin/cos
- `TimeSinceEvent(reference="2020-01-01")` → correct day counts

### Outlier Handling Tests

- `OutlierHandler()` default (IQR clip) → values clipped to [Q1-1.5*IQR, Q3+1.5*IQR]
- `OutlierHandler(method="zscore", threshold=3)` → values clipped to [mean-3σ, mean+3σ]
- `OutlierHandler(method="percentile", lower=0.01, upper=0.99)` → values clipped to [p01, p99]
- `OutlierHandler(action="null")` → outliers become NULL
- `OutlierHandler(action="flag")` → new `col_outlier` binary column added, original unchanged
- `OutlierHandler(action="drop_rows")` → rows with outliers removed (fewer output rows)
- Per-column config: `OutlierHandler({"income": {"method": "iqr"}, "score": "skip"})` → income clipped, score unchanged
- On column with no outliers → all values unchanged
- On constant column → all values unchanged (no division by zero)

### Target Transform Tests

- `TargetTransform(func="log")` → transform: LN(y+1), inverse: EXP(pred)-1
- `TargetTransform(func="sqrt")` → transform: SQRT(y), inverse: pred^2
- `TargetTransform(func="standard")` → learns mean/std, transform: (y-mean)/std
- `TargetTransform(func="minmax")` → learns min/max, transform: (y-min)/(max-min)
- Inverse correctness: `inverse(transform(y)) ≈ y` for all func types
- `TargetTransform` inside regular `Pipeline` → error (only works via `ModelPipeline`)
- `ModelPipeline` with `target_transform` → predictions auto-inverted

### Data Operations Tests

- `Rename({"old": "new"})` → column renamed in output, schema updated
- `Rename` with nonexistent column → `SchemaError`
- `Cast({"zip": "VARCHAR"})` → column type changed in output
- `Cast(columns=sq.numeric(), dtype="FLOAT")` → all numeric columns cast to float
- `Reorder(columns=["b","a","c"])` → output columns in specified order
- `Reorder(sort="alphabetical")` → columns sorted A-Z
- `Filter("price > 0")` → only rows with positive price
- `Filter(columns=["age"], condition="NOT NULL")` → rows where age is not NULL
- `Sample(n=100)` → exactly 100 rows (or fewer if source has fewer)
- `Sample(frac=0.1, seed=42)` → ~10% of rows, reproducible with same seed
- `Deduplicate()` → exact duplicate rows removed
- `Deduplicate(subset=["user_id"], keep="first")` → one row per user_id

### Data Merge & Concat Tests

- `sq.merge("a.parquet", "b.parquet", on="id")` → inner join, correct row count
- `sq.merge(..., how="left")` → left join, preserves all left rows
- `sq.merge(..., how="outer")` → full outer join, NULLs for non-matching
- `sq.merge(..., left_on="uid", right_on="user_id")` → different column names
- `sq.merge` returns a view name (string), not materialized data
- `sq.merge` chaining: `merge(merge(a,b), c)` → three-table join
- `sq.concat(["2023.parquet", "2024.parquet"])` → UNION ALL, correct row count
- `sq.concat(..., source_column="file")` → extra column with source filename
- `sq.concat` with mismatched schemas → `SchemaError`
- `sq.Lookup(table="cities", on="city", features=["population"])` → left join, one new column
- `sq.Lookup` with nonexistent join key → `SchemaError`
- `sq.Lookup` as pipeline step → CTE in compiled SQL

### Feature Name Stability Tests

- Fit the same transformer twice on identical data → assert identical output column names
- Fit → clone → fit clone → assert same feature names (deterministic across instances)
- OneHotEncoder category order must be deterministic (alphabetical sort)

### Property-Based Tests (hypothesis)

Use `hypothesis` for fuzzing transformer inputs. Ensures transformers and SQL generation
never crash on any valid input, including edge cases humans wouldn't think to test.

```python
from hypothesis import given, strategies as st

@given(st.lists(st.floats(allow_nan=True, allow_infinity=False), min_size=1, max_size=100))
def test_standard_scaler_never_crashes(values):
    """StandardScaler must handle any valid float list without crashing."""
    # Create table, fit, transform — no exceptions allowed
    ...
```

- `inverse_transform(transform(X)) ≈ X` for invertible transforms
- `transform(fit(X)) + transform(fit(X))` → identical (deterministic)
- Compiled SQL is valid for target dialect (parse with sqlglot, no errors)
- `+` operator is associative: `(a + b) + c` produces same SQL as `a + (b + c)`
- Random pipeline of N steps → `to_sql()` produces valid SQL (execute without error)
- Random pipeline → `get_feature_names_out()` matches actual output column names
- `AutoFeatures()` on random data → no crashes, output columns ≥ input columns
- `DropCorrelated` + `SelectKBest` → output has ≤ input columns
- `OutlierHandler(action="clip")` → output min ≥ learned lower, output max ≤ learned upper

### Static/Dynamic Classification Tests

**Tier 1 — Built-in declaration tests (CI validation):**
- Every built-in transformer with `_classification="static"` → `discover()` returns `{}`
- Every built-in transformer with `_classification="dynamic"` → `discover()` returns non-empty dict
- Conditionally dynamic: `StringSplit(max_parts=3)._classification == "static"`
- Conditionally dynamic: `StringSplit(max_parts="auto")._classification == "dynamic"`
- Conditionally dynamic: `Clip(lower=0, upper=100)._classification == "static"`
- Conditionally dynamic: `Clip(lower="p01", upper="p99")._classification == "dynamic"`
- Conditionally dynamic: `AutoDatetime(granularity="hour")._classification == "static"`
- Conditionally dynamic: `AutoDatetime(granularity="auto")._classification == "dynamic"`
- All conditionally dynamic: `__init__` declaration matches `discover()` return
- Tier 1 step → `_classify_step` returns without calling `discover()` (verify via mock)

**Tier 2 — Custom declared verification tests:**
- Custom `_classification="static"` + `discover()` returns `{}` → verified, cached
- Custom `_classification="static"` + `discover()` returns non-empty → `ClassificationError`
- Custom `_classification="dynamic"` + `discover()` returns `{}` → `UserWarning`, honored as dynamic
- Custom `_classification="dynamic"` + `discover()` returns non-empty → verified, cached
- After verification: `step._classification_verified == True`
- Second `fit()` on verified step → no `discover()` call for classification (verify via mock)

**Tier 3 — Custom undeclared inspection tests:**
- `discover()` returns `{}` → classified as static
- `discover()` returns `{"a": expr}` → classified as dynamic
- `discover()` returns `None` → classified as dynamic (fallback), `UserWarning` emitted
- `discover()` returns `[]` → classified as dynamic (fallback), `UserWarning` emitted
- `discover()` returns `0` → classified as dynamic (fallback), `UserWarning` emitted
- `discover()` raises `TypeError` → classified as dynamic (fallback), no crash
- `discover()` raises `NotImplementedError` → classified as dynamic (fallback)
- Custom transformer overrides `fit()` directly → classified as dynamic (fallback)
- `output_schema()` returns `None` → classified as schema-changing (fallback)
- `output_schema()` raises → classified as schema-changing (fallback)

**Runtime guard tests:**
- Static step reads `self.params_["key"]` → `StaticViolationError` raised
- Static step calls `self.params_.get("key", default)` → `StaticViolationError` raised (same as `__getitem__` — any params_ access on a static step is a bug)
- Transformer overrides both `expressions()` and `query()` → `query()` wins
- Transformer overrides neither → `CompilationError` raised with helpful message

**Integration tests:**
- Static step in layer → zero aggregation expressions contributed to fit query
- Dynamic step treated as static (forced via test mock) → results differ from correct → proves conservative fallback is necessary
- Pipeline with 5 static + 1 dynamic → fit runs exactly 1 aggregation query (not 0, not 5)
- Pipeline with all static steps → fit runs 0 aggregation queries, transform still works
- Pipeline with 3 built-in + 1 custom undeclared → only 1 `discover()` call for classification
- Audit trail: `pipe.describe()` shows tier + reason for each step

### discover_sets() Tests

- `OneHotEncoder.discover_sets()` returns one query per categorical column
- `OrdinalEncoder.discover_sets()` returns one query per column (ordered distinct)
- `TargetEncoder.discover_sets()` returns category universe per column
- `StandardScaler.discover_sets()` returns `{}` (scalar-only, no sets)
- Static transformer: both `discover()` and `discover_sets()` return `{}`
- Custom transformer with `discover_sets()` returning non-empty → classified as dynamic
- `discover_sets()` result stored in `self.sets_` as `{name: list_of_dicts}`

### Cross-Validation Schema Safety Tests

- OneHotEncoder in 5-fold CV: all folds produce identical output columns
- Synthetic data with rare category (in <20% of data): category column present in ALL folds
- TargetEncoder in CV: all folds have same output columns, different encoded values
- Two-phase discovery: Phase 1 from full data, Phase 2 per fold (verify via query count)
- Self-contained CTE with schema-changing step: output schema matches non-CV fit
- Frozen pipeline from CV: schema includes ALL training categories

### TransformResult Tests

- `transform()` returns `TransformResult`, not raw numpy
- `TransformResult` is NOT a numpy subclass (uses `__array__` protocol instead)
- `TransformResult.shape` matches expected dimensions
- `TransformResult.columns` matches `get_feature_names_out()`
- `TransformResult` works with `model.fit(result, y)` (via `np.asarray()` internally)
- `np.asarray(result)` returns plain numpy array
- `TransformResult[0:10]` slicing works (delegates to internal array)
- `TransformResult.sql` contains valid SQL string
- `TransformResult.dtypes` matches output schema types
- `TransformResult.to_dataframe()` returns pandas DataFrame with correct column names
- `np.concatenate` on TransformResult works (via `__array__` protocol)
- Pickle roundtrip preserves TransformResult metadata

### Freeze Tests

- `pipe.freeze()` returns FrozenPipeline
- `FrozenPipeline.transform()` produces identical output to unfrozen
- `FrozenPipeline.fit()` raises FrozenError
- `FrozenPipeline.sql_` is a pre-compiled SQL string
- `FrozenPipeline.sql_hash_` is deterministic (same pipeline → same hash)
- `FrozenPipeline` validates input schema: missing column → SchemaError
- `FrozenPipeline` validates input schema: extra column → warning (not error)
- `FrozenPipeline` validates input schema: wrong type → SchemaError
- `frozen.save()` + `sq.load_frozen()` roundtrip: identical transform output
- Frozen pipeline with OneHotEncoder: all training categories present even if test data lacks some
- FrozenPipeline serialization includes version metadata

### Drift Detection Tests

- `pipe.detect_drift("new_data")` returns DriftReport
- DriftReport detects mean shift >20% as alert
- DriftReport detects null% increase >5x as alert
- DriftReport detects unseen categories
- DriftReport.passed is True when no alerts
- DriftReport.passed is False when any alert
- Custom thresholds override defaults
- Drift detection on identical data → no alerts, no warnings

### Auto-Passthrough Tests (expressions)

- Custom transformer returning only modified columns → unmodified columns pass through
- Custom transformer adding new column → new column appears in output alongside originals
- Custom transformer returning empty dict → all columns pass through unchanged
- Custom transformer dropping column (via output_schema) → column removed from output
- Base class `_apply_expressions()` merges correctly: modified + passthrough - dropped

### SQL Correctness & Fuzz Tests

- Generate 100 random pipeline configurations → compile → execute → no SQL errors
- Same pipeline compiled to multiple dialects → all parse without error via sqlglot
- Pipeline with 10+ steps → compiled SQL is valid (stress test expression nesting)
- Pipeline with schema-changing steps interleaved with non-schema-changing → correct layers
- Compiled SQL with `to_sql(dialect="postgres")` → valid PostgreSQL syntax
- Compiled SQL with `to_sql(dialect="snowflake")` → valid Snowflake syntax

### FILTER Clause Validation Tests

For every aggregate type used in `discover()`, validate that `FILTER` produces identical
results to computing on a physical subset. This catches any DuckDB edge cases with
`FILTER` semantics for percentiles, MODE, etc.

```python
@pytest.mark.parametrize("agg", [
    "AVG", "MEDIAN", "STDDEV_POP", "MIN", "MAX",
    "PERCENTILE_CONT(0.25)", "PERCENTILE_CONT(0.75)",
    "MODE", "COUNT", "SUM",
])
@pytest.mark.parametrize("fold_size", [3, 5, 10])  # test small and large folds
def test_filter_matches_subset(agg, fold_size, sample_data):
    """FILTER clause must produce identical results to computing on a subset."""
    filtered = f"SELECT {agg}(price) FILTER (WHERE fold != 1) FROM data"
    subset   = f"SELECT {agg}(price) FROM data WHERE fold != 1"
    assert_close(execute(filtered), execute(subset))
```

### Mutation Testing (mutmut)

Tier 3 testing: `make test-full` runs `mutmut` to verify test quality. mutmut generates
code mutations (e.g., `>` → `>=`, `+` → `-`, `True` → `False`) and verifies that tests
detect every mutation. Especially important for SQL generation code where a subtle
operator change could produce valid but incorrect SQL.

```bash
# Run mutation testing on source code
uv run mutmut run --paths-to-mutate=src/sqlearn/

# View surviving mutants (tests that didn't catch the mutation)
uv run mutmut results
```

Focus mutation testing on:
- `compiler.py` — expression composition logic
- `expressions()` methods — arithmetic operators, CASE expressions
- `discover()` methods — aggregate function selection
- Schema change detection logic

### DuckDB CSV Loading Note

When loading CSV data for category discovery tests, use `all_varchar=TRUE` to force all
columns to VARCHAR. This prevents DuckDB's type inference from silently converting
categorical-looking data to integers, which would cause category discovery to miss values.

```python
# BAD: DuckDB might infer "1", "2", "3" as INTEGER, losing category semantics
duckdb.sql("SELECT * FROM 'data.csv'")

# GOOD: All columns loaded as VARCHAR, explicit Cast() where needed
duckdb.sql("SELECT * FROM read_csv('data.csv', all_varchar=TRUE)")
```

### Version Matrix CI Tests

Test against multiple versions of dependencies to catch breaking changes:

```yaml
# .github/workflows/test.yml
jobs:
  test:
    strategy:
      matrix:
        python: ["3.10", "3.11", "3.12", "3.13"]
        duckdb: ["1.0.0", "1.1.0"]      # oldest supported + latest
        sqlglot: ["25.0", "25.8"]        # oldest supported + latest
    steps:
      - run: pip install duckdb==${{ matrix.duckdb }} sqlglot==${{ matrix.sqlglot }}
      - run: pytest
```

sqlglot has breaking changes in minor versions. DuckDB has behavioral changes between
releases. The version matrix catches incompatibilities before users file bugs.

### Performance Benchmark Tests

- `StandardScaler` on 1M rows → completes in <2s (regression guard)
- Pipeline of 5 steps on 1M rows → `fit()` in <5s, `transform()` in <3s
- `AutoFeatures()` on 1M rows × 20 columns → `fit()` in <10s
- `DropCorrelated()` on 50 numeric columns → pairwise correlation in <5s
- Expression composition depth: 10 chained expression-level transforms → one SELECT, zero CTEs
- Memory: `transform()` of 1M rows → peak memory < 2× result size

### Search Tests

- `sq.Search` with `strategy="grid"` matches sklearn `GridSearchCV` results exactly
- `sq.Search` with `strategy="random"` produces valid param combos, scores are reasonable
- Preprocessing dedup: verify same SQL hash → features materialized once (check query count)
- Multi-fidelity rounds: verify correct data sampling, dtype casting, fold count per round
- Early stopping: verify bad trials exit early, good trials complete all folds
- Checkpoint/resume: interrupt mid-search, resume, verify final result matches uninterrupted
- `sq.cross_validate` matches `sklearn.model_selection.cross_val_score` on same data
- `sq.train_test_split` produces correct proportions, stratification preserves class balance
- Feature cache invalidation: different preprocessing params → different cache entries
- Optuna integration: `strategy="optuna"` produces valid results (requires `sqlearn[optuna]`)

### Analysis & Recommendation Tests

- `sq.profile()` returns correct stats (compare to pandas `.describe()`)
- `sq.analyze()` detects multicollinearity on synthetic correlated columns
- `sq.analyze()` detects skewness correctly (compare to scipy.stats.skew)
- `sq.recommend()` suggests scaling for linear models, not for tree models
- `sq.recommend()` suggests HashEncoder for high-cardinality, OneHot for low
- `sq.recommend()` suggests `OutlierHandler` for columns with extreme outliers
- `sq.recommend()` suggests `AutoDatetime` for datetime columns
- `sq.recommend()` suggests `StringSplit` when delimiter detected
- `sq.autopipeline()` generates a Pipeline that fits and transforms without error
- `sq.autopipeline()` includes `DropConstant` when constant columns detected
- `sq.autopipeline()` includes `AutoDatetime` when datetime columns present
- `sq.autopipeline()` includes `AutoSplit` when delimited strings detected
- `sq.autopipeline(as_code=True)` outputs valid Python that `exec()` can run
- Model auto-detection: XGBClassifier → "tree_based", LogisticRegression → "linear"
- Feature suggestions: datetime column → DateParts suggested, skewed numeric → Log suggested

### Studio Tests (`sqlearn[studio]`)

**Free Tier Tests:**
- `sq.studio()` starts Starlette/uvicorn server on available port
- Profile API returns correct stats matching `sq.profile()` output
- Analyze API returns correlations matching `sq.analyze()` output
- Pre-aggregation: histogram endpoint returns ~30 bins, not raw data
- Pre-aggregation: scatter endpoint returns ≤1000 sampled points
- Static frontend files are present and served correctly
- Session management: pipeline state persists across API calls
- Free tier: basic charts render with default styling
- Free tier: data table loads, sorts, filters, paginates
- Free tier: help hints render markdown correctly

**License Gating Tests:**
- Without license: Pro endpoints return `ProFeatureError` with upgrade message
- Without license: frontend shows `<ProGate>` component for locked features
- `sq.activate("valid-key")` unlocks Pro features in same session
- `sq.activate("expired-key")` raises with clear expiry message
- `sq.trial()` starts 14-day trial, Pro features available immediately
- Trial expiry: after 14 days, Pro features re-lock gracefully
- License state persisted in `~/.sqlearn/license.key`
- No internet required for license validation (offline RSA)

### sklearn Compatibility Tests

- `get_params(deep=True)` returns all `__init__` params as dict
- `set_params(**params)` correctly sets params, including nested `__` notation
- `__sklearn_is_fitted__()` returns correct boolean for sklearn's `check_is_fitted()`
- `clone()` produces unfitted copy with same constructor args
- Serialization roundtrip: `pickle.dumps(pipe)` + `pickle.loads()` preserves fitted state
- Serialization nulls connection: `pickle.loads(pickled).connection_ is None`
- joblib roundtrip: `joblib.dump(pipe)` + `joblib.load()` preserves categories and params

### Test Fixture

Standard dataset in `tests/fixtures/`:
- 1000 rows, 20 columns:
  - 5 numeric (including 1 skewed right-skew>2, 1 with outliers, 1 constant)
  - 3 categorical (1 low-cardinality <10, 1 medium 50-100, 1 high >200)
  - 2 datetime (1 DATE, 1 TIMESTAMP with time component)
  - 1 comma-separated string (tags: "a,b,c" style)
  - 1 JSON string (metadata: '{"key": "value"}' style)
  - 1 email column
  - 1 URL column
  - 1 IP address column
  - 1 boolean column
  - 1 ID column (monotonic integer, unique per row)
  - 1 target column (continuous, right-skewed — for regression tests)
- Known NULLs at controlled positions (verifiable null percentages per column)
- Known distributions (verifiable means/stds/medians)
- Known correlations (synthetic columns with controlled r values)
- Known pairwise high-correlation pair (r>0.95 between two numeric columns)
- Stored as Parquet

Additional fixtures:
- `tests/fixtures/second_source.parquet` — 500 rows with `user_id` column for merge tests
- `tests/fixtures/lookup_table.parquet` — 50 rows for `sq.Lookup` tests
- `tests/fixtures/2023.parquet`, `tests/fixtures/2024.parquet` — for concat tests
- `tests/fixtures/large_sample.parquet` — 100K rows for performance benchmarks

### Real Dataset Integration Tests

Include a small real-world dataset (e.g., iris, tips, or similar public domain data) as a
test fixture for end-to-end pipeline tests. Synthetic fixtures catch unit-level issues, but
real data catches integration problems that minimal synthetic data misses:

- Full pipeline end-to-end on realistic data distributions
- Column type inference on real-world messy types
- Edge cases that only appear in production data (mixed nulls, rare categories, etc.)
- Regression guard: output shape and values should not change between releases

### Pro Studio Tests (license-gated, same repo)

**Pro Feature Tests (require active license in test):**
- Pipeline builder: drag-and-drop reorder, add/remove steps via API
- Live code preview: endpoint returns valid Python matching pipeline state
- autopipeline in UI: POST /api/builder/auto → returns pipeline in builder format
- WebSocket search endpoint streams trial results in real-time
- Export endpoint generates valid, executable Python code
- Chart customization: theme switching, color changes persist
- Chart export: PNG/SVG generation produces valid image files
- Chart → Plotly code: generated code produces valid Plotly figure
- Column deep-dive: click endpoint returns full column analysis
- Multiple data sources: upload second file → appears in source manager
- Data source manager: drag CSV → appears in DuckDB → browseable

**Project Generation Tests (Pro):**
- Project generation: all 7 files created, all valid Python (`exec()` works)
- Generated `requirements.txt` includes correct packages
- Generated `config.yaml` contains all parameters from session
- Plotly chart code generates valid figures (`.show()` doesn't error)
- Marimo notebook export: valid `.py` with `@app.cell` decorators
- Jupyter notebook export: valid `.ipynb` that `nbformat.validate()` passes

**Session Persistence Tests (Pro):**
- Session save/resume: save state → reload → all pipeline/search state intact
- Session history: all actions recorded, undo restores previous state
- Model fitting from UI: train → results match training in Python script

---
