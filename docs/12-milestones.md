> **sqlearn docs** — [Index](../CLAUDE.md) | Prev: [Business Model](11-business-model.md) | Next: [Testing Strategy](13-testing.md)

## 13. Implementation Milestones

Nine milestones. Each is independently shippable. Each builds on the previous.
All milestones ship from one repo. Studio Pro features are license-gated, not
in a separate codebase.

**Realistic timeline: 60-70 weeks** (not 40). The original 40-week estimate assumed
zero friction. Real-world factors: debugging edge cases in SQL generation across dialects,
cross-validation schema safety, expression composition depth limits, and the Studio
frontend require more time than greenfield coding. Each milestone includes buffer for
unexpected complexity.

**Priority: Milestones 1-4 are critical.** The core compiler is the make-or-break.
If expression composition at depth 5+, CTE promotion, and the discover→params_ flow
don't work reliably, nothing else matters. Ship v0.1.0 fast. Get real users. File real
bugs. Then build on a validated foundation.

### Milestone 1 — Scaffolding (Week 1-2)

**Goal:** Project skeleton that builds, tests, and lints.

- [ ] Initialize git repository
- [ ] `pyproject.toml` with `src/sqlearn/` layout
- [ ] CI (GitHub Actions: lint + test, Python 3.10-3.13)
- [ ] Pre-commit hooks (ruff format, ruff check, mypy)
- [ ] `CLAUDE.md` with project conventions
- [ ] Empty test suite that passes
- [ ] `.gitignore`, `LICENSE` (MIT), `README.md` (minimal)

**Ship:** `pip install -e .` works, `pytest` runs, CI green.

### Milestone 2 — Core Compiler (Week 3-7)

**Goal:** Prove that compiling ML pipelines to SQL works.

**Week 2 — Engine:**
- [ ] `schema.py` — Schema dataclass, column type detection
- [ ] `transformer.py` — Transformer base class (discover/expressions/output_schema)
  - [ ] `y` as column name string, not numpy array (Decision #93)
  - [ ] `discover(columns, schema, y_column=None)` signature
  - [ ] Target column(s) excluded from `transform()` output by default (`exclude_target=True`)
  - [ ] `exclude_target=False` parameter to include target in output (EDA, debugging)
  - [ ] `_check_thread()` guard + `clone()` method (Decision #92)
  - [ ] `__add__` / `__iadd__` operators for Pipeline composition (Decision #90)
- [ ] `backend.py` — Backend protocol + DuckDB implementation
- [ ] `io.py` — input resolver (table/file/DataFrame → DuckDB)
- [ ] `errors.py` — error hierarchy (`SQLearnError`, `SchemaError`, `FitError`, etc.)
  - [ ] `discover_sets()` method for multi-row discovery (OneHotEncoder, TargetEncoder)
  - [ ] `_apply_expressions()` auto-passthrough wrapper
  - [ ] `TransformResult` wrapper with `__array__` protocol + metadata (columns, dtypes, sql)

**Week 3 — Compiler + Pipeline:**
- [ ] `compiler.py` — expression composition, CTE promotion
- [ ] `pipeline.py` — Pipeline: fit, transform, fit_transform, to_sql

**Week 4 — Three Transformers + Tests:**
- [ ] `Imputer` — proves aggregation discovery, COALESCE composition
- [ ] `StandardScaler` — proves arithmetic composition on top of Imputer
- [ ] `OneHotEncoder` — proves layer boundaries, schema change, CASE generation
  - [ ] OneHotEncoder: `discover_sets()` for distinct categories (proves multi-row discovery)
- [ ] `output.py` — DuckDB result → numpy
- [ ] Integration tests: pipeline output matches sklearn (`np.allclose`)
- [ ] SQL snapshot tests: compiled SQL matches expectations

**Ship `v0.1.0`:**
```python
pipe = sq.Pipeline([sq.Imputer(), sq.StandardScaler(), sq.OneHotEncoder()])
pipe.fit("train.parquet", y="target")   # y is a column name (Decision #93)
X = pipe.transform("test.parquet")      # numpy array, target column auto-excluded
sql = pipe.to_sql()                      # valid DuckDB SQL
assert np.allclose(X, X_sklearn)

# Operators work:
pipe2 = sq.Imputer() + sq.StandardScaler()   # Decision #90
pipe2 += sq.OneHotEncoder()
```

### Milestone 3 — Composition + Breadth (Week 8-14)

**Goal:** Handle real-world pipeline complexity, operators, data operations.

- [ ] `Columns` (replaces ColumnTransformer) with column selectors
- [ ] `Union` (replaces FeatureUnion)
- [ ] Auto column routing: `sq.numeric()`, `sq.categorical()`, `sq.temporal()`, `sq.matching()`
- [ ] `+` operator with flattening, `+=` for incremental building (Decision #90)
- [ ] `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler`, `Normalizer` (via `query()`)
- [ ] `OrdinalEncoder`, `HashEncoder`, `FrequencyEncoder`
- [ ] Schema tracking through full pipeline
- [ ] Error hierarchy with helpful messages
- [ ] Thread safety guard: `_check_thread()`, `clone()` (Decision #92)
- [ ] `get_feature_names_out()`
- [ ] Output formats: arrow, pandas, polars (`out=` parameter)
- [ ] Large result size warning (Decision #91)
- [ ] Pipeline nesting tests
- [ ] `pipe.validate()`, `pipe.describe()`
- [ ] Data operations: `Rename`, `Cast`, `Filter`, `Sample`, `Deduplicate`, `Reorder`
- [ ] Data merging: `sq.merge()`, `sq.concat()` pre-pipeline helpers
- [ ] `sq.Lookup` transformer (mid-pipeline join via `query()`)
- [ ] Feature dropping: `Drop(columns=)`, `Drop(pattern=)`, `Drop(dtype=)`
- [ ] `DropConstant()` — auto-detect and remove constant/all-NULL columns
- [ ] `Columns` compilation: parallel column routing → single SELECT
- [ ] `Union` compilation: parallel features → CTE per branch or merged SELECT
- [ ] Union column name collision resolution (branch prefix)
- [ ] `y` column propagation through Columns and Union branches
- [ ] Non-mutating `+=` operator (creates new Pipeline, no mutation)

**Ship `v0.2.0`:** Handles 90% of real sklearn preprocessing pipelines. Full data
assembly (merge/concat) and cleanup (rename/cast/filter/drop) built in.

### Milestone 4 — Feature Engineering + Selection (Week 15-22)

**Goal:** Feature creation, selection, outlier handling, and auto-features. Extended
to 6 weeks because this milestone now covers significantly more ground.

**Week 9-10 — Core Feature Engineering:**
- [ ] Arithmetic: `Log`, `Sqrt`, `Power`, `Clip`, `Abs`, `Round`, `Sign`
- [ ] Column math: `Add`, `Multiply`, `Ratio`, `Diff`, `Modulo`
- [ ] String basics: `StringLength`, `StringLower`, `StringUpper`, `StringTrim`
- [ ] String advanced: `StringContains`, `StringReplace`, `StringExtract`
- [ ] `StringSplit(by=, max_parts=)` with `max_parts="auto"` (learns from data)
- [ ] Structured extractors: `JsonExtract`, `URLParts`, `EmailParts`, `IPParts`
- [ ] Datetime: `DateParts`, `DateDiff`, `IsWeekend`, `IsHoliday`, `TimeSinceEvent`
- [ ] `CyclicEncode(period=)` — sin/cos for periodic features (hour, day, month)
- [ ] `Expression` (raw SQL escape hatch, parsed through sqlglot)
- [ ] `PolynomialFeatures`
- [ ] Batch processing: `batch_size=` parameter
- [ ] `query()` interface fully tested with CTE promotion

**Week 11-12 — Window, Outliers, Target Transforms:**
- [ ] Window: `Lag`, `Lead`, `RollingMean`, `RollingStd`, `RollingMin`, `RollingMax`, `RollingSum`
- [ ] Advanced window: `EWM`, `Rank`, `PercentRank`, `CumSum`, `CumMax`
- [ ] `GroupFeatures` (window-based, preserves rows)
- [ ] `OutlierHandler(method=, action=)` — IQR/zscore/percentile → clip/null/flag/drop
- [ ] `OutlierHandler` per-column config via dict API
- [ ] `TargetTransform(func=)` — log/sqrt/standard/minmax/quantile with auto-inverse
- [ ] `TargetTransform` integration with `ModelPipeline` (inverse at predict time)
- [ ] `TargetEncoder` with `y_column` propagation (Decision #93)
- [ ] `AutoEncoder`, `BinaryEncoder`

**Week 13-14 — Feature Selection + Auto Features:**
- [ ] `DropCorrelated(threshold=)` — pairwise correlation → drop redundant
- [ ] `DropLowVariance(threshold=)`, `VarianceThreshold`
- [ ] `DropHighNull(threshold=)`, `DropHighCardinality(threshold=)`
- [ ] `SelectKBest(k=, method=)` — top K by correlation/MI/ANOVA (uses `y_column`)
- [ ] `SelectByName(pattern=)` — keep columns matching glob/regex
- [ ] `AutoDatetime(granularity=)` — type-aware auto-expansion with per-column override
- [ ] `AutoSplit()` — auto-detect delimiter, learn max_parts from data
- [ ] `AutoNumeric()` — skew→log, range→clip, distribution→bins
- [ ] `AutoCategorical()` — same as `AutoEncoder` (alias)
- [ ] `AutoFeatures()` — one-liner that chains all auto-* transforms
- [ ] `AutoFeatures` fine-grained control: `datetime=`, `split_strings=`, `numeric=`, etc.
- [ ] `AutoFeatures.to_explicit()` — freeze auto-decisions into explicit pipeline

**Ship `v0.3.0`:** Complete feature engineering toolkit with auto-detection, feature
selection, outlier handling, structured data extraction, and target transforms.

### Milestone 5 — Model Selection + Search (Week 23-28)

**Goal:** Best-in-class hyperparameter search with SQL-native optimizations.

- [ ] CV splits: `KFold`, `StratifiedKFold`, `GroupKFold`, `TimeSeriesSplit`
- [ ] `sq.train_test_split` — SQL views, no data copy, stratification
- [ ] SQL-based metrics: accuracy, f1, rmse, mae, auc, r2
- [ ] `sq.cross_validate` — standalone CV helper
- [ ] `sq.read_column` — target extraction without full data load
- [ ] `sq.Search` — unified search with pluggable strategies
- [ ] Built-in samplers: random, grid, Sobol, TPE
- [ ] Parameter spaces: `IntRange`, `FloatRange`, `LogRange`, `Choice`, `Fixed`
- [ ] Preprocessing SQL dedup via AST hashing (`cache.py`)
- [ ] Multi-fidelity rounds: `fast_explore=True`, manual `rounds=[]`
- [ ] `explore_dtype` / `final_dtype` precision control
- [ ] `USING SAMPLE` for data subsampling in exploration rounds
- [ ] Early stopping, checkpoint/resume, `n_jobs=`
- [ ] `search.summary()`, `search.importance()`, `search.suggest()`
- [ ] Optuna adapter (`strategy="optuna"`)
- [ ] Two-phase discovery for CV: Phase 1 (schema from full data) + Phase 2 (stats per fold)
- [ ] `ModelPipeline` full implementation (fit/predict/predict_proba/score/save/load)
- [ ] Search error recovery: on_error policy, failed_trials_, consecutive error detection
- [ ] Search failure_summary() for debugging bad parameter spaces

**Ship `v0.4.0`:** `sq.Search` replaces GridSearchCV/Optuna for preprocessing + model tuning.

### Milestone 6 — Analysis & Recommendations (Week 29-34)

**Goal:** Data-driven preprocessing recommendations.

- [ ] `sq.profile()` — types, nulls, cardinality, stats, distributions (single SQL query)
- [ ] `sq.analyze()` — correlations, multicollinearity, skewness, target-aware
- [ ] `sq.recommend()` — model + pipeline suggestions via `MODEL_PROFILES`
- [ ] `FEATURE_SUGGESTIONS` — column type → feature engineering ideas
- [ ] `sq.autopipeline()` — generate Pipeline from analysis (`as_code=True`)
- [ ] Auto-detect model family from class name
- [ ] `sq.drift()` — distribution comparison between datasets
- [ ] `sq.correlations()`, `sq.missing()`, `sq.missing_patterns()`
- [ ] `pipe.detect_drift()` — pipeline-integrated drift detection
- [ ] `pipe.validate()` — schema + distribution validation against training stats

**Ship `v0.5.0`:** Smart analysis that tells users what to do.

### Milestone 7 — Export + Polish (Week 35-40)

**Goal:** Production-ready, multi-database, documented.

- [ ] `to_sql(dialect=)` — DuckDB, Postgres, Snowflake, BigQuery, Spark
- [ ] `to_config()` / `from_config()` (YAML)
- [ ] `save()` / `load()` (binary)
- [ ] `export()` (standalone Python)
- [ ] `to_dbt()` (dbt model)
- [ ] `inverse_transform()` for invertible transforms
- [ ] Benchmarks vs sklearn (speed + memory)
- [ ] Documentation site (mkdocs or similar)
- [ ] Tutorials and examples
- [ ] PyPI release
- [ ] Pipeline versioning: format_version, sql_hash, schema_in/schema_out in save format
- [ ] `pipe.freeze()` → FrozenPipeline (immutable, pre-compiled, schema-validated)
- [ ] `sq.load_frozen()` for production deployment
- [ ] YAML config with fitted params + schema (version-controllable)
- [ ] SQL injection documentation and security model

**Ship `v0.6.0` → `v1.0.0`:** Production-ready library. Stable API.


### Milestone 8 — Studio Free + Pro Foundation (Week 41-52)

**Goal:** Interactive EDA dashboard with free tier and Pro license gating.
All in one package (`sqlearn[studio]`). Pro features are built alongside
free features but gated by license check.

**Week 41-43 — Backend + Infrastructure:**
- [ ] `sq.studio()` entry point — Starlette server, browser auto-open
- [ ] API endpoints: profile, analyze, recommend, transform
- [ ] Pre-aggregation layer (SQL → tiny JSON for frontend)
- [ ] Session management (DuckDB connection, pipeline state)
- [ ] File upload endpoint (drag-drop CSV/Parquet — single file, free tier)
- [ ] orjson response serialization
- [ ] License module: `studio/license.py` — RSA signature check, trial logic
- [ ] `sq.activate("SQ-XXXX")` and `sq.trial()` entry points
- [ ] License state exposed to frontend via `/api/license` endpoint
- [ ] `ProFeatureError` exception for gated endpoints
- [ ] `require_pro()` decorator for API endpoint gating

**Week 44-46 — Frontend Core (Free Tier):**
- [ ] Vite + Svelte 5 + Tailwind CSS v4 project setup
- [ ] shadcn-svelte components (dialog, dropdown, tooltip, tabs, sheet, context-menu)
- [ ] Profile view: distributions, null map, warnings, column cards
- [ ] Analysis view: correlation heatmap, target analysis, suggestion sidebar
- [ ] Recommendations view: model + pipeline suggestions (text, read-only)
- [ ] DataTable component (TanStack Table, SQL-powered sort/filter/paginate)
- [ ] Basic charts: histogram, scatter, box, bar (default styling, no customization)
- [ ] ECharts integration: default themes only, no DataZoom/Brush/toolbox (Pro)
- [ ] Help hints (❓) on every transformer, every option (Floating UI + marked)
- [ ] Keyboard shortcuts (tinykeys), toasts (svelte-sonner), dark mode
- [ ] `<ProGate>` component: renders lock overlay + upgrade prompt for Pro features

**Week 47-49 — Pro Features — Builder + Charts:**
- [ ] Pipeline builder: drag-and-drop (svelte-dnd-action), suggestions, reorder
- [ ] Live code preview (highlight.js Python highlighting)
- [ ] autopipeline in UI: one-click analyze → generate → show in builder
- [ ] Full pipeline code export endpoint (`/api/export/pipeline`)
- [ ] Chart customization: color picker, column selector, theme switching
- [ ] Chart export: PNG/SVG via html-to-image
- [ ] DataZoom, Brush selection, toolbox (save image, zoom, data view)
- [ ] Scatter plot reproducibility: USING SAMPLE REPEATABLE(seed)
- [ ] ECharts registerTheme() for runtime theme switching (Pro only)
- [ ] Data preview: transformed output (SQL LIMIT)
- [ ] All builder/chart features gated behind `require_pro()` / `$license.pro`

**Week 50-52 — Pro Features — Search Monitor + Advanced:**
- [ ] WebSocket for live Search progress (`/api/search` WebSocket)
- [ ] Search view: convergence chart (uPlot), param importance (ECharts)
- [ ] Column deep-dive: click column → full stats, distribution, outliers
- [ ] Multiple data sources: additional file uploads, database connections
- [ ] Data source manager: sidebar showing all loaded datasets
- [ ] Advanced analysis: feature importance ranking, class balance
- [ ] All advanced features gated behind `require_pro()` / `$license.pro`

**Ship `v1.1.0` (sqlearn[studio]):** Free EDA dashboard + Pro builder (license-gated).

### Milestone 9 — Studio Pro Completion (Week 53-65)

**Goal:** Project generation, notebook export, model fitting, session persistence.
All features ship in the same `sqlearn[studio]` package, gated by license key.

**Week 53-55 — Project Generation Engine:**
- [ ] Project scaffolder: generate directory structure from template
- [ ] Step-by-step workflow engine (Data → EDA → Clean → Features → Model → Evaluate → Export)
- [ ] Progress bar UI: visual step tracker across top of dashboard
- [ ] Per-step Python file generation with clean imports, comments, docstrings
- [ ] `config.yaml` generation (all parameters centralized)
- [ ] `requirements.txt` generation
- [ ] Export endpoint: `/api/export/project` (Pro-gated)

**Week 56-58 — Model Fitting + Chart Code Export:**
- [ ] Fit models directly from UI (select model, configure, click Train)
- [ ] Training progress bar (callbacks: sklearn, XGBoost, LightGBM)
- [ ] Evaluation view: metrics, confusion matrix, ROC curve, feature importance
- [ ] Chart → Plotly code generation (ECharts renders in browser, Plotly goes to files)
- [ ] Export endpoint: `/api/export/chart-code` (Pro-gated)

**Week 59-62 — Notebook Export + Session Persistence:**
- [ ] Export to Marimo notebook (`.py` with `@app.cell` decorators)
- [ ] Export to Jupyter notebook (`.ipynb` via nbformat)
- [ ] Export to plain Python script
- [ ] Session save/resume: persist all state to disk (DuckDB file)
- [ ] Session history: timeline of all actions, undo, export as script
- [ ] License activation UI in settings panel

**Week 63-65 — Polish + Trial Flow:**
- [ ] 14-day trial flow: `sq.trial()` → full Pro for 14 days → graceful downgrade
- [ ] Trial expiry UI: countdown, upgrade prompts, feature comparison
- [ ] Pro upgrade flow within Studio (enter key, verify, unlock)
- [ ] End-to-end testing of all Pro features
- [ ] Performance optimization: lazy-load Pro components only when licensed
- [ ] Documentation: Pro feature guide, trial guide, license FAQ

**Generated project example (Pro feature):**

```python
# 02_eda.py (generated by sqlearn Studio Pro)
"""
Exploratory Data Analysis
Generated by sqlearn Studio Pro from interactive session.
Dataset: housing_data.parquet (50,000 rows, 15 columns)
"""
import sqlearn as sq
import plotly.express as px

# Profile
profile = sq.profile("housing_data.parquet")
print(profile)

# Target: price (regression)
analysis = sq.analyze("housing_data.parquet", target="price")

# Correlation with target
fig = px.bar(
    x=analysis.correlations.values,
    y=analysis.correlations.index,
    orientation="h",
    title="Correlation with price",
)
fig.show()

# Distribution of target (right-skewed → consider Log transform)
fig = px.histogram(
    sq.read_column("housing_data.parquet", "price"),
    nbins=50,
    title="price distribution (skew=2.3)",
)
fig.show()
```

```python
# 04_feature_engineering.py (generated by sqlearn Studio Pro)
"""
Feature Engineering Pipeline
Generated by sqlearn Studio Pro from interactive session.
"""
import sqlearn as sq

pipe = sq.Pipeline([
    sq.Imputer({"age": "median", "score": "median"}),
    sq.DateParts(columns=["created_at"], parts=["month", "dayofweek"]),
    sq.OneHotEncoder(columns=["city", "type"]),
    sq.HashEncoder(columns=["zip_code"], n_bins=64),
    sq.Log(columns=["price"]),  # target transform (skew correction)
    sq.StandardScaler(),
])

pipe.fit("housing_data.parquet")
X_train = pipe.transform("housing_data.parquet")
print(f"Shape: {X_train.shape}")
print(f"Columns: {pipe.get_feature_names_out()}")

# SQL equivalent (for deployment):
# pipe.to_sql(dialect="postgres")
```

**Ship `v1.2.0` (sqlearn[studio]):** Complete Studio with all Pro features. First revenue.

### Release Strategy Summary

| Version | Tier | Content | Week | Price |
|---|---|---|---|---|
| `0.1.0` | Library | Pipeline + 3 transformers + DuckDB | 1-7 | Free |
| `0.2.0` | Library | Composition + data ops + merge + drop | 8-14 | Free |
| `0.3.0` | Library | Feature engineering + selection + auto-features + outliers | 15-22 | Free |
| `0.4.0` | Library | Model selection + sq.Search | 23-28 | Free |
| `0.5.0` | Library | Analysis + recommendations + drift detection | 29-34 | Free |
| `0.6.0` → `1.0.0` | Library | Export + freeze + polish + stable API | 35-40 | Free |
| `1.1.0` | Studio Free + Pro (gated) | Free EDA + Pro builder/charts (license-gated) | 41-52 | Free / $49-99 |
| `1.2.0` | Studio Pro (complete) | Project generation + notebooks + model fitting | 53-65 | $49-99 |

### What Each Version Delivers

**v0.1.0 — Core Proof (Week 7):**
```python
pipe = sq.Pipeline([sq.Imputer(), sq.StandardScaler(), sq.OneHotEncoder()])
pipe.fit("train.parquet", y="target")
X = pipe.transform("test.parquet")   # numpy array
sql = pipe.to_sql()                   # valid DuckDB SQL
```
Validates: expression composition, CTE promotion, discover→params_ flow.

**v0.2.0 — Real Pipelines (Week 14):**
Handles 90% of real sklearn preprocessing pipelines. Columns, Union, data ops,
merge/concat, drop. Users can replace sklearn preprocessing in real projects.

**v0.3.0 — Feature Engineering (Week 22):**
Complete feature engineering: datetime, string, window, outliers, auto-features.
Feature selection: drop correlated, variance threshold, SelectKBest.
Users can do everything sklearn can't (structured extraction, auto-detection).

**v0.4.0 — Search (Week 28):**
`sq.Search` replaces GridSearchCV/Optuna. SQL-native dedup, multi-fidelity,
preprocessing caching. Unique value proposition no other tool can match.

**v0.5.0 — Intelligence (Week 34):**
`sq.profile()`, `sq.analyze()`, `sq.recommend()`, `sq.autopipeline()`.
The library tells users what to do. Blog post material. Conference talk material.

**v1.0.0 — Production Ready (Week 40):**
Multi-database export, frozen pipelines, documentation site, PyPI release.
Stable API. The library is complete and production-ready.

**v1.1.0 — Studio Launch (Week 52):**
Free Studio: profile, analysis, recommendations, data table, basic charts.
Pro Studio (license-gated): pipeline builder, chart customization, search monitor.
First Pro revenue opportunity via 14-day trial → conversion.

**v1.2.0 — Studio Complete (Week 65):**
Pro Studio complete: project generation, notebook export, model fitting,
session persistence, guided workflow. Full revenue product.

---

## 13a. Non-Goals

Explicitly what sqlearn will NOT do. This prevents scope creep and helps contributors
understand boundaries:

- **No model training** — sqlearn handles 100% of preprocessing. Models (gradient descent,
  tree splitting, backpropagation) are iterative algorithms not expressible as SQL.
- **No distributed execution** — DuckDB is single-node. For distributed, export SQL to
  Spark/BigQuery via `to_sql(dialect=)`.
- **No real-time / streaming preprocessing** — sqlearn is batch-oriented. For streaming,
  export SQL and run it in your streaming engine.
- **No OLAP query optimization** — sqlearn generates SQL; the database optimizes it.
- **No multi-tenant Studio** — Studio is a local tool on `127.0.0.1`.
- **No data versioning** — use DVC, lakeFS, or Delta Lake.
- **No experiment tracking** — use MLflow, W&B, or similar.
- **No GPU acceleration** — DuckDB is CPU-based. GPU would be a different engine.
- **No custom aggregate functions** — use DuckDB UDFs directly if needed.

---

## 13b. Logging Strategy

All logging uses Python's standard `logging` module with hierarchical loggers:

```python
import logging

# Root logger
logger = logging.getLogger("sqlearn")

# Sub-loggers for granularity:
# sqlearn.compiler   — SQL generation, CTE decisions, expression depth
# sqlearn.fit        — classification, aggregation queries, layer resolution
# sqlearn.transform  — query execution, result sizes, warnings
# sqlearn.cache      — feature cache hits/misses (for Search)
# sqlearn.classify   — tier decisions, discover() calls, verification results
# sqlearn.io         — input resolution, DataFrame registration, file reading
```

Users control verbosity:

```python
import logging

# See everything during debugging:
logging.getLogger("sqlearn").setLevel(logging.DEBUG)

# Just classification decisions:
logging.getLogger("sqlearn.classify").setLevel(logging.DEBUG)

# Just SQL queries:
logging.getLogger("sqlearn.fit").setLevel(logging.DEBUG)
```

During `fit()`, the logger emits:

```
DEBUG sqlearn.classify: Step 1/5 Imputer — Tier 1 (built-in), dynamic, trusted
DEBUG sqlearn.classify: Step 2/5 StandardScaler — Tier 1 (built-in), dynamic, trusted
DEBUG sqlearn.fit: Layer 0 fit query: SELECT MEDIAN(price) AS ..., AVG(price) AS ...
DEBUG sqlearn.fit: Layer 0 sets query: SELECT DISTINCT city FROM ...
DEBUG sqlearn.fit: Learned: imputer__price__median=42.5, scaler__price__mean=42.5
DEBUG sqlearn.fit: OneHotEncoder categories: {city: [London, Paris, Berlin]}
```

---

## 13c. Reproducibility Guarantees

| Operation | Deterministic? | Conditions |
|---|---|---|
| `fit()` + `transform()` | Yes | Same DuckDB version, same data, same sqlearn version |
| `to_sql()` | Yes | Same sqlglot version |
| Fold assignment via `HASH()` | Yes | Same DuckDB version, same `random_state` |
| `Sample(seed=42)` | Yes | Same DuckDB version |
| `AutoFeatures()` | No | Threshold changes between sqlearn versions may change decisions. Use `to_explicit()` for production. |

**Cross-version considerations:**
- DuckDB's `HASH()` function may produce different values across major versions.
  Fold assignments created with one DuckDB version may differ on another.
  Pin DuckDB version in production (`duckdb==1.x.y` in requirements).
- sqlglot AST generation may change across versions, producing semantically equivalent
  but textually different SQL. Pin sqlglot in production.
- sqlearn will document any changes that affect output determinism in release notes.

**Recommendation:** For production pipelines, use `pipe.freeze()` which embeds the
SQL string, sql_hash, sqlearn_version, and schema. This makes the pipeline immune to
sqlglot/DuckDB version changes (the frozen SQL is a fixed string).

---

## 13d. Graceful Degradation for Unsupported SQL Features

When `to_sql(dialect=)` targets a database that doesn't support a feature, the
compiler follows this strategy:

| Unsupported feature | Fallback | Example |
|---|---|---|
| `FILTER` clause | `CASE WHEN ... THEN col END` inside aggregate | MySQL, Snowflake |
| `MEDIAN` / `PERCENTILE_CONT` | Approximate or error | MySQL: error with helpful message |
| `HASH(x)` | `MD5(x)` or `CRC32(x)` | Postgres: MD5, MySQL: CRC32 |
| Read Parquet natively | Error: "Export data first" | MySQL, Postgres |
| `USING SAMPLE` | Error: "Not supported" | Some dialects |

**Strategy per unsupported feature:**

```python
pipe.to_sql(dialect="mysql")
# If any transformer uses PERCENTILE_CONT and MySQL can't express it:
# → CompilationError: "RobustScaler uses PERCENTILE_CONT which is not
#   available in MySQL. Options:
#   1. Use a different scaler (StandardScaler, MinMaxScaler)
#   2. Export to a dialect that supports it (DuckDB, Postgres, Snowflake)
#   3. Use strict=False to skip unsupported steps"

pipe.to_sql(dialect="mysql", strict=False)
# → Compiles what it can, skips what it can't, emits warnings per skipped step
```

**Dialect compatibility is documented per transformer.** Each transformer's docstring
lists which dialects it supports. `pipe.to_sql(dialect=X)` validates all steps before
generating SQL.

---

## 13e. `discover_sets()` Execution Model

`discover_sets()` queries are executed **separately from `discover()` aggregates** because
they return multiple rows (not scalar values). Execution order and optimization:

1. **All `discover()` aggregates are batched into ONE query** per layer.
   Multiple steps' `discover()` results are merged into a single SELECT.

2. **Each `discover_sets()` entry becomes a separate query.**
   OneHotEncoder targeting 5 columns = 5 separate `SELECT DISTINCT` queries.

3. **Optimization: `UNION ALL` for same-type set queries.**
   If multiple columns need `SELECT DISTINCT`, the compiler MAY batch them:
   ```sql
   SELECT 'city' AS __sq_col__, city AS __sq_val__ FROM data
   UNION ALL
   SELECT 'state' AS __sq_col__, state AS __sq_val__ FROM data
   ```
   This trades one query for a wider result. Applied when >3 columns need DISTINCT.

4. **Execution order: all `discover()` first, then all `discover_sets()`.**
   This allows `discover_sets()` results to be informed by layer resolution
   (which depends on `discover()` classification).

5. **`discover_sets()` results interact with layers:** If a step has both `discover()`
   and `discover_sets()`, both are executed in the step's layer. Set results are stored
   in `self.sets_` (separate from `self.params_`).

---

