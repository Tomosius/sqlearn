> **sqlearn docs** — [Index](../CLAUDE.md) | Prev: [Error Handling](09-error-handling.md) | Next: [Business Model](11-business-model.md)

## 11. Project Structure

Modular by category. Each transformer type gets its own file. Easy to extend —
adding a new scaler means adding one file in `scalers/`, no touching other code.

```
sqlearn/
├── __init__.py               # public API — exports everything users need
├── core/
│   ├── __init__.py
│   ├── transformer.py        # Transformer base class (the ONE base class)
│   ├── step_info.py          # StepInfo dataclass (static/dynamic, fitted, layer, schemas)
│   ├── pipeline.py           # Pipeline
│   ├── union.py              # Union (replaces FeatureUnion)
│   ├── columns.py            # Columns (replaces ColumnTransformer)
│   ├── compiler.py           # Expression composition, CTE promotion, layer resolution
│   ├── backend.py            # Backend protocol + DuckDB implementation (Postgres etc. later)
│   ├── schema.py             # Schema dataclass, column selectors, __sq_*__ validation
│   ├── io.py                 # Input resolution (table/file/df → DuckDB table name)
│   ├── output.py             # DuckDB result → numpy/arrow/pandas/polars/sparse
│   ├── custom.py             # sq.custom() — template-based transformer factory + validation
│   ├── helpers.py            # sq.read_column, sq.attach, sq.set_backend
│   └── errors.py             # Error hierarchy
│
├── stats/                    # SQL building blocks — reused by ALL transformers
│   ├── __init__.py           # exports all aggregates + statistical functions
│   ├── aggregates.py         # Mean, Median, Mode, Min, Max, Std, Var, Quantile, Sum, Count
│   │                         #   each has .expression(col) and .as_cte(col, source)
│   │                         #   transformers compose from these — DRY, one source of truth
│   ├── correlations.py       # pearson, spearman, kendall, point_biserial, cramers_v, mutual_info
│   ├── tests.py              # chi_squared, anova, normality (Jarque-Bera), mcar
│   └── missing.py            # null_counts, null_patterns, mcar_test
│
├── scalers/                  # one file per scaler — easy to add new ones
│   ├── __init__.py           # exports: StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
│   ├── standard.py           # uses stats.Mean + stats.Std
│   ├── minmax.py             # uses stats.Min + stats.Max
│   ├── robust.py             # uses stats.Median + stats.Quantile
│   └── maxabs.py             # uses stats.Max
│
├── encoders/                 # one file per encoder
│   ├── __init__.py           # exports: OneHotEncoder, OrdinalEncoder, TargetEncoder, etc.
│   ├── onehot.py             # OneHotEncoder (sparse by default, max_categories warning)
│   ├── ordinal.py            # OrdinalEncoder
│   ├── target.py             # TargetEncoder (needs y)
│   ├── hash.py               # HashEncoder (zero-fit)
│   ├── frequency.py          # FrequencyEncoder
│   ├── binary.py             # BinaryEncoder
│   └── auto.py               # AutoEncoder (cardinality-based auto-selection)
│
├── imputers/                 # one file per strategy (or one file if simple)
│   ├── __init__.py           # exports: Imputer
│   └── imputer.py            # uses stats.Mean / stats.Median / stats.Mode
│
├── features/                 # feature engineering — one file per category
│   ├── __init__.py           # exports all feature transforms
│   ├── arithmetic.py         # Add, Multiply, Ratio, Log, Sqrt, Power, Clip, Abs, Diff, Modulo
│   ├── string.py             # StringLength, StringLower, StringUpper, StringTrim, StringContains,
│   │                         #   StringReplace, StringExtract
│   ├── splitting.py          # StringSplit, JsonExtract, URLParts, EmailParts, IPParts
│   ├── datetime.py           # DateParts, DateDiff, IsWeekend, IsHoliday, TimeSinceEvent
│   ├── cyclic.py             # CyclicEncode (sin/cos for periodic features)
│   ├── window.py             # Lag, Lead, RollingMean, RollingStd, RollingMin, RollingMax,
│   │                         #   RollingSum, EWM, Rank, PercentRank, CumSum, CumMax
│   ├── aggregation.py        # GroupFeatures
│   ├── interaction.py        # PolynomialFeatures
│   ├── expression.py         # Expression (raw SQL escape hatch)
│   ├── auto.py               # AutoFeatures, AutoDatetime, AutoSplit, AutoNumeric, AutoCategorical
│   ├── outlier.py            # OutlierHandler (IQR, zscore, percentile → clip/null/flag/drop)
│   └── target.py             # TargetTransform (log, sqrt, standard, minmax, quantile)
│
├── feature_selection/        # feature selection — remove low-value columns
│   ├── __init__.py           # exports: Drop, DropConstant, DropCorrelated, etc.
│   ├── drop.py               # Drop (explicit), DropConstant, DropHighNull, DropHighCardinality
│   ├── correlation.py        # DropCorrelated (pairwise correlation → drop redundant)
│   ├── variance.py           # VarianceThreshold, DropLowVariance
│   ├── select.py             # SelectKBest (top K by correlation/MI/ANOVA), SelectByName
│   └── importance.py         # SelectByImportance (requires fitted model — Phase 5+)
│
├── ops/                      # data operations — non-ML cleanup transforms
│   ├── __init__.py           # exports: Rename, Cast, Reorder, Filter, Sample, Deduplicate
│   ├── rename.py             # Rename(mapping=) — column renaming
│   ├── cast.py               # Cast(mapping=) — type casting
│   ├── reorder.py            # Reorder(columns=, sort=) — column ordering
│   ├── filter.py             # Filter(condition=) — row filtering
│   ├── sample.py             # Sample(n=, frac=, seed=) — row sampling
│   └── deduplicate.py        # Deduplicate(subset=, keep=) — row deduplication
│
├── data/                     # data loading helpers — pre-pipeline operations
│   ├── __init__.py           # exports: merge, concat, read_column, attach
│   ├── merge.py              # sq.merge(left, right, on=, how=) — SQL JOIN wrapper
│   ├── concat.py             # sq.concat(sources, source_column=) — UNION ALL wrapper
│   └── lookup.py             # sq.Lookup(table=, on=, features=) — mid-pipeline join
│
│
├── selection/                # model selection + hyperparameter search
│   ├── __init__.py
│   ├── split.py              # KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit, train_test_split
│   ├── search.py             # sq.Search — unified search (replaces GridSearchCV, Optuna, etc.)
│   ├── cross_validate.py     # sq.cross_validate — standalone CV (also used by Search internals)
│   ├── samplers.py           # Random, Grid, Sobol, TPE samplers + Optuna adapter
│   ├── spaces.py             # IntRange, FloatRange, LogRange, Choice, Fixed
│   ├── rounds.py             # Multi-fidelity round scheduling (Hyperband)
│   ├── cache.py              # Feature cache: SQL AST hash → materialized temp table
│   └── metrics.py            # accuracy, f1, rmse, mae, auc, r2 (SQL aggregation)
│
├── inspection/               # data & pipeline inspection — ALL SQL-based
│   ├── __init__.py
│   ├── profile.py            # sq.profile() — types, nulls, cardinality, stats, distributions
│   ├── quality.py            # sq.quality() — data quality score (0-100) with breakdown
│   ├── analyze.py            # sq.analyze() — target-aware: correlations, imbalance, skewness
│   ├── recommend.py          # sq.recommend() — model + pipeline suggestions
│   ├── autopipeline.py       # sq.autopipeline() — generate complete Pipeline from data analysis
│   ├── check.py              # sq.check() — leakage detection, common mistake prevention
│   ├── audit.py              # pipe.audit() — per-column trace through pipeline steps
│   ├── importance.py         # sq.feature_importance() — pre-model feature ranking (Pearson, MI, ANOVA)
│   ├── models.py             # MODEL_PROFILES knowledge base — what each model family needs
│   ├── suggestions.py        # FEATURE_SUGGESTIONS — column type → feature engineering ideas
│   ├── correlations.py       # sq.correlations() — Pearson, Spearman, Cramér's V, mutual info
│   ├── missing.py            # sq.missing_analysis() — null patterns, MCAR/MAR detection, co-occurrence
│   ├── drift.py              # sq.drift() — distribution comparison between datasets
│   └── lineage.py            # column lineage tracing through pipeline
│
├── plot/                     # OPTIONAL — requires matplotlib (pip install sqlearn[plot])
│   ├── __init__.py           # ImportError with install hint if matplotlib missing
│   ├── correlations.py       # sq.plot.correlations() — heatmap
│   ├── missing.py            # sq.plot.missing() — null pattern matrix
│   ├── distributions.py      # sq.plot.distributions() — histograms per column
│   ├── pipeline.py           # sq.plot.pipeline(pipe) — pipeline structure diagram
│   └── search.py             # sq.plot.convergence(), importance(), parallel() — Search viz
│
├── studio/                   # OPTIONAL — interactive EDA (pip install sqlearn[studio])
│   ├── __init__.py           # sq.studio() entry point — starts server, opens browser
│   ├── app.py                # Starlette application (routes + static file serving)
│   ├── session.py            # DuckDB connection + pipeline state management
│   ├── license.py            # License validation: RSA check, trial logic, require_pro()
│   ├── codegen.py            # Python code generation (pipeline, script, report)
│   ├── api/
│   │   ├── profile.py        # GET /api/profile                         (free)
│   │   ├── analyze.py        # POST /api/analyze                        (free)
│   │   ├── recommend.py      # POST /api/recommend                      (free)
│   │   ├── transform.py      # POST /api/transform — preview            (free)
│   │   ├── license.py        # GET /api/license — license state for UI  (free)
│   │   ├── search.py         # WebSocket /api/search — live progress    (Pro)
│   │   ├── builder.py        # POST /api/builder/* — pipeline builder   (Pro)
│   │   ├── export.py         # POST /api/export/* — code/project/chart  (Pro)
│   │   ├── session.py        # POST /api/session/* — save/resume        (Pro)
│   │   ├── sources.py        # POST /api/sources/* — multi-source       (Pro)
│   │   └── model.py          # POST /api/model/* — fit/evaluate         (Pro)
│   └── static/               # pre-built Svelte frontend (bundled in package)
│       ├── index.html
│       ├── app.js            # compiled Svelte bundle (~5KB runtime)
│       ├── uplot.min.js      # uPlot (~35KB — time series, convergence)
│       └── echarts.min.js    # Apache ECharts (~500KB — heatmaps, box, bar, scatter)
│       # Pro UI components are compiled into app.js (minified, not readable source)
│       # Frontend checks $license.pro store to show/hide Pro features
│
└── export/                   # deployment & serialization
    ├── __init__.py
    ├── sql.py                # to_sql(dialect=) via sqlglot
    ├── dbt.py                # to_dbt()
    ├── config.py             # to_config() / from_config() YAML
    └── standalone.py         # export() standalone Python file
```

**Design principles:**
- `stats/aggregates.py` is the foundation — ALL transformers compose from shared building blocks
- Adding a new transformer = one file in the right folder + `__init__.py` export
- Adding a new statistical test = one function in `stats/` or `inspection/`
- `plot/` is optional — SQL computes data, matplotlib renders. No computation in Python.
- `studio/` is optional — free tier works without license, Pro features gated by `require_pro()`
- No changes to core when extending. Fully modular.

**Modularity — how to extend:**
- **New transformer:** one file in the right folder + `__init__.py` export. Zero core changes.
- **New statistical test:** one function in `stats/`. Transformers compose from it.
- **New chart type:** one ECharts config in frontend. Backend provides pre-aggregated data.
- **New export format:** one file in `export/`. Receives compiled SQL AST.
- **New analysis function:** one file in `inspection/`. Uses `stats/` building blocks.
- **New Studio API endpoint:** one file in `studio/api/`. Free or Pro-gated.
- **New sampler for Search:** implement `Sampler` protocol in `selection/samplers.py`.
- **New backend (Postgres, Spark):** implement `Backend` protocol in `core/backend.py`.

Every extension point follows the same pattern: implement one interface, add one file,
export from `__init__.py`. No monkeypatching, no registration, no core modifications.

Dependencies:

```toml
[project]
requires-python = ">=3.10"
dependencies = [
    "duckdb>=1.0",
    "sqlglot>=25.0",
    "numpy>=1.24",
]

[project.optional-dependencies]
pandas = ["pandas>=1.5"]
polars = ["polars>=0.20"]
arrow = ["pyarrow>=12.0"]
sparse = ["scipy>=1.10"]                # sparse matrix output from OneHotEncoder(sparse=True)
plot = ["matplotlib>=3.5"]
optuna = ["optuna>=3.0"]
yaml = ["pyyaml>=6.0"]                  # to_config() / from_config() YAML serialization
studio = [
    "starlette>=0.36",
    "uvicorn>=0.24",
    "websockets>=12.0",
    "python-multipart>=0.0.9",
    "orjson>=3.9",
    "plotly>=5.18",             # for chart → Plotly code generation (Pro)
    "nbformat>=5.9",            # for Jupyter notebook generation (Pro)
    "cryptography>=41.0",       # for RSA license key validation
]
all = ["sqlearn[pandas,polars,arrow,sparse,plot,optuna,yaml,studio]"]
dev = ["pytest", "pytest-cov", "ruff", "mypy", "hypothesis", "scikit-learn"]
# Frontend deps are NPM (dev-only, compiled to static files at build time)
# Pro deps (plotly, nbformat) are included in studio extras — they install
# for everyone but are only used when license is active. This keeps the
# install simple: one `pip install sqlearn[studio]` for free and Pro users.
```

---

