> **sqlearn docs** — [Index](../CLAUDE.md) | Next: [API Design](02-api-design.md)

# sqlearn — Design & Implementation Plan

*Living document. Open, read, iterate.*

---

## 1. What sqlearn Is

### Design Philosophy: Safe Defaults, Power Opt-In

sqlearn follows one rule: **nothing clever happens unless the user asks for it.**

The base experience is sklearn-equivalent. Same `fit()`, `transform()`, `Pipeline([...])`.
Same correctness guarantees. Same float64 precision. No surprises.

Advanced features exist but are always **explicit opt-in:**

| Feature | Default (safe) | Opt-in (advanced) |
|---|---|---|
| Precision | float64 everywhere | `explore_dtype="float32"` in Search |
| Data sampling | Full dataset | `explore_sample=0.1` in Search |
| Multi-fidelity | Off | `fast_explore=True` or `rounds=[...]` |
| Auto features | Off — user builds pipeline step by step | `sq.AutoFeatures()` explicit step |
| Auto pipeline | Off — user writes their pipeline | `sq.autopipeline()` explicit call |
| Auto encoding | Off — user picks encoder | `sq.AutoEncoder()` explicit step |
| Column routing | Auto for built-ins (numeric/categorical) | User overrides with `columns=` |
| NULL handling | SQL propagation (standard) | `null_policy="error"` or `"warn"` |
| Result size | Eager numpy (sklearn compat) | `out="relation"` for lazy |
| Search sampler | `"random"` | `"bayesian"`, `"sobol"`, `"grid"`, `"auto"` |

**The `Auto*` family (`AutoFeatures`, `AutoDatetime`, `AutoSplit`, `AutoNumeric`,
`AutoEncoder`) are power tools, not defaults.** Users add them to their pipeline
deliberately. They are never injected behind the scenes. The basic pipeline:

```python
# This is the default experience — explicit, predictable, sklearn-like:
pipe = sq.Pipeline([
    sq.Imputer(),
    sq.StandardScaler(),
    sq.OneHotEncoder(),
])
```

If a user wants auto-detection, they explicitly ask for it:

```python
# This is the power-user experience — explicit opt-in:
pipe = sq.Pipeline([
    sq.Imputer(),
    sq.AutoFeatures(),      # user chose this — they know what it does
    sq.StandardScaler(),
])
```

And for production, `AutoFeatures.to_explicit()` freezes all auto-decisions into
a deterministic pipeline with baked parameters. No version-to-version surprises.

**Why this matters for ML correctness:**
- Auto-detection thresholds may change between sqlearn versions
- "Clever" defaults silently change pipeline behavior → hard to debug
- Users must understand their preprocessing to trust their model
- Explicit steps in `pipe.describe()` are auditable and reproducible

---

sqlearn compiles ML preprocessing pipelines to SQL via sqlglot ASTs. You write Python.
The system writes SQL. Every pipeline becomes one query. DuckDB is the default engine,
but any sqlglot-supported database (Postgres, MySQL, Snowflake, BigQuery) is a valid
target. No intermediate numpy arrays, no RAM ceiling on SQL execution, no deployment headache.

**All SQL generation uses sqlglot ASTs — never raw strings.** This is the foundation
for multi-database support. Same pipeline, any database.

**What it replaces in sklearn:**

| sklearn module | sqlearn equivalent | How |
|---|---|---|
| `sklearn.preprocessing` | `sq.StandardScaler`, `sq.MinMaxScaler`, `sq.OneHotEncoder`, ... | SQL arithmetic + CASE expressions |
| `sklearn.impute` | `sq.Imputer` | COALESCE |
| `sklearn.pipeline` | `sq.Pipeline`, `+`, `+=` | Expression composition, operator overloading |
| `sklearn.compose` | `sq.Columns`, `sq.Union` | Column routing, parallel features |
| `sklearn.feature_selection` | `sq.SelectKBest`, `sq.DropCorrelated`, `sq.VarianceThreshold`, ... | SQL aggregation: `CORR()`, `VAR_POP()`, `COUNT(DISTINCT)` |
| `sklearn.model_selection` | `sq.Search`, `sq.cross_validate`, `sq.KFold` | Unified search with multi-fidelity, SQL dedup, feature caching |
| `sklearn.metrics` | `sq.metrics.*` | SQL aggregation |
| pandas data ops | `sq.Rename`, `sq.Cast`, `sq.Filter`, `sq.Drop`, `sq.Deduplicate` | SQL: `AS`, `CAST`, `WHERE`, omit from SELECT, `DISTINCT ON` |
| pandas merge/concat | `sq.merge()`, `sq.concat()`, `sq.Lookup` | SQL: `JOIN`, `UNION ALL`, mid-pipeline `LEFT JOIN` |
| Feature engineering (no sklearn equiv) | `sq.AutoFeatures`, `sq.AutoDatetime`, `sq.StringSplit`, `sq.CyclicEncode`, ... | Type-aware auto-expansion: datetime→parts, strings→split, skew→log |
| Outlier handling (no sklearn equiv) | `sq.OutlierHandler` | SQL: `PERCENTILE_CONT` → `GREATEST(LEAST(...))` IQR/zscore clip |
| Target transforms (no sklearn equiv) | `sq.TargetTransform` | `LN(y+1)` with auto-inverse at predict time |

**What it does NOT replace** (and never will):

| Category | Why not |
|---|---|
| Model training (sklearn, XGBoost, LightGBM, PyTorch) | Not SQL-expressible. Bring your own model. |
| PCA, SVD, t-SNE, UMAP | Matrix decomposition. Not SQL. |
| Text vectorization (TF-IDF, embeddings) | Specialized NLP. Not SQL. |
| Image/audio features | Deep learning territory. |

sqlearn handles **100% of data preprocessing and feature engineering** — everything
between raw data and model training. The only thing outside sqlearn is the model
itself (gradient descent, tree splitting, backpropagation — iterative algorithms
that aren't expressible as SQL).

DuckDB reads all major formats natively — no pandas/manual loading needed:

| Format | Support |
|---|---|
| Parquet, CSV, JSON, NDJSON | Native |
| Arrow / IPC | Native, zero-copy |
| Excel (.xlsx) | Via DuckDB extension |
| S3 / GCS / Azure | Native remote reads |
| Postgres / MySQL / SQLite | Via ATTACH |
| pandas / Polars DataFrame | Zero-copy via Arrow |
| numpy array | Register as table |

The user passes a path. sqlearn handles everything:

```python
pipe = sq.Pipeline([...])
pipe.fit("train.parquet")                # or .csv, .json, s3://, pandas_df, ...
X = pipe.transform("train.parquet")      # numpy array → feed to any model
model = XGBClassifier().fit(X, y)
```

---

## 2. Naming

**Name: `sqlearn`** — `sk` → `sq`. SQL-first. `import sqlearn as sq`.

---
