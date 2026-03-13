> **sqlearn docs** — [Index](../CLAUDE.md) | Prev: [Built-in Transformers](05-transformers.md) | Next: [Analysis & Recommendations](07-analysis.md)

## 7. Model Integration

### 7.1 Basic Handoff + ModelPipeline

sqlearn prepares features. User brings their model.

```python
# Basic — manual handoff
pipe.fit("train.parquet")
X_train = pipe.transform("train.parquet")
y_train = sq.read_column("train.parquet", "target")
model = XGBClassifier().fit(X_train, y_train)

# Wrapped — ModelPipeline for seamless usage
full = sq.ModelPipeline(
    preprocessor=pipe,
    model=XGBClassifier(n_estimators=100),
)
full.fit("train.parquet", y="target")
predictions = full.predict("test.parquet")
```

**`ModelPipeline` — Full Design:**

`ModelPipeline` wraps a sqlearn preprocessing pipeline + any sklearn-compatible model
into a single object with `fit()`, `predict()`, `predict_proba()`, and `score()`:

```python
class ModelPipeline:
    """Combines sqlearn preprocessing with any sklearn-compatible model."""

    def __init__(self, preprocessor, model, *, target_transform=None):
        self.preprocessor = preprocessor     # sq.Pipeline or sq.Transformer
        self.model = model                   # any object with .fit(X, y) + .predict(X)
        self.target_transform = target_transform  # optional sq.TargetTransform

    def fit(self, data, y, *, backend=None):
        """Fit preprocessing + model in one call."""
        self.preprocessor.fit(data, y=y, backend=backend)
        X = self.preprocessor.transform(data)
        y_arr = sq.read_column(data, y, backend=backend)
        if self.target_transform:
            y_arr = self.target_transform.fit_transform_array(y_arr)
        self.model.fit(X, y_arr)
        self._y_column = y
        return self

    def predict(self, data, *, backend=None):
        """Preprocess + predict in one call. Auto-inverts target transform."""
        X = self.preprocessor.transform(data, backend=backend)
        preds = self.model.predict(X)
        if self.target_transform:
            preds = self.target_transform.inverse(preds)
        return preds

    def predict_proba(self, data, *, backend=None):
        """For classifiers: preprocess + predict_proba."""
        X = self.preprocessor.transform(data, backend=backend)
        return self.model.predict_proba(X)

    def score(self, data, *, y=None, scoring="auto", backend=None):
        """Preprocess + score. Auto-detects metric from model type."""
        X = self.preprocessor.transform(data, backend=backend)
        y_col = y or self._y_column
        y_arr = sq.read_column(data, y_col, backend=backend)
        return self.model.score(X, y_arr)

    def to_sql(self, *, dialect="duckdb"):
        """Export PREPROCESSING as SQL. Model is not SQL-expressible."""
        return self.preprocessor.to_sql(dialect=dialect)

    def get_params(self, deep=True):
        """Supports nested param access: preprocessor__scaler__... and model__..."""
        params = {"preprocessor": self.preprocessor, "model": self.model}
        if deep:
            params.update({f"preprocessor__{k}": v
                           for k, v in self.preprocessor.get_params(deep=True).items()})
            params.update({f"model__{k}": v
                           for k, v in self.model.get_params().items()})
        return params

    def set_params(self, **params):
        """Set nested params. Used by sq.Search."""
        ...

    def save(self, path):
        """Save preprocessing (sqlearn format) + model (pickle) together."""
        ...

    @classmethod
    def load(cls, path):
        """Load a saved ModelPipeline.

        WARNING: This loads a pickled model. Only load files from trusted sources.
        Pickled files can execute arbitrary code on load. The preprocessing
        pipeline is loaded from safe JSON, but the model component uses pickle.

        For sharing preprocessing only (no security risk), use:
            pipe.save("pipeline.json")       # JSON, safe to share
            pipe = sq.Pipeline.load("pipeline.json")
        """
        ...
```

**Serialization:** Preprocessing is saved in sqlearn's own format (versioned, portable).
The model is saved via pickle (or joblib). Both are bundled into a single `.sqlearn` file
(a ZIP archive with `pipeline.json` + `model.pkl`). This keeps preprocessing portable
while accepting that models are inherently Python-specific.

**ModelPipeline vs sklearn Pipeline:**

| Feature | sklearn Pipeline | sq.ModelPipeline |
|---|---|---|
| Preprocessing | numpy arrays, step-by-step | SQL, one query |
| Model | Part of the pipeline | Separate object (clear boundary) |
| `to_sql()` | N/A | Exports preprocessing SQL |
| Target transform | TransformedTargetRegressor (separate) | Built-in via `target_transform=` |
| Serialization | pickle everything | Preprocessing: portable JSON. Model: pickle. |
| Cross-database | N/A | Preprocessing works on any sqlglot dialect |

### 7.2 Unified Search — `sq.Search`

One class replaces GridSearchCV, RandomizedSearchCV, Optuna, Hyperopt, and Ray Tune.
We don't wrap them — we build something better using SQL-native advantages they can't match.

**Why not just wrap external tools?**

| External tool | What it does well | What it can't do |
|---|---|---|
| GridSearchCV | Exhaustive, deterministic | Exponential blowup, no adaptivity, refits preprocessing every time |
| Optuna | Smart TPE sampling, pruning | Can't deduplicate preprocessing, no SQL-native fidelity |
| Hyperopt | Bayesian optimization | Older API, same limitations as Optuna |
| Ray Tune | Distributed execution | Massive dependency, DuckDB is single-process anyway |

sqlearn has unique advantages no external tool can replicate:
- **Preprocessing dedup:** Same preprocessing SQL hash → transform once, train N models
- **SQL-native data subsampling:** `USING SAMPLE` is free — no data copying
- **SQL-native precision control:** `CAST(col AS FLOAT)` — zero-cost dtype switching
- **Self-contained CTE:** One query → all fold stats simultaneously
- **Fold column reuse:** `__sq_fold__` persists across all param combos
- **Feature caching:** Materialized features per unique SQL hash, models train from cache

#### The API — Three Levels of Control

**Level 1: Simple (most users)**

```python
search = sq.Search(
    preprocessor=pipe,
    model=XGBClassifier(),
    space={
        "preprocessor__impute__strategy": ["mean", "median"],
        "model__n_estimators": sq.IntRange(50, 500),
        "model__max_depth": sq.IntRange(3, 10),
        "model__learning_rate": sq.LogRange(0.01, 0.3),
    },
    budget=100,
    scoring="f1",
    cv=5,
)
search.fit("train.parquet", y="target")

# Results
search.best_params_          # best parameter combination
search.best_score_           # best CV score
search.best_model_           # fitted model with best params
search.results_              # all trials as DataFrame
```

Default behavior: float64 precision, full data, specified CV folds. Standard.
No surprises, no magic. Same correctness guarantees as sklearn GridSearchCV.

**Level 2: Fast explore (power users)**

```python
search = sq.Search(
    preprocessor=pipe,
    model=XGBClassifier(),
    space={...},
    budget=100,
    scoring="f1",
    cv=5,
    fast_explore=True,         # enables multi-fidelity with sensible defaults
    explore_dtype="float32",   # exploration rounds use float32
    final_dtype="float64",     # final validation uses float64 (this is the default anyway)
    explore_sample=0.1,        # data fraction for exploration rounds
)
```

`fast_explore=True` auto-creates rounds using Hyperband scheduling.
Individual params (`explore_dtype`, `explore_sample`) can be overridden or omitted.

**Level 3: Manual rounds (full control)**

```python
search = sq.Search(
    preprocessor=pipe,
    model=XGBClassifier(),
    space={...},
    rounds=[
        {"n": 60, "sample": 0.1, "dtype": "float32", "cv": 3},   # explore wide
        {"n": 25, "sample": 0.3, "dtype": "float32", "cv": 5},   # narrow down
        {"n": 10, "sample": 1.0, "dtype": "float64", "cv": 5},   # confirm
        {"n":  5, "sample": 1.0, "dtype": "float64", "cv": 10},  # final validation
    ],
    scoring="f1",
)
```

Each round promotes top N from previous round and re-evaluates at higher fidelity.
Round 1 samples N new configs. Rounds 2+ are survival tournaments.

#### Precision Control — Opt-In, Not Default

By default, `sq.Search` uses standard precision (float64). No surprises.
Multi-fidelity precision is strictly opt-in:

```python
# DEFAULT: float64 everywhere. Standard behavior.
search = sq.Search(..., budget=100)

# OPT-IN: fast exploration with lower precision
search = sq.Search(
    ...,
    budget=100,
    explore_dtype="float32",   # early rounds: float32 (faster model training, less memory)
    final_dtype="float64",     # final rounds: float64 (full precision, this is default)
)

# FULL CONTROL: per-round precision
search = sq.Search(
    ...,
    rounds=[
        {"n": 50, "dtype": "float32"},   # explore
        {"n": 10, "dtype": "float64"},   # validate
    ],
)
```

**Why float32 is safe for exploration:**
- Tree models (XGBoost, LightGBM, RF): split on thresholds. Relative ordering
  preserved in float32. Rankings between candidates essentially identical.
- Linear models: convergence barely affected. Same winner.
- Neural nets: already train in float32 (or float16) natively.
- We promote top-K survivors, not pick a single winner from low precision.
  Even if one candidate slips ranking by a spot, it still survives to the next round.

#### Parameter Space Definitions

```python
sq.IntRange(50, 500)                  # uniform integer
sq.IntRange(50, 500, log=True)        # log-uniform integer
sq.FloatRange(0.0, 1.0)              # uniform float
sq.LogRange(0.001, 1.0)              # log-uniform (learning rates, regularization)
sq.Choice(["mean", "median"])         # categorical
sq.Choice([sq.StandardScaler(), sq.RobustScaler()])  # transformer objects
sq.Fixed(42)                          # constant (useful for partial searches)
```

#### Sampling Strategies

```python
sq.Search(..., strategy="random")      # random sampling (default)
sq.Search(..., strategy="grid")        # exhaustive grid (small spaces only)
sq.Search(..., strategy="sobol")       # quasi-random Sobol sequence (better space coverage)
sq.Search(..., strategy="bayesian")    # built-in TPE (Tree-structured Parzen Estimator)
sq.Search(..., strategy="auto")        # auto-select based on space size + data size
```

**Auto strategy selection:**

| Space size | Data size | Auto picks | Why |
|---|---|---|---|
| < 50 combos | any | `"grid"` | Small enough to be exhaustive |
| 50-500 | any | `"sobol"` | Quasi-random covers space better than random |
| > 500 | any | `"bayesian"` | Need intelligent sampling |
| any | > 1M rows | enables `fast_explore` | Big data → multi-fidelity pays off |

All samplers are built-in. Zero external dependencies.

**Optional Optuna integration** for users who want Optuna's TPE or CMA-ES:

```python
# pip install sqlearn[optuna]
sq.Search(..., strategy="optuna")          # uses Optuna's default TPE sampler
sq.Search(..., strategy="optuna:cmaes")   # uses Optuna's CMA-ES sampler
```

Optuna provides the sampler. sqlearn provides the search loop, SQL dedup, fidelity
control, and caching. Best of both — no hard dependency on Optuna, but interoperable.

#### Multi-Fidelity: Three SQL-Native Knobs

No other framework can do this because they work with numpy arrays.
We work with SQL — changing fidelity is just changing a clause.

| Knob | SQL mechanism | Cost | Speedup per trial |
|---|---|---|---|
| Data fraction | `USING SAMPLE 10%` | Free — DuckDB samples without copying | 10-20x |
| Precision | `CAST(col AS FLOAT)` vs `DOUBLE` | Free — one word in SQL | 1.5-2x |
| CV folds | Fewer folds per round | Free — just fewer queries | proportional |

**Combined: 10% data + float32 + 3-fold = ~30x faster per trial than full evaluation.**

```
How rounds work (Hyperband scheduling):

budget=100, fast_explore=True, auto-determined:

Round 1:  60 configs × (10% data, float32, 3-fold) ≈ 2 full-evals    → keep top 25
Round 2:  25 configs × (30% data, float32, 5-fold) ≈ 4 full-evals    → keep top 10
Round 3:  10 configs × (100% data, float64, 5-fold) = 10 full-evals  → keep top 5
Round 4:   5 configs × (100% data, float64, 10-fold) = 10 full-evals → winner

Total cost: ~26 full-eval equivalents for 100 configs explored
```

#### Preprocessing Deduplication (Unique to sqlearn)

```python
space = {
    "preprocessor__impute__strategy": ["mean", "median"],    # 2 preprocessing configs
    "model__n_estimators": [100, 200, 500],                  # 3 model configs
    "model__max_depth": [3, 5, 10],                          # 3 model configs
}

# Total combos: 2 × 3 × 3 = 18
# Unique preprocessing SQL: 2 (mean vs median — model params don't affect SQL)
# sqlearn: transforms data TWICE, trains 18 models from cached features
# sklearn: transforms data 18 TIMES
```

The compiler hashes the preprocessing SQL AST. Same hash → identical transformed features.
Features are materialized once per unique hash, stored in temp tables.
All model fits with the same preprocessing pull from cache.

**This is automatic. No user config needed.**

#### Early Stopping Per Trial

```python
search = sq.Search(
    ...,
    early_stop=True,          # stop unpromising trials mid-CV
    early_stop_patience=2,    # if 2 consecutive folds worse than best, skip remaining
)
```

After each fold completes, check if the trial can mathematically beat the current best.
If not, skip remaining folds. This is free — just a numeric comparison.

With 10-fold CV and aggressive early stopping, most bad configs exit after 2-3 folds.
Only promising configs run all 10 folds.

#### Error Recovery — Trials That Fail

Model training can fail: OOM, numerical errors, incompatible parameters, NaN loss.
`sq.Search` handles failures gracefully without crashing the entire search:

```python
search = sq.Search(
    ...,
    on_error="skip",          # default: skip failed trials, log error, continue
    max_consecutive_errors=10, # stop search if 10 trials fail in a row (bad config)
)
```

**Error policies:**

| Policy | Behavior |
|---|---|
| `on_error="skip"` (default) | Log error, mark trial as failed, continue search. Failed trials don't count against budget. |
| `on_error="raise"` | Stop search immediately on first failure. For debugging. |
| `on_error="warn"` | Like `"skip"` but also emits `UserWarning` per failure. |

**After search — inspect failures:**

```python
search.fit("data", y="target")

search.failed_trials_         # list of FailedTrial objects
len(search.failed_trials_)    # how many failed

for trial in search.failed_trials_:
    print(trial.params)       # what params caused it
    print(trial.error)        # the exception
    print(trial.traceback)    # full traceback string

# Common failure patterns:
search.failure_summary()
# 7 trials failed:
#   5× ValueError: "n_estimators must be > 0" — params: {n_estimators: 0}
#   2× MemoryError — params with n_estimators > 5000
```

**Consecutive error detection:** If `max_consecutive_errors` trials fail in a row,
the search stops with `SearchError`. This catches systematic problems (wrong param
space, broken data) without wasting the full budget. Default: 10.

#### Checkpoint and Resume

```python
# Long search — save progress to DuckDB file
search = sq.Search(..., budget=1000, checkpoint="search_state.db")
search.fit("data", y="target")   # stores trial results + state in DuckDB

# Interrupted? Resume where you left off
search = sq.Search.resume("search_state.db")
search.fit("data", y="target")   # continues from trial 347/1000
```

State stored in DuckDB — queryable, portable, no pickle.

```sql
-- Inspect search results directly
SELECT * FROM search_state.trials ORDER BY score DESC LIMIT 10;
```

#### Progress and Reporting

```python
search.fit("data", y="target", verbose=True)
# Trial 47/100 | Round 2/4 | Best: 0.847 (f1) | ETA: 3m 12s
# ████████████████████░░░░░░░░░░ 47%

# After search
search.summary()                # best params, score, convergence info
search.plot_convergence()       # score vs trial number (requires sqlearn[plot])
search.plot_importance()        # which params matter most (functional ANOVA)
search.plot_parallel()          # parallel coordinates of top configs
```

#### Parallel Model Training

DuckDB runs SQL single-threaded (internally parallelized). Model training is pure Python.
These don't compete for resources:

```python
search = sq.Search(
    ...,
    n_jobs=4,    # train 4 models in parallel after SQL preprocessing
)
```

SQL preprocessing runs sequentially (DuckDB handles its own parallelism).
Model training runs in parallel across CPU cores via joblib.

#### Optuna Integration — Three Paths

**Path 1: `strategy="optuna"` (recommended)**

Optuna is just the sampler. sqlearn handles everything else — dedup, fidelity, caching.
User writes identical code to any other strategy, just changes one string:

```python
# pip install sqlearn[optuna]
search = sq.Search(
    preprocessor=pipe,
    model=XGBClassifier(),
    space={
        "preprocessor__impute__strategy": ["mean", "median"],
        "model__n_estimators": sq.IntRange(50, 500),
        "model__max_depth": sq.IntRange(3, 10),
        "model__learning_rate": sq.LogRange(0.01, 0.3),
        "model__subsample": sq.FloatRange(0.5, 1.0),
    },
    strategy="optuna",           # uses Optuna's TPE sampler
    budget=100,
    scoring="f1",
    cv=5,
)
search.fit("train.parquet", y="target")

print(search.best_params_)
model = search.best_model_
```

sqlearn translates `IntRange` → `trial.suggest_int`, `LogRange` →
`trial.suggest_float(log=True)`, etc. internally. User never touches Optuna's API.

Optuna + multi-fidelity — something Optuna can't do alone:

```python
search = sq.Search(
    ...,
    strategy="optuna",
    fast_explore=True,
    explore_dtype="float32",
    explore_sample=0.1,
)
# Optuna picks params smartly, sqlearn handles SQL-native fidelity
```

Specific Optuna sampler:

```python
sq.Search(..., strategy="optuna:cmaes")   # CMA-ES shorthand

import optuna
sq.Search(                                 # or pass object directly
    ...,
    strategy="optuna",
    optuna_sampler=optuna.samplers.CmaEsSampler(seed=42),
)
```

**Path 2: User drives Optuna, uses `sq.cross_validate` (advanced)**

For Optuna-specific features (multi-objective, custom pruning, dashboards):

```python
import optuna
import sqlearn as sq

pipe = sq.Pipeline([sq.Imputer(), sq.StandardScaler(), sq.OneHotEncoder()])

def objective(trial):
    strategy = trial.suggest_categorical("impute_strategy", ["mean", "median"])
    n_est = trial.suggest_int("n_estimators", 50, 500)
    lr = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)

    p = pipe.set_params(imputer__strategy=strategy)
    model = XGBClassifier(n_estimators=n_est, learning_rate=lr)

    # sq.cross_validate: handles folds, caching, dtype — returns one score
    return sq.cross_validate(
        preprocessor=p,
        model=model,
        data="train.parquet",
        y="target",
        cv=5,
        scoring="f1",
        dtype="float32",
    )

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
```

User drives the loop. sqlearn still provides preprocessing caching, fold column
reuse, self-contained CTE, and dtype control through `sq.cross_validate`.

**Path 3: User drives Optuna, manual everything (escape hatch)**

```python
def objective(trial):
    strategy = trial.suggest_categorical("impute_strategy", ["mean", "median"])
    p = pipe.set_params(imputer__strategy=strategy)
    p.fit("train.parquet")
    X_train = p.transform("train.parquet", dtype="float32")
    X_test = p.transform("test.parquet", dtype="float32")
    y_train = sq.read_column("train.parquet", "target")
    y_test = sq.read_column("test.parquet", "target")

    model = XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 50, 500),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
    )
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)
```

This works but loses preprocessing dedup and fold reuse. Only use when Path 1 or 2
don't cover your Optuna use case (multi-objective, complex pruning).

**Recommendation:** Path 1 for 90% of users. Path 2 for advanced Optuna features.
Path 3 only when you need full Optuna control and accept losing SQL optimizations.

### 7.3 `sq.cross_validate` — Standalone CV Helper

A standalone function for quick cross-validation without building a full `Search`.
Useful on its own, with Optuna, or for quick experiments:

```python
# Quick CV score — one line
score = sq.cross_validate(
    preprocessor=pipe,
    model=XGBClassifier(n_estimators=100),
    data="train.parquet",
    y="target",
    cv=5,
    scoring="f1",
)
print(f"F1: {score:.4f}")  # F1: 0.8472

# With all options
result = sq.cross_validate(
    preprocessor=pipe,
    model=XGBClassifier(n_estimators=100),
    data="train.parquet",
    y="target",
    cv=sq.StratifiedKFold(5, random_state=42),
    scoring="f1",
    dtype="float32",           # precision control
    return_fold_scores=True,   # per-fold breakdown
    return_models=True,        # keep fitted models
    return_times=True,         # fit/score timing
)
print(result.mean)         # 0.8472
print(result.std)          # 0.0031
print(result.fold_scores)  # [0.843, 0.851, 0.847, 0.842, 0.853]
print(result.fit_times)    # [1.2s, 1.1s, 1.3s, 1.2s, 1.1s]
```

**Under the hood:**
- Creates `__sq_fold__` column once
- Uses self-contained CTE for preprocessing (one query per fold)
- Caches transformed features per fold
- Returns aggregate score by default, detailed results when asked

**Without a model — just validate preprocessing:**

```python
# Validate that preprocessing works on all folds
result = sq.cross_validate(
    preprocessor=pipe,
    data="train.parquet",
    cv=5,
)
# result.success = True/False, result.schemas = per-fold output schemas
# Useful for catching schema issues before long model training
```

### 7.4 `sq.read_column` — Target Column Helper

```python
# Read target column without loading full dataset
y = sq.read_column("train.parquet", "target")               # numpy array
y = sq.read_column("train.parquet", "target", dtype="int")   # with cast
y = sq.read_column("train.parquet", ["target", "weight"])    # multiple columns
```

One SQL query: `SELECT target FROM 'train.parquet'`. Avoids loading the full dataset
just to extract the target. Works with any source DuckDB can read.

### 7.5 `sq.train_test_split` — SQL-Native Split

```python
# Split data — SQL TABLESAMPLE, no data copying
train, test = sq.train_test_split("data.parquet", test_size=0.2, random_state=42)

# Returns lightweight split objects, not materialized data
pipe.fit(train)
X_test = pipe.transform(test)
y_test = sq.read_column(test, "target")
```

Split objects are SQL views — `WHERE __sq_split__ = 'train'` / `'test'`.
No data duplication. Works with bigger-than-RAM datasets.

Stratified:

```python
train, test = sq.train_test_split(
    "data.parquet",
    test_size=0.2,
    stratify="target",       # stratified by target column
    random_state=42,
)
```

### 7.6 Search Result Analysis

After a search completes, users need to understand *why* certain params won — not just *what* won.

```python
# Parameter importance — which params actually matter?
search.importance()
# Returns DataFrame:
#   param                         importance    type
#   model__learning_rate          0.42          float
#   model__n_estimators           0.28          int
#   preprocessor__impute__strategy 0.18         categorical
#   model__max_depth              0.12          int

# Interaction effects — which param COMBINATIONS matter?
search.interactions()
# learning_rate × n_estimators: strong interaction (r=0.67)
# impute_strategy × max_depth: weak interaction (r=0.08)

# Top N configs with scores
search.top(10)
# Returns DataFrame of top 10 param combos + scores + fold details

# Score distribution per param value
search.param_scores("model__learning_rate")
# Returns (values, scores) — ready for plotting

# Export all trial results to DuckDB for custom analysis
search.to_duckdb("analysis.db")
# SELECT * FROM trials WHERE score > 0.85 ORDER BY fit_time ASC;
```

Importance is computed via functional ANOVA (same method as Optuna) — measures how
much variance in scores is explained by each parameter. Built-in, no external dependency.

**Smart suggestions after search:**

```python
search.suggest()
# Based on your results:
#   - learning_rate has high importance but narrow best range (0.05-0.08).
#     Consider: sq.FloatRange(0.03, 0.12) for a focused follow-up search.
#   - impute_strategy="median" wins in 92% of top configs.
#     Consider: fixing it and removing from search space.
#   - n_estimators shows diminishing returns above 300.
#     Consider: sq.IntRange(200, 400) instead of (50, 500).
```

### 7.7 Warm-Start Search

Continue a previous search with refined space — no wasted work:

```python
# First search: broad exploration
search1 = sq.Search(
    ...,
    space={
        "model__learning_rate": sq.LogRange(0.001, 1.0),
        "model__n_estimators": sq.IntRange(50, 1000),
    },
    budget=50,
)
search1.fit("data", y="target")

# Second search: narrow in on promising region
search2 = sq.Search(
    ...,
    space={
        "model__learning_rate": sq.LogRange(0.01, 0.1),   # narrowed
        "model__n_estimators": sq.IntRange(200, 500),      # narrowed
    },
    budget=30,
    warm_start=search1,   # reuse previous trials as prior knowledge
)
search2.fit("data", y="target")
# Bayesian sampler starts with 50 prior observations — converges faster
```

With `strategy="bayesian"`, previous trials inform the surrogate model.
With other strategies, warm_start reuses feature cache from the previous search.

#### Full Comparison

```
100 trials, 5-fold CV, 10 preprocessing × 10 model param combos:

Tool              Preproc cost      Model cost        Total
────────────────  ────────────────  ────────────────  ──────
sklearn GridCV    100 × full_fit    100 × 5 folds     600 units
Optuna            100 × full_fit    100 × 5 folds     600 units
sq.Search         10 unique SQL     100 × 5 folds     510 units  (dedup only)
sq.Search + MF    10 unique SQL     ~26 equiv folds   ~36 units  (multi-fidelity)

MF = fast_explore=True. Result: 17x faster than sklearn/Optuna.
```

---

