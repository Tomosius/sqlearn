> **sqlearn docs** — [Index](../CLAUDE.md) | Prev: [Model Integration](06-model-integration.md) | Next: [Export & Deployment](08-export.md)

## 8. Dataset Analysis & Recommendations

Four functions, four levels of depth. Each builds on the previous.
All analysis is SQL — runs on datasets of any size.

### 8.1 `sq.profile()` — Quick Overview (Level 1)

No target needed. Just point at data.

```python
report = sq.profile("data.parquet")
print(report)
```

```
═══ sqlearn Data Profile ═══
Source: data.parquet | Rows: 50,000 | Columns: 15

Column          Type         Nulls    Unique   Mean      Std       Min     Max
──────────────  ───────────  ───────  ───────  ────────  ────────  ──────  ──────
age             INTEGER      2.1%     72       34.5      12.3      18      89
income          DOUBLE       0.0%     48,291   65,420    28,100    12,000  450,000
city            VARCHAR      0.5%     147      —         —         —       —
country         VARCHAR      0.0%     1        —         —         —       —
created_at      TIMESTAMP    0.0%     49,892   —         —         2020-01  2024-12
zip_code        VARCHAR      0.0%     847      —         —         —       —
score           DOUBLE       15.3%    1,203    0.72      0.18      0.0     1.0
...

Warnings:
  ⚠ country has 1 unique value ("US") — constant column, consider dropping
  ⚠ score has 15.3% nulls — needs imputation
  ⚠ zip_code has 847 unique values — high cardinality categorical
```

SQL under the hood — one query:

```sql
SELECT
    COUNT(*) AS n_rows,
    COUNT(DISTINCT age) AS age_unique,
    AVG(age) AS age_mean,
    STDDEV_POP(age) AS age_std,
    MIN(age) AS age_min,
    MAX(age) AS age_max,
    SUM(CASE WHEN age IS NULL THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS age_null_pct,
    ...
FROM 'data.parquet'
```

Access programmatically:

```python
report.columns["age"].mean        # 34.5
report.columns["age"].null_pct    # 0.021
report.columns["city"].unique     # 147
report.warnings                   # list of Warning objects
report.to_dataframe()             # as pandas DataFrame
```

### 8.2 `sq.analyze()` — Target-Aware Analysis (Level 2)

Knows the target. Analyzes relationships.

```python
report = sq.analyze("data.parquet", target="price")
print(report)
```

```
═══ sqlearn Dataset Analysis ═══
Source: data.parquet | Rows: 50,000 | Target: price (DOUBLE) — regression

── Data Quality ──
  Nulls: 3 columns (age: 2.1%, city: 0.5%, score: 15.3%)
  Constants: country (1 value) → drop
  Duplicates: 0 exact duplicate rows
  Near-duplicates: 23 rows differ only in timestamp

── Target Analysis ──
  Distribution: right-skewed (skewness=2.3) → consider Log transform
  Range: $12,000 – $450,000 | Median: $52,000
  Outliers: 127 rows > 3σ (0.25%)

── Feature Correlations with Target ──
  Strong:   sqft (r=0.82), bedrooms (r=0.71), bathrooms (r=0.68)
  Moderate: age (r=0.34), score (r=0.28)
  Weak:     customer_id (r=0.01) → likely irrelevant, consider dropping
  Categorical: city (Cramér's V=0.45), type (Cramér's V=0.62)

── Multicollinearity ──
  bedrooms ↔ bathrooms: r=0.89 — highly correlated, consider dropping one
  sqft ↔ bedrooms: r=0.78 — moderate, keep both but be aware

── Feature Engineering Opportunities ──
  Datetime: created_at → DateParts(parts=["month", "dayofweek", "hour"])
  Ratio: sqft / bedrooms → "sqft_per_bedroom" (potential signal)
  Interaction: sqft × type → interaction strength 0.34
  Text-like: description (avg length 142) → StringLength, StringContains
```

```python
# Programmatic access
report.target.task          # "regression"
report.target.skewness      # 2.3
report.correlations         # DataFrame of all correlations with target
report.multicollinearity    # list of correlated pairs
report.suggestions          # list of Suggestion objects
```

### 8.3 `sq.recommend()` — Model & Pipeline Suggestions (Level 3)

Knows the data AND knows what different models need.

```python
report = sq.recommend("data.parquet", target="price")
print(report)
```

```
═══ sqlearn Recommendations ═══
Source: data.parquet | Target: price — regression | 50K rows, 15 features

── Model Recommendations ──
  ✓ Recommended:
    1. XGBoost / LightGBM
       Why: mixed feature types, robust to outliers, handles moderate size well
       Preprocessing: no scaling needed, encode categoricals, impute nulls
    2. RandomForest
       Why: good baseline, interpretable feature importance
       Preprocessing: same as XGBoost

  ~ Consider:
    3. LinearRegression / Ridge / Lasso
       Why: interpretable, good if relationships are linear
       Preprocessing: MUST scale, encode categoricals, handle multicollinearity
       Warning: target is skewed → apply Log transform first
    4. ElasticNet
       Why: automatic feature selection via L1, handles multicollinearity
       Preprocessing: same as LinearRegression

  ✗ Not recommended:
    - KNN: dataset has 15 features, performance degrades in high dimensions
    - NaiveBayes: assumes feature independence, violated by multicollinearity

── Preprocessing Pipeline (for tree models) ──
  pipe = sq.Pipeline([
      sq.Imputer({"age": "median", "city": "most_frequent", "score": "median"}),
      sq.DateParts(columns=["created_at"], parts=["month", "dayofweek"]),
      sq.OneHotEncoder(columns=["city", "type"]),     # low cardinality
      sq.HashEncoder(columns=["zip_code"], n_bins=64), # high cardinality (847 unique)
      # No scaling — tree models don't need it
  ])

── Preprocessing Pipeline (for linear models) ──
  pipe = sq.Pipeline([
      sq.Imputer({"age": "median", "city": "most_frequent", "score": "median"}),
      sq.Log(columns=["price"]),                       # fix skewed target
      sq.DateParts(columns=["created_at"], parts=["month", "dayofweek"]),
      sq.OneHotEncoder(columns=["city", "type"]),
      sq.HashEncoder(columns=["zip_code"], n_bins=64),
      sq.StandardScaler(),                              # required for linear models
  ])

── Columns to Drop ──
  country: constant (1 value)
  customer_id: near-zero correlation with target (r=0.01)
```

**Model knowledge base** — what each model family needs:

```python
# Built-in knowledge (inspection/models.py)
MODEL_PROFILES = {
    "tree_based": {           # XGBoost, LightGBM, RandomForest, GBM
        "needs_scaling": False,
        "needs_encoding": True,   # except LightGBM native categoricals
        "handles_nulls": "some",  # XGBoost/LightGBM yes, RF no
        "sensitive_to_outliers": False,
        "sensitive_to_multicollinearity": False,
        "max_features_comfortable": 1000,
        "min_rows_recommended": 100,
    },
    "linear": {               # LogisticRegression, Ridge, Lasso, ElasticNet, SVM
        "needs_scaling": True,
        "needs_encoding": True,
        "handles_nulls": False,
        "sensitive_to_outliers": True,
        "sensitive_to_multicollinearity": True,
        "max_features_comfortable": 500,
        "min_rows_recommended": 50,
    },
    "knn": {                  # KNeighborsClassifier/Regressor
        "needs_scaling": True,    # critical — distance-based
        "needs_encoding": True,
        "handles_nulls": False,
        "sensitive_to_outliers": True,
        "sensitive_to_multicollinearity": False,
        "max_features_comfortable": 20,    # curse of dimensionality
        "min_rows_recommended": 100,
    },
    "neural_network": {       # MLPClassifier, PyTorch, TensorFlow
        "needs_scaling": True,
        "needs_encoding": True,
        "handles_nulls": False,
        "sensitive_to_outliers": True,
        "sensitive_to_multicollinearity": False,
        "max_features_comfortable": 10000,
        "min_rows_recommended": 1000,
    },
    "naive_bayes": {          # GaussianNB, MultinomialNB
        "needs_scaling": False,
        "needs_encoding": True,
        "handles_nulls": False,
        "sensitive_to_outliers": False,
        "sensitive_to_multicollinearity": False,
        "max_features_comfortable": 10000,
        "min_rows_recommended": 50,
    },
}
```

Auto-detect model family from class name:

```python
from xgboost import XGBClassifier

# Pass a model to get model-specific recommendations
report = sq.recommend("data.parquet", target="price", model=XGBClassifier())
# Returns only the pipeline relevant to this model family
```

### 8.4 `sq.autopipeline()` — Generate a Complete Pipeline (Level 4)

Actually returns a Pipeline object. Zero manual work.

```python
# Auto-generate pipeline from data analysis
pipe = sq.autopipeline("data.parquet", target="price")
# Returns a fitted sq.Pipeline based on sq.recommend() analysis

# Use it directly
X = pipe.transform("test.parquet")

# Or inspect what it built
print(pipe.describe())
# Pipeline(5 steps):
#   1. Imputer({"age": "median", "city": "most_frequent", "score": "median"})
#   2. DateParts(columns=["created_at"], parts=["month", "dayofweek"])
#   3. OneHotEncoder(columns=["city", "type"])
#   4. HashEncoder(columns=["zip_code"], n_bins=64)
#   5. StandardScaler()    # included because model=LinearRegression

# Model-specific autopipeline
pipe = sq.autopipeline("data.parquet", target="price", model=XGBClassifier())
# Omits StandardScaler because XGBoost doesn't need it

# With feature engineering suggestions applied
pipe = sq.autopipeline("data.parquet", target="price", features=True)
# Also adds: DateParts, Ratio("sqft", "bedrooms"), interaction terms

# Export the generated pipeline as Python code
code = sq.autopipeline("data.parquet", target="price", as_code=True)
print(code)
```

```python
# Generated by sqlearn autopipeline
import sqlearn as sq

pipe = sq.Pipeline([
    sq.Imputer({"age": "median", "city": "most_frequent", "score": "median"}),
    sq.DateParts(columns=["created_at"], parts=["month", "dayofweek"]),
    sq.Ratio("sqft", "bedrooms", name="sqft_per_bedroom"),
    sq.OneHotEncoder(columns=["city", "type"]),
    sq.HashEncoder(columns=["zip_code"], n_bins=64),
])
```

**`as_code=True` outputs clean, editable Python.** User can copy-paste, modify,
and use as a starting point. This is the bridge to the interactive workflow.

### 8.5 Feature Engineering Suggestions

Built-in knowledge of what features to create from what column types:

```python
FEATURE_SUGGESTIONS = {
    "datetime": [
        ("DateParts",  "Extract year, month, day, hour, dayofweek"),
        ("IsWeekend",  "Binary weekend flag"),
        ("DateDiff",   "Days since earliest/reference date"),
    ],
    "numeric_pair": [
        ("Ratio",      "Ratio between correlated numerics"),
        ("Multiply",   "Product — captures interaction effects"),
        ("Add",        "Sum — useful for related quantities"),
    ],
    "numeric_skewed": [
        ("Log",        "Log transform for right-skewed distributions"),
        ("Sqrt",       "Sqrt transform for moderate skew"),
        ("Power",      "Box-Cox style power transform"),
    ],
    "high_cardinality_categorical": [
        ("HashEncoder",      "Fixed-size, no fit needed"),
        ("TargetEncoder",    "Best accuracy, needs y"),
        ("FrequencyEncoder", "Simple count-based, no y needed"),
    ],
    "low_cardinality_categorical": [
        ("OneHotEncoder",    "Standard, sparse by default"),
        ("OrdinalEncoder",   "If natural order exists"),
    ],
    "text_like": [
        ("StringLength",   "Length of text"),
        ("StringContains", "Pattern matching flags"),
        ("StringLower",    "Normalize case"),
    ],
    "numeric_with_outliers": [
        ("Clip",         "Clip extreme values"),
        ("RobustScaler", "Scale using median/IQR, outlier-resistant"),
    ],
}
```

These suggestions are surfaced in `sq.analyze()` and applied in `sq.autopipeline(features=True)`.

### 8.6 `sq.studio()` — Interactive EDA Dashboard (`sqlearn[studio]`)

Not a separate project. An optional install: `pip install sqlearn[studio]`.
One command to launch:

```python
import sqlearn as sq
sq.studio("data.parquet")
# → opens browser to http://localhost:8765

sq.studio("data.parquet", target="price")
# → opens with target pre-selected, analysis auto-runs

sq.studio("s3://bucket/large_data.parquet")
# → works with remote data, DuckDB streams

sq.studio(backend="warehouse.duckdb")
# → browse all tables in a persistent DuckDB database
```

**The core principle: SQL computes, JS renders.**

We NEVER send raw data to the browser. SQL pre-aggregates everything.
A histogram of 50M rows? SQL computes 30 bins + counts → sends 30 data points.
A correlation matrix of 100 columns? SQL computes → sends 100×100 numbers.
This means the dashboard is always fast, regardless of dataset size.

```sql
-- Histogram: 50M rows → 30 data points sent to browser
SELECT FLOOR(price / 10000) * 10000 AS bin, COUNT(*) AS count
FROM data GROUP BY 1 ORDER BY 1;

-- Distribution: 50M rows → 7 numbers sent
SELECT MIN(price), PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY price),
       MEDIAN(price), AVG(price), PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY price),
       MAX(price), STDDEV_POP(price)
FROM data;

-- Scatter (sampled, reproducible): 50M rows → 1000 points sent
SELECT price, sqft FROM data USING SAMPLE 1000 REPEATABLE(42);
```

**Why this beats existing tools:**

| Tool | Data size limit | Smart suggestions? | Pipeline generation? | Charting |
|---|---|---|---|---|
| dtale | RAM-bound (pandas) | No | No | Plotly (heavy) |
| ydata-profiling | RAM-bound (pandas) | Basic | No | Matplotlib (static) |
| sweetviz | RAM-bound (pandas) | No | No | Custom HTML |
| Lux | RAM-bound (pandas) | Intent-based | No | Altair |
| **sq.studio** | **Unlimited (DuckDB)** | **Yes — model-aware** | **Yes — generates code (Pro)** | **uPlot + ECharts (fast, WebGL)** |

Every existing tool loads the entire dataset into pandas. We don't.
DuckDB streams from disk/S3 and pre-aggregates in SQL. Studio works on 100GB datasets
the same way it works on 100KB datasets.

**Free vs Pro in Studio:** Free tier includes profile, analysis, recommendations,
data table, and basic charts (read-only EDA). Pro tier (license key) unlocks
pipeline builder, chart customization/export, search monitor, code generation,
project scaffolding, session persistence, and model fitting. See Section 12 for
the complete feature matrix.

#### Tech Stack — Complete Library Inventory

**Every library is FREE and OPEN SOURCE (MIT, Apache 2.0, BSD, or ISC). No exceptions.**

##### Python Backend

| Package | Purpose | License | Size |
|---|---|---|---|
| **starlette>=0.36** | ASGI web framework (routes, WebSocket, static files) | BSD 3-Clause | small |
| **uvicorn>=0.24** | ASGI server | BSD 3-Clause | small |
| **websockets>=12.0** | WebSocket protocol support | BSD 3-Clause | small |
| **python-multipart>=0.0.9** | File uploads (CSV/Parquet drag-and-drop) | Apache 2.0 | 70KB |
| **orjson>=3.9** | Fast JSON serialization (3-10x faster, native numpy support) | MIT / Apache 2.0 | 300KB |

Why Starlette not FastAPI? FastAPI = Starlette + Pydantic validation + auto-docs. We don't
need Pydantic (data comes from DuckDB, already typed) or auto-docs (local tool, not public API).

Why orjson? Studio sends DuckDB query results (often numpy arrays) as JSON. `orjson.dumps()`
handles numpy natively — no custom encoder needed. 3-10x faster than stdlib `json`.

**Evaluated and SKIPPED:** httptools (marginal on localhost), uvloop (no Windows support),
sse-starlette (WebSocket covers it), msgspec (no numpy support), Jinja2 (not needed yet),
watchfiles (dev-only). Port finding: stdlib `socket`. Browser open: stdlib `webbrowser`.

```
sqlearn/studio/
├── __init__.py          # sq.studio() entry point — starts server, opens browser
├── app.py               # Starlette application (routes + static file serving)
├── license.py           # License validation: RSA check, trial logic, require_pro()
├── api/
│   ├── profile.py       # GET /api/profile — calls sq.profile()          (free)
│   ├── analyze.py       # POST /api/analyze — calls sq.analyze()         (free)
│   ├── recommend.py     # POST /api/recommend — calls sq.recommend()     (free)
│   ├── transform.py     # POST /api/transform — preview pipeline results (free)
│   ├── license.py       # GET /api/license — license state for UI        (free)
│   ├── search.py        # WebSocket /api/search — live sq.Search progress(Pro)
│   ├── builder.py       # POST /api/builder/* — pipeline builder         (Pro)
│   ├── export.py        # POST /api/export/* — code/project/chart        (Pro)
│   ├── session.py       # POST /api/session/* — save/resume              (Pro)
│   ├── sources.py       # POST /api/sources/* — multi-source             (Pro)
│   └── model.py         # POST /api/model/* — fit/evaluate               (Pro)
├── session.py           # DuckDB connection + pipeline state management
├── static/              # pre-built Svelte frontend (Vite → dist/ → bundled in package)
│   ├── index.html
│   ├── app.js           # compiled Svelte bundle (code-split, tree-shaken)
│   └── app.css          # Tailwind CSS (purged, ~10-15KB)
└── codegen.py           # Python code generation (pipeline, script, report)
```

Starlette serves API + static frontend from one process. No Node.js at runtime.

```python
import orjson
from starlette.applications import Starlette
from starlette.routing import Route, WebSocketRoute, Mount
from starlette.responses import Response
from starlette.staticfiles import StaticFiles

class ORJSONResponse(Response):
    media_type = "application/json"
    def render(self, content):
        return orjson.dumps(content)

async def profile(request):
    result = request.app.state.session.profile()
    return ORJSONResponse(result)

async def search_ws(websocket):
    await websocket.accept()
    for trial in search.fit_iter(...):
        await websocket.send_json({"trial": trial.number, "score": trial.score})
    await websocket.close()

app = Starlette(routes=[
    Route("/api/profile", profile),
    Route("/api/analyze", analyze, methods=["POST"]),
    WebSocketRoute("/api/search", search_ws),
    Mount("/", StaticFiles(directory="static", html=True)),
])
```

**Security (localhost-only, no auth library needed):**

```python
sq.studio("data.parquet")
# → http://127.0.0.1:52847    (random available port, bound to localhost)
# Port found via stdlib: socket.bind(("127.0.0.1", 0))

sq.studio("data.parquet", port=8765)          # fixed port
sq.studio("data.parquet", token=True)         # optional token for shared machines
# → http://127.0.0.1:8765?token=a1b2c3d4
```

Binds to `127.0.0.1` (NOT `0.0.0.0`). Origin header validation (5 lines of middleware)
prevents CSRF from malicious websites. CORSMiddleware (built into Starlette) enabled
only during frontend development (Vite dev server on different port). No HTTPS, no
session cookies. Simple.

##### Frontend: Build & CSS

| Package | Purpose | License | Notes |
|---|---|---|---|
| **Vite** + **@sveltejs/vite-plugin-svelte** | Build tool | MIT | `npm run build` → `dist/` → bundled in Python package |
| **Svelte 5** | UI framework | MIT | Compiles to vanilla JS, no virtual DOM, ~5KB runtime |
| **Tailwind CSS v4** | Utility-first CSS | MIT | ~10-15KB purged. Required by shadcn-svelte. No config file in v4. |

**Vite, NOT SvelteKit.** SvelteKit is a full app framework (routing, SSR, adapters). We're
building a single-page app pre-compiled to static files. Vite + vite-plugin-svelte is
all we need: HMR in dev, tree-shaking + minification in build.

##### Frontend: UI Components & Interactions

| Package | Purpose | License | Size (gz) |
|---|---|---|---|
| **shadcn-svelte** (on **Bits UI**) | UI primitives: modals, dropdowns, tooltips, tabs, sheets, accordions, sliders, context menus | MIT | ~0 (copy-paste components, tree-shakeable Bits UI) |
| **Floating UI** (@floating-ui/dom) | Rich tooltip positioning (for help hints with charts/HTML) | MIT | ~3-5KB |
| **Lucide** (lucide-svelte) | Icons: column types, actions, navigation | ISC (MIT-compat) | ~200-400B per icon |
| **svelte-dnd-action** | Drag-and-drop for pipeline builder | MIT | ~15KB |
| **svelte-sonner** | Toast notifications ("Pipeline saved", "Export complete") | MIT | ~5KB |
| **svelte-awesome-color-picker** | Chart color customization (alpha channel support) | MIT | ~8KB |
| **tinykeys** | Keyboard shortcuts (Ctrl+S save, Ctrl+Z undo) | MIT | ~0.65KB |
| **highlight.js** (Python grammar only) | Syntax highlighting for generated code | BSD 3-Clause | ~7KB |
| **marked** | Markdown rendering for help text and tooltips | MIT | ~14KB |

**shadcn-svelte** is the core UI layer. It's copy-paste components built on Bits UI (headless
accessible primitives) + Tailwind CSS. 40+ components. No runtime dependency — code lives
in our project. Covers: Dialog (modals), DropdownMenu, Tooltip, Tabs, Sheet (sidebars),
Accordion, Slider, ContextMenu (right-click), Select, Popover, Command (search palette).

**Help hints (❓ hover):** shadcn-svelte `<Tooltip>` for simple text. Floating UI for rich
tooltips with formatted content, mini charts, or suggestions. Pattern:

```svelte
<div class="flex items-center gap-1">
  <span>Standard Scaler</span>
  <HelpHint>
    Subtracts mean and divides by standard deviation.
    **When to use:** Linear models, KNN, neural networks.
    **Skip if:** Tree-based models (XGBoost, RandomForest).
  </HelpHint>
</div>
```

##### Frontend: Charts & Data

| Package | Purpose | License | Size (gz) |
|---|---|---|---|
| **ECharts** (tree-shaken) | Heatmaps, box plots, bar, scatter, parallel coords | Apache 2.0 | ~135KB |
| **uPlot** | Time series, convergence curves, line charts | MIT | ~35KB |
| **TanStack Table** (@tanstack/svelte-table) | Headless data table (sort, filter, paginate, select) | MIT | ~15KB |
| **TanStack Virtual** (@tanstack/svelte-virtual) | Virtual scrolling for wide/long tables | MIT | ~10-15KB |

**ECharts over AntV G2:** G2 is bigger (319KB gzipped vs 135KB tree-shaken ECharts),
has 15+ deps vs 1 (zrender), 500x fewer npm downloads. Not close.

**TanStack Table over AntV S2:** S2 is ~300KB+ canvas-based pivot table. SQL does our
sort/filter/paginate/pivot. Table just renders + handles clicks. Headless = full Svelte
DOM control for dtale-like interactions.

**Svelte 5 compatibility note:** TanStack Table v8 adapter doesn't work with Svelte 5.
Use `tanstack-table-8-svelte-5` drop-in by dummdidumm (Svelte core team member).
TanStack Table v9 (alpha) will have native Svelte 5 support.

**Chart customization — users can:**

```js
// 1. Runtime theme switching via ECharts registerTheme()
echarts.registerTheme('sqlearn-dark', {
  color: ['#4992ff', '#7cffb2', '#fddd60', '#ff6e76'],
  backgroundColor: '#100c2a',
  textStyle: { color: '#b9b8ce' },
});

// 2. Change colors, titles, axes at runtime via setOption() merge
chart.setOption({
  color: ['#e74c3c', '#2ecc71', '#3498db'],
  title: { text: userTitle },
  xAxis: { name: userXLabel },
});

// 3. Interactive legend — click to toggle series on/off (built-in)
// 4. DataZoom — mouse wheel zoom + slider bar (built-in)
// 5. Brush selection — select regions, get data indices (built-in)
// 6. Magic type switching — toggle between line/bar/stack (built-in toolbox)
```

**Column/series selection:** User picks which columns to display via a multi-select dropdown.
Frontend sends selected columns to backend → SQL query with only those columns →
chart updates. ECharts legend toggles individual series on/off without re-querying.

**Color picker integration:** User clicks a series color swatch → svelte-awesome-color-picker
opens → user picks color → `chart.setOption({ series: [{ itemStyle: { color: newColor } }] })`.

##### Frontend: Export (lazy-loaded — only when user clicks Export)

| Package | Purpose | License | Size (gz) |
|---|---|---|---|
| **jsPDF** | Chart/report → PDF | MIT | ~96KB |
| **ExcelJS** | Data → Excel (.xlsx) with styled cells | MIT | ~330KB |
| **html-to-image** | Dashboard screenshot → PNG | MIT | ~5KB |
| Native `navigator.clipboard` | Copy code/data to clipboard | N/A | 0 |
| Native `canvas.toBlob()` / ECharts `getDataURL()` | Chart → PNG/SVG export | N/A | 0 |
| Manual `Blob` + `URL.createObjectURL` | Data → CSV | N/A | 0 |

**These are dynamically imported only when the user clicks Export.** They add 0 to initial
page load. Total: ~431KB loaded on-demand.

**Chart export workflow:**
```js
// PNG (high-res for presentations)
const dataURL = chart.getDataURL({ type: 'png', pixelRatio: 3, backgroundColor: '#fff' });

// SVG (vector, infinite zoom)
// Use renderer: 'svg' mode, then chart.getDataURL({ type: 'svg' })

// PDF (multi-chart report)
const pdf = new jsPDF();
pdf.addImage(chart1DataURL, 'PNG', 10, 10, 180, 100);
pdf.addPage();
pdf.addImage(chart2DataURL, 'PNG', 10, 10, 180, 100);
pdf.save('analysis-report.pdf');
```

**Why ExcelJS over SheetJS CE:** SheetJS Community Edition can't write cell styles (bold,
colors, borders). Styling is essential for EDA exports. ExcelJS is MIT, full styling.

**Standalone HTML report:** Jinja2 template + inline ECharts JS + embedded JSON data →
single self-contained HTML file (~1MB). Same approach as ydata-profiling but interactive.

##### Evaluated and REJECTED

| Library | Why rejected |
|---|---|
| **AntV G2** (319KB) | Bigger than ECharts, 15+ deps, tiny community |
| **AntV S2** (~300KB+) | Canvas table overkill — SQL does sort/filter/pivot |
| **Plotly** (3.5MB) | 17x larger than our entire frontend |
| **AG Grid Community** (140-298KB) | No official Svelte adapter, context menus = Enterprise-only |
| **Handsontable** | **PROPRIETARY license since v7.0 — DISQUALIFIED** |
| **FastAPI** | Pydantic overhead unnecessary for localhost tool |
| **SvelteKit** | Full framework overkill — Vite alone suffices for SPA |
| **Shiki** (~700KB) | Beautiful but too heavy for client-side highlighting |
| **Tippy.js** | Unmaintained (4+ years). Floating UI is its successor. |
| **borb** (Python PDF) | **AGPL-3.0 — DISQUALIFIED** (copyleft) |
| **fpdf2** (Python PDF) | LGPL-3.0 — flagged, not fully permissive |

##### Bundle Size Summary

| Layer | Size (gzipped) | Loaded |
|---|---|---|
| Svelte 5 runtime | ~5KB | Immediate |
| Tailwind CSS (purged) | ~10-15KB | Immediate |
| UI components (shadcn/Bits) | ~10-20KB (tree-shaken) | Immediate |
| ECharts (tree-shaken) | ~135KB | Immediate |
| uPlot | ~35KB | Immediate |
| TanStack Table + Virtual | ~25-30KB | Immediate |
| UI utilities (icons, DnD, toasts, etc.) | ~50KB | Immediate |
| **Initial page load total** | **~270-330KB** | |
| Export libs (jsPDF + ExcelJS + html-to-image) | ~431KB | Lazy (on Export click) |
| **Maximum total** | **~700-760KB** | |

Compare: Plotly alone = 3.5MB. dtale frontend = ~5MB. Our maximum = 760KB, initial = 330KB.

```
Frontend source tree (Svelte + TypeScript):
├── src/
│   ├── lib/
│   │   ├── components/
│   │   │   ├── ui/               # shadcn-svelte components (copy-pasted, ours to customize)
│   │   │   │   ├── button/
│   │   │   │   ├── dialog/
│   │   │   │   ├── dropdown-menu/
│   │   │   │   ├── tooltip/
│   │   │   │   ├── context-menu/
│   │   │   │   ├── tabs/
│   │   │   │   ├── sheet/        # sidebar
│   │   │   │   ├── select/
│   │   │   │   ├── slider/
│   │   │   │   ├── popover/
│   │   │   │   └── command/      # search palette (Ctrl+K)
│   │   │   ├── Chart.svelte             # ECharts wrapper (resize, theme, export)
│   │   │   ├── TimeSeriesChart.svelte   # uPlot wrapper (convergence curves)
│   │   │   ├── ColumnCard.svelte        # column summary (type icon, stats, sparkline)
│   │   │   ├── Suggestion.svelte        # clickable suggestion with ❓ help hint
│   │   │   ├── PipelineStep.svelte      # draggable step block (svelte-dnd-action)
│   │   │   ├── CodePreview.svelte       # highlight.js Python code + copy button
│   │   │   ├── DataTable.svelte         # TanStack Table wrapper (SQL-powered)
│   │   │   ├── CellPopup.svelte         # right-click cell → stats/filter/histogram
│   │   │   ├── ColorPicker.svelte       # chart color customization
│   │   │   ├── HelpHint.svelte          # ❓ icon → rich tooltip (Floating UI)
│   │   │   └── ExportDialog.svelte      # export options (lazy-loads jsPDF/ExcelJS)
│   │   ├── stores/
│   │   │   ├── session.ts         # DuckDB connection state
│   │   │   ├── pipeline.ts        # current pipeline being built
│   │   │   ├── suggestions.ts     # active suggestions from sq.recommend()
│   │   │   ├── theme.ts           # light/dark mode, chart color palette
│   │   │   └── shortcuts.ts       # tinykeys keyboard shortcut bindings
│   │   └── utils/
│   │       ├── api.ts             # fetch/WebSocket helpers
│   │       ├── sql-to-filter.ts   # convert TanStack filter state → SQL WHERE
│   │       └── export.ts          # lazy import jsPDF/ExcelJS, export helpers
│   ├── views/
│   │   ├── Profile.svelte           # data overview: types, nulls, stats, distributions
│   │   ├── Analysis.svelte          # correlations heatmap, target analysis
│   │   ├── Columns.svelte           # per-column deep dive (distribution, outliers)
│   │   ├── Recommend.svelte         # model + pipeline suggestions
│   │   ├── PipelineBuilder.svelte   # visual pipeline construction (drag-and-drop)
│   │   ├── Search.svelte            # live hyperparameter search + convergence
│   │   └── Export.svelte            # export options: Python, pipeline, report
│   ├── App.svelte                   # root layout: sidebar + main content area
│   └── main.ts                      # entry point
├── vite.config.ts                   # Vite build config
├── tailwind.config.ts               # Tailwind v4 (minimal — mostly in CSS)
├── package.json
└── tsconfig.json
```

#### Data Table — dtale-like but SQL-powered

**TanStack Table** (headless, ~25KB with Virtual) renders a Svelte-native table. Every
interaction maps to a SQL query — the browser never sorts, filters, or aggregates data:

| User action | What happens | SQL |
|---|---|---|
| Click column header | Sort toggle ▲▼ | `ORDER BY col ASC/DESC` |
| Right-click column header | Column menu (rename, hide, type, stats) | `SELECT stats...` |
| Click cell value | Filter to this value | `WHERE col = value` |
| Right-click cell | Popup: value count, % of total, null %, mini histogram | `SELECT COUNT(*), ...` |
| Select multiple cells | Compare: unique values, range, distribution | `SELECT ... WHERE col IN (...)` |
| Column resize/reorder | Pure frontend — drag handles | — |
| Scroll to bottom | Next page auto-loads | `LIMIT 100 OFFSET N` |
| Type in filter bar | Live filter with debounce | `WHERE col LIKE '%query%'` |

**CellPopup** — right-click any cell to get a dtale-like context menu:

```
┌─────────────────────────┐
│ "San Francisco"         │
├─────────────────────────┤
│ Count: 2,847 (5.7%)     │
│ Unique values: 342      │
│ ┌───────────────────┐   │
│ │ ▁▂▇▅▃▂▁ (top 10)  │   │
│ └───────────────────┘   │
├─────────────────────────┤
│ ▶ Filter to this value  │
│ ▶ Exclude this value    │
│ ▶ Sort ascending        │
│ ▶ Add to pipeline       │
│ ▶ Show column stats     │
└─────────────────────────┘
```

Every popup stat is a single SQL query. Even on a 100M-row dataset, the popup appears
instantly because DuckDB aggregates are fast and we query by indexed column values.

**Virtual scrolling** — TanStack Virtual renders only visible rows in the DOM. For wide
tables (50+ columns), this prevents DOM bloat. Combined with SQL pagination, the table
handles any dataset size smoothly.

#### User Experience Flow

**Step 1 — Profile (automatic on launch)**

User runs `sq.studio("data.parquet")`. Browser opens. Dashboard shows:

```
┌─────────────────────────────────────────────────────────────────────┐
│  sqlearn studio                               data.parquet  50K rows │
├──────────────┬──────────────────────────────────────────────────────┤
│              │                                                      │
│  Columns     │   Overview                                           │
│  ─────────   │   ┌──────────────────────────────────────────────┐   │
│  ▣ age       │   │ 15 columns: 8 numeric, 5 categorical, 2 date│   │
│  ▣ income    │   │ 3 columns have nulls                         │   │
│  ▣ sqft      │   │ 1 constant column (country)                  │   │
│  ▣ bedrooms  │   └──────────────────────────────────────────────┘   │
│  ▣ price     │                                                      │
│  # city      │   Distributions          Null Map                    │
│  # type      │   ┌──────────────┐       ┌──────────────┐           │
│  # zip_code  │   │ ▁▂▃▅▇▅▃▂▁   │       │ ██████░░██   │           │
│  ◷ created   │   │   price      │       │ █████████░   │           │
│              │   └──────────────┘       └──────────────┘           │
│  ⚠ Warnings  │                                                      │
│  • country   │   [Select target column ▾]                           │
│    constant  │                                                      │
│  • score 15% │                                                      │
│    nulls     │                                                      │
└──────────────┴──────────────────────────────────────────────────────┘
```

Left sidebar: all columns with type icons (▣=numeric, #=categorical, ◷=datetime).
Click any column → deep dive with full distribution, outlier analysis, null patterns.

**Step 2 — Select target → Analysis auto-runs**

User clicks "price" as target. Dashboard switches to analysis view:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Analysis    Target: price (regression)        ◀ Profile │ Recommend ▶│
├──────────────┬──────────────────────────────────────────────────────┤
│              │                                                      │
│ Correlations │  Correlation with price         Suggestions          │
│ with target  │  ┌─────────────────────┐       ┌──────────────────┐ │
│ ──────────── │  │ sqft      ████ 0.82 │       │ 💡 price is      │ │
│ sqft    0.82 │  │ bedrooms  ███░ 0.71 │       │ right-skewed     │ │
│ beds    0.71 │  │ bathrooms ███░ 0.68 │       │ → try Log()      │ │
│ baths   0.68 │  │ age       ██░░ 0.34 │       │                  │ │
│ age     0.34 │  │ cust_id   ░░░░ 0.01 │       │ 💡 bedrooms and  │ │
│ score   0.28 │  └─────────────────────┘       │ bathrooms r=0.89 │ │
│ cust_id 0.01 │                                │ → drop one       │ │
│              │  Correlation Heatmap            │                  │ │
│ Categorical  │  ┌─────────────────────┐       │ 💡 zip_code has  │ │
│ ──────────── │  │ ■■■□□               │       │ 847 unique vals  │ │
│ type    0.62 │  │ ■■■■□               │       │ → HashEncoder    │ │
│ city    0.45 │  │ □□■■■               │       │                  │ │
│              │  └─────────────────────┘       │ [Apply all ▶]    │ │
└──────────────┴────────────────────────────────┴──────────────────┘
```

Right sidebar shows smart suggestions. Each suggestion is clickable — clicking it
adds the corresponding step to the pipeline being built.

**Step 3 — Build pipeline (click suggestions or drag-and-drop)**

User clicks suggestions or manually adds steps:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Pipeline Builder                              ◀ Analysis │ Search ▶ │
├──────────────┬──────────────────────────────────────────────────────┤
│              │                                                      │
│  Available   │  Current Pipeline                                    │
│  Steps       │  ┌──────────────────────────────────────────────┐   │
│  ──────────  │  │ ┌──────────┐   ┌────────────┐   ┌─────────┐ │   │
│  ▶ Imputer   │  │ │ Imputer  │→ │ DateParts   │→ │ OneHot  │ │   │
│  ▶ Scaler    │  │ │ age:med  │   │ created_at  │   │ city    │ │   │
│  ▶ Encoder   │  │ │ score:med│   │ month,dow   │   │ type    │ │   │
│  ▶ Features  │  │ └──────────┘   └────────────┘   └─────────┘ │   │
│  ▶ Window    │  │     ↓                                        │   │
│              │  │ ┌────────────┐   ┌──────────────────────┐    │   │
│  Drag to add │  │ │ HashEncoder│→ │ StandardScaler       │    │   │
│  or click    │  │ │ zip_code   │   │ (only if linear)     │    │   │
│  suggestions │  │ │ n_bins=64  │   │                      │    │   │
│              │  │ └────────────┘   └──────────────────────┘    │   │
│              │  └──────────────────────────────────────────────┘   │
│              │                                                      │
│              │  Preview (first 5 rows)         Generated Code       │
│              │  ┌─────────────────────┐       ┌──────────────────┐ │
│              │  │ age  sqft  city_NY  │       │ pipe = sq.Pipe([ │ │
│              │  │ 34   1200  1       │       │   sq.Imputer({   │ │
│              │  │ 28   850   0       │       │     "age":"med"  │ │
│              │  │ ...                │       │   }),             │ │
│              │  └─────────────────────┘       │   ...            │ │
│              │                                └──────────────────┘ │
└──────────────┴──────────────────────────────────────────────────────┘
```

Bottom-left: live preview of transformed data (SQL: `SELECT * FROM ... LIMIT 5`).
Bottom-right: live Python code generation — updates as user modifies pipeline.
User can copy the code at any time.

**Step 4 — Search (optional)**

User clicks "Search" → configures and runs `sq.Search` with live WebSocket updates:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Hyperparameter Search                         ◀ Pipeline │ Export ▶ │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Trial 47/100 │ Round 2/4 │ Best: 0.847 (f1) │ ETA: 3m 12s         │
│  ████████████████████░░░░░░░░░░ 47%                                  │
│                                                                      │
│  Convergence              Parameter Importance                       │
│  ┌─────────────────┐     ┌─────────────────────┐                    │
│  │     ╱──────     │     │ lr        ████ 0.42 │                    │
│  │   ╱             │     │ n_est     ███░ 0.28 │                    │
│  │  ╱              │     │ impute    ██░░ 0.18 │                    │
│  │ ╱               │     │ depth     █░░░ 0.12 │                    │
│  └─────────────────┘     └─────────────────────┘                    │
│                                                                      │
│  Top 5 Configs                                                       │
│  ┌────────────────────────────────────────────────────────────┐      │
│  │ #  Score   lr      n_est  depth  impute                   │      │
│  │ 1  0.847   0.052   340    7      median                   │      │
│  │ 2  0.845   0.048   420    6      median                   │      │
│  │ 3  0.843   0.061   280    8      median                   │      │
│  └────────────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────────┘
```

Live WebSocket streaming — charts update in real-time as trials complete.

**Step 5 — Export**

User clicks "Export" and chooses format:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Export                                                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐     │
│  │  Python Script   │  │  Pipeline Only    │  │  Full Report    │    │
│  │  ─────────────   │  │  ──────────────   │  │  ────────────   │    │
│  │  Complete .py    │  │  Just the pipe    │  │  HTML report    │    │
│  │  with imports,   │  │  definition.      │  │  with all       │    │
│  │  data loading,   │  │  Copy-paste into  │  │  charts and     │    │
│  │  fit, transform, │  │  your own code.   │  │  analysis.      │    │
│  │  model training. │  │                   │  │                 │    │
│  │  [Download .py]  │  │  [Copy to clip]   │  │  [Download]     │    │
│  └─────────────────┘  └──────────────────┘  └────────────────┘     │
│                                                                      │
│  Code Preview:                                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ import sqlearn as sq                                         │   │
│  │ from xgboost import XGBClassifier                            │   │
│  │                                                              │   │
│  │ pipe = sq.Pipeline([                                         │   │
│  │     sq.Imputer({"age": "median", "score": "median"}),       │   │
│  │     sq.DateParts(columns=["created_at"], parts=["month"]),  │   │
│  │     sq.OneHotEncoder(columns=["city", "type"]),             │   │
│  │     sq.HashEncoder(columns=["zip_code"], n_bins=64),        │   │
│  │ ])                                                           │   │
│  │                                                              │   │
│  │ pipe.fit("data.parquet")                                     │   │
│  │ X = pipe.transform("data.parquet")                           │   │
│  │ y = sq.read_column("data.parquet", "price")                  │   │
│  │ model = XGBClassifier(n_estimators=340, max_depth=7)         │   │
│  │ model.fit(X, y)                                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

#### Pre-Aggregation: Why It's Always Fast

The secret sauce. Every chart in the dashboard is backed by a SQL aggregate query,
not raw data. The browser never sees more than a few hundred data points.

| Chart type | SQL query | Data sent to browser |
|---|---|---|
| Histogram | `GROUP BY FLOOR(col/bin_width)` | ~30 (bin, count) pairs |
| Box plot | `PERCENTILE_CONT(0.25/0.5/0.75)` | 5 numbers per column |
| Scatter | `USING SAMPLE 1000 REPEATABLE(42)` | 1000 points max (deterministic) |
| Correlation heatmap | `CORR(a, b)` for all pairs | N×N matrix |
| Null pattern matrix | `SUM(CASE WHEN col IS NULL...)` | N×M counts |
| Distribution | `NTILE(100)` percentiles | 100 numbers |
| Value counts | `GROUP BY col ORDER BY count DESC LIMIT 20` | 20 (val, count) pairs |
| Time series | `GROUP BY DATE_TRUNC('month', col)` | ~50 aggregated points |

For a 100GB dataset, the dashboard loads in seconds. Not minutes. Not "please wait."
The same queries that power `sq.profile()` and `sq.analyze()` power the dashboard.

#### WebSocket for Live Updates

Long-running operations (Search, large profile) stream results via WebSocket:

```python
# Starlette WebSocket
async def search_ws(websocket):
    await websocket.accept()
    search = sq.Search(...)
    for trial in search.fit_iter("data", y="target"):
        await websocket.send_json({
            "trial": trial.number,
            "score": trial.score,
            "params": trial.params,
            "best": search.best_score_,
        })
    await websocket.close()
```

Frontend chart updates in real-time — user sees convergence curve grow as trials complete.

#### Dependencies (`sqlearn[studio]`)

```toml
[project.optional-dependencies]
studio = [
    "starlette>=0.36",
    "uvicorn>=0.24",
    "websockets>=12.0",
    "python-multipart>=0.0.9",    # file uploads (CSV/Parquet drag-and-drop)
    "orjson>=3.9",                # fast JSON + native numpy serialization
]
```

That's 5 Python packages. Frontend is pre-compiled Svelte → static files bundled in
the Python package. No Node.js at runtime. No npm.
Just `pip install sqlearn[studio]` and `sq.studio()`.

Frontend NPM dependencies (dev-time only, compiled away at build):
```json
{
  "devDependencies": {
    "@sveltejs/vite-plugin-svelte": "^4.0",
    "svelte": "^5.0",
    "vite": "^6.0",
    "tailwindcss": "^4.0",
    "typescript": "^5.5"
  },
  "dependencies": {
    "bits-ui": "^1.0",
    "@floating-ui/dom": "^1.6",
    "lucide-svelte": "^0.460",
    "svelte-dnd-action": "^0.9",
    "svelte-sonner": "^0.3",
    "svelte-awesome-color-picker": "^4.0",
    "tinykeys": "^3.0",
    "highlight.js": "^11.10",
    "marked": "^15.0",
    "echarts": "^5.5",
    "uplot": "^1.6",
    "@tanstack/svelte-table": "npm:tanstack-table-8-svelte-5@^0.1",
    "@tanstack/svelte-virtual": "^3.13",
    "jspdf": "^2.5",
    "exceljs": "^4.4",
    "html-to-image": "^1.11"
  }
}
```

**Every single dependency is MIT, Apache 2.0, BSD, or ISC. Zero proprietary code.**

#### How It Compares

```
dtale:
  ✗ Loads entire dataset into pandas (RAM-bound)
  ✗ Plotly charts (3.5MB, slow with large data)
  ✗ No model suggestions, no pipeline generation
  ✗ Separate install, separate ecosystem

ydata-profiling:
  ✗ Loads entire dataset into pandas (RAM-bound)
  ✗ Static HTML report (not interactive)
  ✗ Slow on >100K rows
  ✗ No pipeline generation, no model suggestions

sq.studio:
  ✓ DuckDB backend — unlimited data size, streams from S3
  ✓ Pre-aggregated SQL → always fast, any dataset
  ✓ ~330KB initial load (ECharts + uPlot + TanStack + shadcn-svelte + Tailwind)
  ✓ Smart suggestions with ❓ help hints on every option
  ✓ Fully customizable charts — colors, themes, column selection, zoom, brush
  ✓ dtale-like interactive data table — click/right-click cells for stats
  ✓ Pipeline builder — generates working sq.Pipeline code
  ✓ Hyperparameter search — live monitoring via WebSocket
  ✓ Export: Python script, PDF report, Excel, CSV, PNG/SVG charts, HTML report
  ✓ Keyboard shortcuts, toasts, dark mode, drag-and-drop
  ✓ Same package — pip install sqlearn[studio]
  ✓ 100% free, 100% open source — every dependency MIT/Apache/BSD
```

---

### 8.7 `sq.quality()` — Data Quality Score (Level 0)

Before anything else, answer: **"Is my data ready for ML?"**

One function, one number, actionable breakdown.

```python
report = sq.quality("data.parquet", target="price")
print(report)
```

```
═══ sqlearn Data Quality Score ═══
Source: data.parquet | 50,000 rows × 15 columns | Target: price

Score: 68 / 100

Critical Issues (-20):
  ✗ score: 15.3% nulls — imputation will affect 7,650 rows
  ✗ zip_code: 847 unique values — OneHotEncoder would create 847 columns

Warnings (-12):
  ⚠ country: constant column (1 unique value "US") — zero information
  ⚠ bedrooms ↔ bathrooms: r=0.89 — multicollinearity
  ⚠ price (target): right-skewed (skew=2.3) — consider Log transform
  ⚠ customer_id: unique per row (100%) — likely identifier, not feature

Suggestions:
  💡 created_at: datetime → extract DateParts (month, dayofweek, hour)
  💡 tags: contains commas in 78% of rows → StringSplit candidate
  💡 income: right-skewed (1.8) → consider Log transform
```

```python
report.score               # 68
report.critical            # list of Critical objects
report.warnings            # list of Warning objects
report.suggestions         # list of Suggestion objects
report.is_ready            # True if score >= 60 (configurable)

# Scoring breakdown
report.scoring
# {
#   "null_penalty": -10,       # columns with >5% nulls
#   "constant_penalty": -3,    # constant columns
#   "high_cardinality": -10,   # OHE would create >100 columns
#   "multicollinearity": -4,   # pairs with |r| > 0.85
#   "target_skewness": -5,     # |skew| > 2
#   "identifier_columns": -3,  # unique-per-row columns still in data
#   "class_imbalance": 0,      # only for classification (minority < 10%)
# }
```

Scoring rules (SQL-based, all computed in one query):

| Check | Penalty | Threshold | SQL |
|---|---|---|---|
| High nulls | -5 per column | >5% nulls | `SUM(CASE WHEN col IS NULL THEN 1 ELSE 0 END)::FLOAT / COUNT(*)` |
| Constant columns | -3 per column | 1 unique value | `COUNT(DISTINCT col) = 1` |
| High cardinality | -5 per column | >100 unique for categorical | `COUNT(DISTINCT col)` |
| Multicollinearity | -2 per pair | \|r\| > 0.85 | `CORR(a, b)` |
| Target skewness | -5 | \|skew\| > 2 | `(AVG(x) - MEDIAN(x)) / NULLIF(STDDEV(x), 0) * 3` approx |
| Identifier columns | -3 per column | >99% unique | `COUNT(DISTINCT col)::FLOAT / COUNT(*) > 0.99` |
| Class imbalance | -5 | minority < 10% | `COUNT(*) GROUP BY target` |
| Near-zero variance | -2 per column | variance < 0.01 | `VAR_POP(col)` |
| Duplicate rows | -3 | >1% exact duplicates | `COUNT(*) - COUNT(DISTINCT *)` |

### 8.8 `sq.check()` — Leakage & Mistake Detection

Detects common ML mistakes that silently corrupt results. Run before training.

```python
pipe = sq.Pipeline([
    sq.Imputer(),
    sq.StandardScaler(),
    sq.OneHotEncoder(),
])
pipe.fit("train.parquet", y="target")

# Check for problems
issues = sq.check(pipe, "train.parquet", y="target")
print(issues)
```

```
═══ sqlearn Safety Check ═══

LEAKAGE RISKS:
  ✗ customer_id: unique per row (100%) — memorizes, doesn't generalize
    Fix: sq.Drop(columns=["customer_id"])

  ⚠ total_next_month: r=0.97 with target 'price' — suspiciously high
    Is this column known BEFORE the prediction? If not → target leakage
    Fix: verify this isn't from the future, or sq.Drop(columns=["total_next_month"])

COMMON MISTAKES:
  ⚠ StandardScaler applied to all 5 numeric columns, but you're likely using
    a tree-based model (XGBoost/RF don't need scaling)
    Fix: remove StandardScaler, or specify columns= for only linear features

  ⚠ OneHotEncoder on zip_code: 847 categories → 847 columns
    Fix: sq.HashEncoder(columns=["zip_code"], n_bins=64) or sq.TargetEncoder()

  ✓ No target column in features — good
  ✓ No duplicate rows in training data — good
  ✓ Cross-validation will use two-phase discovery — safe
```

```python
issues.leakage           # list of LeakageWarning objects
issues.mistakes          # list of MistakeWarning objects
issues.passed            # True if no critical issues
issues.fix_suggestions   # dict of {column: suggested_code}
```

**Leakage detection rules (all SQL-based):**

| Check | What it catches | SQL |
|---|---|---|
| Identifier columns | Unique-per-row columns (customer_id, row_id) | `COUNT(DISTINCT col) / COUNT(*) > 0.99` |
| Target proxies | Columns with \|r\| > 0.95 to target | `ABS(CORR(col, target)) > 0.95` |
| Constant after split | Column is constant in train but varies in full data | Compare `COUNT(DISTINCT)` between train and full |
| Future data | Timestamp columns after target event | Heuristic: datetime col with values after target timestamp |
| Duplicate rows in train+test | Same row appears in both splits | `INTERSECT` between train and test |

**Common mistake detection rules:**

| Check | What it catches | How |
|---|---|---|
| Scaling + trees | StandardScaler with tree-based model | If `model=` is tree family, warn about unnecessary scaling |
| OHE + high cardinality | OneHotEncoder on >100 categories | `COUNT(DISTINCT)` check |
| Already-encoded | Binary 0/1 column being OneHotEncoded | `COUNT(DISTINCT col) = 2 AND MIN(col) = 0 AND MAX(col) = 1` |
| Already-scaled | Column with mean≈0, std≈1 being re-scaled | `ABS(AVG(col)) < 0.1 AND ABS(STDDEV(col) - 1) < 0.1` |
| Target in features | Target column not excluded | Check `y` column against pipeline columns |
| Redundant steps | Same transformer type applied twice | Pipeline step deduplication check |

**With train/test split — cross-set leakage detection:**

```python
issues = sq.check(pipe, train="train.parquet", test="test.parquet", y="target")
# Additional checks:
#   ✗ 234 rows appear in both train and test — data leakage
#   ⚠ train and test have different 'city' distributions (KL divergence > 0.5)
#   ⚠ 'date' column: test dates overlap with train dates — possible temporal leakage
```

### 8.9 `pipe.audit()` — Preprocessing Audit Trail

After fitting, trace exactly what happened to every column through the pipeline.

```python
pipe.fit("train.parquet", y="target")
pipe.audit("train.parquet")
```

```
═══ Pipeline Audit ═══
15 input columns → 42 output columns (3 steps)

Column: price (DOUBLE)
  Raw:          [12000, 450000] | mean=65,420 | std=28,100 | 23 nulls (0.05%) | skew=2.3
  → Imputer:    filled 23 nulls with median=52,000
  → Scaler:     (x - 65,420) / 28,100 → [-1.90, 13.69] | mean=0.00 | std=1.00
  Output:       DOUBLE, 0 nulls, range [-1.90, 13.69]

Column: city (VARCHAR)
  Raw:          147 unique | "London" (12%), "Paris" (8%), ... | 250 nulls (0.5%)
  → Imputer:    filled 250 nulls with mode="London"
  → OHE:        147 unique → 147 binary columns (city_london, city_paris, ...)
  Output:       147 INTEGER columns, 0 nulls, values {0, 1}
  ⚠ 147 OHE columns is high — consider HashEncoder or TargetEncoder

Column: customer_id (INTEGER)
  Raw:          50,000 unique (100%) | range [1, 50000]
  → Imputer:    no nulls, passed through
  → Scaler:     (x - 25,000) / 14,434 → [-1.73, 1.73]
  ⚠ This column is unique per row — likely an identifier, not a feature
  ⚠ Scaling an ID column is meaningless — consider sq.Drop(columns=["customer_id"])
```

```python
# Programmatic access
audit = pipe.audit("train.parquet")
audit["price"].steps              # [AuditStep(name="Imputer", ...), AuditStep(name="Scaler", ...)]
audit["price"].before             # ColumnStats(mean=65420, nulls=23, ...)
audit["price"].after              # ColumnStats(mean=0.0, nulls=0, ...)
audit["price"].warnings           # ["identifier column", "scaling meaningless"]
audit.all_warnings                # all warnings across all columns
```

**How it works:** For each step in the pipeline, `audit()` runs `sq.profile()` on the
intermediate output (using SQL `LIMIT` for preview). This means N+1 profile queries for
N steps — more expensive than normal transform, but it's a debugging tool, not a
production path. Use `audit(sample=1000)` to profile a sample instead of full data.

### 8.10 `sq.missing_analysis()` — Missing Value Pattern Detection

Beyond "X% null" — understand WHY values are missing.

```python
report = sq.missing_analysis("data.parquet")
print(report)
```

```
═══ Missing Value Analysis ═══
Source: data.parquet | 50,000 rows | 3 columns with nulls

Column      Null %   Pattern     Evidence
─────────   ──────   ────────    ─────────
score       15.3%    MAR         Nulls correlate with age (r=0.42) — older users less likely to have score
age         2.1%     MCAR        No correlation with other columns (max |r|=0.03)
city        0.5%     MCAR        No correlation with other columns (max |r|=0.01)

Missing Co-occurrence:
  score + age: 89 rows have BOTH missing (1.7x expected by chance)
  → These columns may share a missing mechanism

MCAR Test (Little's test):
  Chi-squared = 23.4, p = 0.003
  Interpretation: Nulls are NOT completely random (p < 0.05)
  At least one column has non-random missing pattern (likely 'score')

Recommendations:
  score: MAR — imputation with predictive model (using correlated 'age') is better
         than median imputation. Or: sq.Imputer(strategy="median") + add is_null flag:
         sq.custom("CASE WHEN {col} IS NULL THEN 1 ELSE 0 END AS {col}_was_null",
                   columns=["score"])
  age:   MCAR — simple imputation (median) is appropriate
  city:  MCAR — simple imputation (mode) is appropriate
```

**Missing pattern classification (SQL-based):**

| Pattern | Meaning | Detection (SQL) | Imputation advice |
|---|---|---|---|
| **MCAR** | Missing Completely At Random | No correlation between null indicator and other columns | Any imputation method works |
| **MAR** | Missing At Random (depends on other columns) | Null indicator correlates with observed columns | Use correlated columns to predict missing values, or add `_was_null` flag |
| **MNAR** | Missing Not At Random (depends on the missing value itself) | Cannot detect from data alone — requires domain knowledge | Flag + warn user. Suggest `_was_null` feature. |

**SQL for pattern detection:**

```sql
-- Create null indicator for each column with nulls
-- Then correlate with all other columns
SELECT
    CORR(CASE WHEN score IS NULL THEN 1 ELSE 0 END, age) AS score_null_vs_age,
    CORR(CASE WHEN score IS NULL THEN 1 ELSE 0 END, income) AS score_null_vs_income,
    CORR(CASE WHEN age IS NULL THEN 1 ELSE 0 END, score) AS age_null_vs_score,
    ...
FROM data
-- If any |r| > 0.1 → MAR (nulls depend on other columns)
-- If all |r| < 0.1 → MCAR (nulls are random)
-- MNAR can't be detected from data — always warn about possibility
```

**Little's MCAR test (SQL-approximation):**

Full Little's test requires matrix operations (not SQL). sqlearn uses a SQL-friendly
approximation: chi-squared test comparing observed null patterns against expected
(under MCAR assumption). If significant (p < 0.05), nulls are not completely random.

```sql
-- Count each null pattern (which combination of columns is null)
SELECT
    (score IS NULL)::INT AS score_null,
    (age IS NULL)::INT AS age_null,
    COUNT(*) AS pattern_count
FROM data
GROUP BY 1, 2
-- Compare observed frequencies to expected under independence
```

### 8.11 `sq.feature_importance()` — Pre-Model Feature Ranking

Rank features BEFORE training a model. Helps decide what to keep, drop, or engineer.

```python
report = sq.feature_importance("data.parquet", target="price")
print(report)
```

```
═══ Feature Importance (pre-model) ═══
Target: price (regression) | 50,000 rows × 14 features

Rank  Feature         Score   Method           Action
────  ──────────────  ─────   ──────────────   ──────────
 1    sqft            0.82    Pearson          Keep — strong linear relationship
 2    bedrooms        0.71    Pearson          Keep — but correlated with sqft (r=0.78)
 3    type            0.62    Cramér's V       Keep — strong categorical signal
 4    city            0.45    Cramér's V       Keep — moderate signal
 5    bathrooms       0.68    Pearson          ⚠ Drop? — r=0.89 with bedrooms (redundant)
 6    age             0.34    Pearson          Keep — moderate signal
 7    score           0.28    Pearson          Keep — but 15% nulls
 8    created_month   0.12    Pearson          Weak — consider dropping
 9    zip_code        0.08    Cramér's V       Weak + 847 categories — drop or Hash
10    description_len 0.05    Pearson          Weak — consider dropping
11    customer_id     0.01    Pearson          ✗ Drop — identifier, no signal
12    country         0.00    Zero variance    ✗ Drop — constant column

Multicollinearity groups:
  Group 1: sqft ↔ bedrooms (r=0.78) ↔ bathrooms (r=0.89)
    Keep: sqft (highest target correlation)
    Consider dropping: bathrooms (redundant with bedrooms)

Non-linear relationships (mutual information):
  sqft → price:     MI=0.45  (strong, partially non-linear)
  type → price:     MI=0.38  (strong, categorical)
  zip_code → price: MI=0.31  (moderate — more signal than Pearson suggests!)
    💡 zip_code has low Pearson (0.08) but high MI (0.31) — non-linear relationship
       Consider keeping with HashEncoder or TargetEncoder

Suggested feature set (11 features → 7):
  Keep:  sqft, bedrooms, type, city, age, score, zip_code (with HashEncoder)
  Drop:  bathrooms (redundant), customer_id (identifier), country (constant),
         created_month (weak), description_len (weak)
```

```python
report.ranking                 # DataFrame: feature, score, method, action
report.keep                    # ["sqft", "bedrooms", "type", "city", "age", "score", "zip_code"]
report.drop                    # ["bathrooms", "customer_id", "country", ...]
report.multicollinearity       # list of correlated groups
report.nonlinear               # features with high MI but low Pearson (non-linear signals)

# Apply the suggestion
pipe = sq.Pipeline([
    sq.Drop(columns=report.drop),
    sq.Imputer(),
    sq.StandardScaler(),
    sq.OneHotEncoder(),
])
```

**Methods used (all SQL-based):**

| Method | For | SQL | What it measures |
|---|---|---|---|
| Pearson correlation | Numeric → numeric target | `CORR(feature, target)` | Linear relationship strength |
| Spearman correlation | Ordinal/ranked | `CORR(RANK() OVER ...)` | Monotonic relationship |
| Cramér's V | Categorical → any target | Chi-squared from `COUNT(*) GROUP BY` | Association strength |
| Point-biserial | Binary → numeric target | `CORR(binary_col, target)` | Binary-to-numeric relationship |
| Mutual information | Any → any | Binned `COUNT(*) GROUP BY` → entropy calculation | Non-linear dependence (catches what Pearson misses) |
| ANOVA F-statistic | Numeric → categorical target | `AVG()` per class, between/within variance | Feature separates classes |
| Variance | Any | `VAR_POP(col)` | Near-zero → no information |
| Cardinality ratio | Categorical | `COUNT(DISTINCT) / COUNT(*)` | >0.99 = identifier |

**Mutual information (SQL approximation):**

True MI requires density estimation. sqlearn uses a binned approximation:

```sql
-- Bin both feature and target, compute joint and marginal distributions
WITH binned AS (
    SELECT
        NTILE(20) OVER (ORDER BY feature) AS f_bin,
        NTILE(20) OVER (ORDER BY target) AS t_bin
    FROM data
)
SELECT f_bin, t_bin, COUNT(*) AS joint_count
FROM binned
GROUP BY 1, 2
-- Then compute MI = Σ p(x,y) * log(p(x,y) / (p(x) * p(y))) in Python
-- This is a simple post-SQL calculation on a 20×20 table (400 rows max)
```

The SQL produces a tiny contingency table. The MI calculation (400 cells) happens in
Python — negligible cost. This catches non-linear relationships that Pearson misses
(e.g., U-shaped relationships, categorical interactions).

### 8.12 Enhanced `sq.analyze()` — Class Imbalance & Target Analysis

`sq.analyze()` already handles correlations and multicollinearity. These additions
make it comprehensive for the full target analysis:

**Class imbalance detection (classification targets):**

```python
report = sq.analyze("data.parquet", target="churn")
```

```
── Target Analysis: churn ──
  Task: binary classification
  Distribution:
    churn=0: 47,600 rows (95.2%)
    churn=1:  2,400 rows (4.8%)
  ⚠ SEVERE CLASS IMBALANCE (minority < 10%)

  Impact:
    A model that always predicts 'no churn' gets 95.2% accuracy.
    Accuracy is a MISLEADING metric for this dataset.

  Recommendations:
    1. Use stratified splits: sq.StratifiedKFold(n_splits=5)
    2. Use appropriate metrics: scoring="f1" or scoring="roc_auc" (NOT "accuracy")
    3. Set class weights in your model:
       XGBClassifier(scale_pos_weight=19.8)   # ratio of majority/minority
       LogisticRegression(class_weight="balanced")
    4. Consider: SMOTE oversampling (outside sqlearn — sklearn.imblearn)
```

**Multi-class imbalance:**

```python
report = sq.analyze("data.parquet", target="product_type")
# Target: product_type (5 classes)
#   electronics: 35,000 (70%)
#   clothing:     8,000 (16%)
#   books:        4,000 (8%)
#   sports:       2,500 (5%)
#   toys:           500 (1%)     ← severely underrepresented
#
# ⚠ Class imbalance: toys (1%) is 70x smaller than electronics (70%)
# Recommendations: stratified splits, macro-F1 metric, class weights
```

**Target distribution analysis (regression targets):**

```python
report = sq.analyze("data.parquet", target="price")
# Target: price (regression)
#   Distribution: right-skewed (skewness=2.3, kurtosis=8.1)
#   Range: $12,000 – $450,000 | Median: $52,000 | Mean: $65,420
#   Outliers: 127 rows > 3σ (0.25%)
#
#   ⚠ Skewness > 2 — consider:
#     sq.TargetTransform(func="log")    → skewness after: 0.3
#     sq.TargetTransform(func="sqrt")   → skewness after: 0.8
#
#   Recommendations:
#     1. For linear models: apply Log transform (reduces skew to ~0.3)
#     2. For tree models: no transform needed (trees handle skew)
#     3. Use RMSE or MAE, not R² alone (R² is misleading with outliers)
```

SQL for all of this:

```sql
-- Class imbalance: one query
SELECT target, COUNT(*) AS n, COUNT(*)::FLOAT / SUM(COUNT(*)) OVER () AS pct
FROM data GROUP BY 1 ORDER BY 2 DESC;

-- Target distribution: one query
SELECT
    AVG(target) AS mean,
    MEDIAN(target) AS median,
    STDDEV_POP(target) AS std,
    MIN(target) AS min,
    MAX(target) AS max,
    (AVG(target) - MEDIAN(target)) / NULLIF(STDDEV_POP(target), 0) * 3 AS skewness_approx,
    COUNT(CASE WHEN ABS(target - AVG(target) OVER ()) > 3 * STDDEV_POP(target) OVER ()
          THEN 1 END) AS outlier_count
FROM data;
```

### 8.13 Smart Warnings in `pipe.fit()` — Automatic Mistake Prevention

These warnings trigger automatically during `fit()`. No extra function call needed.
Users can disable with `pipe.fit(..., warnings=False)`.

```python
pipe = sq.Pipeline([
    sq.Imputer(),
    sq.StandardScaler(),
    sq.OneHotEncoder(),
])
pipe.fit("train.parquet", y="target")

# Automatic warnings printed during fit:
# ⚠ zip_code: 847 unique values → OneHotEncoder creates 847 columns.
#   Consider: sq.HashEncoder(columns=["zip_code"], n_bins=64)
#
# ⚠ customer_id: 100% unique values — likely an identifier.
#   Consider: sq.Drop(columns=["customer_id"])
#
# ⚠ country: constant column ("US") — provides zero information.
#   Consider: sq.Drop(columns=["country"])
#
# 💡 created_at: datetime column not being processed.
#   Consider: sq.DateParts(columns=["created_at"])
```

**Warning rules (checked during fit, all SQL-based):**

| Warning | Trigger | SQL check |
|---|---|---|
| High cardinality OHE | OHE on column with >100 unique | `COUNT(DISTINCT col)` |
| Identifier in pipeline | Column with >99% unique values | `COUNT(DISTINCT col) / COUNT(*)` |
| Constant column | Column with 1 unique value | `COUNT(DISTINCT col) = 1` |
| Unused datetime | Datetime column not targeted by any transformer | Schema type check |
| Unused text | VARCHAR with avg length > 20, not processed | `AVG(LENGTH(col))` |
| Redundant scaling | Column already has mean≈0, std≈1 | `ABS(AVG(col)) < 0.1 AND ABS(STDDEV(col)-1) < 0.1` |
| Target in features | y column not excluded | Column name check |
| All-null column | Column is 100% NULL | `COUNT(col) = 0` |
| Near-zero variance | `VAR_POP(col) < 1e-10` for numeric | Variance check |
| Binary re-encoding | Column is already {0,1} being OHE'd | `COUNT(DISTINCT) = 2 AND MIN = 0 AND MAX = 1` |

**Implementation:** Warnings are collected during `fit()` as the pipeline resolves schemas.
Each transformer's `discover()` phase includes column-level checks. Results stored in
`pipe.fit_warnings_` for programmatic access.

```python
pipe.fit_warnings_            # list of FitWarning objects
pipe.fit_warnings_.critical   # warnings that likely cause bad results
pipe.fit_warnings_.suggestions  # helpful tips, not problems

# Suppress specific warnings
pipe.fit("data", y="target", suppress_warnings=["high_cardinality", "identifier"])
```

---

