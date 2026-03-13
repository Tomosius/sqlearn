> **sqlearn docs** — [Index](../CLAUDE.md) | Prev: [Project Structure](10-project-structure.md) | Next: [Implementation Milestones](12-milestones.md)

## 12. Business Model & Product Strategy

### One Package, Three Tiers

Single package, single repo, license key unlocks Pro features. No separate repo.
No separate install. Users see Pro features greyed out in the UI — constant
reminder of what they're missing.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          sqlearn ecosystem                               │
├──────────────────┬───────────────────────┬───────────────────────────────┤
│  sqlearn         │  sqlearn[studio]      │  sqlearn[studio] + license   │
│  (library)       │  (free EDA)           │  (Studio Pro)                │
├──────────────────┼───────────────────────┼───────────────────────────────┤
│  100% free       │  100% free            │  Paid license key            │
│  MIT license     │  MIT license          │  Same package, gated code    │
│  pip install     │  pip install          │  sq.activate("SQ-XXXX")      │
│  sqlearn         │  sqlearn[studio]      │  Unlocks Pro features        │
├──────────────────┼───────────────────────┼───────────────────────────────┤
│  • Pipeline      │  ALL of sqlearn +     │  ALL of free studio +        │
│  • Transformers  │  • Profile view       │  • Pipeline builder (d&d)    │
│  • Search        │  • Analysis view      │  • Live code preview         │
│  • Analysis      │  • Recommendations    │  • autopipeline in UI        │
│  • Export SQL    │  • Data table (EDA)   │  • Full pipeline code export │
│  • to_sql()      │  • Basic charts       │  • Chart customization       │
│  • to_dbt()      │  • Help hints (❓)     │  • Chart export (PNG/SVG)    │
│  • to_config()   │  • Single data source │  • Chart → Plotly code       │
│  • save()/load() │                       │  • Search monitor (live)     │
│  • autopipeline  │                       │  • Column deep-dive          │
│  • freeze()      │                       │  • Multiple data sources     │
│                  │                       │  • Session save/resume       │
│                  │                       │  • Full project generation   │
│                  │                       │  • Notebook export           │
│                  │                       │  • Model fitting from UI     │
│                  │                       │  • Guided workflow           │
│                  │                       │  • Session history + undo    │
│                  │                       │  • 14-day free trial         │
└──────────────────┴───────────────────────┴───────────────────────────────┘
```

### 12.1 Tier 1 — sqlearn Library (100% Free, MIT, Forever)

Everything programmatic. No artificial limits. This is the community engine.

**Included:**
- Full Pipeline, all transformers (scalers, encoders, imputers, features, ops)
- `sq.Search` — unified hyperparameter search with all samplers
- `sq.profile()`, `sq.analyze()`, `sq.recommend()` — all analysis functions
- `sq.autopipeline()` — returns a Pipeline object (programmatic use)
- `sq.autopipeline(as_code=True)` — returns Python source code (programmatic use)
- `to_sql(dialect=)`, `to_dbt()`, `to_config()`, `save()`/`load()`
- `pipe.freeze()` → `FrozenPipeline` for production deployment
- `sq.cross_validate()`, `sq.train_test_split()`, all metrics
- `sq.merge()`, `sq.concat()`, `sq.Lookup()`
- `sq.drift()`, `sq.correlations()`, `sq.missing()`
- `pipe.describe()`, `pipe.lineage()`, `pipe.validate()`
- ModelPipeline (preprocessing + model wrapper)
- All export formats (SQL, dbt, YAML config, standalone Python, binary)
- Optional: `sqlearn[plot]` for matplotlib visualizations

**Install:** `pip install sqlearn`

**Why free:** This builds the ecosystem. Users, blog posts, tutorials,
StackOverflow answers, GitHub stars, conference talks — all from the free library.
No one avoids sqlearn because of pricing. The library IS the product for most
users, and that's fine. Community size drives adoption.

### 12.2 Tier 2 — Studio Free (Read-Only EDA Dashboard)

Interactive data exploration. Users see their data, understand it, get hooked.
The free Studio answers: **"What's in my data and what should I do with it?"**

**Included:**
- **Profile view** — full data quality report: types, nulls, cardinality, stats,
  distributions, warnings (constant columns, high nulls, high cardinality)
- **Analysis view** — correlations with target, multicollinearity detection,
  feature engineering suggestions (as text), target distribution analysis
- **Recommendations view** — model suggestions, preprocessing pipeline
  suggestions (as text — "use StandardScaler for linear models"), feature
  engineering ideas. Read-only: users see what to do, then go write code.
- **Data table** — browse, sort, filter, paginate. SQL-powered (handles any
  dataset size). View transformed output with `LIMIT` preview.
- **Basic charts** — histogram, scatter, box plot, bar chart with default
  styling. No customization (colors, themes, zoom, brush are Pro). No export
  (PNG/SVG/code export is Pro).
- **Help hints (❓)** — on every transformer, every option. Interactive ML
  education. Floating UI tooltips with markdown documentation.
- **Single data source** — one file or table at a time. Drag-drop a CSV or
  Parquet file to load it. No database connections, no S3, no multi-source.

**Not included (greyed out with "Pro" badge):**
- Pipeline builder → shows locked state with preview of what it does
- Chart customization → charts render with defaults, customization panel locked
- Search monitor → shows "Pro feature" overlay
- Code export → "Copy code" buttons show Pro upgrade prompt
- Session save → "Save session" button shows Pro upgrade prompt

**Install:** `pip install sqlearn[studio]` then `sq.studio("data.parquet")`

**Why free:** Discovery. Users install Studio to explore their data. They see
how powerful sqlearn is. They see the locked Pro features. They get curious.
The free Studio is the top of the conversion funnel.

### 12.3 Tier 3 — Studio Pro (Interactive Builder + Generator)

Everything interactive, productive, and generative. The Pro Studio answers:
**"Build it for me and give me the code."**

**Conversion Driver — Pipeline Builder + Code Generation:**
- **Pipeline builder** — drag-and-drop transformers, reorder steps, configure
  parameters. Suggestions panel: "Based on your data, consider adding
  OutlierHandler". Visual flow: data → step → step → output preview.
- **Live code preview** — as you build in the UI, see the Python code update
  in real-time. Copy the full pipeline as a working script at any point.
- **autopipeline in UI** — one-click: analyze data → generate pipeline →
  show in builder. User can then tweak individual steps in the visual builder.
- **Full pipeline code export** — "Export as Python" button generates a
  complete, runnable script with imports, pipeline definition, fit, transform.
- **Chart customization** — color picker, theme switching (light/dark/custom),
  zoom/pan, brush selection, DataZoom for time series, axis labels, titles.
- **Chart export** — PNG, SVG. Click "Export" on any chart.
- **Chart → Plotly code** — click "Export as code" → generates Plotly Python
  code. ECharts renders in browser (fast), Plotly goes into exported files
  (Python standard for reproducible charts).

**Productivity Features:**
- **Search monitor** — live WebSocket convergence chart (uPlot), parameter
  importance visualization (ECharts), top configurations table. Real-time
  updates as `sq.Search` runs.
- **Column deep-dive** — click any column → full statistics, distribution
  chart, outlier analysis, correlation with target, suggested transforms.
- **Advanced analysis** — feature importance ranking, mutual information
  matrix, distribution comparison across splits, class balance analysis.
- **Multiple data sources** — drag-drop multiple files, connect to databases
  (Postgres, MySQL, SQLite via DuckDB ATTACH), S3/GCS paths. All loaded
  into DuckDB, browseable in sidebar. Switch between datasets.
- **Session save/resume** — persist all EDA progress, pipeline state, search
  results, chart configurations to disk. Close browser, come back tomorrow,
  everything is where you left it. Sessions stored as DuckDB files.
- **Session history + undo** — timeline of all actions taken. Click any
  point to restore state. Export history as reproducible Python script.

**Project Generation:**
- **Full ML project scaffolding** — generates complete project structure:
  ```
  my_ml_project/
  ├── 01_data_loading.py       # Data source setup, DuckDB ingestion
  ├── 02_eda.py                # EDA with Plotly charts
  ├── 03_cleaning.py           # Missing values, outliers, type fixes
  ├── 04_feature_engineering.py # Feature creation, encoding, scaling
  ├── 05_modeling.py           # Model training, cross-validation
  ├── 06_evaluation.py         # Metrics, confusion matrix, feature importance
  ├── 07_pipeline_final.py     # Production pipeline (clean, reproducible)
  ├── config.yaml              # All parameters in one place
  └── requirements.txt         # Dependencies
  ```
- **Notebook export** — export entire workflow as:
  - Marimo notebook (`.py` with `@app.cell` decorators)
  - Jupyter notebook (`.ipynb` via nbformat)
  - Plain Python script (`.py`)
- **Model fitting from UI** — select model (XGBoost, LightGBM, RandomForest,
  Linear, etc.), configure hyperparameters, click "Train" → model trains with
  live progress bar (callbacks for sklearn, XGBoost, LightGBM). Evaluation
  view: metrics, confusion matrix, ROC curve, feature importance.
- **Step-by-step guided workflow** — progress bar across top:
  ```
  [Data] → [EDA] → [Clean] → [Features] → [Model] → [Evaluate] → [Export]
    ✓        ✓       ●
  ```
  Each step generates its own Python file. User can jump between steps.

### 12.4 The Conversion Funnel

```
Library users (free)
  │  "sqlearn is great, let me try the Studio"
  ▼
Studio free users
  │  "I can see my data, I see the recommendations..."
  │  "Pipeline Builder looks amazing but it's locked"
  │  "I want to export this chart but it's Pro"
  ▼
Studio Pro trial (14 days free)
  │  "I built 3 pipelines visually, I exported a full project"
  │  "I don't want to go back to writing code manually"
  ▼
Paid Pro user ($49-99)
```

**Why this works:**
1. **No artificial limits.** Free Studio is complete for exploration. The
   boundary is functional (read vs build), not numerical (3 charts vs unlimited).
2. **Loss aversion drives conversion.** 14-day trial lets users experience
   the pipeline builder. After 14 days, losing it feels painful.
3. **Multiple conversion entry points.** Users hit Pro prompts when they try to:
   build pipelines, customize charts, export code, save sessions, connect databases.
4. **Clear value.** Pipeline builder + code generation saves 30-60 min per project.
   At $49-99, the tool pays for itself on the first project.

### 12.5 What's Free vs Paid — Explicit Feature Matrix

| Feature | Library | Studio Free | Studio Pro |
|---|---|---|---|
| **Core** | | | |
| All transformers (scalers, encoders, imputers, features, ops) | Yes | — | — |
| Pipeline composition (`+`, `+=`, nesting, Columns, Union) | Yes | — | — |
| `sq.Search` (all samplers, multi-fidelity, caching) | Yes | — | — |
| `sq.profile()`, `sq.analyze()`, `sq.recommend()` | Yes | — | — |
| `sq.autopipeline()` (returns Pipeline object) | Yes | — | — |
| `sq.autopipeline(as_code=True)` (returns Python code) | Yes | — | — |
| `to_sql()`, `to_dbt()`, `to_config()`, `save()`/`load()` | Yes | — | — |
| `pipe.freeze()` → FrozenPipeline | Yes | — | — |
| ModelPipeline (preprocessing + model wrapper) | Yes | — | — |
| **Studio — Exploration** | | | |
| Profile view (types, nulls, stats, warnings) | — | Yes | Yes |
| Analysis view (correlations, multicollinearity, suggestions) | — | Yes | Yes |
| Recommendations (model + pipeline suggestions as text) | — | Yes | Yes |
| Data table (browse, sort, filter, paginate) | — | Yes | Yes |
| Basic charts (histogram, scatter, box, bar — default styling) | — | Yes | Yes |
| Help hints (❓ on every option) | — | Yes | Yes |
| Single data source (one file at a time) | — | Yes | Yes |
| **Studio — Building (Pro)** | | | |
| Pipeline builder (drag-and-drop, suggestions, reorder) | — | No | Yes |
| Live code preview (Python updates as you build) | — | No | Yes |
| autopipeline in UI (one-click pipeline generation) | — | No | Yes |
| Full pipeline code export (complete runnable script) | — | No | Yes |
| **Studio — Charts (Pro)** | | | |
| Chart customization (colors, themes, zoom, brush) | — | No | Yes |
| Chart export (PNG, SVG) | — | No | Yes |
| Chart → Plotly Python code export | — | No | Yes |
| **Studio — Advanced (Pro)** | | | |
| Search monitor (live convergence, param importance) | — | No | Yes |
| Column deep-dive (click column → full analysis) | — | No | Yes |
| Multiple data sources (files, databases, S3) | — | No | Yes |
| **Studio — Productivity (Pro)** | | | |
| Session save/resume | — | No | Yes |
| Session history + undo | — | No | Yes |
| **Studio — Generation (Pro)** | | | |
| Full ML project scaffolding (multi-file) | — | No | Yes |
| Notebook export (Marimo, Jupyter, Python) | — | No | Yes |
| Model fitting from UI (select, configure, train, evaluate) | — | No | Yes |
| Guided workflow (step-by-step progress bar) | — | No | Yes |

### 12.6 Pricing

| Model | Price | Details |
|---|---|---|
| **One-time license** | $49-99 | Buy once, use forever. All Pro features. |
| **Annual subscription** | $29/year | Includes updates + new features. |
| **Team license** | Per-seat | For organizations. Volume discounts. |
| **Free for students** | $0 | `.edu` email → free license key. |
| **Free for open-source** | $0 | MIT/Apache project → free license key. |
| **14-day trial** | $0 | Full Pro features, no credit card required. |

Decide exact pricing after Studio ships and user feedback is collected. The
14-day trial is critical — let users experience Pro before asking for money.

### 12.7 Repo & Package Strategy

**One repo. One package. License key unlocks Pro.**

```
github.com/your-org/sqlearn          ← PUBLIC, MIT, free forever
```

Everything is in one repo. Pro features are in the codebase but gated by
license validation. The Pro code is included in the distributed package
(minified/obfuscated in the built frontend bundle — the Svelte frontend
compiles to static JS, so Pro UI components are not human-readable source).

```toml
# pyproject.toml
[project]
name = "sqlearn"
dependencies = [
    "duckdb>=1.0",
    "sqlglot>=25.0",
    "numpy>=1.24",
]

[project.optional-dependencies]
studio = [
    "starlette>=0.36",
    "uvicorn>=0.24",
    "websockets>=12.0",
    "python-multipart>=0.0.9",
    "orjson>=3.9",
    "plotly>=5.18",         # for chart → Plotly code generation (Pro)
    "nbformat>=5.9",       # for Jupyter notebook generation (Pro)
]
```

**Why not a separate repo:**
1. **Simpler maintenance.** One CI pipeline, one test suite, one release process.
2. **Better UX.** `pip install sqlearn[studio]` → everything works. No hunting
   for a second package.
3. **Stronger conversion.** Users see Pro features greyed out in the UI. They
   don't need to know sqlearn-pro exists — they just enter a license key.
4. **Easier development.** Pro features share types, components, and utilities
   with free features. No cross-repo dependency management.

### 12.8 License Key Mechanism

```python
# Activate Pro features
import sqlearn as sq
sq.activate("SQ-XXXX-XXXX-XXXX")   # saves to ~/.sqlearn/license.key

# Or set via environment variable
# SQLEARN_LICENSE=SQ-XXXX-XXXX-XXXX

# Start a 14-day trial (no key needed)
sq.trial()                           # saves trial start date to ~/.sqlearn/trial

# License validation: RSA signature check (offline)
# No phone-home, no telemetry, no internet required
# Key contains: email, expiry_date, tier → signed by private key
# sqlearn checks signature with embedded public key → works offline forever
```

**License gating in Python (backend API endpoints):**

```python
# studio/license.py
def require_pro(feature_name: str) -> None:
    """Check Pro license. Raise if not activated."""
    if not _license_valid():
        raise ProFeatureError(
            f"'{feature_name}' requires Studio Pro. "
            f"Activate with sq.activate('SQ-XXXX') or start a free trial: "
            f"sq.trial()"
        )

# Used in Pro-gated endpoints:
@app.route("/api/export/project")
async def export_project(request):
    require_pro("Project generation")
    ...
```

**License gating in Svelte (frontend UI):**

```svelte
<!-- Pro features show a lock icon + upgrade prompt in free mode -->
{#if $license.pro}
    <PipelineBuilder {pipeline} />
{:else}
    <ProGate feature="Pipeline Builder"
             description="Build pipelines visually with drag-and-drop">
        <PipelineBuilderPreview />  <!-- screenshot/animation of feature -->
    </ProGate>
{/if}
```

### 12.9 Why This Model Works

1. **Library is always free.** Builds ecosystem. Users, blog posts, tutorials,
   conference talks, GitHub stars — all from the free version. No one avoids
   sqlearn because of pricing.

2. **Free Studio drives discovery.** Users install `sqlearn[studio]`, explore
   their data, see how powerful it is. They're hooked on the EDA. They see
   the locked Pro features and get curious.

3. **Pro converts power users.** Once someone builds their first pipeline
   visually and exports a full project in 5 minutes, the $49-99 price is
   obvious value. The tool pays for itself on the first project.

4. **Clear value boundary.** Free = explore + learn + understand. Pro = build
   + generate + export. No artificial limitations (no "3 charts per session"
   or "max 5 transformers"). The boundary is functional, not numerical.

5. **14-day trial eliminates friction.** Users try everything. Loss aversion
   kicks in after 14 days. They've built pipelines visually, they don't want
   to go back to writing code by hand.

6. **One repo protects simplicity.** No cross-repo dependencies, no version
   mismatches, no "which package do I install?" confusion. Pro is just a
   license key away.

