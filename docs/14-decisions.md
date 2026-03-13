> **sqlearn docs** — [Index](../CLAUDE.md) | Prev: [Testing Strategy](13-testing.md)

## 15. Technical Risks

| Risk | Severity | Mitigation |
|---|---|---|
| sqlglot API breaks between versions | High | Wrap in internal `sql_utils.py`, pin version range |
| Numerical precision mismatch vs numpy | High | `np.allclose(atol=1e-10)`, use `STDDEV_POP` (sklearn uses ddof=0) |
| Expression nesting blowup (10+ steps) | Medium | Depth counter, auto-CTE at depth > 8 |
| OneHotEncoder column explosion | Medium | `max_categories=30` default, warn + suggest alternatives |
| Cross-dialect SQL correctness | Medium | Phase 5 concern, dialect-specific test suites |
| DuckDB version coupling | Low | Pin minimum version, test against latest in CI |
| Custom transformer returns wrong type from `discover()` | Medium | Tier 3 conservative inspection: anything that isn't exactly `{}` → dynamic fallback. `None`, wrong type, exception → all safe. Emit `UserWarning` so author notices. |
| Custom transformer overrides both `expressions()` and `query()` | Low | `query()` wins (conservative). Documented. If `query()` returns `None`, fall back to `expressions()`. Both fail → `CompilationError` with clear message. |
| Built-in `_classification` declaration is wrong | High | **Caught in CI, never at runtime.** Every built-in has a test: if `_classification="static"`, assert `discover()=={}`. If declaration is wrong, CI fails before release. Users never see a wrong declaration. |
| Custom `_classification` declaration is wrong | Medium | Tier 2: verified at first `fit()`. If `_classification="static"` but `discover()` returns params → hard `ClassificationError`. If `_classification="dynamic"` but `discover()` returns `{}` → `UserWarning` (safe, just wasteful). |
| AutoFeatures false positives (wrong transform) | High | Conservative thresholds by default. `AutoNumeric` skew threshold=2 (not 1). `AutoSplit` requires >50% of rows containing the delimiter. All auto-* decisions logged via `pipe.describe()` so user can inspect + override per-column. |
| AutoSplit delimiter detection on messy data | Medium | Only check common delimiters (`,;|\t`). Require consistent count across sampled rows (stddev of delimiter count < 1). If ambiguous, skip column (no split). User can always force via `AutoSplit({"col": ","})`. |
| StringSplit max_parts="auto" on variable-length data | Medium | Use `MAX(parts)` from sample of 1000 rows. Columns beyond max_parts in other rows → NULL. Warn if stddev of part count is high: "Column 'tags' has variable part count (1-47). Consider setting max_parts explicitly." |
| JsonExtract on malformed / deeply nested JSON | Medium | Top-level keys only (no nested extraction). Malformed JSON → NULL per field, not error. Warn: "12% of rows have invalid JSON in column 'metadata'." Nested extraction is a future feature, not Phase 1. |
| Thread guard false positive in async frameworks | Medium | `_check_thread()` uses `threading.current_thread().ident` + `os.getpid()`. Async code (asyncio) runs on one thread — no false positives. If someone uses `run_in_executor()` → correctly raises (they need `clone()`). Multiprocessing with `fork` or `spawn` → correctly raises with process-specific error message. Document async patterns in docs. |
| DropCorrelated removes useful feature | Medium | Only drops when `|r| > threshold` (default 0.95 — very conservative). When `y=` provided, keeps the column with higher target correlation. User can inspect via `pipe.describe()` which shows "dropped X because r(X,Y)=0.97, kept Y (higher target corr)". |
| SelectKBest correlation ≠ predictive power | Medium | Correlation captures linear relationships only. Offer `method="mutual_info"` for non-linear. `method="anova"` for classification. Default is `"correlation"` with clear docstring: "Linear correlation only. Use mutual_info for non-linear relationships." |
| OutlierHandler IQR on non-normal distributions | Low | IQR is distribution-agnostic (no normality assumption). For highly skewed data, `method="percentile"` is safer. `sq.recommend()` suggests percentile method when skewness > 2. Document trade-offs in help hints. |
| CyclicEncode period="auto" wrong guess | Low | Only auto-detects from well-known column names: `hour`→24, `dayofweek`/`dow`→7, `month`→12, `minute`→60, `day_of_year`→365. Unknown names → error: "Cannot auto-detect period for column 'X'. Specify period= explicitly." |
| Lookup table join explosion (many-to-many) | Medium | `sq.Lookup` defaults to `how="left"` and warns if join produces more rows than input: "Lookup on 'city' increased row count from 50K to 73K — join key is not unique in lookup table. Use `sq.Lookup(..., validate='one_to_one')` to enforce." |
| TargetTransform inverse precision loss | Low | Log/exp roundtrip has floating-point error ~1e-15. For `func="standard"` and `func="minmax"`, inverse uses same learned params — error proportional to float64 precision. Add `np.allclose` assertion in tests. |
| Scope creep: too many transformers to maintain | Medium | Each transformer is one file, one class, one test file. No shared mutable state between transformers. `auto.py` delegates to individual transformers (doesn't duplicate logic). New transformers can be community-contributed with minimal review surface. |
| Cross-fold schema inconsistency | **Critical** | Two-phase discovery: schema from full data (Phase 1), values per fold (Phase 2). Rare categories always present in output. See Section 4.7b. |
| Pipeline version mismatch on load | Medium | Embedded version metadata. Same major version → info log. Different major → warning. Incompatible format → error with migration instructions. SQL hash verification optional via `verify=True`. |
| `expressions()` forgetting passthrough | **Eliminated** | Base class `_apply_expressions()` handles passthrough automatically. User returns only modified/new columns. |
| `+=` mutation in notebooks | **Eliminated** | `__iadd__` creates new Pipeline (non-mutating). Rerunning cells is safe. |
| Frozen pipeline stale after data changes | Medium | `detect_drift()` compares new data against training stats. Alerts on distribution shifts. User decides when to refit. |
| ModelPipeline pickle security | Medium | Preprocessing saved as JSON (safe). Model saved as pickle (inherent risk). Document: "Only load models from trusted sources." Same risk as sklearn. |

---

## 16. Decisions Log

Decided. Not revisiting unless evidence says otherwise.

| # | Decision | Rationale |
|---|---|---|
| 1 | One base class (`Transformer`), not sklearn's 8+ mixins | Simpler, no historical baggage |
| 2 | Auto column routing via `_default_columns` | Eliminates ColumnTransformer for 80% of cases |
| 3 | `discover()` + `expressions()` + `output_schema()` interface | Three methods cover all transformer types |
| 4 | Static/dynamic auto-detected from `discover()` | User doesn't think about it |
| 5 | Schema-change auto-detected from `output_schema()` | User doesn't think about it |
| 6 | `expressions()` for inline, `query()` for CTE-requiring transforms | Clean separation of composable vs query-level |
| 7 | Layers during fit, one query for transform | Minimum queries, maximum performance |
| 8 | In-memory DuckDB default, persistent opt-in | Zero config for common path |
| 9 | numpy output default | sklearn compatibility |
| 10 | Pipeline accepts list, tuple-list, or dict | Flexibility without confusion |
| 11 | `Columns` replaces `ColumnTransformer` | Shorter name, same concept |
| 12 | `Union` replaces `FeatureUnion` | Shorter name, same concept |
| 13 | `AutoEncoder` auto-selects encoding per column | No other library does this |
| 14 | `GroupFeatures` defaults to window mode (preserves rows) | No silent grain change |
| 15 | `handle_unknown="error"` default on encoders | Safe by default, opt into permissive |
| 16 | `null_policy="propagate"` default | SQL semantics, explicit when needed |
| 17 | Multi-source via DuckDB ATTACH, not split pipelines | One pipeline = one engine |
| 18 | Name: `sqlearn` | SQL-first positioning, `sk` → `sq` parallel |
| 19 | All SQL via sqlglot ASTs, never raw strings | Multi-database from day one |
| 20 | Backend protocol defined in Phase 1 | Adding Postgres/MySQL = implementing one interface |
| 21 | Baked for fit+transform+export, self-contained CTE for CV | Best perf in each context, user doesn't choose |
| 22 | Fold column: `__sq_fold__`, reserved prefix `__sq_*__` | Collision-proof, validated on schema read |
| 23 | Unified `sq.Search` with pluggable strategies (grid, random, sobol, bayesian, optuna) | One class, SQL-native fidelity + fold reuse. Replaced separate GridSearchCV etc. |
| 24 | `stats/aggregates.py` as shared building blocks | All transformers compose from Mean, Median, Std, etc. DRY. |
| 25 | Step metadata (`StepInfo`) tracked per step | static/dynamic, fitted, layer, schemas — works with nesting |
| 26 | SQL-based statistical analysis (correlations, missing, tests) | No scipy/statsmodels needed |
| 27 | Optional `plot/` module (matplotlib) | SQL computes, matplotlib renders. Clean separation. |
| 28 | Minimal deps: duckdb + sqlglot + numpy only | No dependency conflicts. Everything else optional. |
| 29 | Imputer dict API for per-column strategies | One step replaces 3+ Imputer steps. String=strategy, number=constant. |
| 30 | `dtype=` parameter on `transform()` | CAST numeric output to float32/64. Halves memory in GridSearch. |
| 31 | `out="relation"` returns DuckDB Relation | Lazy, chainable. No materialization until user calls fetch. |
| 32 | Unified `sq.Search` replaces GridSearchCV, Optuna, etc. | One class, pluggable samplers, SQL-native advantages |
| 33 | Multi-fidelity precision opt-in, float64 default | No surprises by default. `explore_dtype="float32"` for power users. |
| 34 | Built-in samplers: random, grid, Sobol, TPE | Zero external deps. Optuna is optional integration (`sqlearn[optuna]`). |
| 35 | Preprocessing dedup via SQL AST hash | Same preprocessing SQL → transform once, train N models from cache. Automatic. |
| 36 | Multi-fidelity rounds via `fast_explore` or `rounds=[]` | Hyperband scheduling. Three knobs: data%, dtype, cv folds. |
| 37 | Checkpoint/resume in DuckDB file | Queryable, portable, no pickle. Long searches are interruptible. |
| 38 | `strategy="auto"` selects sampler based on space + data size | Grid for small, Sobol for medium, Bayesian for large. Enables `fast_explore` for big data. |
| 39 | `sq.cross_validate` as standalone function | Works standalone, with Optuna, and internally by `sq.Search`. One source of truth for CV logic. |
| 40 | `sq.read_column` for target extraction | One SQL query, no full data load. Avoids pandas dependency for simple target reads. |
| 41 | `sq.train_test_split` returns SQL views, not data copies | No data duplication. Works with bigger-than-RAM. Stratified via SQL window functions. |
| 42 | Optuna as optional sampler, three integration paths | Path 1: `strategy="optuna"`. Path 2: user-driven + `sq.cross_validate`. Path 3: full manual. |
| 43 | Feature cache via SQL AST hash in `cache.py` | Automatic. Same preprocessing SQL → materialize once → all model fits reuse. |
| 44 | `search.importance()` via functional ANOVA | Built-in param importance. Same method as Optuna, no dependency. |
| 45 | `search.suggest()` recommends refined search space | Actionable suggestions based on trial results. Unique to sqlearn. |
| 46 | `warm_start=` for continuing searches | Reuses prior trials as Bayesian prior + feature cache. No wasted work. |
| 47 | Four analysis levels: profile → analyze → recommend → autopipeline | Each builds on previous. All SQL-based. Separate functions, clear purpose. |
| 48 | `MODEL_PROFILES` knowledge base for model-aware recommendations | Static dict: what each model family needs (scaling, encoding, nulls). Extensible. |
| 49 | `sq.autopipeline()` returns actual Pipeline object or Python code | `as_code=True` outputs editable Python. Bridge to interactive workflow. |
| 50 | `FEATURE_SUGGESTIONS` maps column types to feature engineering ideas | Datetime → DateParts, numeric pairs → Ratio, skewed → Log, etc. |
| 51 | ~~sqlearn-studio is a SEPARATE project~~ → superseded by #52 | Library = intelligence, Studio = interactive UI. Different deps, different release cycles. |
| 52 | Studio is `sqlearn[studio]` optional install, not separate repo | One pip install, one `sq.studio()` command. Pre-built Svelte frontend bundled in package. |
| 53 | uPlot (35KB, time series) + ECharts (~135KB tree-shaken, everything else) | Two charting libs, each best at its job. Combined ~170KB vs Plotly's 3.5MB. |
| 54 | SQL pre-aggregation: browser never sees raw data | Histogram = 30 points, scatter = 1000 sampled, heatmap = N×N. Always fast on any dataset size. |
| 55 | Starlette (not FastAPI) + pre-compiled Svelte | No Pydantic overhead for local tool. No Node.js at runtime. Frontend ~200KB total. |
| 56 | No authentication. `127.0.0.1` only. Optional token for shared machines. | Local tool, localhost binding. No HTTPS, no CORS, no cookies. Simple. |
| 57 | WebSocket for live Search progress | Real-time convergence chart, param importance, top configs. |
| 58 | Code generation throughout studio | Pipeline builder shows live Python code. Export as complete script. User always sees what's generated. |
| 59 | ECharts over AntV G2 | G2 is bigger (319KB gzipped vs 135KB ECharts tree-shaken), 15+ deps vs 1 (zrender), 500x fewer npm downloads. ECharts wins every metric. |
| 60 | TanStack Table (25KB) over AntV S2 (~300KB+) for data tables | SQL does sort/filter/paginate/pivot. Table just renders + handles clicks. Headless = full Svelte-native DOM control for dtale-like interactions. |
| 61 | shadcn-svelte + Bits UI for UI components | Copy-paste components, 0 runtime dep, 40+ components (modals, dropdowns, tooltips, context menus, tabs, sheets). MIT. |
| 62 | Floating UI for rich tooltips (help hints with ❓) | Successor to Tippy.js (unmaintained). 3-5KB, MIT. Simple tooltips via shadcn, rich ones via Floating UI. |
| 63 | Lucide icons (lucide-svelte) | ISC license, ~200-400B per icon, tree-shakeable, standard for shadcn-svelte ecosystem. |
| 64 | svelte-dnd-action for pipeline builder drag-and-drop | Svelte-native `use:dndzone` action. MIT, ~15KB. Handles nested containers, touch, keyboard accessibility. |
| 65 | highlight.js (Python only) for code preview | BSD 3-Clause, ~7KB. Shiki (700KB) too heavy for client-side. Prism.js less maintained. |
| 66 | tinykeys for keyboard shortcuts | MIT, 650 bytes. Dead simple API. hotkeys-js also good but 4x larger for same job. |
| 67 | svelte-sonner for toast notifications | MIT, ~5KB. Modern Sonner port. Svelte 5 native. |
| 68 | svelte-awesome-color-picker for chart color customization | MIT, ~8KB. Svelte-native, alpha channel, Svelte 5 rewrite. |
| 69 | marked for markdown rendering (help text, tooltips) | MIT, ~14KB, zero deps. Runtime markdown parsing for documentation panels. |
| 70 | Vite (NOT SvelteKit) for frontend build | SvelteKit is full framework overkill. Vite → static dist/ → bundled in Python package. |
| 71 | Tailwind CSS v4 for styling | MIT, ~10-15KB purged. Required by shadcn-svelte. No config file in v4. |
| 72 | orjson for fast JSON + numpy serialization | MIT/Apache 2.0, 300KB. 3-10x faster than stdlib, native numpy array serialization. |
| 73 | python-multipart for file uploads | Apache 2.0, 70KB. Required by Starlette for drag-and-drop CSV/Parquet upload. |
| 74 | jsPDF for PDF export (browser-side) | MIT, ~96KB lazy-loaded. Charts → PDF via getDataURL() + addImage(). |
| 75 | ExcelJS for Excel export | MIT, ~330KB lazy-loaded. Full cell styling (bold, colors, borders). SheetJS CE can't style. |
| 76 | html-to-image for dashboard screenshots | MIT, ~5KB. SVG foreignObject approach — 20x faster than html2canvas. |
| 77 | ECharts built-in toolbox for chart interaction | Save image, zoom, data view, chart type switch, brush — all free, zero extra code. |
| 78 | ECharts registerTheme() for runtime theme switching | Light/dark mode + custom palettes. User can customize chart colors at runtime via setOption(). |
| 79 | Export libraries lazy-loaded | jsPDF (96KB) + ExcelJS (330KB) + html-to-image (5KB) = 431KB loaded ONLY when user clicks Export. Initial load stays ~330KB. |
| 80 | Handsontable DISQUALIFIED | Proprietary license since v7.0 (2019). Not open source. |
| 81 | borb (Python PDF) DISQUALIFIED | AGPL-3.0 copyleft — requires releasing entire app source. |
| 82 | ~~Three-product ecosystem~~ → superseded by #82b | Original: sqlearn + studio + sqlearn-pro (separate repo). Replaced by single-package model. |
| 82b | Three-tier, one-package model: sqlearn (library, free) + sqlearn[studio] free tier (read-only EDA) + sqlearn[studio] Pro (license-gated builder/generator) | One repo, one package, one install. Pro features gated by license key. Free Studio is read-only EDA (explore/learn). Pro Studio adds pipeline builder, code generation, charts, search monitor, project scaffolding, notebooks, model fitting, sessions. |
| 83 | ~~Free = explore/learn/build manually. Paid = generate projects.~~ → superseded by #83b | Original split put pipeline builder in free tier — too generous. |
| 83b | Free = explore + learn + understand ("What's in my data?"). Paid = build + generate + export ("Build it for me and give me the code.") | Pipeline builder is the conversion driver. Free users see data, get recommendations, copy snippets. Pro users build visually, export code, generate projects. Functional boundary, not numerical limits. |
| 84 | ~~sqlearn-pro in separate PRIVATE repo~~ → superseded by #84b | Separate repo adds maintenance overhead, cross-repo deps, user confusion. |
| 84b | Pro features in same repo, gated by license key. Frontend Pro components compiled/minified into app.js. Backend Pro endpoints use `require_pro()`. | One CI, one release, one install. Users see Pro features greyed out → constant conversion prompt. `sq.activate("SQ-XXXX")` or `sq.trial()` unlocks. |
| 85 | Pro generates Plotly code (not ECharts) for Python files | ECharts renders in browser for speed. Plotly is the Python standard for reproducible charts. |
| 86 | Pro exports to Marimo, Jupyter, and plain Python | Three notebook formats. nbformat for Jupyter, native .py for Marimo. |
| 87 | Offline license key via RSA signature | No phone-home, no telemetry, no internet required. Simple, respectful. |
| 88 | Multi-file project generation: 01_data → 02_eda → 03_clean → ... → 07_pipeline | Each step = separate file. Clean, documented, reproducible. |
| 89 | 9 milestones: scaffolding → compiler → breadth → features → search → analysis → export → studio (free+pro gated) → studio pro completion | Each independently shippable. Studio ships with free tier + license-gated Pro in one package. Milestone 9 completes remaining Pro features (project gen, notebooks, model fitting, sessions). |
| 90 | `+` operator for sequential Pipeline composition, with flattening. `+=` for incremental append. No `\|` operator — Union is explicit. | `a + b` = "a then b" — universally understood. Flattening avoids gratuitous nesting. `\|` saves 10 chars but costs discoverability. Union is rare enough for explicit `sq.Union()`. |
| 91 | Eager numpy output by default with large-result size warning. `out="relation"` is the lazy escape hatch. | sklearn compat is non-negotiable. `model.fit(X, y)` must just work. Size warning catches accidental OOM. Lazy via `__array__` protocol is fragile (breaks `.shape`, indexing, `isinstance`). |
| 92 | Pipelines are NOT thread-safe. Runtime thread guard detects cross-thread access and raises with `clone()` suggestion. | In-memory DuckDB requires shared connection (temp tables from fit must be visible to transform). Locking gives false confidence. Fail loudly + offer `clone()` for thread-safe copies. |
| 93 | `y` is a column name (string), not a numpy array. Flows from `pipeline.fit("data", y="price")` into `discover(columns, schema, y_column)`. Target auto-excluded from transform output. | SQL-native: target is already in the table. No round-tripping through numpy. Multi-target = `y=["t1","t2"]`. No `needs_target` flag needed — steps that use `y_column` raise if it's None. |
| 93b | Three-tier classification: Tier 1 (built-in) declares `_classification` and is trusted at runtime, validated by CI. Tier 2 (custom with declaration) is verified once at first fit, then cached. Tier 3 (custom without declaration) gets full conservative inspection every fit. | Built-in transformers are our code — inspecting `Log()` every fit is wasted work. Trust declarations, validate in CI. Custom code gets full inspection. If any doubt → dynamic. Cost of false "static" = data corruption. Cost of false "dynamic" = one cheap query. |
| 94 | Feature selection as pipeline steps (`Drop`, `DropConstant`, `DropCorrelated`, `SelectKBest`, etc.) in `feature_selection/` module. | SQL-native: `COUNT(DISTINCT)`, `CORR()`, `VAR_POP()` run in DuckDB. Selection happens after encoding so expanded features are visible. |
| 95 | Auto feature engineering family: `AutoFeatures`, `AutoDatetime`, `AutoSplit`, `AutoNumeric`. Type-aware, override-able per column. | The biggest DX win: one step auto-expands datetime, splits delimited strings, log-transforms skewed numerics, encodes categoricals. User can override per column or disable specific sub-transforms. |
| 96 | `OutlierHandler(method=, action=)` with IQR/zscore/percentile methods and clip/null/flag/drop actions. | SQL-native: IQR bounds from `PERCENTILE_CONT`. Clip via `GREATEST(LEAST(...))`. Default is conservative (IQR clip). Per-column config via dict API. |
| 97 | `TargetTransform(func=)` only via `ModelPipeline`, not inside regular `Pipeline`. Auto-inverts predictions. | Target transforms are model-training concerns, not feature engineering. Keeping them in `ModelPipeline` avoids confusion. Inverse is automatic at predict time. |
| 98 | Data operations (`Rename`, `Cast`, `Filter`, `Sample`, `Deduplicate`) as pipeline steps in `ops/` module. | Cleanup is part of the pipeline, not a separate step. Having them as transforms means they participate in `to_sql()`, schema tracking, and lineage. |
| 99 | `sq.merge()`, `sq.concat()` as pre-pipeline helpers. `sq.Lookup()` as mid-pipeline transformer. | Merge/concat are data assembly (pre-pipeline). Lookup is enrichment (mid-pipeline join). Clear separation: `merge()` returns a view name, `Lookup` is a pipeline step. |
| 100 | `CyclicEncode(period=)` for sin/cos encoding of periodic features (hour, day, month). `period="auto"` infers from column name. | Hour 23 is close to hour 0 — ordinal encoding misses this. Sin/cos preserves cyclical distance. Auto-detection from column name is a convenience, not magic. |
| 101 | String splitting via `StringSplit(by=, max_parts=)` with `max_parts="auto"` that learns from data. Structured extractors: `JsonExtract`, `URLParts`, `EmailParts`, `IPParts`. | Common real-world need: comma-separated tags, JSON metadata, URLs, emails. Auto-detecting delimiter and max parts during `discover()` avoids manual counting. |

| 102 | `expressions()` auto-passthrough: user returns only modified/new columns, base class merges with untouched columns via `_apply_expressions()`. | Eliminates the most common custom transformer bug (forgetting passthrough). Every custom transformer in the old API had to write `else: result[col] = expr` — forget once → silent column drop → data corruption. |
| 103 | `discover_sets()` for multi-row discovery alongside `discover()` for scalar aggregates. | `discover()` → `fetch_one()` can't handle OneHotEncoder (DISTINCT categories) or TargetEncoder (per-category means). Clean split: scalars in `discover()`, sets in `discover_sets()`. Both contribute to static/dynamic classification. |
| 104 | Two-phase discovery for cross-validation: Phase 1 (schema from full data) + Phase 2 (values per fold). | Without this, different CV folds produce different output schemas when rare categories are absent from a fold's training split. Phase 1 learns the UNIVERSE of categories from all data (schema-only, not leakage). Phase 2 learns per-fold statistics via self-contained CTE. All folds guaranteed identical output columns. |
| 105 | `__iadd__` is non-mutating: `pipe += step` creates a new Pipeline. | Mutation causes real bugs in notebooks where cells are re-run. `base = Pipeline([...]); v1 = base; v1 += step` would corrupt `base` with mutation. Non-mutating `+=` follows Python numeric convention (int += 1 creates new int). |
| 106 | `TransformResult` wrapper with `__array__` protocol (NOT numpy subclass). | numpy subclassing is fragile across versions (`__array_ufunc__`, `__array_wrap__`, `__array_finalize__` interactions, slicing loses metadata, `np.concatenate` strips subclass). Wrapper with `__array__` protocol + property delegation (`shape`, `__getitem__`, `__len__`) gives sklearn compatibility via `np.asarray()` while being robust. Debugging value preserved: `result.columns[7]`, `result.sql`. |
| 107 | `pipe.freeze()` → `FrozenPipeline`: immutable, pre-compiled SQL, schema-validated, versioned. | Production deployments need guarantees: no accidental refit, no version drift, no schema surprises. Frozen pipeline is the deployment artifact. Development uses mutable Pipeline. |
| 108 | Pipeline versioning: format_version, sql_hash, sqlearn_version, schema_in/schema_out embedded in save format. | Without versioning, a sqlearn update that changes SQL generation silently changes pipeline output. SQL hash detects this. Format version enables migration. Schema embedding enables validation. |
| 109 | `pipe.detect_drift()` — pipeline-integrated drift detection comparing new data against training statistics. | The pipeline already knows training stats (from fit). Comparing new data stats is one SQL query. Catches distribution shifts, null increases, unseen categories. SQL-based, works on any data size. |
| 110 | `Columns` compiles to single SELECT (expression branches merged). `Union` compiles to CTE-per-branch or merged SELECT when all branches are expression-level. | Parallel composition must integrate cleanly with the single-query compilation strategy. Column routing is disjoint → safe to merge. Union branches may produce different columns → need CTEs unless all are expression-level. |
| 111 | Search error recovery: `on_error="skip"` (default), `failed_trials_`, `max_consecutive_errors=10`. | Model training fails in practice (OOM, NaN, bad params). Crashing the entire search on one failure wastes all progress. Skip + log is the right default. Consecutive error detection catches systematic problems. |
| 112 | `ModelPipeline` wraps preprocessing + model with `fit/predict/predict_proba/score/save/load`. Preprocessing serialized as JSON, model as pickle. | Clear boundary: SQL preprocessing (portable) vs model (Python-specific). `to_sql()` exports preprocessing only. Save format bundles both. `get_params`/`set_params` supports nested access for `sq.Search`. |
| 113 | `AutoFeatures.to_explicit()` converts auto-detected decisions into an explicit Pipeline with baked parameters. | Auto-detection is great for exploration but risky for production (thresholds may change between versions). `to_explicit()` freezes decisions into a deterministic pipeline. `autopipeline(as_code=True)` uses this internally. |
| 114 | `EWM` and `Deduplicate` reclassified as static (from dynamic). EWM's recursive CTE structure is determined by constructor args. Deduplicate's `ROW_NUMBER()` window is determined by `subset`/`keep` args. | Neither learns aggregate values from data. They use `query()` for SQL structure (CTE/window), not for data-dependent parameters. Static + `query()` = CTE without aggregation query. |
| 115 | Security model: sqlearn trusts all SQL inputs. `Expression()` and `Filter()` are NOT sanitized. Documented: never pass untrusted user input. | sqlearn is a data science library, not a web framework. SQL injection is the caller's responsibility. Studio runs on localhost only. Explicit documentation prevents false assumptions. |
| 116 | Scatter plots in Studio use `USING SAMPLE REPEATABLE(seed)` for consistent renders across refreshes. | Random sampling without seed means the plot changes on every refresh — confusing for users. Seeded sampling is deterministic. User can re-randomize explicitly. |
| 117 | 14-day free trial for Studio Pro. `sq.trial()` starts trial, no key needed. | Let users experience Pro before paying. Loss aversion after 14 days drives conversion. No credit card required. Trial state stored in `~/.sqlearn/trial`. |
| 118 | `<ProGate>` Svelte component for locked feature UI. Shows preview/screenshot + upgrade prompt. | Consistent UX across all Pro-gated features. Users see what they're missing. Every locked feature is a conversion touchpoint. |
| 119 | `require_pro(feature_name)` Python decorator for backend API gating. | Single point of enforcement. Raises `ProFeatureError` with activation instructions. Used on all Pro endpoints. |
| 120 | Free for students (`.edu` email) and open-source projects (MIT/Apache). | Builds goodwill, trains future paying users. Used at universities = credibility. Open-source projects = ecosystem growth. |
| 121 | `sq.autopipeline()` and `as_code=True` are FREE in the library API. Only the Studio UI pipeline builder and UI-based autopipeline are Pro. | Library users should never hit a paywall. The conversion driver is the visual experience, not the programmatic API. Keep the library complete. |
| 122 | Three levels of custom transformers: Level 1 `sq.Expression()` (static one-liner), Level 2 `sq.custom()` (template-based, covers 90%), Level 3 `sq.Transformer` subclass (full power). | Progressive complexity. The gap between "use built-ins" and "write sqlglot ASTs" was too wide. `sq.custom()` bridges it with SQL template strings that are parsed through sqlglot (safe, multi-database) but readable. |
| 123 | `sq.custom()` uses `{col}` and `{param}` placeholders in SQL templates. `learn=` dict makes it dynamic. Templates parsed through sqlglot at creation time. | Fail fast on bad SQL. `{col}` expands per target column. `{param}` expands to learned values (`{col}__{param}` naming). All SQL goes through sqlglot — never raw strings to the database. |
| 124 | `_validate_custom()` runs on first `fit()` for all non-built-in transformers. Checks: return types (sqlglot ASTs not strings/values), schema consistency (new columns need `output_schema()`), param consistency (expressions can't reference undiscovered params), classification consistency (static can't discover). | Correctness by construction. These are the exact bugs that cause silent data corruption in sklearn custom transformers. Catching them at fit time with clear error messages + fix suggestions prevents hours of debugging. |
| 125 | `sq.custom()` mode parameter: `"per_column"` (default, iterates over target columns) vs `"combine"` (uses all listed columns in one expression). | Most custom transforms apply the same logic per column (scaling, encoding, flagging). But some combine columns (BMI = weight/height^2). The mode makes both cases clean without overloading syntax. |
| 126 | Design philosophy: safe defaults, power opt-in. Base experience is sklearn-equivalent with zero surprises. `Auto*` features, `fast_explore`, precision control, multi-fidelity — all require explicit opt-in. Nothing clever happens unless the user asks for it. | ML correctness requires predictability. Auto-detection thresholds may change between versions. Clever defaults silently change pipeline behavior. Users must understand their preprocessing to trust their model. Explicit steps are auditable and reproducible. |
| 127 | `sq.quality()` — single data quality score (0-100) as the very first entry point. Answers "is my data ready for ML?" before any pipeline work. All checks are SQL-based aggregates in one query. | Users need a starting point. Not "read 50 lines of profile output" but one number + actionable breakdown. Scoring rules are deterministic and transparent (penalty table in docs). |
| 128 | `sq.check()` — leakage + mistake detection as a dedicated safety function. Separate from `fit()` warnings because it can compare train/test splits and check model-specific advice. | Leakage is the #1 silent killer in ML. No other preprocessing library detects it. sqlearn controls the full pipeline flow + uses SQL to check identifier columns, target proxies, train/test overlap, temporal leakage. |
| 129 | `pipe.audit()` — per-column trace showing before/after stats through every pipeline step. Debugging tool, not production path (runs N+1 profile queries). Supports `sample=` for speed. | "My model sucks and I don't know why" is the #2 ML pain point. Audit shows exactly what happened to every column. Catches: identifiers being scaled, high-cardinality OHE, redundant encoding, null handling effects. |
| 130 | `sq.feature_importance()` — pre-model feature ranking using SQL-based methods: Pearson, Spearman, Cramér's V, point-biserial, mutual information (binned), ANOVA F-stat. All computed via SQL aggregates + tiny Python post-processing. | Users create 50 features but most are noise. Ranking before training saves time and improves models. Mutual information (binned SQL approximation) catches non-linear relationships that Pearson misses — e.g., U-shaped or categorical interactions. |
| 131 | `sq.missing_analysis()` — MCAR/MAR pattern detection via null-indicator correlations. Little's MCAR test via chi-squared on null pattern frequencies. MNAR cannot be detected from data — always warn about possibility. | "15% nulls" isn't enough. Users need to know: are nulls random (MCAR → simple imputation is fine) or do they correlate with other columns (MAR → predictive imputation or add `_was_null` flag). All detection is SQL: CORR(null_indicator, other_columns). |
| 132 | Smart `fit()` warnings — automatic, checked during schema resolution. Warns about: identifiers, constants, high cardinality OHE, unused datetime/text, redundant scaling, already-binary columns, near-zero variance. All checks are SQL aggregates piggybacked on the discover phase. Opt-out via `warnings=False`. | Prevention is cheaper than debugging. These warnings catch the exact mistakes that cause "my model always predicts the same thing" or "my model has 99% accuracy" (because it memorized an ID column). Zero cost — checks run during discover() which already scans the data. |

---

## 17. Open Questions

All previous open questions have been resolved (see Decisions Log #90-#116).

Remaining open questions (resolve before affected milestone):

| # | Question | Affects | Notes |
|---|---|---|---|
| Q1 | ~~Should `TransformResult` subclass `np.ndarray` directly or use `__array__` protocol?~~ **RESOLVED:** Use `__array__` protocol wrapper (NOT subclass). numpy subclassing is fragile across versions. Wrapper implements `shape` as property, `__getitem__`, `__len__` for ergonomics. | Milestone 2 | Resolved. |
| Q2 | Should frozen pipeline embed the SQL string or the AST? | Milestone 7 | String is simpler and truly immutable. AST allows dialect switching post-freeze. Leaning toward string (dialect chosen at freeze time). |
| Q3 | How should `Union` handle branches with different row counts (e.g., one branch filters)? | Milestone 3 | Currently assumes all branches produce same rows. A filtering branch would break the side-by-side merge. Options: error on filter in Union branch, or use window-based row numbering. |

---

## 18. Resolved Decisions

Decisions that were open but are now settled:

**`pipe + pipe` operator** — DECIDED: `+` for sequential, with flattening.
- `a + b` → `Pipeline([a, b])`. `Pipeline + Pipeline` → flat merge of steps.
- `+=` for incremental building: `pipe += sq.StandardScaler()`.
- No `|` operator. Union must be explicit: `sq.Union([a, b])`.
- Flattening is safe — nesting produces identical SQL. If grouping is needed,
  use explicit `Pipeline([pipe_a, pipe_b])`.

**Lazy vs eager transform** — DECIDED: eager numpy by default.
- `transform()` returns numpy array immediately (sklearn compatible).
- `out="relation"` returns lazy DuckDB Relation (escape hatch for power users).
- Large result warning: if estimated size > 1GB, emit `UserWarning` with alternatives
  (`batch_size=`, `out="parquet"`, `out="relation"`). Threshold configurable via
  `sq.set_option("warn_result_size", "2GB")` or `None` to disable.
- No `__array__` protocol magic — it's fragile (breaks `.shape`, indexing, `isinstance`).

**Thread safety** — DECIDED: not thread-safe, with runtime guard.
- One connection per pipeline (required: in-memory DuckDB shares state via connection).
- `_check_thread()` detects cross-thread access and raises `SQLearnError` with
  message: "Use `.clone()` to create a thread-safe copy."
- `clone()` creates new Pipeline with same fitted params + new connection.
- `sq.Search(n_jobs=4)` handles parallel model training internally (SQL is sequential,
  model fits are parallel via joblib — no connection sharing).
- No locks, no pools, no thread-local hacks. Fail loudly.

**`y` parameter propagation** — DECIDED: `y` is a column name, not an array.
- `pipeline.fit("data.parquet", y="price")` — `y` is a string (column name).
- Multi-target: `y=["target1", "target2"]` — list of column names.
- `discover(columns, schema, y_column=None)` — steps that need target access it here.
- Steps that don't need `y` ignore `y_column`. No `needs_target` flag needed.
- Target column(s) excluded from `transform()` output by default (`exclude_target=True`).
- `pipe.transform("data", exclude_target=False)` includes target in output (for EDA,
  debugging, or chaining pipelines).
- If a step tries to use `y_column` and it's `None`, it raises `FitError`.
- This is the SQL-native answer: the target is already a column in the table.

**Sparse output** — DECIDED: dense by default for OneHotEncoder.
- `OneHotEncoder(sparse=False)` is the default. Returns dense numpy array.
  sklearn changed to dense default in v1.2 (2023) because sparse caused too many
  downstream issues: many models silently convert to dense, numpy operations don't
  work on sparse, pandas DataFrame surprises. Dense-by-default reduces friction at
  the model boundary. Users can opt in: `OneHotEncoder(sparse=True)` for high-cardinality.
- When cardinality > `max_categories` (default 30), emit a warning (not an error):
  ```
  UserWarning: OneHotEncoder: column 'product_id' has 847 categories.
  This creates 847 sparse columns. Consider using:
    - sq.TargetEncoder()      → 1 column  (needs y, best accuracy)
    - sq.HashEncoder(n=64)    → 64 columns (no fit needed, fast)
    - sq.FrequencyEncoder()   → 1 column  (simple, no y needed)
    - sq.AutoEncoder()        → auto-selects per column
  ```
- User can override: `OneHotEncoder(max_categories=None, sparse=False)` for dense + no warning.
- `sq.analyze("data")` (Phase 4) will recommend optimal encoder per column.

**Modular file structure** — DECIDED: one file per transformer type.
- `scalers/standard.py`, `scalers/minmax.py`, etc.
- `encoders/onehot.py`, `encoders/target.py`, etc.
- Adding a new transformer = one new file + `__init__.py` export. No touching core.
