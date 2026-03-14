# sqlearn Backlog

Tracking features, improvements, bugs, and ideas. Organized by priority and category.
Update this file as items are added, started, or completed.

**Status key:** `[ ]` todo, `[~]` in progress, `[x]` done, `[!]` blocked, `[-]` dropped

---

## Active — Next Up

These are ready to work on. Ordered by priority.

### Milestone 1 — Scaffolding
- [x] Initialize git repository
- [x] `pyproject.toml` with `src/sqlearn/` layout
- [x] CI (GitHub Actions: lint + test, Python 3.10-3.14)
- [x] Pre-commit hooks (ruff format, ruff check, mypy)
- [x] `CLAUDE.md` with project conventions (update from current draft)
- [x] Empty test suite that passes
- [x] `.gitignore`, `LICENSE` (MIT), `README.md` (minimal)
- [x] **Ship:** `pip install -e .` works, `pytest` runs, CI green

---

## Planned — Future Milestones

### Milestone 2 — Core Compiler (v0.1.0)
- [x] `schema.py` — Schema dataclass, column type detection
- [x] `transformer.py` — Transformer base class with `_validate_custom()` method
- [ ] `custom.py` — `sq.custom()` template-based transformer factory
- [ ] `backend.py` — Backend protocol + DuckDB implementation
- [ ] `io.py` — input resolver (table/file/DataFrame → DuckDB)
- [ ] `errors.py` — error hierarchy (incl. `ClassificationError`, `ProFeatureError`)
- [ ] `compiler.py` — expression composition, CTE promotion
- [ ] `pipeline.py` — Pipeline: fit, transform, fit_transform, to_sql
- [ ] `Imputer` — proves aggregation discovery, COALESCE composition
- [ ] `StandardScaler` — proves arithmetic composition
- [ ] `OneHotEncoder` — proves layer boundaries, CASE generation
- [ ] `sq.Expression()` — static one-liner custom transform
- [ ] `sq.custom()` — template-based custom transform (static + dynamic)
- [ ] Custom transformer validation tests (type checking, schema, params)
- [ ] Integration tests: pipeline output matches sklearn (`np.allclose`)
- [ ] SQL snapshot tests

### Milestone 3 — Composition + Breadth (v0.2.0)
- [ ] `Columns` (replaces ColumnTransformer)
- [ ] `Union` (replaces FeatureUnion)
- [ ] Auto column routing selectors
- [ ] `+` operator with flattening
- [ ] More scalers: MinMax, Robust, MaxAbs, Normalizer
- [ ] More encoders: Ordinal, Hash, Frequency
- [ ] Data operations: Rename, Cast, Filter, Sample, Deduplicate
- [ ] Data merging: merge(), concat(), Lookup

### Milestone 4 — Feature Engineering (v0.3.0)
- [ ] Arithmetic transforms (Log, Sqrt, Power, Clip, etc.)
- [ ] String transforms (Length, Lower, Split, Extract, etc.)
- [ ] Datetime transforms (DateParts, DateDiff, IsWeekend, etc.)
- [ ] Window transforms (Lag, Lead, Rolling*, Rank, etc.)
- [ ] OutlierHandler, TargetTransform
- [ ] Feature selection (DropCorrelated, VarianceThreshold, SelectKBest)
- [ ] AutoFeatures family

### Milestone 5 — Search (v0.4.0)
- [ ] CV splits (KFold, Stratified, Group, TimeSeries)
- [ ] SQL-based metrics
- [ ] `sq.Search` with pluggable samplers
- [ ] Multi-fidelity rounds
- [ ] Preprocessing dedup via AST hashing

### Milestone 6 — Analysis & Safety (v0.5.0)
- [ ] `sq.quality()` — data quality score (0-100) with breakdown
- [ ] `sq.profile()`, `sq.analyze()`, `sq.recommend()`
- [ ] `sq.autopipeline()`
- [ ] `sq.feature_importance()` — pre-model ranking (Pearson, MI, ANOVA, Cramér's V)
- [ ] `sq.missing_analysis()` — MCAR/MAR detection, null co-occurrence matrix
- [ ] `sq.check()` — leakage detection, common mistake warnings
- [ ] `pipe.audit()` — per-column trace through pipeline steps
- [ ] `sq.drift()`, `sq.correlations()`
- [ ] Smart warnings in `pipe.fit()` (identifiers, high cardinality, constants, etc.)
- [ ] Class imbalance detection + recommendations in `sq.analyze()`
- [ ] Target distribution analysis (skewness, outliers, transform suggestions)

### Milestone 7 — Export + Polish (v0.6.0 → v1.0.0)
- [ ] Multi-dialect export (Postgres, Snowflake, BigQuery, Spark)
- [ ] `to_config()` / `from_config()` YAML
- [ ] `save()` / `load()` binary
- [ ] `pipe.freeze()` → FrozenPipeline
- [ ] Documentation site
- [ ] PyPI release

### Milestone 8 — Studio Free + Pro Foundation (v1.1.0)
- [ ] Starlette backend + API endpoints
- [ ] License module (RSA validation, trial logic)
- [ ] Svelte frontend (profile, analysis, table, charts)
- [ ] Pro pipeline builder (license-gated)
- [ ] Pro chart customization/export (license-gated)
- [ ] Pro search monitor (license-gated)

### Milestone 9 — Studio Pro Completion (v1.2.0)
- [ ] Project scaffolder (multi-file generation)
- [ ] Notebook export (Marimo, Jupyter)
- [ ] Model fitting from UI
- [ ] Session save/resume
- [ ] 14-day trial flow

---

## Open Questions

Unresolved decisions. See `docs/14-decisions.md` for full context.

- [ ] Q2: Should frozen pipeline embed SQL string or AST? (Milestone 7)
- [ ] Q3: How should Union handle branches with different row counts? (Milestone 3)

---

## Ideas — Not Yet Planned

Future features and improvements to consider after v1.0.

- [ ] `sq generate` CLI command (generate project from command line, Pro feature)
- [ ] LLM-powered `sq.suggest()` (natural language → pipeline, uses Claude API)
- [ ] Pipeline templates/recipes (Kaggle starter, time series, high-cardinality, etc.)
- [ ] Spark backend (implement Backend protocol for PySpark)
- [ ] Postgres backend (implement Backend protocol for psycopg)
- [ ] Community plugin system (third-party transformer packages)
- [ ] VS Code extension (pipeline visualization, SQL preview)
- [ ] Benchmark suite published on website (sqlearn vs sklearn speed/memory)
- [ ] Team features for Studio Pro (shared sessions, collaboration)

---

## Bugs

None yet (pre-implementation).

---

## Improvements — Code Quality / DX

Tracked improvements that aren't features or bugs.

- [ ] After v0.1.0: restructure plan docs to reduce duplication (review item #23)
- [ ] After v0.1.0: documentation tiers for two audiences (review item #13)
- [ ] Validate expression composition at depth 5+ after Milestone 2 ships (review item #9)

---

## Review Items Incorporated

Items from the original plan review that have been addressed:

| # | Item | Status |
|---|---|---|
| 9 | Ship v0.1.0 fast, validate compiler | Priority note added to milestones |
| 13 | Two audiences (expert + beginner) | Deferred to after v0.1.0 |
| 14 | Free version too good | Business model restructured (Section 12 rewrite) |
| 15 | Timeline aggressive | 60-70 week realistic estimate, priority on M1-4 |
| 23 | Plan duplicates content | Deferred to after v0.1.0 |
| 1-8, 10-12, 16-22, 24-26 | Various fixes | All resolved in plan (see plan-review record) |
