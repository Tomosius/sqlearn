# sqlearn — Agent Guide

> This file is for coding agents (Claude, Copilot, etc.). It explains what sqlearn is,
> how it works, and where to find detailed information. Read this first.

## What is sqlearn?

sqlearn compiles ML preprocessing pipelines to SQL via sqlglot ASTs. You write Python,
the system writes SQL. Every pipeline becomes one query. DuckDB is the default engine,
but any sqlglot-supported database (Postgres, MySQL, Snowflake, BigQuery) is a valid target.

```python
import sqlearn as sq

pipe = sq.Pipeline([
    sq.Imputer(),
    sq.StandardScaler(),
    sq.OneHotEncoder(),
])
pipe.fit("train.parquet", y="target")
X = pipe.transform("test.parquet")    # numpy array
sql = pipe.to_sql()                    # valid DuckDB SQL
```

**Name:** `sqlearn` (`import sqlearn as sq`). SQL-first sklearn alternative.

## Design Philosophy

**Safe defaults, power opt-in.** Nothing clever happens unless the user asks for it.

- Base experience is sklearn-equivalent: `fit()`, `transform()`, float64, no surprises
- `Auto*` features (`AutoFeatures`, `AutoEncoder`, etc.) are explicit pipeline steps, never injected
- `fast_explore`, precision control, multi-fidelity — all require explicit opt-in parameters
- `autopipeline()` is a convenience function, not a default behavior
- For production: `AutoFeatures.to_explicit()` freezes auto-decisions into deterministic pipeline

## Core Concepts

1. **One base class** — all transformers extend `Transformer` with three methods:
   `discover()` (learn stats via SQL), `expressions()` (inline SQL), `query()` (CTE SQL)
2. **Auto column routing** — `StandardScaler` defaults to numeric, `OneHotEncoder` to
   categorical. No `ColumnTransformer` needed for 80% of cases.
3. **Expression composition** — transformers compose into nested sqlglot ASTs that
   compile to a single SELECT statement. CTE promotion when nesting gets deep.
4. **Static vs dynamic** — transformers that need no data stats (Log, Rename) are static.
   Those that learn from data (StandardScaler, OneHotEncoder) are dynamic. Auto-detected.
5. **`y` is a column name** — `pipe.fit("data", y="price")`, not a numpy array.

## Three-Tier Product

| Tier | Install | What it does |
|---|---|---|
| **Library** (free, MIT) | `pip install sqlearn` | Pipeline, transformers, Search, analysis, all exports |
| **Studio Free** (free, MIT) | `pip install sqlearn[studio]` | Read-only EDA: profile, analysis, charts, data table |
| **Studio Pro** (paid, license key) | `sq.activate("SQ-XXXX")` | Pipeline builder, code gen, chart export, project scaffolding |

One repo. One package. Pro features are license-gated, not in a separate repo.

## Project Structure

```
sqlearn/
├── core/           # Transformer base, Pipeline, Compiler, Backend, Schema, IO, Output
├── stats/          # SQL building blocks: aggregates, correlations, tests, missing
├── scalers/        # StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
├── encoders/       # OneHotEncoder, OrdinalEncoder, TargetEncoder, HashEncoder, ...
├── imputers/       # Imputer (strategy per column)
├── features/       # Arithmetic, string, datetime, window, cyclic, auto, outlier, target
├── feature_selection/  # Drop, DropCorrelated, VarianceThreshold, SelectKBest, ...
├── ops/            # Rename, Cast, Filter, Sample, Deduplicate, Reorder
├── data/           # merge(), concat(), Lookup (mid-pipeline join)
├── selection/      # Search, CV splits, samplers, spaces, metrics, cache
├── inspection/     # profile(), analyze(), recommend(), autopipeline(), drift()
├── plot/           # Optional matplotlib visualizations (sqlearn[plot])
├── studio/         # Optional interactive dashboard (sqlearn[studio])
│   ├── license.py  # RSA license validation, require_pro(), trial logic
│   ├── api/        # REST + WebSocket endpoints (free and Pro-gated)
│   └── static/     # Pre-built Svelte frontend
└── export/         # to_sql(), to_dbt(), to_config(), export() standalone
```

**Extension pattern:** New transformer = one file in the right folder + `__init__.py` export.
No core changes needed. All transformers compose from `stats/aggregates.py` building blocks.

## Dependencies

Core: `duckdb>=1.0`, `numpy>=1.24`, `sqlglot[rs]>=25.0` (that's it)

Optional extras: `pandas`, `polars`, `arrow`, `sparse`, `plot` (matplotlib),
`optuna`, `yaml` (pyyaml), `studio` (starlette + uvicorn + websockets + plotly + nbformat)

## Custom Transformers — Three Levels

| Level | API | When to use |
|---|---|---|
| 1 | `sq.Expression("SQL")` | Static one-liner: `"price * qty AS revenue"` |
| 2 | `sq.custom(sql, columns=, learn=)` | Per-column, optionally learn stats. Covers 90% of needs. |
| 3 | `class MyTransformer(sq.Transformer)` | Full power: sets, CTEs, joins, windows |

All levels are validated, type-checked, and compose safely with built-in transformers.
See `docs/02-api-design.md` section 3.10 for the full guide.

## Safety & Analysis Functions

| Function | Purpose | When to use |
|---|---|---|
| `sq.quality()` | Data quality score (0-100) with breakdown | First thing — "is my data ready?" |
| `sq.profile()` | Types, nulls, stats, distributions | Explore data |
| `sq.analyze()` | Correlations, imbalance, skewness, multicollinearity | Understand target relationship |
| `sq.feature_importance()` | Pre-model feature ranking (Pearson, MI, ANOVA, Cramér's V) | Decide what to keep/drop |
| `sq.missing_analysis()` | MCAR/MAR pattern detection, null co-occurrence | Understand why values are missing |
| `sq.check()` | Leakage detection, common mistake warnings | Before training — catches silent errors |
| `pipe.audit()` | Per-column trace through every pipeline step | Debug bad model results |
| `pipe.detect_drift()` | Distribution comparison against training stats | Production monitoring |
| `pipe.fit()` warnings | Automatic: identifiers, high cardinality, constants, etc. | Always on (opt-out) |

All analysis is SQL-based. Runs on datasets of any size.

## Key Architecture Decisions

- All SQL via sqlglot ASTs, never raw strings (multi-database from day one)
- `discover()` returns aggregates (scalar stats), `discover_sets()` returns sets (categories)
- `expressions()` for inline transforms, `query()` for CTE-requiring transforms
- Expression composition merges nested ASTs into single SELECT
- CTE auto-promotion at depth > 8
- Two-phase discovery for CV: schema from full data (Phase 1), values per fold (Phase 2)
- `pipe.freeze()` → FrozenPipeline for production (immutable, pre-compiled, schema-validated)
- Pipelines are NOT thread-safe. Use `clone()` for thread-safe copies.
- `TransformResult` uses `__array__` protocol wrapper, NOT numpy subclass
- Custom transformers validated on first fit (type checking, schema consistency, param consistency)
- `sq.custom()` templates parsed through sqlglot at creation time (fail fast on bad SQL)

## Documentation Map

All design docs are in `docs/`. Read what you need:

| File | Content | When to read |
|---|---|---|
| `docs/01-overview.md` | What sqlearn is, what it replaces, what it doesn't | Start here for context |
| `docs/02-api-design.md` | Pipeline API, column routing, input/output, custom transformers | Building or modifying the public API |
| `docs/03-architecture.md` | Base class, static/dynamic classification, discover/expressions | Core engine work, new transformers |
| `docs/04-compiler.md` | Expression composition, CTEs, layer resolution | Compiler changes, SQL generation |
| `docs/05-transformers.md` | All built-in transformers with SQL examples | Adding/modifying transformers |
| `docs/06-model-integration.md` | ModelPipeline, sq.Search, CV, metrics | Model integration, hyperparameter search |
| `docs/07-analysis.md` | profile(), analyze(), recommend(), autopipeline(), Studio | Analysis features, Studio UI |
| `docs/08-export.md` | to_sql(), to_dbt(), frozen pipelines, versioning | Export formats, deployment |
| `docs/09-error-handling.md` | Error hierarchy, drift detection | Error handling patterns |
| `docs/10-project-structure.md` | File layout, dependencies, modularity guide | Project organization, adding new modules |
| `docs/11-business-model.md` | Three tiers, pricing, licensing, conversion funnel | Business decisions, Pro feature gating |
| `docs/12-milestones.md` | 9 milestones, release strategy, non-goals | Planning, what to build when |
| `docs/13-testing.md` | Test strategy per module, fixtures, CI matrix | Writing tests |
| `docs/14-decisions.md` | 121 decisions, technical risks, open questions | Understanding why things are the way they are |
| `docs/15-documentation.md` | Documentation strategy, phased rollout, docstring standards | Documentation work, adding examples |
| `docs/16-performance.md` | Performance strategy, caching, optimization rollout | Compiler optimizations, CV performance work |
| `BACKLOG.md` | Feature requests, bugs, improvements, ideas | What to work on next |

## Code + Docs + Tests = One Unit

Every code change ships with documentation and tests. No exceptions.

| Change type | Required docs | Required tests |
|---|---|---|
| New transformer | Docstring + API page + mkdocs nav entry + SQL example | All 11 mandatory tests + cross-library validation |
| New parameter | Docstring update | Parameter-specific tests |
| Bug fix | Update docs if behavior changes | Regression test proving the fix |
| New module | `__init__.py` docstring + API page | Module-level tests |
| Behavior change | Update all affected docs | Before/after tests |

Documentation standard: scikit-learn quality. Every public function has Args, Returns,
Raises, Examples. Every example is runnable. Generated SQL is shown. Edge cases documented.

Testing standard: compare with sklearn/scipy where possible. Test extreme edges (single row,
all NULLs, 1e308 values, unicode columns, SQL keyword columns). Run mutation testing on
critical modules. Zero surviving mutants in compiler.py and transformer.py.

## Coding Conventions

- Python 3.10+ (use `X | Y` union types, not `Union[X, Y]`)
- `src/sqlearn/` layout with `pyproject.toml`
- Tests: `pytest`, one test file per source file, `tests/` mirrors `src/`
- All SQL via sqlglot ASTs — if you're writing raw SQL strings, stop
- Transformers: implement `discover()`, `expressions()`, and/or `query()`
- Output: Arrow by default (built into DuckDB), optional numpy/pandas/polars
- Docstrings: Google style, required on all public classes/functions
- Type annotations: required on all function signatures (strict mode)

## Toolchain

| Tool | Purpose | Config location |
|---|---|---|
| `uv` | Package manager, virtualenv, lockfile | `pyproject.toml` |
| `ruff` | Linting + formatting (replaces black, isort, flake8, pylint, bandit) | `pyproject.toml [tool.ruff]` |
| `pyright` | Primary type checker (strict mode, Zed integration) | `pyproject.toml [tool.pyright]` |
| `mypy` | Secondary type checker (strict mode, CI gate) | `pyproject.toml [tool.mypy]` |
| `pytest` | Testing + coverage | `pyproject.toml [tool.pytest]` |
| `hypothesis` | Property-based testing (fuzz inputs) | Used in test files |
| `mutmut` | Mutation testing (verifies test quality) | `Makefile` |
| `interrogate` | Docstring coverage (95% minimum) | `pyproject.toml [tool.interrogate]` |
| `vulture` | Dead code detection | `pyproject.toml [tool.vulture]` |
| `pre-commit` | Git hooks (ruff + mypy + interrogate + vulture + hygiene) | `.pre-commit-config.yaml` |
| `make` | Dev workflow entry point | `Makefile` |

### Quick commands

```bash
make check                            # run ALL checks (lint + type + docs + dead code + test)
make lint                             # ruff check
make format                           # ruff format
make typecheck                        # pyright + mypy (strict)
make interrogate                      # docstring coverage
make vulture                          # dead code detection
make test                             # Tier 1: fast tests (failed first)
make cov                              # Tier 2: full coverage
make mutant                           # Tier 3: mutation testing
make test-full                        # Tier 2+3: coverage + mutation
make prerelease                       # test across Python 3.10-3.14
make clean                            # remove build artifacts
```

### Three-tier testing strategy

| Tier | Command | When to use |
|---|---|---|
| 1 | `make test` | During development — fast, failed-first, stops on first failure |
| 2 | `make cov` | Before committing — full coverage report |
| 3 | `make test-full` | Before releases — coverage + mutation testing (slow) |

### Editor

Zed project settings in `.zed/settings.json`:
- Pyright LSP (strict mode, workspace diagnostics)
- Ruff formatter on save (auto-fix + organize imports)
- Ruler at column 99

## Current Status

Milestone 1 (scaffolding) and Milestone 2 (core compiler, v0.1.0) complete.
Milestone 3 (composition + breadth, v0.2.0) is next.
See `docs/12-milestones.md` for the full roadmap.
See `BACKLOG.md` for tracked items.
