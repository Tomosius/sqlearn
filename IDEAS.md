# sqlearn Ideas

Raw thoughts, half-baked concepts, shower ideas. Dump anything here.
No evaluation needed — just capture it before you forget.

When an idea matures, move it to `BACKLOG.md` under the right category.

**Flow:** IDEAS.md → BACKLOG.md → docs/12-milestones.md

---

## Library

- `sync_python_version.py` script: auto-sync Python version strings across pyproject.toml
  sections (requires-python, ruff target-version, mypy python_version, pyright pythonVersion,
  classifiers). Prevents version drift. Low priority since Python version changes rarely.
  *(from ducklearn_conda)*
- `_get_tags()` / `_more_tags()` extensible metadata system: base tags (`requires_fit`,
  `returns_cte`, `engine`) merged with subclass overrides. Could be useful for plugin
  ecosystem. *(from ducklearn_pycharm)*
- `consume_fit_result(df)` hook on base class: generic method for storing learned values
  from fit SQL results. Simple pattern: `self.values_ = df.to_dict()`. Consider if this
  simplifies the discover() → params_ flow. *(from ducklearn_pycharm)*
- Static vs dynamic classification via AST inspection: use sqlglot to inspect discover()
  return values and auto-classify. Could be an additional validation layer beyond the
  current three-tier system. Full heuristic implementation exists in ducklearn1's
  `FitInspector` (SELECT *, DISTINCT, GROUP BY, window functions, scalar aggregates,
  two-pass CTE analysis). See `docs/03-architecture.md` Section 4.3.12b for details.
  *(from ducklearn_conda, ducklearn1)*
- `make release` workflow: single command that bumps version via commitizen, pushes tags,
  and deploys versioned MkDocs docs via mike. *(from ducklearn-old)*
- CTE-per-step mental model: each pipeline step maps to exactly one CTE (`step_00`,
  `step_01`). Simpler alternative to expression composition for debugging. Consider as
  a `compile_mode="debug"` option. *(from ducklearn-old)*
- Three-phase planner as explicit objects: `FitInspectionPlanner` → `FitExecutionPlanner`
  → `TransformPlanner`. Currently implicit in compiler. Making them explicit classes with
  `StepFitInfo` / `FitInspectionPlan` dataclasses improves testability and enables
  per-phase optimization. See `docs/04-compiler.md` Section 5.3b. *(from ducklearn1)*
- Fit query batching optimization: combine multiple static discover() queries into one
  multi-CTE query to minimize DuckDB round-trips. Each CTE maps back to its originating
  step. See `docs/04-compiler.md` Section 5.3b. *(from ducklearn1)*
- `HasGetParams` runtime-checkable Protocol: enables `get_params(deep=True)` to recurse
  into any object implementing `get_params()`, not just Transformer subclasses. Useful
  for third-party estimator interop. See `docs/03-architecture.md` Section 4.3.12c.
  *(from ducklearn1)*
- SQL lint pre-commit tool: Python AST extracts SQL-like strings from source files
  (string literals, `execute()`/`query()`/`sql()` calls), validates via `sqlglot.parse()`
  with DuckDB dialect. Less relevant since sqlearn uses sqlglot ASTs not raw SQL strings,
  but useful for catching bad SQL in tests or examples. *(from ducklearn1)*
- Commitizen for conventional commits: automated version bumping and changelog generation.
  Pairs with the `make release` workflow idea. Could integrate with pre-commit hooks for
  commit message validation. *(from ducklearn1)*
- MkDocs Material + mkdocstrings: auto-generated API docs from docstrings. ducklearn1 had
  a working setup with Material theme, section-index, and awesome-pages plugins. Reference
  implementation for Milestone 7 documentation site. *(from ducklearn1)*
- Extended static analysis pipeline: Bandit (security scanning), Vulture (dead code),
  pip-audit (dependency vulnerabilities), Pyright (additional type checking) alongside
  ruff + mypy. Consider adding to CI or pre-commit. *(from ducklearn1)*
- Exhaustive combinatorial testing: generate all non-empty subsets of feature flags
  via `itertools.combinations` to test every combination of transformer features. Catches
  interaction bugs that individual tests miss. See `docs/13-testing.md`.
  *(from ducklearn1)*

### Cross-Validation Performance (M5)

- **Compile Once, Execute Many (QueryTemplate):** Pipeline compiles the self-contained CTE
  query into a frozen `QueryTemplate` once — a parameterized AST with `:k` as the only
  variable. The CV system binds fold values and executes. No fold-specific state lives on
  the pipeline (no `params_` overwrite problem). The pipeline is the compiler, the CV system
  is the executor. One compilation instead of K. See `docs/16-performance.md` Section 16.6.
- **Zero-variance column skipping at output boundary:** When superset discovery (Phase 1)
  creates columns absent from some folds (e.g., OHE category not in fold 3's training data),
  the SQL schema stays consistent (all folds produce same columns), but constant columns are
  detected and excluded at the numpy handoff before model training. Detection via DuckDB zone
  maps: `MIN(col) = MAX(col)` is O(1) metadata lookup, no scan. Not the same as
  VarianceThreshold (explicit step) — this is an implicit optimization at the output layer.
  See `docs/16-performance.md` Section 16.6 and `docs/03-architecture.md` Section 4.7b.
- **Materialize fold source view once:** Create `__sq_source__` temp view with `__sq_fold__`
  column baked in at CV start. All subsequent queries (discovery + transform) read from this
  view. The `NTILE(k) OVER (ORDER BY HASH(rowid || seed))` computation runs exactly once,
  not per query. Every fold filter is a simple `WHERE __sq_fold__ != :k` against the
  pre-computed column.
- **FILTER clause for single-scan fold stats:** DuckDB's `AVG(x) FILTER (WHERE __sq_fold__
  != :k)` computes per-fold aggregates in ONE table scan. Much faster than K separate queries
  with `WHERE __sq_fold__ != :k` each. Combined with QueryTemplate, this means: one scan for
  all fold stats + one scan per fold for transform = K+1 scans total vs 2K scans (K fit + K
  transform). Already designed in `docs/03-architecture.md` Section 4.7, but this is the
  single biggest CV performance win and should be emphasized in implementation.
- **Fold-column indexing for persistent tables:** If source is a persistent DuckDB table
  (not a file), an index on `__sq_fold__` makes fold filtering near-free. For parquet files,
  DuckDB handles this automatically via row group pruning — no user action needed. Consider
  documenting this as a "performance tip" for users with very large persistent tables.

## Studio

-

## Business / Marketing

-

## Integrations

- `sqlearn[rs]` — Rust backend as optional install. Port compiler hot paths (expression
  composition, aggregation batching, CTE optimization) to Rust via PyO3. Python API stays
  identical — users don't touch the Rust layer. Biggest win: DuckDB extension that accepts
  query plans directly, skipping SQL string serialization/parsing round trip. Could also
  port Studio UI backend to Rust (Axum/Actix) for faster response times. Ship after v1.0
  when the Python design is proven and stable. Two codebases maintained in parallel: Python
  for readability/contributions, Rust for production performance.
- Tiered SQL parsing in sqlearn[rs]: use `sqlparser-rs` (Rust-native, DuckDB dialect support)
  for the default DuckDB path — avoids sqlglot's Python overhead for AST construction and
  SQL generation. Fall back to sqlglot for unsupported dialects (Postgres, Snowflake, BigQuery).
  Three tiers: (1) sqlearn[rs]+DuckDB → sqlparser-rs native, (2) sqlearn[rs]+other DB →
  sqlglot fallback, (3) sqlearn pure Python → sqlglot for everything. Same API surface
  regardless of backend. Could also explore `duckdb-rs` for direct plan injection bypassing
  SQL strings entirely.
- Rust-native Studio UI: replace Starlette/FastAPI with Axum or Actix-web. Rust backend
  communicates with DuckDB via `duckdb-rs` (direct in-memory access, no serialization
  overhead). Svelte frontend stays the same — just faster API responses. Benefits: single
  Rust process handles both compiler and UI serving, shared DuckDB connection pool in-process,
  zero-copy data transfer between compiler output and API responses. Could use `arrow-rs` for
  Arrow IPC between DuckDB and the UI layer without Python intermediary.
- DuckDB extensions for sqlearn: custom C/Rust extensions that register sqlearn-specific
  functions (e.g., `sqlearn_transform(plan, table)` that accepts a serialized FitPlan and
  executes it natively inside DuckDB). Eliminates the Python→SQL→DuckDB round trip entirely.
  Could also register custom aggregate functions for complex discovery patterns. Explore
  DuckDB's extension API (`duckdb_extension`) for loadable modules. Ultimate goal: `LOAD
  sqlearn; SELECT * FROM sqlearn_transform('plan.json', 'data.parquet');` — pure SQL,
  no Python runtime needed for inference/scoring.

## Random

-
