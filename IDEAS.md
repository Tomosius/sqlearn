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

## Studio

-

## Business / Marketing

-

## Integrations

-

## Random

-
