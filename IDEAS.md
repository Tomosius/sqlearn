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
  current three-tier system. *(from ducklearn_conda)*
- `make release` workflow: single command that bumps version via commitizen, pushes tags,
  and deploys versioned MkDocs docs via mike. *(from ducklearn-old)*
- CTE-per-step mental model: each pipeline step maps to exactly one CTE (`step_00`,
  `step_01`). Simpler alternative to expression composition for debugging. Consider as
  a `compile_mode="debug"` option. *(from ducklearn-old)*

## Studio

-

## Business / Marketing

-

## Integrations

-

## Random

-
