---
name: commit
description: Use when committing changes, creating git commits, or when the user says commit, /commit, or asks to save progress
disable-model-invocation: false
user-invocable: true
---

# Semantic Commit Protocol — sqlearn

**ROLE:** Senior Release Engineer.
**TASK:** Produce the best possible semantic commit message for staged changes.

## Hard Rules

1. One logical change per commit. Mixed concerns → split.
2. Subject: imperative mood, ≤72 chars, no trailing `.!?`, no emojis.
3. Scope: domain name (not filename), ≤24 chars, lowercase, or omitted.
4. Never hallucinate ticket numbers, versions, or benchmarks.
5. Output raw commit text only — no commentary, no wrappers.
6. **No Co-Authored-By lines.** Never add co-author trailers.

## Before Writing the Message

1. `git diff --cached --name-only` — see what files changed
2. Read each changed file in full (not just the diff) — understand context
3. `git diff --cached` — see the precise changes
4. Check `BACKLOG.md` — is this part of a tracked milestone item?
5. Check `CLAUDE.md` — understand project conventions

**Never write a commit message from the diff alone.**

## Type Selection (Priority Order)

| Priority | Type | When |
|---|---|---|
| 1 | **fix** | Corrects incorrect behavior |
| 2 | **feat** | New user-facing functionality |
| 3 | **perf** | Performance improvement |
| 4 | **refactor** | No behavior change (restructure, extract, rename) |
| 5 | **build** | Build system, deps, toolchain |
| 6 | **test** | Tests only (no source changes) |
| 7 | **ci** | CI config only |
| 8 | **docs** | Documentation only |
| 9 | **style** | Formatting only |
| 10 | **chore** | Maintenance (.gitignore, cleanup) |

## Project Scopes

| Scope | Domain |
|---|---|
| `core` | transformer.py, pipeline.py, compiler.py, backend.py, schema.py, io.py |
| `custom` | custom.py — sq.custom() template-based transformers |
| `scalers` | StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler |
| `encoders` | OneHotEncoder, OrdinalEncoder, TargetEncoder, HashEncoder, etc. |
| `imputers` | Imputer |
| `features` | Arithmetic, string, datetime, window, cyclic, auto, outlier, target |
| `selection` | Drop, DropCorrelated, VarianceThreshold, SelectKBest |
| `ops` | Rename, Cast, Filter, Sample, Deduplicate |
| `data` | merge(), concat(), Lookup |
| `search` | sq.Search, samplers, spaces, metrics, cache, CV splits |
| `inspection` | profile, quality, analyze, recommend, check, audit, importance |
| `export` | to_sql, to_dbt, to_config, save/load, freeze |
| `studio` | Studio backend, API, frontend, license |
| `stats` | aggregates, correlations, tests, missing |
| `plot` | matplotlib visualizations |
| `docs` | Documentation files |
| `tests` | Test files only |
| `build` | pyproject.toml, CI, toolchain |
| `skills` | Agent skills (.claude/skills/) |

**Scope selection:** Use domain of the primary change. Source + matching tests → use source domain. >3 unrelated domains → omit scope.

**Banned scopes:** `utils`, `common`, `helpers`, `misc`, `general`, `lib`, `shared`

## Message Format

```
<type>(<scope>): <subject>

<WHY — motivation, cause, impact>

- <HOW bullet 1>
- <HOW bullet 2>
- <HOW bullet 3 max>

<footers>
```

## Subject Rules

- Imperative: "add" not "added"
- ≤72 chars total (including `type(scope): `)
- Lowercase first letter
- No trailing punctuation
- Breaking: `feat(core)!: require backend argument`

**Banned verbs:** update, change, modify, improve, adjust, tweak, handle, ensure, address

**Use instead:** add, remove, extract, validate, reject, enforce, replace, rename, implement, expose, generate, split, consolidate, port

## Body Rules

| Type | Body required? |
|---|---|
| feat, fix, perf | Yes |
| refactor, build | Recommended |
| test, docs, style, ci, chore | Optional |

## Atomicity

One commit = one logical change. Tests with their implementation = one commit.

If diff is not atomic, warn:
```
WARNING: This diff is not atomic.
Reason: <reason>
Recommended split:
1. type(scope): first change
2. type(scope): second change
```

## Pre-Output Checks

1. Type is highest valid match?
2. Scope is valid domain, not filename?
3. No banned verbs?
4. Subject ≤72 chars?
5. No trailing punctuation?
6. Atomic?
7. Body present if required?
8. Breaking `!` ↔ `BREAKING CHANGE:` consistent?
9. No guessed issue numbers?
10. No Co-Authored-By lines?

## After Committing

- Update `BACKLOG.md` if a tracked item was completed (mark `[x]`)
- If milestone item completed, note it

## Commit Message Template

Pass via HEREDOC:
```bash
git commit -m "$(cat <<'EOF'
type(scope): subject here

Why this change was needed.

- How bullet 1
- How bullet 2
EOF
)"
```
