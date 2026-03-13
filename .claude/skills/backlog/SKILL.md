---
name: backlog
description: Use when completing tasks, closing issues, updating progress, marking items done, or when the user mentions backlog, progress, or done
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Edit, Bash, Grep
---

# Backlog & Progress Tracking — sqlearn

## Files

| File | Purpose |
|---|---|
| `BACKLOG.md` | Tracked items: milestones, planned work, bugs, improvements |
| `IDEAS.md` | Raw unstructured ideas (not evaluated yet) |
| `CLAUDE.md` | Current project status |

## When Something Is Completed

1. **Read** `BACKLOG.md` to find the item
2. **Mark** `[x]` on the completed item
3. **Check** if the milestone section has all items done
4. If milestone complete, note it in the `CLAUDE.md` current status section

```markdown
# Before
- [ ] `schema.py` — Schema dataclass, column type detection

# After
- [x] `schema.py` — Schema dataclass, column type detection
```

## When a Bug Is Found

Add to the `## Bugs` section in `BACKLOG.md`:

```markdown
## Bugs
- [ ] StandardScaler produces NaN when std=0 on constant column
```

## When an Improvement Is Identified

Add to `## Improvements` section in `BACKLOG.md`:

```markdown
## Improvements
- [ ] Compiler should deduplicate identical CTE expressions
```

## When a New Idea Comes Up

Add to `IDEAS.md` under the appropriate section. Don't evaluate — just capture.

When an idea matures (discussed, validated), move it to `BACKLOG.md` under the right category.

## Status Key

```
[ ]  — todo
[~]  — in progress
[x]  — done
[!]  — blocked
[-]  — dropped
```

## Milestone Completion

When all items in a milestone are `[x]`:

1. Update version in `pyproject.toml`
2. Commit: `chore(release): bump version to X.Y.Z`
3. Update `CLAUDE.md` current status
4. Note the milestone as complete in `BACKLOG.md`
