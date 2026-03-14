---
name: backlog
description: Use when completing tasks, closing issues, updating progress, marking items done, or when the user mentions backlog, progress, or done. Also trigger when checking what to work on next, reviewing project status, starting new work items, or when any work item transitions state (starting, blocking, dropping). If the user asks "what's next?" or "what should I work on?", use this skill.
disable-model-invocation: false
user-invocable: true
---

# Backlog & Progress Tracking — sqlearn

## Files

| File | Purpose | Flow |
|---|---|---|
| `IDEAS.md` | Raw unstructured ideas (not evaluated) | Capture first |
| `BACKLOG.md` | Tracked items: milestones, planned work, bugs, improvements | Evaluated, prioritized |
| `CLAUDE.md` | Current project status (milestone progress) | Updated on milestone completion |

**Flow:** `IDEAS.md` → `BACKLOG.md` → implementation → mark `[x]`

## Status Key

```
[ ]  — todo
[~]  — in progress
[x]  — done
[!]  — blocked
[-]  — dropped
```

## When Something Is Completed

1. **Read** `BACKLOG.md` to find the item
2. **Mark** `[x]` on the completed item
3. **Check** if the milestone section has all items done
4. If milestone complete:
   - Update version in `pyproject.toml`
   - Update `CLAUDE.md` current status section
   - Note the milestone as complete in `BACKLOG.md`

## When Starting Work on an Item

Mark it `[~]` so progress is visible:

```markdown
- [~] `schema.py` — Schema dataclass, column type detection
```

## When a Bug Is Found

Add to `## Bugs` in `BACKLOG.md`:

```markdown
## Bugs
- [ ] StandardScaler produces NaN when std=0 on constant column
```

## When an Improvement Is Identified

Add to `## Improvements — Code Quality / DX` in `BACKLOG.md`:

```markdown
## Improvements — Code Quality / DX
- [ ] Compiler should deduplicate identical CTE expressions
```

## When a New Idea Comes Up

Add to `IDEAS.md` under the appropriate section (`Library`, `Studio`, `Business / Marketing`,
`Integrations`, `Random`). Don't evaluate — just capture.

When an idea matures (discussed, validated), move it to `BACKLOG.md` under the right category.

## When Something Is Blocked

Mark `[!]` with a reason:

```markdown
- [!] `compiler.py` — expression composition (blocked: needs schema.py first)
```

## When Something Is Dropped

Mark `[-]` with a reason so we know why:

```markdown
- [-] Spark backend (dropped: not enough demand, revisit post-v2)
```

## BACKLOG.md Structure

```markdown
## Active — Next Up              # current milestone items
## Planned — Future Milestones   # milestone 2, 3, 4, ...
## Open Questions                # unresolved decisions
## Ideas — Not Yet Planned       # future features post-v1.0
## Bugs                          # known bugs
## Improvements — Code Quality   # DX improvements
```

## Milestone Completion Checklist

When all items in a milestone are `[x]`:

1. Update version in `pyproject.toml`
2. Update `CLAUDE.md` current status section
3. Mark milestone header as complete in `BACKLOG.md`
4. Move next milestone items to `## Active — Next Up` if appropriate
