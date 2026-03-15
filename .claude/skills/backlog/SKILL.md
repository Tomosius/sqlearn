---
name: backlog
description: Use when completing tasks, closing issues, updating progress, marking items done, or when the user mentions backlog, progress, or done. Also trigger when checking what to work on next, reviewing project status, starting new work items, creating GitHub issues, or when any work item transitions state (starting, blocking, dropping). If the user asks "what's next?" or "what should I work on?", use this skill. IMPORTANT — also trigger after any commit that completes a tracked item, to ensure GitHub stays in sync.
disable-model-invocation: false
user-invocable: true
---

# Backlog & Progress Tracking — sqlearn

Four systems must stay in sync. If any one falls behind, the project board becomes useless and the user has to fix it manually. Every state transition (start, complete, block, drop) must propagate to ALL relevant systems.

## Systems

| System | What | How to update |
|--------|------|---------------|
| `BACKLOG.md` | Local tracking with status markers | Edit the file |
| GitHub Issues | Per-task tracking | `gh issue close/create` |
| GitHub Milestones | Per-milestone grouping | `gh api` to close when all issues done |
| GitHub Project #22 | Visual board (owner: Tomosius) | `gh project item-add/item-edit` |

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

---

## When Something Is Completed

This is the most failure-prone workflow — all four systems must update. Do all steps, not just some.

1. **BACKLOG.md** — mark `[x]` on the completed item
2. **GitHub Issue** — close with a summary comment:
   ```bash
   gh issue close <number> --comment "<what was done, test count, key details>"
   ```
3. **Project Board** — verify the issue is on the board and status is Done:
   ```bash
   # Check if on board (look for the issue number in output)
   gh project item-list 22 --owner Tomosius --format json | jq '.items[] | select(.content.number == <N>)'

   # If missing, add it
   gh project item-add 22 --owner Tomosius --url https://github.com/Tomosius/sqlearn/issues/<N>

   # Set status to Done (fetch IDs dynamically)
   ITEM_ID=$(gh project item-list 22 --owner Tomosius --format json | jq -r '.items[] | select(.content.number == <N>) | .id')
   PROJECT_ID=$(gh project list --owner Tomosius --format json | jq -r '.projects[] | select(.number == 22) | .id')
   FIELD_ID=$(gh project field-list 22 --owner Tomosius --format json | jq -r '.fields[] | select(.name == "Status") | .id')
   OPTION_ID=$(gh project field-list 22 --owner Tomosius --format json | jq -r '.fields[] | select(.name == "Status") | .options[] | select(.name == "Done") | .id')
   gh project item-edit --project-id "$PROJECT_ID" --id "$ITEM_ID" --field-id "$FIELD_ID" --single-select-option-id "$OPTION_ID"
   ```
4. **Milestone check** — if all items in the milestone are now `[x]`:
   - See "Milestone Completion Checklist" below

## When Starting Work on an Item

1. **BACKLOG.md** — mark `[~]`:
   ```markdown
   - [~] `schema.py` — Schema dataclass, column type detection
   ```
2. **GitHub Issue** — if no issue exists, create one and add to board:
   ```bash
   gh issue create --title "<title>" --body "<description>" --milestone "<milestone name>"
   gh project item-add 22 --owner Tomosius --url <issue-url>
   ```

## When Creating a New GitHub Issue

Always do both — create the issue AND add it to the project board. Forgetting the board add is the #1 source of sync drift.

```bash
# Create with milestone
gh issue create --title "<title>" --body "<body>" --milestone "M<N> — <name>"

# Add to project board immediately
gh project item-add 22 --owner Tomosius --url <issue-url>
```

## When a Bug Is Found

Add to `## Bugs` in `BACKLOG.md`:

```markdown
## Bugs
- [ ] StandardScaler produces NaN when std=0 on constant column
```

Create a GitHub issue if it warrants tracking.

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

---

## Milestone Completion Checklist

When all items in a milestone are `[x]`, propagate to ALL systems:

1. **`pyproject.toml`** — bump version
2. **`src/sqlearn/__init__.py`** — bump `__version__`
3. **`CLAUDE.md`** — update current status section
4. **`BACKLOG.md`** — mark milestone header as complete
5. **GitHub Milestone** — close it:
   ```bash
   # Find milestone number
   gh api repos/Tomosius/sqlearn/milestones --jq '.[] | select(.title | startswith("M<N>")) | .number'
   # Close it
   gh api repos/Tomosius/sqlearn/milestones/<number> -X PATCH -f state=closed
   ```
6. **Project Board** — verify all milestone issues show "Done" status
7. **Project memory** — update `project_current_status.md`

## Sync Verification

When in doubt, run this to check all systems are aligned:

```bash
echo "=== Issues ===" && gh issue list --state all --milestone "<milestone>" --json number,state,title --jq '.[] | "\(.number) [\(.state)] \(.title)"'
echo "=== Milestone ===" && gh api repos/Tomosius/sqlearn/milestones --jq '.[] | select(.title | startswith("M<N>")) | "\(.title) — state: \(.state), \(.closed_issues) closed"'
echo "=== Board ===" && gh project item-list 22 --owner Tomosius --format json | jq -r '.items[] | "\(.content.number // "draft") [\(.status)] \(.title)"'
```
