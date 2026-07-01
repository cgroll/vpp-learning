# Session Wrap-up

Run the following checks in order, then produce a consolidated report and suggest next actions. Fix nothing automatically — report and suggest.

## 1. Pipeline status

Run `dvc status` and note which stages (if any) are out of date or have missing outputs. If a stage is out of date, name it explicitly.

## 2. Git status

Run `git status` to list all modified, staged, and untracked files.

Group the pending changes by concern — for example:
- Pipeline scripts (`pipeline/`)
- Package code (`vpp/`)
- Notebooks / images (generated outputs)
- Config / infrastructure (`dvc.yaml`, `pyproject.toml`, `Makefile`, etc.)
- Docs (`README.md`, `CONVENTIONS.md`, `PROJECT.md`)

Flag if the pending changes span unrelated concerns (e.g., both new analysis scripts and infrastructure changes). In that case, suggest splitting into separate commits and propose groupings.

## 3. Code quality

Run these three checks and report pass/fail with any error details:

```bash
uv run ruff check vpp/ pipeline/
uv run ruff format --check vpp/ pipeline/
uv run pyright vpp/
```

## 4. Doc freshness

Run `git diff HEAD -- vpp/ pipeline/ dvc.yaml pyproject.toml Makefile` to understand what changed this session.

Read the current contents of `README.md`, `CONVENTIONS.md`, and `PROJECT.md`.

For each file, assess whether it needs updating based on the diff:

- **README.md** — update if setup steps, tool commands, or entry points changed
- **CONVENTIONS.md** — update if new pipeline patterns, path conventions, DVC stage patterns, or tooling were introduced
- **PROJECT.md** — almost always needs updating: current state, what was done, what was learned, revised next step

Say for each file: "looks current" or "suggest update: <brief reason>".

Then ask: *What should be recorded as the key outcome of this session and what is the next step?*

Use the answer to draft the `## Current state` and `## Next steps` sections of `PROJECT.md` and show the draft for approval before writing it.

## 5. Summary

Report a final checklist:

- [ ] Pipeline up to date (`dvc status` clean)
- [ ] No uncommitted relevant files (or commit plan is clear)
- [ ] `ruff check` passing
- [ ] `ruff format` passing
- [ ] `pyright` passing
- [ ] `README.md` current
- [ ] `CONVENTIONS.md` current
- [ ] `PROJECT.md` current

If all pass, suggest a commit message but do not run `git commit`.

If anything is failing or stale, list what needs to be addressed first.
