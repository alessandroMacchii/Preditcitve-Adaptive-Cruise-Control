# ClaudeExplained.md — what Claude Code can do in this repo (and what it asks first)

> A practical description of how Claude Code's permissions work **in this project**, written
> by Claude itself. It is not official documentation: it is "how things stand here", verified against the
> real configuration of the repo (`.claude/settings.local.json`) and the behavior observed
> across sessions.

---

## 1. How the permission system works (in short)

Every Claude action goes through a **tool** (read a file, modify it, run a command…).
For each call the system decides among three outcomes: **allowed without asking**, **requires
confirmation** (a yes/no prompt appears to the user), or **denied**. The decision depends on:

1. **The type of tool**: read-only operations are free; those that change state
   ask.
2. **The session's permission mode** (changed with `Shift+Tab` or `/permissions`):
   - *default* — file edits and shell commands ask for confirmation;
   - *accept edits* — edits to project files are auto-approved, commands ask;
   - *plan mode* — read-only: Claude can explore but touch nothing;
   - *bypass permissions* — everything auto-approved (use with caution).
3. **The saved allowlists** in the configuration files: `~/.claude/settings.json` (user, applies
   everywhere), `.claude/settings.json` (project, shared via git) and `.claude/settings.local.json`
   (project, local only). When you answer "yes, always" to a confirmation request, the rule
   lands there and from then on it does not ask again.
4. A **user's no** to a prompt counts as an instruction: Claude does not retry the same identical
   command, it changes approach.

---

## 2. What Claude can do WITHOUT asking (in this repo)

### Always, by the nature of the tool (read-only)
- **Read any project file**: code, notebooks (cell by cell, outputs included),
  markdown, PDFs (e.g. the course slides), images, small CSV/parquet files.
- **Search**: grep on contents, glob on file names, explore the folder structure.
- **Query git read-only**: `git status`, `git log`, `git diff`, `git show` (read-only
  commands run in a sandbox without a prompt).
- **Write to the session scratchpad** (a temporary folder isolated from the project) and to
  Claude's **persistent memory** (`~/.claude/projects/<project>/memory/`): notes that
  survive between sessions, outside the repo.

### Because authorized in the past in THIS repo (`.claude/settings.local.json`)
Every "yes, always" given in previous sessions left a rule. The most relevant today:
- **`PowerShell(Remove-Item *)`** → Claude can **delete files and folders without a prompt**.
  ⚠️ It is the broadest and most delicate rule in the list: it is why the removal of
  two folders earlier this session did not ask for confirmation. To go back to the prompt, remove
  the line from `.claude/settings.local.json` (or use `/permissions`).
- **`Bash(git rm *)`** and **`Bash(git restore *)`** → removal of tracked files and restoring
  files from a commit (the latter can **discard local changes**: this one too is broader than
  it looks).
- **`Bash(python -c '…)`** and **`Bash(./.venv/Scripts/python.exe -c '…)`** → run arbitrary
  Python snippets without a prompt.
- Now-historical point rules: running specific scripts (`build_elevation_cache.py`,
  `build_presentation.py`, the now-deleted `_edit_nb*.py` scripts), `pip install python-pptx`,
  reading the LibreOffice/Office folders, `jq --version`, two `git log`/`git status` commands
  with an explicit path.

### Workflow behaviors (do not require permission by definition)
- Reasoning, answering, proposing plans, asking questions.
- Using multiple tools in parallel, launching read-only research sub-agents.

---

## 3. What it ASKS first (in default mode)

- **Creating or modifying project files** (Edit/Write): every change shows the diff and waits for
  confirmation — unless the session is in *accept edits* or a saved rule exists.
- **Shell commands that change state** and are not in an allowlist: `git add/commit/push`,
  `pip install`, moving/renaming files, starting long processes.
- **Network access** (WebFetch/WebSearch, API calls via script): requires confirmation per
  domain/command, except for saved rules.
- **Anything that touches files outside the project** (apart from scratchpad and memory).

## 4. Rules Claude gives ITSELF in this project (beyond the technical permissions)

These are not imposed by the system but by `CLAUDE.md` (project conventions) — they hold even
when the technical permission would be there:
- **Ask Alex before long/costly actions** (massive API calls like the elevation
  cache, heavy fits like Optuna) **or destructive ones**, even if the allowlist would allow them.
- **Do not commit/push on its own initiative**: git commit only when requested.
- **Alex runs the notebooks** in the `.venv` kernel: Claude prepares the code, does not launch long
  runs on its own.
- Before deleting/overwriting something it did not create, **look inside** and
  flag it if the content does not match how it was described.

## 5. What Claude cannot do anyway

- Answer interactive terminal prompts (no `git rebase -i`, `Read-Host`, interactive
  logins): commands must be non-interactive.
- Bypass a user's "no" or a block imposed by a hook.
- See the screen, other apps, or files never indicated/reachable from the filesystem.
- Remember between sessions what is not written (in the repo files or in its memory):
  every session restarts from the documents — this is why STATE.md exists.

---

## 6. Operational summary

| Action | Asks? |
|---|---|
| Read/search files, git read-only | No |
| Write to scratchpad / own memory | No |
| Delete files (`Remove-Item`) | **No** (⚠️ rule saved in this repo) |
| `git rm` / `git restore` / `python -c` | No (saved rules) |
| Create/modify project files | Yes (diff + confirmation; no in *accept edits*) |
| `git commit` / `git push` | Yes (and by convention only on request) |
| Install packages, new commands that change state | Yes |
| Network (fetch/search/API) | Yes, except for saved rules |
| Long runs (Optuna, elevation cache) | Technical permission aside, **asks by convention** |

> Hygiene tip: the list in `.claude/settings.local.json` should be cleaned up now and then — the
> point rules on deleted scripts are harmless but dead, while `Remove-Item *`,
> `git restore *` and the free `python -c` deserve a second thought: convenient, but broad.
