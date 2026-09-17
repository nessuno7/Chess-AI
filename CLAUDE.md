# CLAUDE.md
 
Project instructions for Claude Code. Follow these rules on every task in this repository.
 
## Directory permissions
 
These rules control when you may edit files versus when you must ask first. They are scoped by directory and always apply.
 
### `tests/` — full autonomy
 
You may create, modify, refactor, or delete any code under `tests/` **without asking for permission**. Make the changes directly and describe what you did afterward.
 
### `python-engine/` — propose only
 
For any code under `python-engine/`, you must **not** modify files on your own. Instead:
 
- **Propose** the change (show the diff or describe exactly what you would edit and why).
- **Flag** any bugs, risks, or issues you notice.
- Only actually edit a file in `python-engine/` if I **explicitly** tell you to (e.g. "go ahead", "apply that change", "make the edit").
If a task seems to require changing `python-engine/`, stop at the proposal and wait for my explicit approval before touching those files.
 
## Default
 
If a file is outside both `tests/` and `python-engine/`, ask before making non-trivial changes unless I've said otherwise for that task.
 