"""Workspace-scoped shell for the research junior.

Behavior follows the UCE reader surface: work_root is the commissioned repo,
read-only ``os_bash`` uses the same command classes as
``src/scout/pathfinder/shell_router.py``, and output is head+tail capped
like ``src/scout/shell_output.py``. Shadow and Hub routing are absent because
these clones are the live tree.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path

DISCOVERY_SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".env",
        "dist",
        "build",
        ".next",
        ".nuxt",
        "target",
        ".tox",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "coverage",
        ".coverage",
        ".idea",
        ".vscode",
        "endpoints.dist",
        "endpoints.build",
    }
)

READ_COMMANDS = frozenset(
    {
        "cd",
        "pwd",
        "ls",
        "cat",
        "head",
        "tail",
        "less",
        "more",
        "find",
        "grep",
        "rg",
        "file",
        "stat",
        "echo",
        "wc",
        "tree",
        "mdls",
        "which",
        "open",
    }
)
WRITE_COMMANDS = frozenset({"mv", "cp", "rm", "mkdir", "touch", "chmod", "sed", "tee"})
INTERPRETER_COMMANDS = frozenset(
    {"python", "python3", "uv", "node", "ruby", "perl", "php", "bash", "sh", "zsh"}
)
TESTRUNNER_COMMANDS = frozenset(
    {
        "pytest",
        "mypy",
        "ruff",
        "black",
        "isort",
        "pyright",
        "pylint",
        "flake8",
        "npm",
        "pnpm",
        "yarn",
        "npx",
        "tsc",
        "vitest",
        "jest",
        "eslint",
        "prettier",
        "tsx",
        "make",
        "cargo",
        "go",
    }
)
BLOCKED_COMMANDS = frozenset({"sudo", "su", "dd", "mkfs", "diskutil"})
GIT_WRITE = frozenset(
    {"add", "commit", "push", "reset", "checkout", "switch", "merge", "rebase", "clean", "am", "apply"}
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_DEFAULT_BUDGET = 24000


class CommandClass:
    READ = "read"
    WRITE = "write"
    BLOCKED = "blocked"


def classify_os_bash(command: str) -> str:
    stripped = command.strip()
    if not stripped:
        return CommandClass.READ
    try:
        tokens = shlex.split(stripped)
    except ValueError:
        tokens = stripped.split()
    first = tokens[0].lower() if tokens else ""
    if first in BLOCKED_COMMANDS:
        return CommandClass.BLOCKED
    if first == "git":
        sub = ""
        for tok in tokens[1:]:
            if tok.startswith("-"):
                continue
            sub = tok.lower()
            break
        if sub in GIT_WRITE:
            return CommandClass.WRITE
        return CommandClass.READ
    if first in WRITE_COMMANDS or first in INTERPRETER_COMMANDS or first in TESTRUNNER_COMMANDS:
        return CommandClass.WRITE
    if ">" in stripped or ">>" in stripped:
        return CommandClass.WRITE
    return CommandClass.READ


def cap_command_output(output: str, budget: int = _DEFAULT_BUDGET) -> tuple[str, dict]:
    if not isinstance(output, str) or not output:
        return output, {"truncated": False}
    cleaned = _ANSI_RE.sub("", output) if "\x1b" in output else output
    if "\r" in cleaned:
        cleaned = "\n".join(
            line.rsplit("\r", 1)[-1] if "\r" in line else line for line in cleaned.split("\n")
        )
    if len(cleaned) <= budget:
        return cleaned, {"truncated": False}
    head_budget = int(budget * 0.30)
    tail_budget = budget - head_budget
    head = cleaned[:head_budget]
    tail = cleaned[-tail_budget:]
    if "\n" in head:
        head = head[: head.rindex("\n")]
    if "\n" in tail:
        tail = tail[tail.index("\n") + 1 :]
    omitted = len(cleaned) - len(head) - len(tail)
    marker = (
        f"\n\n… [ {omitted} characters omitted from the MIDDLE to fit the "
        f"model's context. The head and the tail are both shown — the tail "
        f"includes the command's final summary and any errors, so trust it "
        f"for the outcome. For full detail, re-run a narrower command. ] …\n\n"
    )
    return head + marker + tail, {
        "truncated": True,
        "total_chars": len(cleaned),
        "shown_chars": len(head) + len(tail),
    }


class Workspace:
    """cwd persists for the commission. Every path must stay under work_root."""

    def __init__(self, work_root: Path):
        self.work_root = work_root.resolve()
        self.cwd = self.work_root

    def resolve(self, path: str | None, *, allow_dir: bool) -> tuple[Path | None, dict | None]:
        cleaned = (path or "").strip()
        if not cleaned:
            return None, {
                "error": "missing_path",
                "detail": "path is required — relative to cwd, or absolute under the workspace.",
                "cwd": str(self.cwd),
            }
        if cleaned in {".", "./"} and not allow_dir:
            return None, {
                "error": "invalid_path",
                "path": cleaned,
                "cwd": str(self.cwd),
                "detail": "path must name a file. Pass a filename or a relative path.",
            }
        expanded = os.path.expanduser(cleaned)
        candidate = Path(expanded)
        if not candidate.is_absolute():
            candidate = self.cwd / candidate
        try:
            resolved = candidate.resolve()
        except OSError as exc:
            return None, {"error": "invalid_path", "path": cleaned, "detail": str(exc)}
        if not _is_under(resolved, self.work_root):
            return None, {
                "error": "outside_work_root",
                "path": cleaned,
                "work_root": str(self.work_root),
                "detail": (
                    f"{resolved} is outside the commissioned work_root {self.work_root}. "
                    "Use paths relative to work_root."
                ),
            }
        return resolved, None

    def change_directory(self, path: str) -> dict:
        resolved, err = self.resolve(path, allow_dir=True)
        if err:
            if err.get("error") == "missing_path":
                return {"error": "missing_path", "detail": "path is required."}
            return err
        assert resolved is not None
        if not resolved.is_dir():
            return {
                "error": "not_a_directory",
                "path": path,
                "cwd": str(self.cwd),
                "detail": f"Not a directory: {resolved}",
            }
        previous = str(self.cwd)
        self.cwd = resolved
        return {"cwd": str(self.cwd), "previous": previous}

    def bash(self, command: str, timeout: float = 30.0) -> dict:
        kind = classify_os_bash(command)
        if kind == CommandClass.BLOCKED:
            return {
                "command": command,
                "output": "Command blocked.",
                "exit_code": 1,
                "cwd": str(self.cwd),
            }
        if kind == CommandClass.WRITE:
            return {
                "command": command,
                "output": (
                    "Write commands are not allowed in read-only OS mode. "
                    "Use pathfinder(task) or enable OS write access."
                ),
                "exit_code": 1,
                "cwd": str(self.cwd),
            }
        tokens = _tokens(command)
        first = tokens[0].lower() if tokens else ""
        if first == "cd":
            target = tokens[1] if len(tokens) > 1 else str(self.work_root)
            moved = self.change_directory(target)
            if moved.get("error"):
                return {
                    "command": command,
                    "output": str(moved.get("detail") or moved["error"]),
                    "exit_code": 1,
                    "cwd": str(self.cwd),
                }
            return {"command": command, "output": "", "exit_code": 0, "cwd": str(self.cwd)}
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=self.cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            partial = exc.stdout or ""
            if isinstance(partial, bytes):
                partial = partial.decode("utf-8", "replace")
            capped, meta = cap_command_output(f"Timed out after {timeout:.0f}s.\n{partial}")
            result = {
                "command": command,
                "output": capped,
                "exit_code": 1,
                "cwd": str(self.cwd),
            }
            if meta.get("truncated"):
                result["output_truncated"] = True
                result["output_total_chars"] = meta["total_chars"]
            return result
        output = (proc.stdout or "") + (proc.stderr or "")
        capped, meta = cap_command_output(output)
        result = {
            "command": command,
            "output": capped,
            "exit_code": proc.returncode,
            "cwd": str(self.cwd),
        }
        if meta.get("truncated"):
            result["output_truncated"] = True
            result["output_total_chars"] = meta["total_chars"]
        return result


def _tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def should_skip_dir(name: str, include_hidden: bool) -> bool:
    if name in DISCOVERY_SKIP_DIRS:
        return True
    if not include_hidden and name.startswith("."):
        return True
    return False
