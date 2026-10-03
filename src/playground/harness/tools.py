"""Reader tools.

Names, argument names, and result shapes follow the UCE research junior
(``research_junior_tools`` in ``src/scout/code/commissions.py`` plus the
pathfinder implementations). ``read_file`` is the exception: UCE's research
junior binds ``_make_smart_read_file``, which summarizes files over 400
lines. Here ``read_file`` is a pager and returns raw lines for
``start_line``..``end_line``.
"""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
from pathlib import Path

from playground.harness.shell import DISCOVERY_SKIP_DIRS, Workspace, should_skip_dir

LIST_FILES_MAX_ENTRIES = 500
GLOB_MAX_RESULTS = 500
GREP_MAX_RESULTS = 200
GREP_MAX_FILE_BYTES = 2_000_000
GREP_LINE_PREVIEW_LIMIT = 400

# One page can be long enough that a transcript lands in the 32k–96k band,
# and short enough that the rest of the turn still fits under 96k.
DEFAULT_PAGE_LINES = 500
MAX_PAGE_LINES = 4000
MAX_PAGE_CHARS = 200_000

_REGEX_METACHAR_ESCAPE = re.compile(r"\\([().\[\]{}*+?|^$])")
_BINARY_EXT = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".pdf",
        ".zip",
        ".gz",
        ".wasm",
        ".woff",
        ".woff2",
        ".ico",
        ".mp4",
        ".mp3",
        ".pyc",
        ".so",
        ".dylib",
        ".bin",
    }
)


def _missing(name: str, detail: str) -> dict:
    return {"error": f"missing_{name}", "detail": detail}


def _deescape_substring_pattern(pattern: str) -> str:
    return _REGEX_METACHAR_ESCAPE.sub(r"\1", pattern)


def _rel(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


class ReaderTools:
    def __init__(self, shell: Workspace):
        self.shell = shell

    def dispatch(self, name: str, arguments: dict) -> dict:
        fn = {
            "match_path": self.match_path,
            "get_cwd": self.get_cwd,
            "change_directory": self.change_directory,
            "list_files": self.list_files,
            "glob": self.glob,
            "grep": self.grep,
            "read_file": self.read_file,
            "os_bash": self.os_bash,
            "web_search": self.web_search,
            "fetch_url": self.fetch_url,
        }.get(name)
        if fn is None:
            return {
                "error": f"Unknown tool {name!r}. No Python callable bound.",
                "available": sorted(_TOOL_SCHEMAS),
            }
        try:
            return fn(**_filter_args(fn, arguments or {}))
        except TypeError as exc:
            return {"error": "bad_arguments", "detail": str(exc)}

    def get_cwd(self) -> dict:
        return {"cwd": str(self.shell.cwd)}

    def change_directory(self, path: str | None = None) -> dict:
        return self.shell.change_directory(path or "")

    def match_path(self, query: str = "", limit: int = 5, kind: str = "directory") -> dict:
        cleaned = (query or "").strip()
        if not cleaned:
            return {
                "error": "missing_query",
                "detail": "query is required — e.g. 'documents', 'downloads', 'unified_compute_engine'.",
            }
        try:
            limit_n = int(limit)
        except (TypeError, ValueError):
            limit_n = 5
        limit_n = max(1, min(limit_n, 20))
        kind_norm = (kind or "directory").strip().lower()
        if kind_norm not in {"directory", "file", "any"}:
            kind_norm = "directory"
        needle = cleaned.casefold()
        tokens = [t for t in re.split(r"[\s/_-]+", needle) if t]
        scored: list[tuple[int, Path, str]] = []
        root = self.shell.work_root
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not should_skip_dir(d, include_hidden=False)]
            current = Path(dirpath)
            candidates: list[tuple[str, Path]] = []
            if kind_norm in {"directory", "any"} and current != root:
                candidates.append(("directory", current))
            if kind_norm in {"file", "any"}:
                candidates.extend(("file", current / name) for name in filenames if not name.startswith("."))
            for item_kind, path in candidates:
                name = path.name.casefold()
                score = 0
                if name == needle:
                    score += 100
                elif needle in name:
                    score += 60
                elif needle in _rel(root, path).casefold():
                    score += 40
                score += sum(15 for tok in tokens if tok in name)
                if score <= 0:
                    continue
                scored.append((score, path, item_kind))
            if len(scored) > 4000:
                break
        scored.sort(key=lambda item: (-item[0], _rel(root, item[1])))
        matches = []
        for score, path, item_kind in scored[:limit_n]:
            matches.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "score": score,
                    "kind": item_kind,
                    "source": "workspace",
                    "reason": "name match under the commissioned workspace",
                }
            )
        return {
            "query": cleaned,
            "kind": kind_norm,
            "matches": matches,
            "advice": (
                "Pick the top match and continue with list_files, grep, or read_file on that path."
                if matches
                else "No strong match — try a different query."
            ),
        }

    def list_files(
        self,
        path: str | None = None,
        recursive: bool = False,
        max_depth: int = 3,
        include_hidden: bool = False,
    ) -> dict:
        root, err = self._walk_root(path, default_to_work_root=False)
        if err:
            return err
        assert root is not None
        try:
            depth_cap = max(1, int(max_depth))
        except (TypeError, ValueError):
            depth_cap = 3
        entries: list[dict] = []
        truncated = False

        def add(child: Path, depth: int) -> None:
            try:
                kind = "directory" if child.is_dir() else "file"
                size = child.stat().st_size if kind == "file" else None
            except OSError:
                return
            entries.append(
                {
                    "name": child.name,
                    "path": _rel(root, child) or ".",
                    "absolute_path": str(child),
                    "kind": kind,
                    "depth": depth,
                    "size": size,
                }
            )

        def walk(current: Path, depth: int) -> bool:
            nonlocal truncated
            try:
                children = sorted(current.iterdir(), key=lambda c: (0 if c.is_dir() else 1, c.name.casefold()))
            except OSError:
                return True
            for child in children:
                if len(entries) >= LIST_FILES_MAX_ENTRIES:
                    truncated = True
                    return False
                if child.is_dir() and should_skip_dir(child.name, include_hidden):
                    continue
                if not include_hidden and child.name.startswith("."):
                    continue
                add(child, depth)
                if recursive and child.is_dir() and depth < depth_cap:
                    if not walk(child, depth + 1):
                        return False
            return True

        walk(root, 1)
        unexplored: list[str] = []
        if not recursive:
            unexplored = [e["name"] for e in entries if e.get("kind") == "directory"]
        hint = None
        if unexplored:
            sample = ", ".join(unexplored[:3])
            more = f" and {len(unexplored) - 3} other(s)" if len(unexplored) > 3 else ""
            hint = (
                f"You have a SHALLOW listing only. The contents of {sample}{more} are NOT in this "
                f"response. Do NOT describe what is inside those directories — you have not seen "
                f"them. To actually look, call list_files again with path='<dirname>' OR with recursive=True."
            )
        return {
            "root": str(root),
            "recursive": bool(recursive),
            "max_depth": depth_cap if recursive else 1,
            "include_hidden": bool(include_hidden),
            "entries": entries,
            "count": len(entries),
            "truncated": truncated,
            "subdirectories_unexplored": unexplored,
            "next_step_hint": hint,
            "cwd": str(self.shell.cwd),
        }

    def glob(self, pattern: str | None = None, path: str | None = None) -> dict:
        cleaned = (pattern or "").strip()
        if not cleaned:
            return _missing("pattern", "pattern is required — e.g. '**/*.py' or 'src/**/test_*.ts'.")
        root, err = self._walk_root(path, default_to_work_root=True)
        if err:
            return err
        assert root is not None
        matches: list[dict] = []
        truncated = False
        try:
            iterator = root.glob(cleaned)
        except (ValueError, OSError) as exc:
            return {"error": "invalid_pattern", "pattern": cleaned, "detail": str(exc)}
        for candidate in iterator:
            rel = _rel(root, candidate)
            parts = Path(rel).parts
            if any(part in DISCOVERY_SKIP_DIRS or part.startswith(".") for part in parts):
                continue
            if len(matches) >= GLOB_MAX_RESULTS:
                truncated = True
                break
            kind = "directory" if candidate.is_dir() else "file"
            try:
                size = candidate.stat().st_size if kind == "file" else None
            except OSError:
                continue
            matches.append(
                {
                    "path": rel,
                    "absolute_path": str(candidate),
                    "kind": kind,
                    "size": size,
                }
            )
        matches.sort(key=lambda item: item["path"])
        hint = None
        if truncated:
            hint = (
                f"RESULTS WERE TRUNCATED at the {GLOB_MAX_RESULTS}-match cap. "
                f"The {len(matches)} matches above are NOT all matches. "
                f"Do NOT claim to have listed every match. "
                f"Call glob again with a more specific pattern or a deeper path."
            )
        return {
            "root": str(root),
            "source_root": str(self.shell.work_root),
            "pattern": cleaned,
            "matches": matches,
            "count": len(matches),
            "truncated": truncated,
            "truncation_hint": hint,
            "cwd": str(self.shell.cwd),
        }

    def grep(
        self,
        pattern: str | None = None,
        path: str | None = None,
        file_glob: str | None = None,
        regex: bool = False,
        ignore_case: bool = False,
        max_results: int = GREP_MAX_RESULTS,
    ) -> dict:
        cleaned = pattern or ""
        if not cleaned.strip():
            return _missing("pattern", "pattern is required — the text or regex to search for.")
        if not regex:
            cleaned = _deescape_substring_pattern(cleaned)
        root, err = self._walk_root(path, default_to_work_root=True)
        if err:
            return err
        assert root is not None
        try:
            cap = min(int(max_results), 200)
        except (TypeError, ValueError):
            cap = GREP_MAX_RESULTS
        if cap <= 0:
            cap = GREP_MAX_RESULTS
        multiline = "\n" in cleaned
        rg = _run_ripgrep(
            root=root,
            pattern=cleaned,
            file_glob=file_glob,
            regex=bool(regex),
            ignore_case=bool(ignore_case),
            max_results=cap,
            multiline=multiline,
        )
        if rg is not None:
            rg["cwd"] = str(self.shell.cwd)
            rg["source_root"] = str(self.shell.work_root)
            return rg
        return self._grep_python(
            root, cleaned, file_glob, bool(regex), bool(ignore_case), cap, multiline
        )

    def read_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> dict:
        """Read raw lines. ``start_line`` and ``end_line`` are 1-indexed and inclusive."""
        cleaned = (path or "").strip()
        if not cleaned:
            return _missing("path", "path is required — relative to cwd or absolute under the workspace.")
        resolved, err = self.shell.resolve(cleaned, allow_dir=False)
        if err:
            return err
        assert resolved is not None
        if not resolved.exists():
            return {"path": cleaned, "error": f"File not found: {cleaned}"}
        if resolved.is_dir():
            return {"path": cleaned, "error": f"Path is a directory, not a file: {cleaned}"}
        if resolved.suffix.lower() in _BINARY_EXT or _looks_binary(resolved):
            try:
                size_bytes = resolved.stat().st_size
            except OSError:
                size_bytes = None
            return {
                "path": cleaned,
                "kind": "binary",
                "size_bytes": size_bytes,
                "error": "binary_file",
                "detail": "This is a binary file, not text. read_file returns raw text lines only.",
                "cwd": str(self.shell.cwd),
            }
        try:
            start = int(start_line)
        except (TypeError, ValueError):
            return {"path": cleaned, "error": "start_line and end_line must be integers."}
        if start < 1:
            start = 1
        if end_line is None:
            end = start + DEFAULT_PAGE_LINES - 1
        else:
            try:
                end = int(end_line)
            except (TypeError, ValueError):
                return {"path": cleaned, "error": "start_line and end_line must be integers."}
        if end < start:
            return {"path": cleaned, "error": "end_line must be greater than or equal to start_line."}
        capped_end = min(end, start + MAX_PAGE_LINES - 1)
        try:
            text = resolved.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return {"path": cleaned, "error": f"Failed to read: {exc}"}
        lines = text.splitlines()
        total = len(lines)
        selected = lines[start - 1 : capped_end]
        content_lines: list[str] = []
        used = 0
        last_line = start - 1
        char_truncated = False
        for offset, line in enumerate(selected):
            piece = line + "\n"
            if content_lines and used + len(piece) > MAX_PAGE_CHARS:
                char_truncated = True
                break
            content_lines.append(piece)
            used += len(piece)
            last_line = start + offset
        content = "".join(content_lines)
        shown_end = last_line if content_lines else start - 1
        range_truncated = capped_end < end or shown_end < min(end, total)
        truncated = range_truncated or char_truncated or (shown_end < total and end >= total and char_truncated)
        # Truncated when the caller did not receive every requested line that exists.
        requested_end = min(end, total)
        truncated = bool(content_lines) and shown_end < requested_end or (not content_lines and total >= start)
        if total < start:
            truncated = False
            content = ""
            shown_end = 0
        next_start = shown_end + 1 if truncated and shown_end < total else None
        payload = {
            "path": str(resolved),
            "content": content,
            "start_line": start if content_lines else start,
            "end_line": shown_end if content_lines else 0,
            "total_lines": total,
            "shown_lines": f"{start}-{shown_end}" if content_lines else "0-0",
            "truncated": bool(truncated and next_start),
            "cwd": str(self.shell.cwd),
        }
        if next_start:
            payload["next_start_line"] = next_start
            payload["detail"] = (
                f"Page stopped at line {shown_end}. Call read_file again with "
                f"start_line={next_start} to continue this file."
            )
        return payload

    def os_bash(self, command: str | None = None) -> dict:
        cleaned = (command or "").strip()
        if not cleaned:
            return _missing("command", "command is required — pass a non-empty shell command string.")
        return self.shell.bash(cleaned)

    def web_search(self, query: str, count: int = 5) -> dict:
        from playground.harness.web import web_search

        return web_search(query, count)

    def fetch_url(self, url: str, max_chars: int = 20000) -> dict:
        from playground.harness.web import fetch_url

        return fetch_url(url, max_chars)

    def _walk_root(self, path: str | None, *, default_to_work_root: bool) -> tuple[Path | None, dict | None]:
        if default_to_work_root and not (path or "").strip():
            cleaned = str(self.shell.work_root)
        else:
            cleaned = (path or "").strip() or "."
        resolved, err = self.shell.resolve(cleaned, allow_dir=True)
        if err:
            return None, err
        assert resolved is not None
        if not resolved.exists():
            return None, {
                "error": "not_found",
                "path": cleaned,
                "cwd": str(self.shell.cwd),
                "detail": f"Path does not exist: {resolved}",
            }
        if not resolved.is_dir():
            return None, {
                "error": "not_a_directory",
                "path": cleaned,
                "cwd": str(self.shell.cwd),
                "detail": f"Not a directory: {resolved}",
            }
        return resolved, None

    def _grep_python(
        self,
        root: Path,
        pattern: str,
        file_glob: str | None,
        regex: bool,
        ignore_case: bool,
        max_results: int,
        multiline: bool,
    ) -> dict:
        flags = re.IGNORECASE if ignore_case else 0
        matcher = None
        if regex:
            try:
                matcher = re.compile(pattern, flags | (re.MULTILINE if multiline else 0))
            except re.error as exc:
                return {"error": "invalid_regex", "pattern": pattern, "detail": str(exc)}
        needle = pattern.lower() if ignore_case else pattern
        globs = _expand_brace_glob(file_glob) if file_glob else []
        matches: list[dict] = []
        files_scanned = 0
        truncated = False

        def accept(rel: str, line_no: int, text: str) -> bool:
            preview = text.rstrip("\n")
            if len(preview) > GREP_LINE_PREVIEW_LIMIT:
                preview = preview[:GREP_LINE_PREVIEW_LIMIT] + "…"
            matches.append(
                {
                    "path": rel,
                    "absolute_path": str(root / rel),
                    "line": line_no,
                    "text": preview,
                }
            )
            return len(matches) < max_results

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not should_skip_dir(d, False)]
            for name in filenames:
                if name.startswith("."):
                    continue
                if globs and not any(fnmatch.fnmatch(name, g) for g in globs):
                    continue
                file_path = Path(dirpath) / name
                try:
                    if file_path.stat().st_size > GREP_MAX_FILE_BYTES:
                        continue
                except OSError:
                    continue
                files_scanned += 1
                rel = _rel(root, file_path)
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                if regex and matcher is not None:
                    for found in matcher.finditer(content):
                        line_no = content.count("\n", 0, found.start()) + 1
                        if not accept(rel, line_no, content.splitlines()[line_no - 1] if content.splitlines() else ""):
                            truncated = True
                            break
                else:
                    for line_no, line in enumerate(content.splitlines(), 1):
                        hay = line.lower() if ignore_case else line
                        if needle in hay:
                            if not accept(rel, line_no, line):
                                truncated = True
                                break
                if truncated or len(matches) >= max_results:
                    truncated = len(matches) >= max_results
                    break
            if truncated:
                break
        hint = None
        if truncated:
            hint = (
                f"RESULTS WERE TRUNCATED at the {max_results}-match cap. "
                f"There are MORE matches in the codebase that you have not seen. "
                f"Do NOT report a total count from this response alone. "
                f"If aggregating across multiple grep calls, DEDUPE by (path, line) before counting."
            )
        return {
            "root": str(root),
            "pattern": pattern,
            "regex": regex,
            "ignore_case": ignore_case,
            "file_glob": file_glob,
            "matches": matches,
            "count": len(matches),
            "files_scanned": files_scanned,
            "truncated": truncated,
            "truncation_hint": hint,
            "engine": "python",
            "cwd": str(self.shell.cwd),
            "source_root": str(self.shell.work_root),
        }


def _looks_binary(path: Path) -> bool:
    try:
        chunk = path.read_bytes()[:1024]
    except OSError:
        return False
    return b"\0" in chunk


def _expand_brace_glob(pattern: str) -> list[str]:
    match = re.match(r"^(.*)\{([^{}]+)\}(.*)$", pattern)
    if not match:
        return [pattern]
    head, body, tail = match.groups()
    return [head + part + tail for part in body.split(",")]


def _run_ripgrep(
    *,
    root: Path,
    pattern: str,
    file_glob: str | None,
    regex: bool,
    ignore_case: bool,
    max_results: int,
    multiline: bool,
) -> dict | None:
    rg = shutil.which("rg")
    if rg is None:
        return None
    cmd = [
        rg,
        "--no-heading",
        "--line-number",
        "--with-filename",
        "--no-config",
        "--max-count",
        str(max_results),
        "--max-columns",
        str(GREP_LINE_PREVIEW_LIMIT),
        "--color",
        "never",
    ]
    if ignore_case:
        cmd.append("--ignore-case")
    if not regex:
        cmd.append("--fixed-strings")
    if multiline:
        cmd.append("--multiline")
    if file_glob:
        cmd.extend(["--glob", file_glob])
    for skip in DISCOVERY_SKIP_DIRS:
        cmd.extend(["--glob", f"!**/{skip}/**"])
    cmd.extend(["--", pattern, str(root)])
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    except (subprocess.SubprocessError, OSError):
        return None
    if proc.returncode not in (0, 1):
        return None
    matches: list[dict] = []
    truncated = False
    for line in proc.stdout.splitlines():
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
        path_str, line_no_str, text = parts
        try:
            line_no = int(line_no_str)
        except ValueError:
            continue
        try:
            rel = Path(path_str).resolve().relative_to(root).as_posix()
        except ValueError:
            rel = path_str
        matches.append(
            {
                "path": rel,
                "absolute_path": path_str,
                "line": line_no,
                "text": text,
            }
        )
        if len(matches) >= max_results:
            truncated = True
            break
    hint = None
    if truncated:
        hint = (
            f"RESULTS WERE TRUNCATED at the {max_results}-match cap. "
            f"There are MORE matches in the codebase that you have not seen. "
            f"Do NOT report a total count from this response alone. "
            f"If aggregating across multiple grep calls, DEDUPE by (path, line) before counting. "
            f"To get the rest, call grep again with a narrower path, a more specific file_glob, "
            f"or a more distinctive pattern."
        )
    return {
        "root": str(root),
        "pattern": pattern,
        "regex": bool(regex),
        "ignore_case": bool(ignore_case),
        "file_glob": file_glob,
        "matches": matches,
        "count": len(matches),
        "files_scanned": None,
        "truncated": truncated,
        "truncation_hint": hint,
        "engine": "ripgrep",
    }


def _filter_args(fn, arguments: dict) -> dict:
    import inspect

    params = inspect.signature(fn).parameters
    cleaned = {}
    aliases = {
        "startLine": "start_line",
        "endLine": "end_line",
        "fileGlob": "file_glob",
        "ignoreCase": "ignore_case",
        "maxResults": "max_results",
        "maxDepth": "max_depth",
        "includeHidden": "include_hidden",
        "maxChars": "max_chars",
    }
    for key, value in arguments.items():
        name = aliases.get(key, key)
        if name in params:
            cleaned[name] = value
    # Models trained on the UCE pager sometimes send offset/limit. Map them
    # only when the paged names were omitted, so the schema the model is
    # shown stays start_line/end_line.
    if "start_line" not in cleaned and "offset" in arguments and "start_line" in params:
        try:
            cleaned["start_line"] = int(arguments["offset"]) + 1
        except (TypeError, ValueError):
            pass
    if "end_line" not in cleaned and "limit" in arguments and "start_line" in cleaned and "end_line" in params:
        try:
            cleaned["end_line"] = int(cleaned["start_line"]) + int(arguments["limit"]) - 1
        except (TypeError, ValueError):
            pass
    return cleaned


_TOOL_SCHEMAS: dict[str, dict] = {
    "match_path": {
        "description": (
            "Resolve natural language to absolute paths under the commissioned workspace. "
            "Use to turn vague folder names into paths. This does not search file contents."
        ),
        "properties": {
            "query": {"type": "string", "description": "Spoken or partial path hint (folder name, alias, or project name)."},
            "limit": {"type": "integer", "description": "Max ranked matches (1-20)."},
            "kind": {"type": "string", "description": "directory (default), file, or any.", "enum": ["directory", "file", "any"]},
        },
        "required": ["query"],
    },
    "get_cwd": {
        "description": "Return the working directory. File tools resolve relative paths against it.",
        "properties": {},
        "required": [],
    },
    "change_directory": {
        "description": "Change the working directory inside the workspace. Prefer this over bare cd in os_bash.",
        "properties": {
            "path": {"type": "string", "description": "Absolute or path relative to current cwd, inside the workspace."},
        },
        "required": ["path"],
    },
    "list_files": {
        "description": (
            "List files and subdirectories at a path. Default mode is non-recursive (like ls) and returns "
            "ONLY the immediate children. The response includes subdirectories_unexplored — every name in "
            "there is a directory whose contents you have not seen. Skips node_modules, .git, .venv, build, and similar."
        ),
        "properties": {
            "path": {"type": "string", "description": "Directory path. Defaults to cwd."},
            "recursive": {"type": "boolean", "description": "Walk into subdirectories. Default false."},
            "max_depth": {"type": "integer", "description": "Max walk depth when recursive. Default 3."},
            "include_hidden": {"type": "boolean", "description": "Include dotfiles. Default false."},
        },
        "required": [],
    },
    "glob": {
        "description": (
            "Find files matching a glob pattern, like **/*.py. Skips heavy dirs and hidden files. "
            "Omit path to glob the whole workspace."
        ),
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern. Required."},
            "path": {"type": "string", "description": "Root directory. Omit to glob the whole workspace."},
        },
        "required": ["pattern"],
    },
    "grep": {
        "description": (
            "Search file contents for a pattern. Exact substring by default; set regex=true for a regular "
            "expression. Omit path to search the whole workspace. Caps at max_results."
        ),
        "properties": {
            "pattern": {"type": "string", "description": "Text or regex to search for. Required."},
            "path": {"type": "string", "description": "Directory to search. Omit to search the whole workspace."},
            "file_glob": {"type": "string", "description": "Glob filter such as *.py or *.{ts,tsx}."},
            "regex": {"type": "boolean", "description": "Treat pattern as a regex. Default false."},
            "ignore_case": {"type": "boolean", "description": "Case-insensitive match. Default false."},
            "max_results": {"type": "integer", "description": "Max matches. Default 200, capped at 200."},
        },
        "required": ["pattern"],
    },
    "read_file": {
        "description": (
            "Read a page of raw file lines. Does not summarize. start_line and end_line are 1-indexed "
            "and inclusive. Omit end_line for a default page of about 500 lines. A wide range is appropriate "
            "when the objective names a long region. If the page is truncated, call again with start_line "
            "set to next_start_line."
        ),
        "properties": {
            "path": {"type": "string", "description": "File path relative to cwd, or absolute under the workspace."},
            "start_line": {"type": "integer", "description": "First line to return. 1-indexed. Default 1."},
            "end_line": {"type": "integer", "description": "Last line to return, inclusive. Omit for the default page."},
        },
        "required": ["path"],
    },
    "os_bash": {
        "description": (
            "Run a read-only shell command in the workspace. Write commands (rm, git commit, python, redirects) "
            "are refused. Prefer typed file tools over bash for listing or reading files."
        ),
        "properties": {
            "command": {"type": "string", "description": "Shell command, e.g. git status or git log -1."},
        },
        "required": ["command"],
    },
    "web_search": {
        "description": "Search the web via Brave Search. Returns ranked results with titles, URLs, and snippets.",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "count": {"type": "integer", "description": "Number of results (1-10, default 5)."},
        },
        "required": ["query"],
    },
    "fetch_url": {
        "description": (
            "Fetch a URL and return its readable text. HTML is stripped to text. "
            "github.com /blob/ URLs are rewritten to the raw file. Binary content is rejected."
        ),
        "properties": {
            "url": {"type": "string", "description": "HTTP or HTTPS URL."},
            "max_chars": {"type": "integer", "description": "Truncate content to this many characters. Default 20000."},
        },
        "required": ["url"],
    },
}


def openai_tools() -> list[dict]:
    tools = []
    for name, spec in _TOOL_SCHEMAS.items():
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": spec["description"],
                    "parameters": {
                        "type": "object",
                        "properties": spec["properties"],
                        "required": spec["required"],
                        "additionalProperties": False,
                    },
                },
            }
        )
    return tools


TOOL_NAMES = frozenset(_TOOL_SCHEMAS)
