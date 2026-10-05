"""SubSWE: a commissioned-reader benchmark on public repos.

The published number is a deterministic pass rate from a rescore of the
trace files. A judge grade does not move that number.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

from playground.benchmark import check_task, gold_fields
from playground.cloud import CloudError, MidiumCloud, OpenRouter
from playground.config import (
    FLASH_TEACHER,
    GRADER_MODEL,
    ROOT,
    STUDENT_MODEL,
    SUBSWE_DIR,
    SUBSWE_REPORT_PATH,
    SUBSWE_MAX_ITERS,
    SUBSWE_TASKS_PATH,
    Settings,
)
from playground.harness.loop import run_reader
from playground.run import teacher_slug
from playground.traces import append_jsonl, read_jsonl

# Display name, route, concurrency. Local is one at a time. Cloud stays at 2.
PUBLISHED_MODELS: tuple[tuple[str, str, int], ...] = (
    (STUDENT_MODEL, "local", 1),
    (FLASH_TEACHER, "openrouter", 8),
    ("Laguna XS 2.1", "cloud", 2),
)
LORA_MODEL = "Gemma 4 E4B LoRA"
F16_MODEL = "Gemma 4 E4B F16"
READER_4BIT_MODEL = "Gemma 4 E4B Reader 4bit"


def load_subswe_tasks() -> list[dict]:
    raw = json.loads(SUBSWE_TASKS_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) != 40:
        raise SystemExit(f"SubSWE expected 40 tasks in {SUBSWE_TASKS_PATH}")
    return raw


def trace_path(model: str, run: int) -> Path:
    return SUBSWE_DIR / f"{teacher_slug(model)}.run{run}.jsonl"


def model_spec(model: str) -> tuple[str, int]:
    if model == READER_4BIT_MODEL:
        return "fused", 1
    if model in {LORA_MODEL, F16_MODEL}:
        return "lora", 1
    for name, route, concurrency in PUBLISHED_MODELS:
        if name == model:
            return route, concurrency
    known = ", ".join([*(name for name, _, _ in PUBLISHED_MODELS), F16_MODEL, LORA_MODEL, READER_4BIT_MODEL])
    raise SystemExit(f"Unknown SubSWE model {model!r}. Choose one of: {known}")


async def run_subswe(
    cfg: Settings,
    model: str,
    run: int,
    *,
    grade: bool,
    adapter: Path | None = None,
    model_path: Path | None = None,
) -> Path:
    route, concurrency = model_spec(model)
    if run not in {1, 2, 3}:
        raise SystemExit("SubSWE run must be 1, 2, or 3.")
    tasks = load_subswe_tasks()
    dest = trace_path(model, run)
    done = {row.get("id") for row in read_jsonl(dest)}
    pending = [task for task in tasks if task["id"] not in done]
    if grade:
        cfg.require_midium()
    student = _student(cfg, model, route, adapter, model_path=model_path)
    grader = MidiumCloud(cfg, GRADER_MODEL, timeout=120.0) if grade else None
    sem = asyncio.Semaphore(concurrency)

    async def one(index: int, task: dict) -> None:
        async with sem:
            print(f"  {index}/{len(tasks)} {task['id']}", flush=True)
            row = await _episode(student, grader, model, run, task)
            append_jsonl(dest, row)

    try:
        await asyncio.gather(*(one(tasks.index(task) + 1, task) for task in pending))
    finally:
        await student.aclose()
        if grader is not None:
            await grader.aclose()
    print(f"Wrote {dest}", flush=True)
    return dest


def score_model(model: str, run: int = 1) -> dict:
    """Rescore one trace file. Does not require the other published models."""
    path = trace_path(model, run)
    if not path.is_file():
        raise SystemExit(f"No trace file for {model} run {run}: {path}")
    tasks = {task["id"]: task for task in load_subswe_tasks()}
    scored = _rescore_file(path, tasks, model, run)
    _print_report({"runs": {path.name: scored}, "published": {}})
    return scored


def fetch_subswe_repos(tasks: list[dict] | None = None, *, dest_root: Path = ROOT, runner=None) -> None:
    """Shallow-clone each pinned SubSWE repo. data/repos stays gitignored."""
    run = runner or subprocess.run
    seen: dict[str, tuple[str, str]] = {}
    for task in tasks or load_subswe_tasks():
        seen[task["repo"]] = (task["commit"], task["path"])
    for repo, (commit, rel) in seen.items():
        dest = dest_root / rel
        if not (dest / ".git").is_dir():
            dest.parent.mkdir(parents=True, exist_ok=True)
            _git(
                run,
                ["git", "clone", "--filter=blob:none", "--no-checkout", f"https://github.com/{repo}.git", str(dest)],
            )
        _git(run, ["git", "-C", str(dest), "fetch", "--depth", "1", "origin", commit])
        _git(run, ["git", "-C", str(dest), "checkout", "--detach", commit])
        print(f"  {repo} @ {commit[:12]}", flush=True)


def _git(run, cmd: list[str]) -> None:
    proc = run(cmd, check=False)
    code = getattr(proc, "returncode", 0)
    if code not in (0, None):
        raise SystemExit(f"Command failed ({code}): {' '.join(cmd)}")


def write_report() -> dict:
    """Rescore every saved trace. Refuse a published rate if a model is missing."""
    tasks = {task["id"]: task for task in load_subswe_tasks()}
    runs: dict[str, dict] = {}
    missing_models = []
    for model, _, _ in PUBLISHED_MODELS:
        files = [trace_path(model, run) for run in (1, 2) if trace_path(model, run).is_file()]
        if not files:
            missing_models.append(model)
        for path in files:
            run_no = 1 if path.name.endswith(".run1.jsonl") else 2
            runs[f"{teacher_slug(model)}.run{run_no}"] = _rescore_file(path, tasks, model, run_no)
    published = {}
    for run_no in (1, 2):
        block = {}
        absent = []
        for model, _, _ in PUBLISHED_MODELS:
            key = f"{teacher_slug(model)}.run{run_no}"
            if key not in runs:
                absent.append(model)
            else:
                block[model] = runs[key]
        if absent:
            published[f"run{run_no}"] = {"published": False, "missing": absent}
        else:
            published[f"run{run_no}"] = {"published": True, "models": block}
    payload = {"runs": runs, "published": published}
    if missing_models:
        payload["refused"] = missing_models
    SUBSWE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUBSWE_REPORT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _print_report(payload)
    if missing_models:
        names = ", ".join(missing_models)
        raise SystemExit(f"Refusing a published rate. Missing traces for: {names}")
    return payload


def _print_report(payload: dict) -> None:
    for key, row in payload["runs"].items():
        print(
            f"{row['model']} run {row['run']}: pass_rate={row['pass_rate']} "
            f"calls={row['call_count']} errors={row['error_count']}",
            flush=True,
        )
        for kind, stats in row["by_kind"].items():
            print(f"  {kind} {stats['passed']}/{stats['attempted']}", flush=True)
        for item in row["tasks"]:
            if item.get("error"):
                print(f"  error {item['id']}: {item['error']}", flush=True)
                continue
            flags = item.get("deterministic") or {}
            if flags.get("invalid"):
                continue
            if flags.get("passed"):
                continue
            failed = [name for name, value in flags.items() if value is False and name not in {"passed", "invalid"}]
            print(f"  fail {item['id']}: {', '.join(failed)}", flush=True)
    for label, block in payload["published"].items():
        if not block.get("published"):
            missing = ", ".join(block.get("missing") or [])
            print(f"{label}: not published, missing {missing}", flush=True)
    print(f"Wrote {SUBSWE_REPORT_PATH}", flush=True)


def _rescore_file(path: Path, tasks: dict[str, dict], model: str, run: int) -> dict:
    rows = []
    calls = 0
    errors = 0
    for saved in read_jsonl(path):
        task = tasks.get(saved.get("id") or "")
        if task is None:
            continue
        if saved.get("error"):
            errors += 1
            rows.append(
                {
                    "id": task["id"],
                    "kind": task.get("kind"),
                    "error": saved.get("error"),
                    "deterministic": {"passed": False, "invalid": True},
                    "semantic": saved.get("semantic"),
                }
            )
            continue
        work = Path(task["path"])
        if not work.is_absolute():
            work = ROOT / work
        flags = check_task(task, saved, work)
        calls += len(saved.get("tool_rounds") or [])
        semantic = saved.get("semantic")
        if semantic and semantic.get("grade") is None and semantic.get("reason"):
            semantic = {**semantic, "judge_error": True}
        rows.append(
            {
                "id": task["id"],
                "kind": task.get("kind"),
                "error": None,
                "deterministic": flags,
                "semantic": semantic,
            }
        )
    attempted = [row for row in rows if not (row.get("deterministic") or {}).get("invalid")]
    passed = [row for row in attempted if (row.get("deterministic") or {}).get("passed")]
    by_kind: dict[str, dict] = {}
    for row in attempted:
        bucket = by_kind.setdefault(row["kind"], {"passed": 0, "attempted": 0})
        bucket["attempted"] += 1
        if (row.get("deterministic") or {}).get("passed"):
            bucket["passed"] += 1
    return {
        "model": model,
        "run": run,
        "path": str(path),
        "pass_rate": round(len(passed) / len(attempted), 4) if attempted else None,
        "attempted": len(attempted),
        "passed": len(passed),
        "error_count": errors,
        "call_count": calls,
        "by_kind": by_kind,
        "tasks": rows,
    }


async def _episode(student, grader, model: str, run: int, task: dict) -> dict:
    work = ROOT / task["path"]
    try:
        result = await run_reader(
            work_root=work,
            objective=task["objective"],
            complete=student.complete,
            max_iters=SUBSWE_MAX_ITERS,
        )
    except (CloudError, OSError) as exc:
        return {
            "id": task["id"],
            "model": model,
            "run": run,
            "kind": task.get("kind"),
            "repo": task.get("repo"),
            "commit": task.get("commit"),
            "objective": task.get("objective"),
            "report": "",
            "stop_reason": "",
            "tool_rounds": [],
            "error": str(exc),
            "semantic": None,
        }
    semantic = None
    if grader is not None:
        semantic = await _grade(grader, task, result.get("report") or "")
    return {
        "id": task["id"],
        "model": model,
        "run": run,
        "kind": task.get("kind"),
        "repo": task.get("repo"),
        "commit": task.get("commit"),
        "objective": task.get("objective"),
        "report": result.get("report") or "",
        "stop_reason": result.get("stop_reason") or "",
        "tool_rounds": result.get("tool_rounds") or [],
        "error": None,
        "semantic": semantic,
    }


async def _grade(grader: MidiumCloud, task: dict, report: str) -> dict:
    from playground.benchmark import _GRADER_SYSTEM
    from playground.cloud import parse_json_object

    user = (
        f"Objective:\n{task.get('objective')}\n\n"
        f"Gold note:\n{gold_fields(task)['note']}\n\n"
        f"Report:\n{report or '(empty)'}"
    )
    try:
        message = await grader.complete(
            [
                {"role": "system", "content": _GRADER_SYSTEM},
                {"role": "user", "content": user},
            ],
            None,
        )
    except CloudError as exc:
        return {"grade": None, "reason": str(exc), "judge_error": True}
    try:
        parsed = parse_json_object(message.get("content") or "")
    except CloudError as exc:
        return {"grade": None, "reason": str(exc), "judge_error": True}
    grade = str(parsed.get("grade") or "").strip().lower()
    if grade not in {"good", "bad"}:
        return {"grade": None, "reason": str(parsed.get("reason") or ""), "judge_error": True}
    return {"grade": grade, "reason": str(parsed.get("reason") or "").strip()}


def _student(
    cfg: Settings,
    model: str,
    route: str,
    adapter: Path | None = None,
    model_path: Path | None = None,
):
    if route == "fused":
        from playground.export_reader import FOUR_BIT_DIR
        from playground.lora_reader import LoraReader

        path = model_path or FOUR_BIT_DIR
        if not path.is_dir():
            raise SystemExit(f"4-bit reader not found: {path}")
        return LoraReader(None, model_path=str(path))
    if route == "lora":
        from playground.lora_reader import LoraReader
        from playground.train import TOOLS_CHECKPOINT_DIR

        if model == F16_MODEL:
            return LoraReader(None)
        return LoraReader(adapter or TOOLS_CHECKPOINT_DIR)
    if route == "local":
        cfg.require_local()
        return MidiumCloud(cfg, model, api_key=cfg.local_api_key, base_url=cfg.local_base_url)
    if route == "openrouter":
        cfg.require_openrouter()
        return OpenRouter(cfg, timeout=180.0)
    cfg.require_midium()
    return MidiumCloud(cfg, model)
