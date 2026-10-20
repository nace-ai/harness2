"""A/B driver: does a Drex document helper raise JobBench scores for OpenCode?

Arm `base` is the stock JobBench OpenCode harness. Arm `drex` is the same tree plus
scripts/drex_read.py. The driver calls the adapter's execute() and score() directly.

Run from the harness2 checkout (so ROOT, .env and dependencies/ resolve):
    python experiments/jb_drex/ab.py --out runs/jb-ab --tasks <profession>/task<N> ... \
        --repeats 3 --concurrency 4 [--arms base drex] [--dry-run] [--judge-only]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

ARMS = ("base", "drex")
JUDGE_MODEL = "claude-sonnet-5-5"
JUDGE_API_BASE = "https://api.anthropic.com/v1"
HELPER = Path(__file__).with_name("drex_read.py")
HELPER_CALL = re.compile(r"python3?\s+\S*drex_read\.py")
JUDGE_SHIM = """\
import functools
import sys

if sys.argv and sys.argv[0].endswith("judge.py"):
    from openai.resources.chat.completions import Completions

    _create = Completions.create

    @functools.wraps(_create)
    def create(self, *args, **kwargs):
        kwargs.pop("temperature", None)
        return _create(self, *args, **kwargs)

    Completions.create = create
"""


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out", default="runs/jb-ab", help="run store, harness_drex/ and results")
    p.add_argument("--tasks", nargs="+", required=True, help="task ids, or `all`")
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--dry-run", action="store_true", help="print jobs and the helper line only")
    p.add_argument("--judge-only", action="store_true", help="score existing run dirs only")
    return p


# harness2.config reads HARNESS2_RUNS_DIR once, at import, so it is set before the imports below.
_ARGS = _arg_parser().parse_args()
OUT = Path(_ARGS.out).expanduser().resolve()
os.environ["HARNESS2_RUNS_DIR"] = str(OUT)

from harness2 import config  # noqa: E402
from harness2.benchmarks.jobbench import JobBenchAdapter, compose_agent_body  # noqa: E402
from harness2.cli._common import make_adapter  # noqa: E402
from harness2.core import runmark as RM  # noqa: E402
from harness2.core import tree as HT  # noqa: E402


@dataclass(frozen=True)
class Job:
    task: str
    arm: str
    rep: int

    @property
    def run_id(self) -> str:
        return f"ab-{self.arm}-r{self.rep}"

    @property
    def key(self) -> str:
        return f"{self.task}|{self.arm}|{self.rep}"


def wire_judge() -> None:
    """Point JobBench's judge at Anthropic's OpenAI-compatible endpoint.

    jobbench._run_judge reads config.JB_JUDGE_* when it builds each judge command, so
    patching the module constants after import is what reaches the judge subprocess.
    """
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        sys.exit("ANTHROPIC_API_KEY is not set (expected in .env)")
    os.environ["HARNESS2_JUDGE_MODEL"] = JUDGE_MODEL
    os.environ["HARNESS2_JUDGE_BASE_URL"] = JUDGE_API_BASE
    config.JUDGE_MODEL = config.JB_JUDGE_MODEL = JUDGE_MODEL
    config.JB_JUDGE_API_BASE = JUDGE_API_BASE
    config.JB_JUDGE_API_KEY = key


def install_judge_shim() -> Path:
    """Make judge.py omit `temperature`, which claude-sonnet-5-5 rejects with a 400.

    judge.py hardcodes temperature=0.0. A sitecustomize on the judge subprocess's
    PYTHONPATH strips it from the one call, leaving the JobBench checkout untouched.
    """
    shim = OUT / "judge_shim"
    shim.mkdir(parents=True, exist_ok=True)
    (shim / "sitecustomize.py").write_text(JUDGE_SHIM, encoding="utf-8")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(shim), os.environ.get("PYTHONPATH")) if p
    )
    return shim


def build_drex_harness(adapter: JobBenchAdapter, dest: Path) -> Path:
    """The base harness plus scripts/drex_read.py; rebuilt only when the content differs."""
    scratch = Path(tempfile.mkdtemp(prefix="harness_drex_", dir=dest.parent))
    try:
        shutil.copytree(adapter.base_harness(), scratch, dirs_exist_ok=True)
        shutil.copy(HELPER, scratch / "scripts" / HELPER.name)
        if dest.is_dir() and RM.fingerprint(dest) == RM.fingerprint(scratch):
            return dest
        shutil.rmtree(dest, ignore_errors=True)
        scratch.rename(dest)
        return dest
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def validate_harness(adapter: JobBenchAdapter, harness: Path) -> dict:
    surface = adapter.substrate.surface()
    tree = HT.read_tree(surface, harness)
    census = HT.validate(surface, tree, allow_subagents=adapter.spec.allow_subagents)
    adapter.validate_tree(tree)
    return census


def rendered_helper_line(adapter: JobBenchAdapter, harness: Path) -> str:
    """The AGENTS.md line the agent will see for drex_read.py, from the real materialise path."""
    install = adapter.substrate.materialise(harness, adapter.spec, {})
    try:
        body = compose_agent_body(install.staging_dir, {})
    finally:
        shutil.rmtree(install.staging_dir, ignore_errors=True)
    lines = [ln for ln in body.splitlines() if HELPER.name in ln]
    if len(lines) != 1:
        raise RuntimeError(f"expected one {HELPER.name} line in the rendered AGENTS.md: {lines}")
    return lines[0]


def run_dir_for(adapter: JobBenchAdapter, job: Job) -> Path:
    return adapter.store / job.task / "model_output" / job.run_id.replace("/", "__")


def traj_dir_for(adapter: JobBenchAdapter, job: Job) -> Path:
    return adapter.store / job.task / "model_traj" / job.run_id.replace("/", "__")


def helper_calls(trajectory: str) -> int:
    """Tool actions in the newest attempt that run drex_read.py; reading the file does not count."""
    return sum(
        len(HELPER_CALL.findall(ln)) for ln in trajectory.splitlines() if ln.startswith("[ACTION]")
    )


def usage_from_events(traj_dir: Path) -> dict:
    """Cost, tokens and agent wall time summed over every attempt's OpenCode event stream."""
    cost, wall_ms, attempts = 0.0, 0, 0
    tokens = {"input": 0, "output": 0, "reasoning": 0, "cache_read": 0, "cache_write": 0}
    for f in sorted(traj_dir.glob("*.jsonl")):
        stamps: list[int] = []
        for ln in f.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                ev = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if not isinstance(ev, dict):
                continue
            if isinstance(ev.get("timestamp"), int):
                stamps.append(ev["timestamp"])
            part = ev.get("part") if isinstance(ev.get("part"), dict) else {}
            if part.get("type") != "step-finish":
                continue
            cost += float(part.get("cost") or 0)
            t = part.get("tokens") or {}
            cache = t.get("cache") or {}
            tokens["input"] += int(t.get("input") or 0)
            tokens["output"] += int(t.get("output") or 0)
            tokens["reasoning"] += int(t.get("reasoning") or 0)
            tokens["cache_read"] += int(cache.get("read") or 0)
            tokens["cache_write"] += int(cache.get("write") or 0)
        if stamps:
            attempts += 1
            wall_ms += max(stamps) - min(stamps)
    return {
        "cost_usd": round(cost, 4),
        "tokens": tokens,
        "agent_wall_s": round(wall_ms / 1000, 1),
        "attempts": attempts,
    }


def judge_failure(adapter: JobBenchAdapter, job: Job) -> str:
    """Why a delivered run has no score: the judge's own quarantined report, first failure."""
    safe = config.JB_JUDGE_MODEL.split("/")[-1].replace(".", "-")
    label = job.run_id.replace("/", "__")
    path = adapter.store / job.task / "eval_result" / f"eval_{label}" / f"{safe}_judge.json.judge_failed"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        for rubric in report.get("rubrics") or []:
            for c in (rubric.get("result") or {}).get("criteria_results") or []:
                if "Failed" in str(c.get("reasoning")):
                    return str(c["reasoning"])[:400]
    except (OSError, json.JSONDecodeError):
        pass
    return "judge produced no usable report; see the runner log"


def pass_fraction(score: dict | None) -> float | None:
    if not score or not score.get("n_criteria"):
        return None
    return score["n_passed"] / score["n_criteria"]


def run_job(
    adapter: JobBenchAdapter, job: Job, harnesses: dict[str, Path | None], judge_only: bool
) -> dict:
    row: dict = {
        "task": job.task, "arm": job.arm, "rep": job.rep, "run_id": job.run_id,
        "delivered": None, "score": None, "pass_frac": None, "drex_calls": None,
        "cost_usd": None, "tokens": None, "agent_wall_s": None, "attempts": None, "error": None,
        "judge_error": None,
    }
    try:
        if judge_only:
            run_dir = run_dir_for(adapter, job)
            if not run_dir.is_dir():
                raise FileNotFoundError(f"no run dir to judge: {run_dir}")
        else:
            run_dir = adapter.execute(job.task, harnesses[job.arm], job.run_id)
        row["delivered"] = adapter.delivered(run_dir)
        row["score"] = adapter.score(run_dir, job.task)
        row["pass_frac"] = pass_fraction(row["score"])
        if row["score"] is None:
            row["judge_error"] = judge_failure(adapter, job)
        trajectory = adapter.trajectory(run_dir, obs_trunc=adapter.spec.obs_trunc)
        row["drex_calls"] = helper_calls(trajectory)
        row.update(usage_from_events(traj_dir_for(adapter, job)))
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


class ResultStore:
    """ab_results.jsonl keyed by (task, arm, rep); a rerun replaces its row."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.rows: dict[str, dict] = {}
        if path.is_file():
            for ln in path.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    r = json.loads(ln)
                    self.rows[f"{r['task']}|{r['arm']}|{r['rep']}"] = r

    def put(self, job: Job, row: dict) -> None:
        with self.lock:
            self.rows[job.key] = row
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text("".join(json.dumps(r) + "\n" for r in self.rows.values()), encoding="utf-8")
            os.replace(tmp, self.path)


def _mean(xs: list[float]) -> float | None:
    return fmean(xs) if xs else None


def summarise(rows: list[dict], tasks: list[str], arms: list[str]) -> dict:
    per_task: dict[str, dict[str, dict]] = {}
    for t in tasks:
        per_task[t] = {}
        for a in arms:
            rs = [r for r in rows if r["task"] == t and r["arm"] == a]
            scored = [r["pass_frac"] for r in rs if r["pass_frac"] is not None]
            calls = [r["drex_calls"] for r in rs if r["drex_calls"] is not None]
            per_task[t][a] = {
                "n_jobs": len(rs),
                "n_scored": len(scored),
                "mean_pass_frac": _mean(scored),
                "delivered": sum(1 for r in rs if r["delivered"]),
                "errors": sum(1 for r in rs if r["error"]),
                "mean_drex_calls": _mean(calls),
                "cost_usd": round(sum(r["cost_usd"] or 0 for r in rs), 4),
            }
    overall = {}
    for a in arms:
        means = [per_task[t][a]["mean_pass_frac"] for t in tasks
                 if per_task[t][a]["mean_pass_frac"] is not None]
        pooled = [r["pass_frac"] for r in rows if r["arm"] == a and r["pass_frac"] is not None]
        overall[a] = {"mean_of_task_means": _mean(means), "n_tasks": len(means),
                      "pooled_mean": _mean(pooled), "n_scored": len(pooled)}
    paired = None
    if set(arms) == set(ARMS):
        diffs = {
            t: per_task[t]["drex"]["mean_pass_frac"] - per_task[t]["base"]["mean_pass_frac"]
            for t in tasks
            if per_task[t]["drex"]["mean_pass_frac"] is not None
            and per_task[t]["base"]["mean_pass_frac"] is not None
        }
        paired = {"per_task": diffs, "mean_diff": _mean(list(diffs.values())), "n_tasks": len(diffs)}
    return {"headline": "pass_frac = n_passed / n_criteria (weighted rubric score)",
            "per_task": per_task, "overall": overall, "paired_drex_minus_base": paired}


def _fmt(x: float | None, spec: str = ".3f") -> str:
    return "-" if x is None else format(x, spec)


def print_summary(s: dict, arms: list[str]) -> None:
    print("\n== per task (mean pass_frac over scored repeats [n scored/n jobs], delivered, drex calls) ==")
    for t, by_arm in s["per_task"].items():
        cells = []
        for a in arms:
            c = by_arm[a]
            cells.append(
                f"{a}: {_fmt(c['mean_pass_frac'])} [{c['n_scored']}/{c['n_jobs']}] "
                f"deliv {c['delivered']} err {c['errors']} drex {_fmt(c['mean_drex_calls'], '.1f')}"
            )
        print(f"{t}\n    " + "\n    ".join(cells))
    print("\n== overall ==")
    for a in arms:
        o = s["overall"][a]
        print(f"{a}: task-mean {_fmt(o['mean_of_task_means'])} over {o['n_tasks']} tasks, "
              f"pooled {_fmt(o['pooled_mean'])} over {o['n_scored']} scored runs")
    p = s["paired_drex_minus_base"]
    if p:
        print(f"paired drex - base: {_fmt(p['mean_diff'], '+.3f')} over {p['n_tasks']} tasks")
        for t, d in p["per_task"].items():
            print(f"    {t}: {d:+.3f}")


def main() -> int:
    args = _ARGS
    wire_judge()
    install_judge_shim()
    adapter = make_adapter("jb", None, "oc")
    if not isinstance(adapter, JobBenchAdapter):
        raise TypeError("expected the JobBench adapter")
    adapter.preflight()

    known = adapter.tasks()
    tasks = known if args.tasks == ["all"] else args.tasks
    unknown = [t for t in tasks if t not in known]
    if unknown:
        sys.exit(f"unknown task ids {unknown}; adapter.tasks() has {len(known)}, e.g. {known[:2]}")
    arms = list(dict.fromkeys(args.arms))
    jobs = [Job(t, a, r) for t in tasks for a in arms for r in range(1, args.repeats + 1)]
    if "drex" in arms and not os.environ.get("DREX_API_KEY") and not args.dry_run:
        sys.exit("DREX_API_KEY is not set (expected in .env)")

    OUT.mkdir(parents=True, exist_ok=True)
    harnesses: dict[str, Path | None] = {"base": None}
    helper_line = None
    if "drex" in arms:
        drex_dir = build_drex_harness(adapter, OUT / "harness_drex")
        harnesses["drex"] = drex_dir
        census = validate_harness(adapter, drex_dir)
        helper_line = rendered_helper_line(adapter, drex_dir)
    validate_harness(adapter, adapter.base_harness())

    print(f"out: {OUT}\nstore: {adapter.store}")
    print(f"judge: model={config.JB_JUDGE_MODEL} base={config.JB_JUDGE_API_BASE} "
          f"effort={config.JB_JUDGE_REASONING_EFFORT}")
    print(f"solver: {config.SOLVER_MODEL_OC}")
    if helper_line:
        print(f"drex tree: {census}\nrendered helper line: {helper_line}")
    print(f"{len(jobs)} jobs ({len(tasks)} tasks x {len(arms)} arms x {args.repeats} repeats), "
          f"concurrency {args.concurrency}")
    for j in jobs:
        print(f"  {j.run_id:14s} {j.task}")
    if args.dry_run:
        return 0

    store = ResultStore(OUT / "ab_results.jsonl")
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(run_job, adapter, j, harnesses, args.judge_only): j for j in jobs}
        for fut in as_completed(futures):
            job, row = futures[fut], fut.result()
            store.put(job, row)
            print(f"done {job.run_id} {job.task}: delivered={row['delivered']} "
                  f"pass_frac={_fmt(row['pass_frac'])} drex_calls={row['drex_calls']} "
                  f"cost={row['cost_usd']} wall={row['agent_wall_s']}s "
                  f"error={row['error']} judge_error={row['judge_error']}", flush=True)

    keys = {j.key for j in jobs}
    summary = summarise([r for k, r in store.rows.items() if k in keys], tasks, arms)
    (OUT / "ab_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_summary(summary, arms)
    return 0


if __name__ == "__main__":
    sys.exit(main())
