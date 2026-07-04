#!/usr/bin/env python3
"""
Evaluate existing model outputs with OpenCompass.

This is the CPU/local half of the split workflow. It reads infer_manifest.jsonl
created by run_model.py, then runs OpenCompass in eval/viz mode with -r to reuse
already generated predictions.

Typical usage:
  python run_outputs.py --root model_outputs/arness/mbpp_s6
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from run_test import OPENCOMPASS_DIR, ROOT, copy_aliases_for_visualizer

CONTROL_ARGS_WITH_VALUE = {
    "-m",
    "--mode",
    "-w",
    "--work-dir",
    "-r",
    "--reuse",
}
CONTROL_ARGS_NO_VALUE = {
    "--dry-run",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_under_root(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def strip_opencompass_control_args(args: Sequence[Any] | None) -> List[str]:
    cleaned: List[str] = []
    items = [str(x) for x in (args or [])]
    i = 0
    while i < len(items):
        item = items[i]
        if item in CONTROL_ARGS_WITH_VALUE:
            i += 2
            continue
        if any(item.startswith(flag + "=") for flag in CONTROL_ARGS_WITH_VALUE):
            i += 1
            continue
        if item in CONTROL_ARGS_NO_VALUE:
            i += 1
            continue
        cleaned.append(item)
        i += 1
    return cleaned


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    fieldnames = list(row.keys())
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def find_manifests(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob("infer_manifest.jsonl"))


def build_env() -> Dict[str, str]:
    env = os.environ.copy()
    pythonpath = [str(ROOT), str(OPENCOMPASS_DIR)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return env


def run_command(command: Sequence[str], cwd: Path, env: Dict[str, str]) -> int:
    print("$ " + " ".join(command), flush=True)
    proc = subprocess.Popen(
        list(command),
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    return proc.wait()


def resolve_from_manifest(record: Dict[str, Any], manifest_path: Path, key: str) -> Path:
    """Resolve an absolute path, falling back to *_rel under manifest directory.

    This makes outputs portable after copying the model_outputs tree from GPU to local.
    """
    raw = record.get(key)
    if raw:
        path = Path(raw)
        if path.exists():
            return path

    rel_key = f"{key}_rel"
    rel = record.get(rel_key)
    if rel:
        candidate = manifest_path.parent / rel
        if candidate.exists() or key == "work_dir":
            return candidate

    if raw:
        return Path(raw)
    raise KeyError(f"Cannot resolve `{key}` from manifest record: {record.get('run_name')}")


def resolve_artifact_path(record: Dict[str, Any], manifest_path: Path, key: str, default: Path) -> Path:
    raw = record.get(key)
    if raw:
        path = Path(raw)
        if path.exists():
            return path

    rel = record.get(f"{key}_rel")
    if rel:
        return manifest_path.parent / rel

    if raw:
        return Path(raw)
    return default


def timestamp_exists(work_dir: Path, timestamp: str) -> bool:
    return (work_dir / timestamp).exists()


def latest_timestamp(work_dir: Path) -> Optional[str]:
    if not work_dir.exists():
        return None
    candidates = []
    for child in work_dir.iterdir():
        if child.is_dir() and (child / "configs").exists():
            candidates.append(child.name)
    return sorted(candidates)[-1] if candidates else None


def read_jsonl_count(path: Path) -> int:
    if not path.exists() or path.is_dir():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def mode_sequence(mode: str) -> List[str]:
    if mode == "both":
        return ["eval", "viz"]
    return [mode]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate saved model outputs with OpenCompass reuse mode."
    )
    parser.add_argument(
        "--root",
        default="model_outputs",
        help="model_outputs root, an experiment folder, or an infer_manifest.jsonl file.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional filter by run_name, experiment name, task name, or benchmark.",
    )
    parser.add_argument(
        "--mode",
        choices=["eval", "viz", "both"],
        default="eval",
        help="OpenCompass mode. `eval` already writes summary in this OpenCompass version.",
    )
    parser.add_argument(
        "--reuse",
        default=None,
        help="Override OpenCompass reuse timestamp. Default: timestamp recorded by run_model.py.",
    )
    parser.add_argument(
        "--extra-opencompass-args",
        nargs="*",
        default=None,
        help="Extra args passed to OpenCompass after removing mode/work-dir/reuse controls.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even if a previous eval_manifest.jsonl entry exists for the same run/mode/timestamp.",
    )
    args = parser.parse_args()

    root = resolve_under_root(args.root)
    manifests = find_manifests(root)
    if not manifests:
        raise SystemExit(f"No infer_manifest.jsonl found under: {root}")

    selected = set(args.only or [])
    env = build_env()
    extra_args = strip_opencompass_control_args(args.extra_opencompass_args or [])
    total = 0

    for manifest_path in manifests:
        exp_root = manifest_path.parent
        eval_manifest = exp_root / "eval_manifest.jsonl"
        eval_csv = exp_root / "eval_runs.csv"

        done_keys: Set[str] = set()
        if eval_manifest.exists() and not args.force:
            for item in read_jsonl(eval_manifest):
                done_keys.add(
                    f"{item.get('run_name')}|{item.get('mode')}|{item.get('reuse_timestamp')}"
                )

        for record in read_jsonl(manifest_path):
            run_name = record.get("run_name")
            experiment = record.get("experiment")
            task = record.get("task")
            benchmark = record.get("benchmark")

            if selected and not ({run_name, experiment, task, benchmark} & selected):
                continue

            if record.get("returncode") not in (0, None):
                print(
                    f"[skip:{run_name}] inference returncode={record.get('returncode')}",
                    flush=True,
                )
                continue

            config_path = resolve_from_manifest(record, manifest_path, "config")
            work_dir = resolve_from_manifest(record, manifest_path, "work_dir")
            outputs_jsonl = resolve_artifact_path(record, manifest_path, "outputs_jsonl", work_dir / "outputs.jsonl")
            summary_jsonl = resolve_artifact_path(record, manifest_path, "summary_jsonl", work_dir / "summary.jsonl")
            trace_jsonl = resolve_artifact_path(record, manifest_path, "trace_jsonl", work_dir / "trace.jsonl")

            reuse_timestamp = args.reuse or record.get("opencompass_reuse_timestamp")
            if not reuse_timestamp:
                reuse_timestamp = latest_timestamp(work_dir)
            if not reuse_timestamp:
                raise SystemExit(f"No reusable OpenCompass timestamp found for {run_name}.")
            if not timestamp_exists(work_dir, reuse_timestamp):
                raise SystemExit(
                    f"Reuse timestamp does not exist for {run_name}: "
                    f"{work_dir / reuse_timestamp}"
                )

            for oc_mode in mode_sequence(args.mode):
                done_key = f"{run_name}|{oc_mode}|{reuse_timestamp}"
                if done_key in done_keys:
                    print(f"[skip:{run_name}] already evaluated mode={oc_mode} reuse={reuse_timestamp}")
                    continue

                total += 1
                command = [
                    sys.executable,
                    str(OPENCOMPASS_DIR / "run.py"),
                    str(config_path),
                    "-w",
                    str(work_dir),
                    "-m",
                    oc_mode,
                    "-r",
                    str(reuse_timestamp),
                ]
                command.extend(extra_args)

                print(f"\n[{oc_mode}:{run_name}] work_dir: {work_dir}")
                print(f"[{oc_mode}:{run_name}] reuse: {reuse_timestamp}")

                if args.dry_run:
                    returncode = None
                    elapsed = None
                else:
                    started = time.perf_counter()
                    returncode = run_command(command, OPENCOMPASS_DIR, env)
                    elapsed = round(time.perf_counter() - started, 3)
                    copy_aliases_for_visualizer(outputs_jsonl, summary_jsonl)

                eval_record = {
                    "created_at": utc_now(),
                    "mode": oc_mode,
                    "dry_run": args.dry_run,
                    "task": task,
                    "experiment": experiment,
                    "benchmark": benchmark,
                    "run_label": record.get("run_label"),
                    "run_name": run_name,
                    "config": str(config_path),
                    "work_dir": str(work_dir),
                    "outputs_jsonl": str(outputs_jsonl),
                    "summary_jsonl": str(summary_jsonl),
                    "trace_jsonl": str(trace_jsonl),
                    "visual_command": record.get("visual_command"),
                    "reuse_timestamp": reuse_timestamp,
                    "returncode": returncode,
                    "elapsed_seconds": elapsed,
                    "num_output_records": read_jsonl_count(outputs_jsonl),
                    "num_summary_records": read_jsonl_count(summary_jsonl),
                    "trace_exists": trace_jsonl.exists(),
                    "command": command,
                }
                append_jsonl(eval_manifest, eval_record)
                append_csv(
                    eval_csv,
                    {
                        "created_at": eval_record["created_at"],
                        "mode": oc_mode,
                        "run_name": run_name,
                        "benchmark": benchmark,
                        "reuse_timestamp": reuse_timestamp,
                        "returncode": "" if returncode is None else returncode,
                        "elapsed_seconds": "" if elapsed is None else elapsed,
                        "work_dir": str(work_dir),
                    },
                )

                if returncode not in (0, None):
                    return returncode

    print(f"\nEvaluated {total} OpenCompass run(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
