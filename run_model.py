#!/usr/bin/env python3
"""
Run model inference only for the YAML experiment config.

This is the GPU-side half of the split workflow:
  test_config.yaml -> generated OpenCompass config -> OpenCompass -m infer

Outputs are written under the same condition layout as run_test.py, rooted at
outputs/ by default:
  outputs/<task_name>/<compact_experiment>/<visual_condition>/

It intentionally does not run OpenCompass evaluation. Use run_outputs.py later
on the saved outputs directory to evaluate/reuse the inference outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from run_test import (  # reuse the existing repo config/rendering logic
    OPENCOMPASS_DIR,
    ROOT,
    arness_visual_condition_label,
    as_list,
    build_model_cfg,
    compact_experiment_name,
    compact_run_label,
    collect_experiments,
    condition_run_name,
    copy_aliases_for_visualizer,
    deep_merge,
    expand_matrix,
    load_yaml,
    render_opencompass_config,
    safe_name,
    unique_label,
    varying_param_keys,
    write_visual_command,
)

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


def rel_to(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def strip_opencompass_control_args(args: Sequence[Any] | None) -> List[str]:
    """Keep user extra OpenCompass args, but remove mode/workdir/reuse controls."""
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


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def append_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    fieldnames = list(row.keys())
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


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


def current_gpu_snapshot() -> Dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"available": False}

    if result.returncode != 0:
        return {"available": False, "error": result.stderr.strip()}

    rows: List[Dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 6:
            rows.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "memory_used_mb": parts[2],
                    "memory_total_mb": parts[3],
                    "utilization_gpu_percent": parts[4],
                    "power_draw_w": parts[5],
                }
            )
    return {"available": True, "gpus": rows}


class GpuTelemetry:
    def __init__(self, output_path: Path, interval_seconds: float = 1.0):
        self.output_path = output_path
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 2))

    def _run(self) -> None:
        fields = [
            "timestamp_utc",
            "gpu_index",
            "gpu_name",
            "utilization_gpu_percent",
            "memory_used_mb",
            "memory_total_mb",
            "power_draw_w",
            "temperature_gpu_c",
        ]
        query = [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        with self.output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            while not self._stop.is_set():
                timestamp = utc_now()
                try:
                    result = subprocess.run(
                        query, capture_output=True, text=True, check=False, timeout=10
                    )
                    if result.returncode == 0:
                        for line in result.stdout.splitlines():
                            parts = [part.strip() for part in line.split(",")]
                            if len(parts) == 7:
                                writer.writerow(
                                    {
                                        "timestamp_utc": timestamp,
                                        "gpu_index": parts[0],
                                        "gpu_name": parts[1],
                                        "utilization_gpu_percent": parts[2],
                                        "memory_used_mb": parts[3],
                                        "memory_total_mb": parts[4],
                                        "power_draw_w": parts[5],
                                        "temperature_gpu_c": parts[6],
                                    }
                                )
                    else:
                        writer.writerow({"timestamp_utc": timestamp})
                    f.flush()
                except (OSError, subprocess.TimeoutExpired):
                    writer.writerow({"timestamp_utc": timestamp})
                    f.flush()
                self._stop.wait(self.interval_seconds)


def opencompass_timestamp_dirs(work_dir: Path) -> Set[str]:
    if not work_dir.exists():
        return set()
    names: Set[str] = set()
    for child in work_dir.iterdir():
        if child.is_dir() and (child / "configs").exists():
            names.add(child.name)
    return names


def latest_opencompass_timestamp(work_dir: Path) -> Optional[str]:
    names = sorted(opencompass_timestamp_dirs(work_dir))
    return names[-1] if names else None


def experiment_root(output_root: Path, experiment: Dict[str, Any], compact_exp_name: str) -> Path:
    task = safe_name(str(experiment.get("task") or "runs"))
    return output_root / task / compact_exp_name


def read_jsonl_count(path: Path) -> int:
    if not path.exists() or path.is_dir():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def build_env() -> Dict[str, str]:
    env = os.environ.copy()
    pythonpath = [str(ROOT), str(OPENCOMPASS_DIR)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return env


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run model inference only and save reusable outputs in the run_test/visual layout."
    )
    parser.add_argument("--config", default="test_config.yaml", help="YAML experiment config.")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Task section or experiment name. Same semantics as run_test.py.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Root for saved inference outputs. Default: outputs",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    args = parser.parse_args()

    config_path = resolve_under_root(args.config)
    config = load_yaml(config_path)

    execution_cfg = config.get("execution", {}) or {}
    data_cfg = config.get("data", {}) or {}
    global_model = config.get("model", {}) or {}
    runner_cfg = config.get("runner", {}) or {}
    default_params = config.get("defaults", {}) or {}

    selected = set(args.only or [])
    experiments = collect_experiments(config, selected)
    if not experiments:
        raise SystemExit("No experiments matched.")

    output_root = resolve_under_root(args.output_root)
    dry_run = args.dry_run or bool(execution_cfg.get("dry_run", False))
    extra_opencompass_args = strip_opencompass_control_args(
        execution_cfg.get("opencompass_args", []) or []
    )

    telemetry_cfg = execution_cfg.get("gpu_telemetry", {}) or {}
    telemetry_enabled = bool(telemetry_cfg.get("enabled", True))
    telemetry_interval = float(telemetry_cfg.get("interval_seconds", 1.0))

    env = build_env()
    planned = 0

    for experiment in experiments:
        exp_name = experiment.get("name")
        if not exp_name:
            raise SystemExit("Every experiment needs a `name`.")

        benchmark_list = as_list(experiment.get("benchmark"))
        first_benchmark = str(benchmark_list[0]) if benchmark_list else None
        compact_exp_dir = compact_experiment_name(experiment, first_benchmark)
        exp_root = experiment_root(output_root, experiment, compact_exp_dir)
        generated_dir = exp_root / "generated_configs"
        generated_dir.mkdir(parents=True, exist_ok=True)
        infer_manifest = exp_root / "infer_manifest.jsonl"
        infer_csv = exp_root / "infer_runs.csv"
        expanded_params = [deep_merge(default_params, p) for p in expand_matrix(experiment)]
        include_keys = varying_param_keys(expanded_params)
        used_labels: Set[str] = set()

        for benchmark in benchmark_list:
            for idx, params in enumerate(expand_matrix(experiment), start=1):
                planned += 1
                merged_params = deep_merge(default_params, params)
                run_name = condition_run_name(exp_name, benchmark, merged_params, idx)
                visual_label = arness_visual_condition_label(benchmark, merged_params)
                run_label = unique_label(
                    visual_label or compact_run_label(merged_params, include_keys, compact_exp_dir, idx),
                    used_labels,
                    idx,
                )
                work_dir = exp_root / run_label
                work_dir.mkdir(parents=True, exist_ok=True)

                model_cfg = build_model_cfg(
                    deepcopy(global_model), merged_params, benchmark, run_label
                )
                model_cfg["task_id"] = experiment.get("task")

                # Keep raw model output and optional trace outside OpenCompass timestamp
                # folders, so they are easy to find and copy back locally.
                outputs_jsonl = work_dir / "outputs.jsonl"
                summary_jsonl = work_dir / "summary.jsonl"
                trace_jsonl = work_dir / "trace.jsonl"
                model_cfg.setdefault("per_sample_output", str(outputs_jsonl))
                model_cfg.setdefault("metrics_output", str(summary_jsonl))
                if (
                    model_cfg.get("return_trace")
                    or model_cfg.get("trace_token_snapshots")
                    or model_cfg.get("trace_decode_snapshots")
                ):
                    model_cfg.setdefault("step_trace_output", str(trace_jsonl))
                    model_cfg["arness_trace_output"] = str(work_dir / "sample_traces")

                generated_config = work_dir / "oc_config.py"
                config_text = render_opencompass_config(
                    benchmark=benchmark,
                    model_cfg=deepcopy(model_cfg),
                    runner_cfg=runner_cfg,
                    sample_limit=merged_params.get("sample_limit"),
                    sample_indices=merged_params.get("sample_indices"),
                    experiment_params=merged_params,
                    data_cfg=data_cfg,
                )
                generated_config.write_text(config_text, encoding="utf-8")

                run_config = {
                    "mode": "infer",
                    "created_at": utc_now(),
                    "source_config": str(config_path),
                    "task": experiment.get("task"),
                    "experiment": exp_name,
                    "compact_experiment": compact_exp_dir,
                    "benchmark": benchmark,
                    "run_label": run_label,
                    "run_name": run_name,
                    "opencompass_run_name": run_label,
                    "params": merged_params,
                    "model": model_cfg,
                    "runner": runner_cfg,
                    "generated_opencompass_config": str(generated_config),
                    "work_dir": str(work_dir),
                    "output_files": {
                        "outputs_jsonl": str(outputs_jsonl),
                        "summary_jsonl": str(summary_jsonl),
                        "trace_jsonl": str(trace_jsonl),
                        "gpu_csv": str(work_dir / "gpu.csv"),
                    },
                }
                write_json(work_dir / "run.json", run_config)
                write_visual_command(work_dir / "visual_command.txt", work_dir, merged_params)

                before_timestamps = opencompass_timestamp_dirs(work_dir)
                command = [
                    sys.executable,
                    str(OPENCOMPASS_DIR / "run.py"),
                    str(generated_config),
                    "-w",
                    str(work_dir),
                    "-m",
                    "infer",
                ]
                command.extend(extra_opencompass_args)

                manifest_record: Dict[str, Any] = {
                    "created_at": utc_now(),
                    "mode": "infer",
                    "dry_run": dry_run,
                    "task": experiment.get("task"),
                    "experiment": exp_name,
                    "benchmark": benchmark,
                    "compact_experiment": compact_exp_dir,
                    "run_label": run_label,
                    "run_name": run_name,
                    "params": merged_params,
                    "source_config": str(config_path),
                    "experiment_root": str(exp_root),
                    "config": str(generated_config),
                    "config_rel": rel_to(generated_config, exp_root),
                    "work_dir": str(work_dir),
                    "work_dir_rel": rel_to(work_dir, exp_root),
                    "outputs_jsonl": str(outputs_jsonl),
                    "outputs_jsonl_rel": rel_to(outputs_jsonl, exp_root),
                    "summary_jsonl": str(summary_jsonl),
                    "summary_jsonl_rel": rel_to(summary_jsonl, exp_root),
                    "trace_jsonl": str(trace_jsonl),
                    "trace_jsonl_rel": rel_to(trace_jsonl, exp_root),
                    "visual_command": (work_dir / "visual_command.txt").read_text(encoding="utf-8").strip(),
                    "command": command,
                    "gpu_before": current_gpu_snapshot(),
                }

                print(f"\n[infer:{run_name}] config: {generated_config}")
                print(f"[infer:{run_name}] work_dir: {work_dir}")

                if dry_run:
                    print("[dry-run] " + " ".join(command))
                    manifest_record.update(
                        {
                            "returncode": None,
                            "elapsed_seconds": None,
                            "opencompass_reuse_timestamp": None,
                        }
                    )
                else:
                    telemetry: Optional[GpuTelemetry] = None
                    if telemetry_enabled:
                        telemetry = GpuTelemetry(
                            work_dir / "gpu.csv",
                            interval_seconds=telemetry_interval,
                        )
                        telemetry.start()

                    started = time.perf_counter()
                    try:
                        returncode = run_command(command, OPENCOMPASS_DIR, env)
                    finally:
                        if telemetry is not None:
                            telemetry.stop()

                    elapsed = round(time.perf_counter() - started, 3)
                    copy_aliases_for_visualizer(outputs_jsonl, summary_jsonl)
                    after_timestamps = opencompass_timestamp_dirs(work_dir)
                    new_timestamps = sorted(after_timestamps - before_timestamps)
                    reuse_timestamp = (
                        new_timestamps[-1]
                        if new_timestamps
                        else latest_opencompass_timestamp(work_dir)
                    )

                    manifest_record.update(
                        {
                            "returncode": returncode,
                            "elapsed_seconds": elapsed,
                            "opencompass_reuse_timestamp": reuse_timestamp,
                            "gpu_after": current_gpu_snapshot(),
                            "num_output_records": read_jsonl_count(outputs_jsonl),
                            "num_summary_records": read_jsonl_count(summary_jsonl),
                            "trace_exists": trace_jsonl.exists(),
                        }
                    )

                    append_csv(
                        infer_csv,
                        {
                            "created_at": manifest_record["created_at"],
                            "run_label": run_label,
                            "run_name": run_name,
                            "benchmark": benchmark,
                            "returncode": returncode,
                            "elapsed_seconds": elapsed,
                            "opencompass_reuse_timestamp": reuse_timestamp or "",
                            "output_records": manifest_record["num_output_records"],
                            "summary_records": manifest_record["num_summary_records"],
                            "work_dir": str(work_dir),
                            "visual_command": manifest_record["visual_command"],
                        },
                    )

                    if returncode != 0 and execution_cfg.get("stop_on_error", True):
                        append_jsonl(infer_manifest, manifest_record)
                        return returncode

                append_jsonl(infer_manifest, manifest_record)

    print(f"\nPlanned {planned} inference run(s).")
    print(f"Outputs root: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
