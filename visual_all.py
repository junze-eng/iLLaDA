#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arness_visual_all.py

Visualize iLLaDA ARness experiments from a run root such as outputs/arness_all.

This version includes BOTH:
  - Task1 / parallel experiments as the no-threshold baseline, usually n=50
  - Task3 / threshold counterfactual experiments, usually n=10

It reads test_config.yaml to find expected Task1 + Task3 conditions, audits
repeated appended runs, writes clean plot-ready CSVs, and saves figures under
visual/arness_all.

Usage:
  python arness_visual_all.py
  python arness_visual_all.py --root outputs/arness_all --config test_config.yaml --out visual/arness_all

Important interpretation:
- Task1 and Task3 can be plotted together as ARness evidence, but they have
  different sample sizes. Task1 is the no-threshold baseline; Task3 isolates
  the threshold effect.
- Some run dirs may contain duplicated summary.jsonl rows because a condition
  was re-run into the same work_dir. This script counts raw rows and unique
  sample_idx, and de-duplicates at the condition level.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import yaml


# -----------------------------
# General helpers
# -----------------------------

def as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a or {})
    for k, v in (b or {}).items():
        if isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def expand_experiment(exp: Dict[str, Any], defaults: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    params = deep_merge(defaults, exp.get("params") or {})
    sweep = exp.get("sweep") or {}
    if not sweep:
        yield params
        return

    keys = list(sweep.keys())
    values = [as_list(sweep[k]) for k in keys]
    for combo in product(*values):
        item = dict(params)
        for k, v in zip(keys, combo):
            item[k] = v
        yield item


def norm_float_or_none(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"none", "null", "nan"}:
        return None
    return float(s)


def threshold_label(x: Any) -> str:
    v = norm_float_or_none(x)
    if v is None:
        return "none"
    return f"{v:g}"


def num_or_nan(x: Any) -> float:
    if x is None:
        return float("nan")
    try:
        if pd.isna(x):
            return float("nan")
    except Exception:
        pass
    s = str(x).strip()
    if s == "":
        return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def latest_summary_csv_value(run_dir: Optional[Path]) -> Tuple[Optional[str], Optional[float], Optional[Path]]:
    """Find the latest OpenCompass summary csv under a run dir and return a best metric."""
    if run_dir is None or not run_dir.exists():
        return None, None, None
    candidates = sorted(run_dir.glob("**/summary/summary_*.csv"))
    if not candidates:
        return None, None, None

    csv_path = candidates[-1]
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None, None, csv_path

    preferred_cols = [
        "accuracy", "score", "pass", "acc", "gsm8k", "mbpp",
        "opencompass_score", "result",
    ]
    for col in df.columns:
        if str(col).lower() in preferred_cols:
            val = num_or_nan(df[col].iloc[0])
            if not math.isnan(val):
                return str(col), float(val), csv_path

    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    for col in reversed(numeric_cols):
        val = num_or_nan(df[col].iloc[0])
        if not math.isnan(val):
            return str(col), float(val), csv_path

    return None, None, csv_path


def infer_run_dir(root: Path, run_name: str) -> Optional[Path]:
    if not run_name or run_name == "nan":
        return None
    direct = root / run_name
    if direct.exists():
        return direct
    # Sometimes run_summary.csv is inside the run dir and root aggregate lacks run_name alignment.
    matches = [p for p in root.rglob(run_name) if p.is_dir()]
    if matches:
        return matches[0]
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(run_name)]
    return candidates[0] if candidates else None


def infer_source(value: Any) -> str:
    s = str(value or "")
    if s.startswith("task1_") or "task1_" in s:
        return "task1"
    if s.startswith("task3_") or "task3_" in s:
        return "task3"
    return "unknown"


def source_label(source: str, threshold: str) -> str:
    if source == "task1":
        return "task1 none"
    if threshold == "none":
        return "task3 none"
    return f"task3 th={threshold}"


# -----------------------------
# Config expected-condition parsing
# -----------------------------

@dataclass(frozen=True)
class ExpectedCondition:
    source: str
    task: str
    experiment: str
    benchmark: str
    decoding_config_name: str
    gen_length: int
    gen_steps: int
    gen_blocksize: int
    threshold_label: str
    threshold_value: Optional[float]
    sample_limit: Optional[int]

    @property
    def key(self) -> Tuple[str, str, int, str]:
        return (self.source, self.benchmark, self.gen_steps, self.threshold_label)


def _condition_source(task_name: str, exp_name: str) -> str:
    if exp_name.startswith("task1_") or task_name == "parallel":
        return "task1"
    if exp_name.startswith("task3_") or task_name == "arness":
        return "task3"
    return "unknown"


def load_expected_conditions(config_path: Path) -> List[ExpectedCondition]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    defaults = cfg.get("defaults") or {}
    tasks = cfg.get("tasks") or {}

    expected: List[ExpectedCondition] = []
    for task_name in ["parallel", "arness"]:
        task = tasks.get(task_name)
        if not task:
            continue
        for exp in task.get("experiments", []):
            exp_name = str(exp.get("name") or "")
            source = _condition_source(task_name, exp_name)
            if source not in {"task1", "task3"}:
                continue
            for bench in as_list(exp.get("benchmark")):
                for params in expand_experiment(exp, defaults):
                    gen_length = int(params.get("gen_length"))
                    gen_steps = int(params.get("gen_steps"))
                    gen_blocksize = int(params.get("gen_blocksize", params.get("block_length", gen_length)))
                    th = norm_float_or_none(params.get("token_selection_confidence_threshold"))
                    sample_limit_raw = params.get("sample_limit", params.get("num_samples"))
                    sample_limit = int(sample_limit_raw) if sample_limit_raw not in (None, "") else None
                    decoding_config_name = str(params.get("decoding_config_name") or exp_name)
                    expected.append(
                        ExpectedCondition(
                            source=source,
                            task=task_name,
                            experiment=exp_name,
                            benchmark=str(bench),
                            decoding_config_name=decoding_config_name,
                            gen_length=gen_length,
                            gen_steps=gen_steps,
                            gen_blocksize=gen_blocksize,
                            threshold_label=threshold_label(th),
                            threshold_value=th,
                            sample_limit=sample_limit,
                        )
                    )

    seen = set()
    deduped: List[ExpectedCondition] = []
    for c in expected:
        full_key = (
            c.source, c.experiment, c.benchmark, c.decoding_config_name,
            c.gen_length, c.gen_steps, c.threshold_label,
        )
        if full_key not in seen:
            deduped.append(c)
            seen.add(full_key)
    return deduped


# -----------------------------
# Condition data collection
# -----------------------------

def normalize_aggregate_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    pairs = {
        "gen_length": ["gen_length", "param_gen_length"],
        "gen_steps": ["gen_steps", "param_gen_steps"],
        "gen_blocksize": ["gen_blocksize", "param_gen_blocksize", "block_length"],
        "threshold": [
            "token_selection_confidence_threshold",
            "param_token_selection_confidence_threshold",
        ],
        "sample_limit": ["sample_limit", "param_sample_limit"],
        "num_samples": ["num_samples"],
        "latency_mean_s": ["latency_mean_s", "latency_mean", "latency_mean_seconds"],
        "tokens_per_second_mean": ["tokens_per_second_mean", "tps_mean"],
        "visible_tps_mean": ["visible_tps_mean"],
        "actual_commit_tps_mean": ["actual_commit_tps_mean"],
        "completion_rate": ["completion_rate", "completion_rate_mean"],
        "actual_parallelism": ["actual_parallelism", "actual_parallelism_mean"],
        "actual_arness_mean": ["actual_arness_mean", "actual_arness"],
        "threshold_pass_rate_mean": ["threshold_pass_rate_mean", "threshold_pass_rate"],
        "fallback_rate_mean": ["fallback_rate_mean", "fallback_rate"],
        "cuda_max_memory_allocated_mb": ["cuda_max_memory_allocated_mb", "peak_vram"],
        "cuda_max_memory_reserved_mb": ["cuda_max_memory_reserved_mb"],
        "peak_vram": ["peak_vram", "cuda_max_memory_reserved_mb"],
        "gpu_util_mean": ["gpu_util_mean"],
        "gpu_util_max": ["gpu_util_max"],
        "gpu_memory_used_max_mb": ["gpu_memory_used_max_mb"],
        "gpu_power_mean_w": ["gpu_power_mean_w"],
        "gpu_temperature_max_c": ["gpu_temperature_max_c"],
        "primary_metric_name": ["primary_metric_name"],
        "primary_metric_value": ["primary_metric_value"],
        "returncode": ["returncode"],
        "run_name": ["run_name"],
        "experiment": ["experiment"],
        "benchmark": ["benchmark"],
        "decoding_config_name": ["decoding_config_name"],
    }

    for dst, srcs in pairs.items():
        if dst in df.columns:
            continue
        for src in srcs:
            if src in df.columns:
                df[dst] = df[src]
                break

    numeric_cols = [
        "gen_length", "gen_steps", "gen_blocksize", "sample_limit", "num_samples",
        "latency_mean_s", "tokens_per_second_mean", "visible_tps_mean",
        "actual_commit_tps_mean", "completion_rate", "actual_parallelism",
        "actual_arness_mean", "threshold_pass_rate_mean", "fallback_rate_mean",
        "cuda_max_memory_allocated_mb", "cuda_max_memory_reserved_mb", "peak_vram",
        "gpu_util_mean", "gpu_util_max", "gpu_memory_used_max_mb",
        "gpu_power_mean_w", "gpu_temperature_max_c", "primary_metric_value",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "threshold" in df.columns:
        df["threshold_label"] = df["threshold"].apply(threshold_label)
    else:
        df["threshold_label"] = "none"

    # Source comes from experiment/run_name. Task1 has no threshold.
    exp_s = df.get("experiment", pd.Series([""] * len(df))).astype(str)
    run_s = df.get("run_name", pd.Series([""] * len(df))).astype(str)
    df["source"] = [infer_source(e) if infer_source(e) != "unknown" else infer_source(r) for e, r in zip(exp_s, run_s)]
    df.loc[df["source"] == "task1", "threshold_label"] = "none"

    if "decoding_config_name" not in df.columns:
        df["decoding_config_name"] = df.get("experiment", "")

    return df


def load_aggregate_rows(root: Path) -> pd.DataFrame:
    paths: List[Path] = []
    for name in ["aggregate.csv", "run_summary.csv"]:
        p = root / name
        if p.exists():
            paths.append(p)

    # Also read nested run summaries. Avoid visual/ and generated config folders.
    for p in sorted(root.rglob("run_summary.csv")):
        if p not in paths and "generated_configs" not in str(p):
            paths.append(p)
    for p in sorted(root.rglob("aggregate.csv")):
        if p not in paths and "generated_configs" not in str(p):
            paths.append(p)

    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df["source_csv"] = str(p)
            # If nested run_summary lacks run_name, use parent dir.
            if "run_name" not in df.columns or df["run_name"].isna().all():
                df["run_name"] = p.parent.name
            frames.append(df)
        except Exception as e:
            print(f"[WARN] failed reading {p}: {e}")

    if not frames:
        raise FileNotFoundError(f"No aggregate.csv / run_summary.csv found under {root}")

    df = pd.concat(frames, ignore_index=True)
    df = normalize_aggregate_columns(df)

    mask = df["source"].isin(["task1", "task3"])
    return df[mask].copy()


def add_sample_audit(root: Path, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    raw_counts = []
    unique_counts = []
    duplicate_counts = []
    run_dirs = []
    latest_metric_names = []
    latest_metric_values = []
    latest_metric_paths = []

    for _, row in df.iterrows():
        run_name = str(row.get("run_name") or "")
        run_dir = infer_run_dir(root, run_name)
        run_dirs.append(str(run_dir) if run_dir else "")

        summary_rows = read_jsonl(run_dir / "summary.jsonl") if run_dir else []
        raw = len(summary_rows)
        ids = []
        for r in summary_rows:
            if "sample_idx" in r:
                ids.append(str(r.get("sample_idx")))
            elif "sample_id" in r:
                ids.append(str(r.get("sample_id")))
        unique = len(set(ids)) if ids else raw
        n_from_row = int(row.get("num_samples") or 0) if not math.isnan(num_or_nan(row.get("num_samples"))) else 0
        raw_counts.append(raw if raw else n_from_row)
        unique_counts.append(unique if unique else n_from_row)
        duplicate_counts.append(max(0, (raw if raw else 0) - (unique if unique else 0)))

        m_name, m_val, m_path = latest_summary_csv_value(run_dir)
        latest_metric_names.append(m_name or "")
        latest_metric_values.append(m_val if m_val is not None else float("nan"))
        latest_metric_paths.append(str(m_path) if m_path else "")

    df["run_dir"] = run_dirs
    df["raw_sample_records"] = raw_counts
    df["unique_sample_count"] = unique_counts
    df["duplicate_sample_records"] = duplicate_counts
    df["latest_oc_metric_name"] = latest_metric_names
    df["latest_oc_metric_value"] = latest_metric_values
    df["latest_oc_summary_csv"] = latest_metric_paths
    return df


def select_one_row_per_condition(df: pd.DataFrame, expected: List[ExpectedCondition]) -> pd.DataFrame:
    rows = []
    for cond in expected:
        sub = df[
            (df["source"].astype(str) == cond.source)
            & (df["benchmark"].astype(str) == cond.benchmark)
            & (pd.to_numeric(df["gen_steps"], errors="coerce") == float(cond.gen_steps))
            & (df["threshold_label"].astype(str) == cond.threshold_label)
        ].copy()

        if sub.empty:
            continue

        # If multiple rows exist because both root aggregate and nested run_summary were read,
        # prefer a row with metric, then num_samples close to unique sample count, then highest unique count.
        def rank_tuple(r: pd.Series) -> Tuple[int, float, float, str]:
            metric_missing = 1 if math.isnan(num_or_nan(r.get("primary_metric_value"))) else 0
            unique = num_or_nan(r.get("unique_sample_count"))
            nrow = num_or_nan(r.get("num_samples"))
            if math.isnan(unique):
                unique = cond.sample_limit or 0
            if math.isnan(nrow):
                nrow = unique
            # Prefer rows whose reported num_samples matches unique count; avoid duplicated appended n=20.
            return (metric_missing, abs(nrow - unique), -unique, str(r.get("source_csv") or ""))

        sub["_rank"] = sub.apply(rank_tuple, axis=1)
        sub = sub.sort_values("_rank")
        chosen = sub.iloc[0].copy()

        chosen["expected_source"] = cond.source
        chosen["expected_task"] = cond.task
        chosen["expected_experiment"] = cond.experiment
        chosen["expected_decoding_config_name"] = cond.decoding_config_name
        chosen["source"] = cond.source
        chosen["benchmark"] = cond.benchmark
        chosen["gen_length"] = cond.gen_length
        chosen["gen_steps"] = cond.gen_steps
        chosen["gen_blocksize"] = cond.gen_blocksize
        chosen["threshold_label"] = cond.threshold_label
        chosen["threshold_value"] = cond.threshold_value if cond.threshold_value is not None else ""
        chosen["expected_sample_limit"] = cond.sample_limit if cond.sample_limit is not None else ""
        chosen["series_label"] = source_label(cond.source, cond.threshold_label)

        # Use latest OpenCompass summary metric as fallback only when aggregate primary metric is missing.
        if math.isnan(num_or_nan(chosen.get("primary_metric_value"))) and not math.isnan(num_or_nan(chosen.get("latest_oc_metric_value"))):
            chosen["primary_metric_name"] = chosen.get("latest_oc_metric_name") or "latest_oc_metric"
            chosen["primary_metric_value"] = chosen.get("latest_oc_metric_value")

        rows.append(chosen.drop(labels=["_rank"], errors="ignore"))

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["planned_parallelism"] = out["gen_length"].astype(float) / out["gen_steps"].astype(float)
    out["nominal_arness"] = out["gen_steps"].astype(float) / out["gen_length"].astype(float)
    if "actual_parallelism" in out.columns:
        out["actual_parallelism_gap"] = out["planned_parallelism"] - pd.to_numeric(out["actual_parallelism"], errors="coerce")
    else:
        out["actual_parallelism_gap"] = float("nan")
    return out


# -----------------------------
# Plotting
# -----------------------------

def _series_sort_key(label: str) -> Tuple[int, str]:
    # Show task1 baseline first, then threshold lines.
    if label.startswith("task1"):
        return (0, label)
    if label.endswith("0.8"):
        return (1, label)
    if label.endswith("0.9"):
        return (2, label)
    return (3, label)


def save_line_by_parallelism(df: pd.DataFrame, metric: str, ylabel: str, out_dir: Path, stem: str, title: str) -> None:
    if metric not in df.columns:
        print(f"[WARN] skip {metric}: missing column")
        return
    for bench in sorted(df["benchmark"].dropna().unique()):
        sub = df[df["benchmark"] == bench].copy()
        if sub.empty:
            continue
        plt.figure(figsize=(7.6, 4.8))
        for label in sorted(sub["series_label"].astype(str).unique(), key=_series_sort_key):
            s = sub[sub["series_label"].astype(str) == label].sort_values("planned_parallelism")
            plt.plot(s["planned_parallelism"], s[metric], marker="o", label=label)
        plt.xlabel("Planned parallelism = gen_length / gen_steps")
        plt.ylabel(ylabel)
        plt.title(f"{title}: {bench}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        path = out_dir / f"{stem}_{bench}.png"
        plt.savefig(path, dpi=220)
        plt.close()
        print(f"[OK] wrote {path}")


def save_speed_quality(df: pd.DataFrame, out_dir: Path) -> None:
    if "primary_metric_value" not in df.columns or "latency_mean_s" not in df.columns:
        return
    for bench in sorted(df["benchmark"].dropna().unique()):
        sub = df[df["benchmark"] == bench].copy()
        plt.figure(figsize=(6.6, 5.2))
        for label in sorted(sub["series_label"].astype(str).unique(), key=_series_sort_key):
            s = sub[sub["series_label"].astype(str) == label].sort_values("latency_mean_s")
            plt.plot(s["latency_mean_s"], s["primary_metric_value"], marker="o", label=label)
            for _, r in s.iterrows():
                plt.annotate(str(int(r["gen_steps"])), (r["latency_mean_s"], r["primary_metric_value"]), fontsize=8)
        plt.xlabel("Latency per sample (s)")
        plt.ylabel("Score / accuracy")
        plt.title(f"Speed-quality tradeoff: {bench}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        path = out_dir / f"speed_quality_{bench}.png"
        plt.savefig(path, dpi=220)
        plt.close()
        print(f"[OK] wrote {path}")


def save_planned_vs_actual(df: pd.DataFrame, out_dir: Path) -> None:
    if "actual_parallelism" not in df.columns:
        return
    for bench in sorted(df["benchmark"].dropna().unique()):
        sub = df[df["benchmark"] == bench].copy()
        plt.figure(figsize=(6.6, 5.2))
        for label in sorted(sub["series_label"].astype(str).unique(), key=_series_sort_key):
            s = sub[sub["series_label"].astype(str) == label].sort_values("planned_parallelism")
            plt.plot(s["planned_parallelism"], s["actual_parallelism"], marker="o", label=label)
            for _, r in s.iterrows():
                plt.annotate(str(int(r["gen_steps"])), (r["planned_parallelism"], r["actual_parallelism"]), fontsize=8)
        plt.xlabel("Planned parallelism")
        plt.ylabel("Actual parallelism")
        plt.title(f"Planned vs actual parallelism: {bench}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        path = out_dir / f"planned_vs_actual_parallelism_{bench}.png"
        plt.savefig(path, dpi=220)
        plt.close()
        print(f"[OK] wrote {path}")


def make_plots(clean_all: pd.DataFrame, clean_task3: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Combined Task1 baseline + Task3 threshold counterfactuals.
    combined_specs = [
        ("primary_metric_value", "Score / accuracy", "all_score_vs_parallelism", "Score vs ARness"),
        ("latency_mean_s", "Latency per sample (s)", "all_latency_vs_parallelism", "Latency vs ARness"),
        ("tokens_per_second_mean", "Tokens per second", "all_tps_vs_parallelism", "Throughput vs ARness"),
        ("completion_rate", "Completion rate", "all_completion_vs_parallelism", "Completion vs ARness"),
        ("actual_parallelism", "Actual parallelism", "all_actual_parallelism_vs_parallelism", "Actual parallelism vs planned"),
        ("actual_arness_mean", "Actual ARness", "all_actual_arness_vs_parallelism", "Actual ARness vs planned"),
    ]
    for metric, ylabel, stem, title in combined_specs:
        save_line_by_parallelism(clean_all, metric, ylabel, out_dir, stem, title)
    save_speed_quality(clean_all, out_dir)
    save_planned_vs_actual(clean_all, out_dir)

    # Task3-only process metrics where Task1 has no threshold interpretation.
    if not clean_task3.empty:
        task3_specs = [
            ("threshold_pass_rate_mean", "Threshold pass rate", "task3_threshold_pass_vs_parallelism", "Task3 threshold pass rate"),
            ("fallback_rate_mean", "Fallback rate", "task3_fallback_vs_parallelism", "Task3 fallback rate"),
            ("completion_rate", "Completion rate", "task3_completion_vs_parallelism", "Task3 completion rate"),
        ]
        for metric, ylabel, stem, title in task3_specs:
            save_line_by_parallelism(clean_task3, metric, ylabel, out_dir, stem, title)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/arness_all", help="Root run folder, e.g. outputs/arness_all.")
    parser.add_argument("--config", default="test_config.yaml", help="Config file containing tasks.parallel and tasks.arness.")
    parser.add_argument("--out", default="visual/arness_all", help="Output visualization folder.")
    args = parser.parse_args()

    root = Path(args.root)
    config_path = Path(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    expected = load_expected_conditions(config_path)
    expected_df = pd.DataFrame([c.__dict__ for c in expected])
    expected_df["condition_key"] = expected_df.apply(
        lambda r: f"{r['source']}|{r['benchmark']}|{r['gen_steps']}|{r['threshold_label']}", axis=1
    )
    expected_df.to_csv(out_dir / "expected_arness_all_conditions.csv", index=False)

    print(f"[INFO] expected conditions from config: {len(expected)}")
    print("[INFO] expected by source:")
    if not expected_df.empty:
        print(expected_df.groupby("source").size().to_string())

    raw = load_aggregate_rows(root)
    raw = add_sample_audit(root, raw)
    raw.to_csv(out_dir / "arness_raw_rows_with_audit.csv", index=False)
    print(f"[INFO] raw aggregate rows task1/task3: {len(raw)}")

    clean = select_one_row_per_condition(raw, expected)
    if clean.empty:
        raise SystemExit("[ERROR] No matching Task1/Task3 rows found. Check --root and --config.")

    preferred_cols = [
        "source", "series_label", "expected_task", "expected_experiment", "run_name", "experiment",
        "benchmark", "expected_decoding_config_name", "decoding_config_name",
        "gen_length", "gen_steps", "gen_blocksize", "threshold_label", "threshold_value",
        "expected_sample_limit", "num_samples", "raw_sample_records", "unique_sample_count",
        "duplicate_sample_records", "primary_metric_name", "primary_metric_value",
        "planned_parallelism", "nominal_arness", "actual_parallelism", "actual_parallelism_gap",
        "actual_arness_mean", "completion_rate", "threshold_pass_rate_mean", "fallback_rate_mean",
        "latency_mean_s", "tokens_per_second_mean", "visible_tps_mean", "actual_commit_tps_mean",
        "cuda_max_memory_allocated_mb", "cuda_max_memory_reserved_mb", "peak_vram",
        "gpu_util_mean", "gpu_util_max", "gpu_memory_used_max_mb", "gpu_power_mean_w",
        "gpu_temperature_max_c", "returncode", "run_dir", "source_csv", "latest_oc_summary_csv",
    ]
    for col in preferred_cols:
        if col not in clean.columns:
            clean[col] = ""
    clean = clean[preferred_cols + [c for c in clean.columns if c not in preferred_cols]]
    clean = clean.sort_values(["benchmark", "source", "gen_steps", "threshold_label"], ascending=[True, True, False, True])

    clean_all_path = out_dir / "arness_all_clean.csv"
    clean_task1_path = out_dir / "arness_task1_clean.csv"
    clean_task3_path = out_dir / "arness_task3_clean.csv"
    audit_path = out_dir / "arness_sample_count_audit.csv"

    clean.to_csv(clean_all_path, index=False)
    clean[clean["source"] == "task1"].to_csv(clean_task1_path, index=False)
    clean[clean["source"] == "task3"].to_csv(clean_task3_path, index=False)
    clean[[
        "source", "benchmark", "gen_steps", "threshold_label", "series_label", "run_name",
        "num_samples", "raw_sample_records", "unique_sample_count", "duplicate_sample_records", "run_dir",
    ]].to_csv(audit_path, index=False)

    got_keys = set(zip(clean["source"].astype(str), clean["benchmark"].astype(str), clean["gen_steps"].astype(int), clean["threshold_label"].astype(str)))
    missing = []
    for c in expected:
        if c.key not in got_keys:
            missing.append(c.__dict__)
    pd.DataFrame(missing).to_csv(out_dir / "missing_arness_conditions.csv", index=False)

    print(f"[OK] wrote {clean_all_path} rows={len(clean)}")
    print(f"[OK] wrote {clean_task1_path} rows={(clean['source'] == 'task1').sum()}")
    print(f"[OK] wrote {clean_task3_path} rows={(clean['source'] == 'task3').sum()}")
    print(f"[OK] wrote {audit_path}")
    if missing:
        print(f"[WARN] missing expected conditions: {len(missing)} -> {out_dir / 'missing_arness_conditions.csv'}")
    else:
        print("[OK] all expected Task1/Task3 conditions found")

    dup = clean[pd.to_numeric(clean["duplicate_sample_records"], errors="coerce").fillna(0) > 0]
    if not dup.empty:
        print("[WARN] Some run dirs contain duplicated summary rows from reruns.")
        print("       Figures use one condition row each; unique_sample_count reports real unique sample_idx count.")
        print(dup[["source", "benchmark", "gen_steps", "threshold_label", "raw_sample_records", "unique_sample_count", "duplicate_sample_records"]].to_string(index=False))

    clean_task3 = clean[clean["source"] == "task3"].copy()
    make_plots(clean, clean_task3, out_dir)
    print(f"[DONE] visual outputs in: {out_dir}")


if __name__ == "__main__":
    main()
