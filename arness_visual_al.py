#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arness_visual_al.py

Read an iLLaDA ARness run root such as outputs/arness_all, select the 16 Task3
conditions from the config, de-duplicate repeated appended runs, and write
plot-ready CSV + figures to visual/arness_all.

Usage:
  python arness_visual_al.py
  python arness_visual_al.py --root outputs/arness_all --config test_config.yaml --out visual/arness_all

Design notes:
- A "condition" is benchmark × gen_steps × token_selection_confidence_threshold.
- Some run folders may contain duplicated summary.jsonl rows because the same
  condition was re-run into the same work_dir. This script counts both raw
  records and unique sample_idx, then uses unique_sample_count for audit.
- Figures are generated from condition-level metrics; official score is taken
  from aggregate.csv / run_summary.csv, while generation diagnostics are taken
  from the selected condition row.
"""

from __future__ import annotations

import argparse
import csv
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


def latest_summary_csv_value(run_dir: Path) -> Tuple[Optional[str], Optional[float], Optional[Path]]:
    """Find the latest OpenCompass summary csv under a run dir and return a best metric."""
    candidates = sorted(run_dir.glob("**/summary/summary_*.csv"))
    if not candidates:
        return None, None, None

    # Last lexical timestamp is latest.
    csv_path = candidates[-1]
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None, None, csv_path

    # OpenCompass summaries vary. Try common forms.
    # Case 1: columns include dataset/version/metric columns.
    metric_candidates = [
        "accuracy", "score", "pass", "acc", "gsm8k", "mbpp",
    ]
    numeric_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)

    # Prefer a column that looks like final metric.
    for col in df.columns:
        if str(col).lower() in metric_candidates:
            val = num_or_nan(df[col].iloc[0])
            if not math.isnan(val):
                return col, float(val), csv_path

    # Otherwise choose the last numeric column.
    for col in reversed(numeric_cols):
        val = num_or_nan(df[col].iloc[0])
        if not math.isnan(val):
            return col, float(val), csv_path

    return None, None, csv_path


def infer_run_dir(root: Path, run_name: str) -> Optional[Path]:
    if not run_name:
        return None
    direct = root / run_name
    if direct.exists():
        return direct
    matches = [p for p in root.rglob(run_name) if p.is_dir()]
    if matches:
        return matches[0]
    # Fallback: find dir containing run_name as prefix.
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(run_name)]
    return candidates[0] if candidates else None


# -----------------------------
# Config expected-condition parsing
# -----------------------------

@dataclass(frozen=True)
class ExpectedCondition:
    experiment: str
    benchmark: str
    gen_length: int
    gen_steps: int
    threshold_label: str
    threshold_value: Optional[float]
    sample_limit: Optional[int]

    @property
    def key(self) -> Tuple[str, int, str]:
        return (self.benchmark, self.gen_steps, self.threshold_label)


def load_expected_task3_conditions(config_path: Path) -> List[ExpectedCondition]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    defaults = cfg.get("defaults") or {}
    task = (cfg.get("tasks") or {}).get("arness")
    if not task:
        raise ValueError(f"No tasks.arness found in {config_path}")

    expected: List[ExpectedCondition] = []
    for exp in task.get("experiments", []):
        name = str(exp.get("name") or "")
        if not name.startswith("task3_"):
            continue

        for bench in as_list(exp.get("benchmark")):
            for params in expand_experiment(exp, defaults):
                gen_length = int(params.get("gen_length"))
                gen_steps = int(params.get("gen_steps"))
                th = norm_float_or_none(params.get("token_selection_confidence_threshold"))
                sample_limit_raw = params.get("sample_limit", params.get("num_samples"))
                sample_limit = int(sample_limit_raw) if sample_limit_raw not in (None, "") else None
                expected.append(
                    ExpectedCondition(
                        experiment=name,
                        benchmark=str(bench),
                        gen_length=gen_length,
                        gen_steps=gen_steps,
                        threshold_label=threshold_label(th),
                        threshold_value=th,
                        sample_limit=sample_limit,
                    )
                )

    # De-duplicate while preserving order.
    seen = set()
    deduped: List[ExpectedCondition] = []
    for c in expected:
        full_key = (c.experiment, c.benchmark, c.gen_length, c.gen_steps, c.threshold_label)
        if full_key not in seen:
            deduped.append(c)
            seen.add(full_key)
    return deduped


# -----------------------------
# Condition data collection
# -----------------------------

def normalize_aggregate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize aggregate/run_summary column names produced by different code versions."""
    df = df.copy()

    # Prefer param_* if plain column is missing.
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
        "cuda_max_memory_allocated_mb": ["cuda_max_memory_allocated_mb"],
        "cuda_max_memory_reserved_mb": ["cuda_max_memory_reserved_mb"],
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
    }

    for dst, srcs in pairs.items():
        if dst in df.columns:
            continue
        for src in srcs:
            if src in df.columns:
                df[dst] = df[src]
                break

    for c in [
        "gen_length", "gen_steps", "gen_blocksize", "sample_limit", "num_samples",
        "latency_mean_s", "tokens_per_second_mean", "visible_tps_mean",
        "actual_commit_tps_mean", "completion_rate", "actual_parallelism",
        "actual_arness_mean", "threshold_pass_rate_mean", "fallback_rate_mean",
        "cuda_max_memory_allocated_mb", "cuda_max_memory_reserved_mb",
        "gpu_util_mean", "gpu_util_max", "gpu_memory_used_max_mb",
        "gpu_power_mean_w", "gpu_temperature_max_c", "primary_metric_value",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "threshold" in df.columns:
        df["threshold_label"] = df["threshold"].apply(threshold_label)
    else:
        df["threshold_label"] = "none"

    return df


def load_aggregate_rows(root: Path) -> pd.DataFrame:
    paths = []
    # Prefer root aggregate. If absent, also read nested run_summary/aggregate.
    if (root / "aggregate.csv").exists():
        paths.append(root / "aggregate.csv")
    if (root / "run_summary.csv").exists():
        paths.append(root / "run_summary.csv")

    for p in sorted(root.glob("task3_*/run_summary.csv")):
        paths.append(p)
    for p in sorted(root.glob("task3_*/aggregate.csv")):
        paths.append(p)

    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df["source_csv"] = str(p)
            frames.append(df)
        except Exception as e:
            print(f"[WARN] failed reading {p}: {e}")

    if not frames:
        raise FileNotFoundError(f"No aggregate.csv / run_summary.csv found under {root}")

    df = pd.concat(frames, ignore_index=True)
    df = normalize_aggregate_columns(df)

    # Keep only task3 rows.
    mask = pd.Series(False, index=df.index)
    if "experiment" in df.columns:
        mask |= df["experiment"].astype(str).str.startswith("task3_")
    if "run_name" in df.columns:
        mask |= df["run_name"].astype(str).str.startswith("task3_")
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
        raw_counts.append(raw if raw else int(row.get("num_samples") or 0))
        unique_counts.append(unique if unique else int(row.get("num_samples") or 0))
        duplicate_counts.append(max(0, (raw if raw else 0) - (unique if unique else 0)))

        if run_dir:
            m_name, m_val, m_path = latest_summary_csv_value(run_dir)
        else:
            m_name, m_val, m_path = None, None, None
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
    """Select one row for each expected condition.

    If a repeated run produced both num_samples=10 and num_samples=20 rows but only
    10 unique sample IDs, prefer the row whose num_samples matches unique_sample_count.
    """
    rows = []
    for cond in expected:
        sub = df[
            (df["benchmark"].astype(str) == cond.benchmark)
            & (df["gen_steps"].astype(float) == float(cond.gen_steps))
            & (df["threshold_label"].astype(str) == cond.threshold_label)
        ].copy()

        if sub.empty:
            continue

        # Score row preference:
        # 1) row num_samples closest to unique_sample_count
        # 2) largest unique sample count
        # 3) non-null primary metric
        # 4) first row stable
        def rank_tuple(r: pd.Series) -> Tuple[float, float, int, float]:
            unique = num_or_nan(r.get("unique_sample_count"))
            nrow = num_or_nan(r.get("num_samples"))
            if math.isnan(unique):
                unique = cond.sample_limit or 0
            if math.isnan(nrow):
                nrow = unique
            metric_missing = 1 if math.isnan(num_or_nan(r.get("primary_metric_value"))) else 0
            return (
                abs(nrow - unique),
                -unique,
                metric_missing,
                nrow,
            )

        sub["_rank"] = sub.apply(rank_tuple, axis=1)
        sub = sub.sort_values("_rank")
        chosen = sub.iloc[0].copy()

        # Fill stable condition fields from config.
        chosen["expected_experiment"] = cond.experiment
        chosen["benchmark"] = cond.benchmark
        chosen["gen_length"] = cond.gen_length
        chosen["gen_steps"] = cond.gen_steps
        chosen["threshold_label"] = cond.threshold_label
        chosen["threshold_value"] = cond.threshold_value if cond.threshold_value is not None else ""
        chosen["expected_sample_limit"] = cond.sample_limit if cond.sample_limit is not None else ""

        # If official metric missing in aggregate but latest OC summary has metric, use it.
        if math.isnan(num_or_nan(chosen.get("primary_metric_value"))) and not math.isnan(num_or_nan(chosen.get("latest_oc_metric_value"))):
            chosen["primary_metric_name"] = chosen.get("latest_oc_metric_name") or "latest_oc_metric"
            chosen["primary_metric_value"] = chosen.get("latest_oc_metric_value")

        rows.append(chosen.drop(labels=["_rank"], errors="ignore"))

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)

    # Derived fields.
    out["planned_parallelism"] = out["gen_length"].astype(float) / out["gen_steps"].astype(float)
    if "actual_parallelism" in out.columns:
        out["actual_parallelism_gap"] = out["planned_parallelism"] - pd.to_numeric(out["actual_parallelism"], errors="coerce")
    else:
        out["actual_parallelism_gap"] = float("nan")

    return out


# -----------------------------
# Plotting
# -----------------------------

def save_line_plot(df: pd.DataFrame, metric: str, ylabel: str, out_path: Path, title_prefix: str) -> None:
    if metric not in df.columns:
        print(f"[WARN] skip {metric}: missing column")
        return

    for bench in sorted(df["benchmark"].dropna().unique()):
        sub = df[df["benchmark"] == bench].copy()
        if sub.empty:
            continue

        plt.figure(figsize=(7.5, 4.8))
        for th in sorted(sub["threshold_label"].astype(str).unique(), key=lambda x: (x == "none", x)):
            s = sub[sub["threshold_label"].astype(str) == th].sort_values("gen_steps")
            plt.plot(s["gen_steps"], s[metric], marker="o", label=f"threshold={th}")
        plt.gca().invert_xaxis()
        plt.xlabel("Diffusion steps, lower means more parallel")
        plt.ylabel(ylabel)
        plt.title(f"{title_prefix}: {bench}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        path = out_path.with_name(f"{out_path.stem}_{bench}{out_path.suffix}")
        plt.savefig(path, dpi=220)
        plt.close()
        print(f"[OK] wrote {path}")


def save_planned_vs_actual(df: pd.DataFrame, out_dir: Path) -> None:
    if "actual_parallelism" not in df.columns:
        return
    for bench in sorted(df["benchmark"].dropna().unique()):
        sub = df[df["benchmark"] == bench].copy()
        plt.figure(figsize=(6.2, 5.2))
        for th in sorted(sub["threshold_label"].astype(str).unique(), key=lambda x: (x == "none", x)):
            s = sub[sub["threshold_label"].astype(str) == th].sort_values("planned_parallelism")
            plt.plot(s["planned_parallelism"], s["actual_parallelism"], marker="o", label=f"threshold={th}")
            for _, r in s.iterrows():
                plt.annotate(str(int(r["gen_steps"])), (r["planned_parallelism"], r["actual_parallelism"]), fontsize=8)
        plt.xlabel("Planned parallelism = gen_length / gen_steps")
        plt.ylabel("Actual parallelism")
        plt.title(f"Planned vs actual parallelism: {bench}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        path = out_dir / f"planned_vs_actual_parallelism_{bench}.png"
        plt.savefig(path, dpi=220)
        plt.close()
        print(f"[OK] wrote {path}")


def save_speed_quality(df: pd.DataFrame, out_dir: Path) -> None:
    if "primary_metric_value" not in df.columns or "latency_mean_s" not in df.columns:
        return
    for bench in sorted(df["benchmark"].dropna().unique()):
        sub = df[df["benchmark"] == bench].copy()
        plt.figure(figsize=(6.2, 5.2))
        for th in sorted(sub["threshold_label"].astype(str).unique(), key=lambda x: (x == "none", x)):
            s = sub[sub["threshold_label"].astype(str) == th].sort_values("latency_mean_s")
            plt.plot(s["latency_mean_s"], s["primary_metric_value"], marker="o", label=f"threshold={th}")
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


def make_plots(clean: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_specs = [
        ("primary_metric_value", "Score / accuracy", "score_vs_steps", "Task3 score vs ARness"),
        ("latency_mean_s", "Latency per sample (s)", "latency_vs_steps", "Task3 latency vs ARness"),
        ("tokens_per_second_mean", "Tokens per second", "tps_vs_steps", "Task3 throughput vs ARness"),
        ("completion_rate", "Completion rate", "completion_vs_steps", "Task3 completion vs ARness"),
        ("actual_parallelism", "Actual parallelism", "actual_parallelism_vs_steps", "Task3 actual parallelism vs ARness"),
        ("threshold_pass_rate_mean", "Threshold pass rate", "threshold_pass_vs_steps", "Task3 threshold pass vs ARness"),
        ("fallback_rate_mean", "Fallback rate", "fallback_vs_steps", "Task3 fallback vs ARness"),
        ("cuda_max_memory_allocated_mb", "CUDA max allocated memory (MB)", "cuda_memory_vs_steps", "Task3 CUDA memory vs ARness"),
    ]

    for metric, ylabel, stem, title in metric_specs:
        save_line_plot(clean, metric, ylabel, out_dir / f"{stem}.png", title)

    save_planned_vs_actual(clean, out_dir)
    save_speed_quality(clean, out_dir)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/arness_all", help="Root run folder renamed from context_full.")
    parser.add_argument("--config", default="test_config.yaml", help="Config file containing tasks.arness.")
    parser.add_argument("--out", default="visual/arness_all", help="Output visualization folder.")
    args = parser.parse_args()

    root = Path(args.root)
    config_path = Path(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    expected = load_expected_task3_conditions(config_path)
    expected_df = pd.DataFrame([c.__dict__ for c in expected])
    expected_df["condition_key"] = expected_df.apply(lambda r: f"{r['benchmark']}|{r['gen_steps']}|{r['threshold_label']}", axis=1)
    expected_df.to_csv(out_dir / "expected_task3_conditions.csv", index=False)

    print(f"[INFO] expected Task3 conditions from config: {len(expected)}")
    if len(expected) != 16:
        print(f"[WARN] expected condition count is {len(expected)}, not 16. Check config if this is unexpected.")

    raw = load_aggregate_rows(root)
    raw = add_sample_audit(root, raw)
    raw.to_csv(out_dir / "task3_raw_rows_with_audit.csv", index=False)
    print(f"[INFO] raw task3 aggregate rows: {len(raw)}")

    clean = select_one_row_per_condition(raw, expected)
    if clean.empty:
        raise SystemExit("[ERROR] No matching Task3 rows found. Check --root and --config.")

    # Reorder useful columns.
    preferred_cols = [
        "expected_experiment", "run_name", "experiment", "benchmark",
        "gen_length", "gen_steps", "threshold_label", "threshold_value",
        "expected_sample_limit", "num_samples", "raw_sample_records",
        "unique_sample_count", "duplicate_sample_records",
        "primary_metric_name", "primary_metric_value",
        "planned_parallelism", "actual_parallelism", "actual_parallelism_gap",
        "actual_arness_mean", "completion_rate", "threshold_pass_rate_mean",
        "fallback_rate_mean", "latency_mean_s", "tokens_per_second_mean",
        "visible_tps_mean", "actual_commit_tps_mean",
        "cuda_max_memory_allocated_mb", "cuda_max_memory_reserved_mb",
        "gpu_util_mean", "gpu_util_max", "gpu_memory_used_max_mb",
        "gpu_power_mean_w", "gpu_temperature_max_c",
        "returncode", "run_dir", "source_csv", "latest_oc_summary_csv",
    ]
    for col in preferred_cols:
        if col not in clean.columns:
            clean[col] = ""

    clean = clean[preferred_cols + [c for c in clean.columns if c not in preferred_cols]]
    clean = clean.sort_values(["benchmark", "gen_steps", "threshold_label"], ascending=[True, False, True])

    clean_path = out_dir / "task3_arness_clean.csv"
    audit_path = out_dir / "task3_sample_count_audit.csv"
    clean.to_csv(clean_path, index=False)
    clean[[
        "benchmark", "gen_steps", "threshold_label", "run_name",
        "num_samples", "raw_sample_records", "unique_sample_count",
        "duplicate_sample_records", "run_dir"
    ]].to_csv(audit_path, index=False)

    # Missing expected conditions.
    got_keys = set(zip(clean["benchmark"].astype(str), clean["gen_steps"].astype(int), clean["threshold_label"].astype(str)))
    missing = []
    for c in expected:
        if c.key not in got_keys:
            missing.append(c.__dict__)
    pd.DataFrame(missing).to_csv(out_dir / "missing_task3_conditions.csv", index=False)

    print(f"[OK] wrote {clean_path} rows={len(clean)}")
    print(f"[OK] wrote {audit_path}")
    if missing:
        print(f"[WARN] missing expected conditions: {len(missing)} -> {out_dir / 'missing_task3_conditions.csv'}")
    else:
        print("[OK] all expected Task3 conditions found")

    # Important audit message for duplicated reruns.
    dup = clean[pd.to_numeric(clean["duplicate_sample_records"], errors="coerce").fillna(0) > 0]
    if not dup.empty:
        print("[WARN] Some run dirs contain duplicated summary rows from reruns.")
        print("       Figures use one condition row each; unique_sample_count reports real unique sample_idx count.")
        print(dup[["benchmark", "gen_steps", "threshold_label", "raw_sample_records", "unique_sample_count", "duplicate_sample_records"]].to_string(index=False))

    make_plots(clean, out_dir)
    print(f"[DONE] visual outputs in: {out_dir}")


if __name__ == "__main__":
    main()
