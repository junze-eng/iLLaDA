#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize dynamic-threshold GSM8K experiments for iLLaDA ARness analysis.

Default input root is outputs/dynamic_threshold_gsm8k. The script also accepts a
single condition directory, the whole dynamic_threshold_gsm8k directory, or a .zip
archive exported from that directory.

It reads:
  - aggregate.csv / summary_all.csv
  - sample_traces/*/sample_metrics.json
  - sample_traces/*/step_events.csv
  - sample_traces/*/block_metrics.csv

Outputs:
  - PNG figures under <out>/figures
  - merged CSVs under <out>/tables
  - summary markdown under <out>/dynamic_threshold_visual_summary.md
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("outputs/dynamic_threshold_gsm8k")


def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path)
    except Exception as exc:
        print(f"[warn] failed to read CSV {path}: {exc}")
    return None


def safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[warn] failed to read JSON {path}: {exc}")
        return None


def parse_list_like(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return []
    try:
        return ast.literal_eval(s)
    except Exception:
        # Last-resort parser for strings like [0.8, 0.6, None]
        s2 = s.strip("[]")
        out: List[Any] = []
        for part in s2.split(","):
            p = part.strip()
            if not p:
                continue
            if p in {"None", "null", "NaN", "nan"}:
                out.append(None)
            else:
                try:
                    out.append(float(p))
                except ValueError:
                    out.append(p.strip("'\""))
        return out


def infer_steps_from_name(text: str) -> Optional[int]:
    m = re.search(r"steps(\d+)", text or "")
    return int(m.group(1)) if m else None


def infer_thr_label(text: str) -> str:
    text = text or ""
    # Match known schedule labels first. Directory names can contain many underscores,
    # so a greedy regex after "thr" would otherwise capture too much.
    known = ["high_to_low", "static_mid", "low_to_high", "none"]
    for k in known:
        if f"thr{k}" in text or k in text:
            return k
    m = re.search(r"thr([^/\\]+)", text)
    if m:
        raw = m.group(1)
        # Stop at common run-name separators if present.
        raw = re.split(r"(?:_gsm8k|_len|_block|_steps)", raw)[0]
        return raw or "unknown"
    return "unknown"


def infer_sample_idx(path: Path) -> Optional[int]:
    for part in reversed(path.parts):
        m = re.match(r"sample_(\d+)$", part)
        if m:
            return int(m.group(1))
    return None


def prepare_root(input_path: Path) -> Tuple[Path, Optional[tempfile.TemporaryDirectory]]:
    """Return readable root. If zip, extract to tempdir and return extracted top dir."""
    if input_path.suffix.lower() == ".zip":
        td = tempfile.TemporaryDirectory(prefix="dynthr_vis_")
        out = Path(td.name)
        with zipfile.ZipFile(input_path, "r") as zf:
            zf.extractall(out)
        # If exactly one directory, use it. Otherwise use temp root.
        children = [p for p in out.iterdir() if p.is_dir()]
        return (children[0] if len(children) == 1 else out), td
    return input_path, None


def collect_aggregate(root: Path) -> pd.DataFrame:
    files = sorted(root.rglob("aggregate.csv")) + sorted(root.rglob("summary_all.csv"))
    rows = []
    for f in files:
        df = safe_read_csv(f)
        if df is None or df.empty:
            continue
        df = df.copy()
        df["source_file"] = str(f)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    agg = pd.concat(rows, ignore_index=True, sort=False)

    # Normalize key columns.
    if "decoding_config_name" not in agg.columns and "run_name" in agg.columns:
        agg["decoding_config_name"] = agg["run_name"]
    if "threshold_schedule_label" not in agg.columns:
        agg["threshold_schedule_label"] = agg["decoding_config_name"].map(infer_thr_label)
    agg["threshold_schedule_label"] = agg["threshold_schedule_label"].fillna(
        agg.get("decoding_config_name", pd.Series([""] * len(agg))).map(infer_thr_label)
    )
    if "gen_steps" not in agg.columns:
        agg["gen_steps"] = agg.get("decoding_config_name", pd.Series([""] * len(agg))).map(infer_steps_from_name)
    else:
        agg["gen_steps"] = pd.to_numeric(agg["gen_steps"], errors="coerce")
        missing = agg["gen_steps"].isna()
        if missing.any():
            agg.loc[missing, "gen_steps"] = agg.loc[missing, "decoding_config_name"].map(infer_steps_from_name)

    # Remove failed placeholder rows and duplicates. Some aggregate.csv files contain
    # a first failed row with num_samples=0/returncode=1 and a second successful row.
    if "num_samples" in agg.columns:
        agg["num_samples"] = pd.to_numeric(agg["num_samples"], errors="coerce")
    if "returncode" in agg.columns:
        agg["returncode"] = pd.to_numeric(agg["returncode"], errors="coerce")
    if "primary_metric_value" in agg.columns:
        agg["primary_metric_value"] = pd.to_numeric(agg["primary_metric_value"], errors="coerce")

    # Prefer successful, non-empty rows when duplicated.
    agg["_success_pref"] = 0
    if "returncode" in agg.columns:
        agg["_success_pref"] += (agg["returncode"].fillna(999) == 0).astype(int) * 10
    if "num_samples" in agg.columns:
        agg["_success_pref"] += agg["num_samples"].fillna(0).clip(lower=0)
    if "primary_metric_value" in agg.columns:
        agg["_success_pref"] += agg["primary_metric_value"].notna().astype(int) * 5
    agg = agg.sort_values("_success_pref", ascending=False)

    dedupe_cols = [c for c in ["run_name", "decoding_config_name", "gen_steps", "threshold_schedule_label"] if c in agg.columns]
    if dedupe_cols:
        agg = agg.drop_duplicates(subset=dedupe_cols, keep="first")
    agg = agg.drop(columns=["_success_pref"], errors="ignore")
    return agg.reset_index(drop=True)


def collect_sample_metrics(root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for f in sorted(root.rglob("sample_metrics.json")):
        obj = safe_read_json(f)
        if not obj:
            continue
        row = dict(obj)
        row["sample_trace_dir"] = str(f.parent)
        row["condition_dir"] = str(f.parents[2]) if len(f.parents) >= 3 else str(f.parent)
        if row.get("sample_idx") is None:
            row["sample_idx"] = infer_sample_idx(f)
        name = str(row.get("decoding_config_name") or f)
        row.setdefault("threshold_schedule_label", infer_thr_label(name))
        row.setdefault("gen_steps", infer_steps_from_name(name))
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Fill labels from parent directory when missing.
    if "threshold_schedule_label" in df.columns:
        bad = df["threshold_schedule_label"].isna() | (df["threshold_schedule_label"].astype(str).isin(["", "nan", "None", "unknown"]))
        df.loc[bad, "threshold_schedule_label"] = df.loc[bad, "sample_trace_dir"].map(infer_thr_label)
    for col in ["gen_steps", "sample_idx", "completion_rate", "actual_parallelism", "actual_arness", "threshold_pass_rate", "fallback_rate", "final_mask_count", "tokens_per_second", "elapsed_seconds"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def collect_step_events(root: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(root.rglob("step_events.csv")):
        df = safe_read_csv(f)
        if df is None or df.empty:
            continue
        sample_idx = infer_sample_idx(f)
        parent_text = str(f)
        df = df.copy()
        df["sample_idx"] = sample_idx
        df["sample_trace_dir"] = str(f.parent)
        df["condition_dir"] = str(f.parents[2]) if len(f.parents) >= 3 else str(f.parent)
        df["threshold_schedule_label"] = infer_thr_label(parent_text)
        df["gen_steps"] = infer_steps_from_name(parent_text)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True, sort=False)
    for col in ["global_step_idx", "block_idx", "block_local_round", "selected_count", "mean_selected_confidence", "min_selected_confidence", "max_selected_confidence", "block_completion_rate", "mask_count_before", "mask_count_after", "scheduled_transfer_count", "threshold_passed_count", "fallback_forced_count", "actual_transfer_count", "current_completion_rate", "cumulative_transferred_tokens", "gen_steps"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def collect_block_metrics(root: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(root.rglob("block_metrics.csv")):
        df = safe_read_csv(f)
        if df is None or df.empty:
            continue
        sample_idx = infer_sample_idx(f)
        parent_text = str(f)
        df = df.copy()
        df["sample_idx"] = sample_idx
        df["sample_trace_dir"] = str(f.parent)
        df["condition_dir"] = str(f.parents[2]) if len(f.parents) >= 3 else str(f.parent)
        df["threshold_schedule_label"] = infer_thr_label(parent_text)
        df["gen_steps"] = infer_steps_from_name(parent_text)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True, sort=False)
    for col in ["block_idx", "block_length", "local_rounds", "selected_tokens_total", "final_completion_rate", "final_visible_tokens", "final_mask_tokens", "threshold_pass_steps", "fallback_steps", "mean_selected_count_per_round", "sample_idx", "gen_steps"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def label_condition(row: pd.Series) -> str:
    steps = int(row["gen_steps"]) if pd.notna(row.get("gen_steps")) else "?"
    thr = str(row.get("threshold_schedule_label", "unknown"))
    return f"steps{steps}\n{thr}"


def condition_order(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return []
    tmp = df[["gen_steps", "threshold_schedule_label"]].drop_duplicates().copy()
    # Prefer larger steps first; within same steps, high_to_low then static_mid then others.
    pref = {"high_to_low": 0, "static_mid": 1, "low_to_high": 2, "none": 3, "unknown": 9}
    tmp["_pref"] = tmp["threshold_schedule_label"].map(lambda x: pref.get(str(x), 5))
    tmp = tmp.sort_values(["gen_steps", "_pref"], ascending=[False, True])
    return [f"steps{int(r.gen_steps)}\n{r.threshold_schedule_label}" for r in tmp.itertuples()]


def ensure_out_dirs(out: Path) -> Tuple[Path, Path]:
    figs = out / "figures"
    tables = out / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    return figs, tables


def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[ok] wrote {path}")


def plot_aggregate_overview(agg: pd.DataFrame, figs: Path):
    if agg.empty:
        return
    df = agg.copy()
    df["condition"] = df.apply(label_condition, axis=1)
    order = condition_order(df)
    df["condition"] = pd.Categorical(df["condition"], categories=order, ordered=True)
    df = df.sort_values("condition")

    metrics = [
        ("primary_metric_value", "Accuracy", "%"),
        ("completion_rate", "Completion rate", ""),
        ("actual_parallelism", "Actual transfer tokens/step", ""),
        ("tokens_per_second_mean", "Wall-clock tokens/sec", ""),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()
    for ax, (col, title, suffix) in zip(axes, metrics):
        if col not in df.columns:
            ax.axis("off")
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        ax.bar(df["condition"].astype(str), vals)
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=30)
        ax.grid(axis="y", alpha=0.25)
        if suffix == "%":
            ax.set_ylabel("Percent")
        elif col == "completion_rate":
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Fraction")
    fig.suptitle("Dynamic threshold GSM8K: aggregate overview", y=1.02, fontsize=14)
    savefig(figs / "01_aggregate_overview.png")


def plot_tradeoff(agg: pd.DataFrame, figs: Path):
    if agg.empty or "primary_metric_value" not in agg.columns:
        return
    df = agg.copy()
    df["condition"] = df.apply(label_condition, axis=1)
    xcol = "actual_parallelism" if "actual_parallelism" in df.columns else "effective_parallelism"
    if xcol not in df.columns:
        return
    plt.figure(figsize=(8, 6))
    for label, sub in df.groupby("threshold_schedule_label"):
        sub = sub.sort_values(xcol)
        plt.plot(pd.to_numeric(sub[xcol], errors="coerce"), pd.to_numeric(sub["primary_metric_value"], errors="coerce"), marker="o", label=str(label))
        for _, r in sub.iterrows():
            plt.annotate(f"s{int(r['gen_steps'])}", (float(r[xcol]), float(r["primary_metric_value"])), textcoords="offset points", xytext=(4, 4), fontsize=8)
    plt.xlabel("Actual transfer tokens per step")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs actual parallelism")
    plt.grid(alpha=0.25)
    plt.legend(title="Threshold schedule")
    savefig(figs / "02_accuracy_parallelism_tradeoff.png")


def plot_sample_heatmaps(metrics: pd.DataFrame, figs: Path):
    if metrics.empty:
        return
    df = metrics.copy()
    df["condition"] = df.apply(label_condition, axis=1)
    order = condition_order(df)
    samples = sorted([int(x) for x in df["sample_idx"].dropna().unique()])
    plot_specs = [
        ("completion_rate", "Final completion rate", 0, 1),
        ("threshold_pass_rate", "Threshold pass rate", 0, 1),
        ("fallback_rate", "Fallback rate", 0, 1),
        ("actual_parallelism", "Actual transfer tokens/step", None, None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.ravel()
    for ax, (col, title, vmin, vmax) in zip(axes, plot_specs):
        if col not in df.columns:
            ax.axis("off")
            continue
        mat = df.pivot_table(index="sample_idx", columns="condition", values=col, aggfunc="mean")
        mat = mat.reindex(index=samples, columns=order)
        im = ax.imshow(mat.values, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_yticks(range(len(samples)))
        ax.set_yticklabels(samples)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=30, ha="right")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat.values[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Per-sample dynamic-threshold diagnostics", y=1.02, fontsize=14)
    savefig(figs / "03_sample_metric_heatmaps.png")


def plot_block_completion(blocks: pd.DataFrame, figs: Path):
    if blocks.empty or "final_completion_rate" not in blocks.columns:
        return
    df = blocks.copy()
    df["condition"] = df.apply(label_condition, axis=1)
    order = condition_order(df)
    mat = df.pivot_table(index="condition", columns="block_idx", values="final_completion_rate", aggfunc="mean")
    mat = mat.reindex(index=order)
    plt.figure(figsize=(12, max(3.5, 0.55 * len(mat))))
    im = plt.imshow(mat.values, aspect="auto", vmin=0, vmax=1)
    plt.title("Mean final block completion by condition")
    plt.xlabel("Block index")
    plt.ylabel("Condition")
    plt.xticks(range(mat.shape[1]), [int(c) for c in mat.columns])
    plt.yticks(range(mat.shape[0]), mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.values[i, j]
            if pd.notna(val):
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, label="Completion rate")
    savefig(figs / "04_block_completion_heatmap.png")


def plot_step_dynamics(steps: pd.DataFrame, figs: Path):
    if steps.empty:
        return
    df = steps.copy()
    needed = {"gen_steps", "threshold_schedule_label", "global_step_idx"}
    if not needed.issubset(df.columns):
        return
    # Use normalized progress so 64-step and 128-step runs are comparable.
    df["step_progress"] = df["global_step_idx"] / df.groupby(["gen_steps", "threshold_schedule_label", "sample_idx"])["global_step_idx"].transform("max").replace(0, np.nan)
    df["progress_bin"] = (df["step_progress"] * 20).round() / 20
    group_cols = ["gen_steps", "threshold_schedule_label", "progress_bin"]
    agg = df.groupby(group_cols, dropna=False).agg(
        current_completion_rate=("current_completion_rate", "mean"),
        actual_transfer_count=("actual_transfer_count", "mean"),
        fallback_forced_count=("fallback_forced_count", "mean"),
        threshold_passed_count=("threshold_passed_count", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    specs = [
        ("current_completion_rate", "Mean current completion"),
        ("actual_transfer_count", "Mean actual transfer count"),
        ("fallback_forced_count", "Mean fallback-forced count"),
        ("threshold_passed_count", "Mean threshold-passed count"),
    ]
    for ax, (col, title) in zip(axes.ravel(), specs):
        for (gsteps, label), sub in agg.groupby(["gen_steps", "threshold_schedule_label"]):
            sub = sub.sort_values("progress_bin")
            ax.plot(sub["progress_bin"], sub[col], marker="o", markersize=3, label=f"steps{int(gsteps)} {label}")
        ax.set_title(title)
        ax.set_xlabel("Normalized generation progress")
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Fraction")
    axes[0, 1].set_ylabel("Tokens")
    axes[1, 0].set_ylabel("Tokens")
    axes[1, 1].set_ylabel("Tokens")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Step-level dynamics averaged across samples", y=1.02, fontsize=14)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(figs / "05_step_dynamics.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[ok] wrote {figs / '05_step_dynamics.png'}")


def plot_per_sample_completion(steps: pd.DataFrame, figs: Path, max_samples: int = 12):
    if steps.empty or "current_completion_rate" not in steps.columns:
        return
    samples = sorted([int(x) for x in steps["sample_idx"].dropna().unique()])[:max_samples]
    for sample in samples:
        sub0 = steps[steps["sample_idx"] == sample].copy()
        if sub0.empty:
            continue
        plt.figure(figsize=(10, 5))
        for (gsteps, label), sub in sub0.groupby(["gen_steps", "threshold_schedule_label"]):
            sub = sub.sort_values("global_step_idx")
            plt.plot(sub["global_step_idx"], sub["current_completion_rate"], label=f"steps{int(gsteps)} {label}")
        plt.title(f"Sample {sample}: completion over denoising steps")
        plt.xlabel("Global denoising step")
        plt.ylabel("Current completion rate")
        plt.ylim(0, 1.05)
        plt.grid(alpha=0.25)
        plt.legend()
        savefig(figs / f"06_sample_{sample:04d}_completion_curve.png")


def write_summary(out: Path, agg: pd.DataFrame, metrics: pd.DataFrame):
    lines = ["# Dynamic threshold GSM8K visualization summary", ""]
    if not agg.empty:
        cols = [c for c in ["gen_steps", "threshold_schedule_label", "primary_metric_value", "completion_rate", "actual_parallelism", "actual_arness_mean", "tokens_per_second_mean"] if c in agg.columns]
        lines.append("## Aggregate conditions")
        lines.append(agg[cols].sort_values(["gen_steps", "threshold_schedule_label"], ascending=[False, True]).to_markdown(index=False))
        lines.append("")
    if not metrics.empty:
        lines.append("## Per-sample means by condition")
        group = metrics.groupby(["gen_steps", "threshold_schedule_label"], dropna=False).agg(
            n=("sample_idx", "count"),
            completion_rate=("completion_rate", "mean"),
            final_mask_count=("final_mask_count", "mean"),
            actual_parallelism=("actual_parallelism", "mean"),
            threshold_pass_rate=("threshold_pass_rate", "mean"),
            fallback_rate=("fallback_rate", "mean"),
        ).reset_index()
        lines.append(group.to_markdown(index=False))
        lines.append("")
    lines.append("## Figure guide")
    lines.append("- `01_aggregate_overview.png`: performance, completion, actual parallelism, and wall-clock throughput.")
    lines.append("- `02_accuracy_parallelism_tradeoff.png`: whether more actual parallel transfer trades off against accuracy.")
    lines.append("- `03_sample_metric_heatmaps.png`: sample-level sensitivity and unfinished-generation pattern.")
    lines.append("- `04_block_completion_heatmap.png`: which blocks remain incomplete under each threshold schedule.")
    lines.append("- `05_step_dynamics.png`: average generation dynamics across normalized denoising progress.")
    lines.append("- `06_sample_XXXX_completion_curve.png`: per-sample completion trajectories.")
    (out / "dynamic_threshold_visual_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] wrote {out / 'dynamic_threshold_visual_summary.md'}")


def main():
    parser = argparse.ArgumentParser(description="Visualize dynamic-threshold GSM8K ARness traces.")
    parser.add_argument("path", nargs="?", default=str(DEFAULT_ROOT), help="Input root/condition directory or .zip archive. Default: outputs/dynamic_threshold_gsm8k")
    parser.add_argument("--out", default=None, help="Output directory. Default: <input_root>/visual_dynamic_threshold or ./visual_dynamic_threshold for zip")
    parser.add_argument("--no-sample-curves", action="store_true", help="Skip per-sample completion curves.")
    args = parser.parse_args()

    input_path = Path(args.path)
    root, tempdir = prepare_root(input_path)
    try:
        if not root.exists():
            raise FileNotFoundError(f"Input path does not exist: {root}")
        if args.out:
            out = Path(args.out)
        elif input_path.suffix.lower() == ".zip":
            out = Path.cwd() / "visual_dynamic_threshold"
        else:
            out = root / "visual_dynamic_threshold"
        figs, tables = ensure_out_dirs(out)

        agg = collect_aggregate(root)
        metrics = collect_sample_metrics(root)
        steps = collect_step_events(root)
        blocks = collect_block_metrics(root)

        if agg.empty and metrics.empty and steps.empty and blocks.empty:
            raise RuntimeError(f"No recognizable dynamic-threshold artifacts found under {root}")

        if not agg.empty:
            agg.to_csv(tables / "merged_aggregate.csv", index=False)
        if not metrics.empty:
            metrics.to_csv(tables / "merged_sample_metrics.csv", index=False)
        if not steps.empty:
            steps.to_csv(tables / "merged_step_events.csv", index=False)
        if not blocks.empty:
            blocks.to_csv(tables / "merged_block_metrics.csv", index=False)

        plot_aggregate_overview(agg, figs)
        plot_tradeoff(agg, figs)
        plot_sample_heatmaps(metrics, figs)
        plot_block_completion(blocks, figs)
        plot_step_dynamics(steps, figs)
        if not args.no_sample_curves:
            plot_per_sample_completion(steps, figs)
        write_summary(out, agg, metrics)
        print(f"\nDone. Open: {out}")
    finally:
        if tempdir is not None:
            tempdir.cleanup()


if __name__ == "__main__":
    main()
