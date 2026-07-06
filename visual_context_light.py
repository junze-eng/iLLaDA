#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize lightweight context probe results.

Default project layout:
  input : outputs/context_light
  output: visual/context_light

It accepts either a single context_light output directory containing
predictions.jsonl, or a parent directory containing multiple context_light runs.

Main outputs:
  context_light_combined.png
  context_light_accuracy_heatmap.png
  context_light_summary_by_condition.csv / .md
  context_light_summary_overall.csv / .md
  context_light_examples.jsonl
  context_light_report.txt
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_VERSION = "visual_context_light_v1"
POSITION_ORDER = ["front", "middle", "back"]
PASS_VALUES = {"1", "true", "t", "yes", "y", "pass", "passed", "correct", "right", "ok"}
FAIL_VALUES = {"0", "false", "f", "no", "n", "fail", "failed", "incorrect", "wrong", "bad"}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[visual_context_light] skip bad jsonl line {path}:{line_no}: {exc}")
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def safe_read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def discover_runs(root: Path) -> List[Path]:
    if (root / "predictions.jsonl").exists():
        return [root]
    runs = []
    for p in sorted(root.rglob("predictions.jsonl")):
        if "visual" not in p.parts:
            runs.append(p.parent)
    if runs:
        return runs
    for p in sorted(root.rglob("summary_by_condition.csv")):
        if "visual" not in p.parts:
            runs.append(p.parent)
    return runs


def short_run_name(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
        name = rel.as_posix()
        return name if name not in ("", ".") else path.name
    except Exception:
        return path.name


def normalize_position(x: Any) -> str:
    text = str(x or "").strip().lower()
    if text in {"begin", "beginning", "start", "0", "0.0"}:
        return "front"
    if text in {"mid", "50", "50.0"}:
        return "middle"
    if text in {"end", "100", "100.0"}:
        return "back"
    return text


def effective_match(row: Dict[str, Any]) -> Optional[bool]:
    mj = str(row.get("manual_judgment", "") or "").strip().lower()
    if mj in PASS_VALUES:
        return True
    if mj in FAIL_VALUES:
        return False
    val = row.get("keyword_match")
    if isinstance(val, bool):
        return val
    if val is None:
        return None
    text = str(val).strip().lower()
    if text in PASS_VALUES:
        return True
    if text in FAIL_VALUES:
        return False
    return None


def to_float(x: Any) -> Optional[float]:
    if x in (None, ""):
        return None
    try:
        if isinstance(x, str) and x.lower() == "nan":
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def first_nonempty(series: pd.Series):
    for x in series:
        if x not in (None, "") and not (isinstance(x, float) and math.isnan(x)):
            return x
    return None


def load_predictions_run(run_dir: Path, root: Path) -> pd.DataFrame:
    run_config = safe_read_json(run_dir / "run_config.json")
    args = run_config.get("args", {}) if isinstance(run_config.get("args"), dict) else {}
    rows = read_jsonl(run_dir / "predictions.jsonl")
    out_rows = []
    for row in rows:
        match = effective_match(row)
        out_rows.append({
            "run_name": short_run_name(run_dir, root),
            "run_dir": str(run_dir),
            "sample_id": row.get("sample_id"),
            "context_length": int(row.get("context_length")) if row.get("context_length") not in (None, "") else None,
            "needle_position": normalize_position(row.get("needle_position")),
            "question": row.get("question"),
            "groundtruth": row.get("groundtruth"),
            "prediction": row.get("prediction"),
            "keyword_match": bool(row.get("keyword_match")) if row.get("keyword_match") is not None else None,
            "manual_judgment": row.get("manual_judgment", ""),
            "effective_match": match,
            "exact_match": bool(row.get("exact_match")) if row.get("exact_match") is not None else None,
            "normalized_match": bool(row.get("normalized_match")) if row.get("normalized_match") is not None else None,
            "latency_sec": to_float(row.get("latency_sec")),
            "tokens_per_second": to_float(row.get("tokens_per_second")),
            "prompt_tokens": to_float(row.get("prompt_tokens")),
            "prompt_chars": to_float(row.get("prompt_chars")),
            "max_memory_gb": to_float(row.get("max_memory_gb")),
            "gen_length": row.get("gen_length", args.get("gen_length")),
            "gen_steps": row.get("gen_steps", args.get("gen_steps")),
            "gen_blocksize": row.get("gen_blocksize", args.get("gen_blocksize")),
            "mask_id": row.get("mask_id", args.get("mask_id")),
            "threshold": args.get("token_selection_confidence_threshold"),
            "sample_selection": args.get("sample_selection"),
            "num_samples_per_condition": args.get("num_samples_per_condition"),
            "seed": args.get("seed"),
        })
    return pd.DataFrame(out_rows)


def load_summary_only_run(run_dir: Path, root: Path) -> pd.DataFrame:
    path = run_dir / "summary_by_condition.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["run_name"] = short_run_name(run_dir, root)
    df["run_dir"] = str(run_dir)
    df["needle_position"] = df["needle_position"].map(normalize_position)
    return df


def load_all(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, List[Path]]:
    runs = discover_runs(root)
    if not runs:
        raise FileNotFoundError(
            f"No context_light outputs found under {root}. Expected predictions.jsonl or summary_by_condition.csv."
        )
    pred_parts = []
    summary_parts = []
    for run in runs:
        if (run / "predictions.jsonl").exists():
            pred_parts.append(load_predictions_run(run, root))
        elif (run / "summary_by_condition.csv").exists():
            summary_parts.append(load_summary_only_run(run, root))
    pred_df = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    summary_df = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    return pred_df, summary_df, runs


def summarize_predictions(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pred.empty:
        return pd.DataFrame(), pd.DataFrame()
    work = pred.copy()
    work["context_length"] = pd.to_numeric(work["context_length"], errors="coerce").astype("Int64")
    work["needle_position"] = work["needle_position"].astype(str)
    work["effective_match_num"] = work["effective_match"].map(lambda x: np.nan if x is None else (1.0 if bool(x) else 0.0))
    work["exact_match_num"] = work["exact_match"].map(lambda x: np.nan if x is None else (1.0 if bool(x) else 0.0))
    work["normalized_match_num"] = work["normalized_match"].map(lambda x: np.nan if x is None else (1.0 if bool(x) else 0.0))

    rows = []
    for (run_name, ctx, pos), g in work.groupby(["run_name", "context_length", "needle_position"], dropna=False):
        rows.append({
            "run_name": run_name,
            "context_length": int(ctx) if pd.notna(ctx) else None,
            "needle_position": pos,
            "n": int(len(g)),
            "keyword_accuracy": float(g["effective_match_num"].mean()) if g["effective_match_num"].notna().any() else np.nan,
            "exact_match_rate": float(g["exact_match_num"].mean()) if g["exact_match_num"].notna().any() else np.nan,
            "normalized_match_rate": float(g["normalized_match_num"].mean()) if g["normalized_match_num"].notna().any() else np.nan,
            "mean_latency_sec": float(g["latency_sec"].mean()) if g["latency_sec"].notna().any() else np.nan,
            "mean_tokens_per_second": float(g["tokens_per_second"].mean()) if g["tokens_per_second"].notna().any() else np.nan,
            "mean_prompt_tokens": float(g["prompt_tokens"].mean()) if g["prompt_tokens"].notna().any() else np.nan,
            "mean_max_memory_gb": float(g["max_memory_gb"].mean()) if g["max_memory_gb"].notna().any() else np.nan,
            "gen_length": first_nonempty(g["gen_length"]),
            "gen_steps": first_nonempty(g["gen_steps"]),
            "gen_blocksize": first_nonempty(g["gen_blocksize"]),
            "threshold": first_nonempty(g["threshold"]),
        })
    by_condition = pd.DataFrame(rows)
    sort_condition(by_condition)

    overall_rows = []
    for (run_name, ctx), g in work.groupby(["run_name", "context_length"], dropna=False):
        overall_rows.append({
            "run_name": run_name,
            "context_length": int(ctx) if pd.notna(ctx) else None,
            "n": int(len(g)),
            "keyword_accuracy": float(g["effective_match_num"].mean()) if g["effective_match_num"].notna().any() else np.nan,
            "exact_match_rate": float(g["exact_match_num"].mean()) if g["exact_match_num"].notna().any() else np.nan,
            "normalized_match_rate": float(g["normalized_match_num"].mean()) if g["normalized_match_num"].notna().any() else np.nan,
            "mean_latency_sec": float(g["latency_sec"].mean()) if g["latency_sec"].notna().any() else np.nan,
            "mean_tokens_per_second": float(g["tokens_per_second"].mean()) if g["tokens_per_second"].notna().any() else np.nan,
            "mean_prompt_tokens": float(g["prompt_tokens"].mean()) if g["prompt_tokens"].notna().any() else np.nan,
            "mean_max_memory_gb": float(g["max_memory_gb"].mean()) if g["max_memory_gb"].notna().any() else np.nan,
            "gen_length": first_nonempty(g["gen_length"]),
            "gen_steps": first_nonempty(g["gen_steps"]),
            "gen_blocksize": first_nonempty(g["gen_blocksize"]),
            "threshold": first_nonempty(g["threshold"]),
        })
    overall = pd.DataFrame(overall_rows)
    sort_overall(overall)
    return by_condition, overall


def sort_condition(df: pd.DataFrame) -> None:
    if df.empty:
        return
    df["needle_position"] = pd.Categorical(df["needle_position"].astype(str), POSITION_ORDER, ordered=True)
    df.sort_values(["run_name", "context_length", "needle_position"], inplace=True)
    df["needle_position"] = df["needle_position"].astype(str)


def sort_overall(df: pd.DataFrame) -> None:
    if df.empty:
        return
    df.sort_values(["run_name", "context_length"], inplace=True)


def merge_summary_only(summary_only: pd.DataFrame, by_condition: pd.DataFrame) -> pd.DataFrame:
    if summary_only.empty:
        return by_condition
    keep = [
        "run_name", "context_length", "needle_position", "n", "keyword_accuracy",
        "exact_match_rate", "normalized_match_rate", "mean_latency_sec",
        "mean_tokens_per_second", "mean_prompt_tokens", "mean_max_memory_gb",
    ]
    cols = [c for c in keep if c in summary_only.columns]
    extra = summary_only[cols].copy()
    out = pd.concat([by_condition, extra], ignore_index=True) if not by_condition.empty else extra
    sort_condition(out)
    return out


def make_overall_from_condition(by_condition: pd.DataFrame) -> pd.DataFrame:
    if by_condition.empty:
        return pd.DataFrame()
    rows = []
    for (run, ctx), g in by_condition.groupby(["run_name", "context_length"], dropna=False):
        weights = pd.to_numeric(g["n"], errors="coerce").fillna(0)
        def wmean(col):
            vals = pd.to_numeric(g[col], errors="coerce")
            mask = vals.notna() & (weights > 0)
            if not mask.any():
                return np.nan
            return float(np.average(vals[mask], weights=weights[mask]))
        rows.append({
            "run_name": run,
            "context_length": int(ctx) if pd.notna(ctx) else None,
            "n": int(weights.sum()),
            "keyword_accuracy": wmean("keyword_accuracy"),
            "exact_match_rate": wmean("exact_match_rate"),
            "normalized_match_rate": wmean("normalized_match_rate"),
            "mean_latency_sec": wmean("mean_latency_sec"),
            "mean_tokens_per_second": wmean("mean_tokens_per_second"),
            "mean_prompt_tokens": wmean("mean_prompt_tokens"),
            "mean_max_memory_gb": wmean("mean_max_memory_gb"),
        })
    out = pd.DataFrame(rows)
    sort_overall(out)
    return out


def table_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_csv(index=False)


def set_context_axis(ax, df: pd.DataFrame) -> None:
    xs = sorted([int(x) for x in pd.to_numeric(df["context_length"], errors="coerce").dropna().unique()])
    if len(xs) >= 2 and min(xs) > 0:
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs, [str(x) for x in xs])
    elif xs:
        ax.set_xticks(xs, [str(x) for x in xs])


def plot_combined(by_condition: pd.DataFrame, overall: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [
        ("keyword_accuracy", "Accuracy"),
        ("mean_latency_sec", "Mean latency (s)"),
        ("mean_tokens_per_second", "Mean TPS"),
        ("mean_max_memory_gb", "Mean max memory (GB)"),
    ]
    run_names = sorted(by_condition["run_name"].dropna().unique().tolist()) if not by_condition.empty else []
    multi_run = len(run_names) > 1

    for ax, (metric, ylabel) in zip(axes.ravel(), metrics):
        if multi_run:
            data = overall
            for run in run_names:
                sub = data[data["run_name"] == run].copy()
                if sub.empty or metric not in sub.columns:
                    continue
                ax.plot(sub["context_length"], sub[metric], marker="o", linewidth=2, label=run)
            ax.legend(title="Run", fontsize=8)
        else:
            data = by_condition
            for pos in POSITION_ORDER:
                sub = data[data["needle_position"].astype(str) == pos].copy()
                if sub.empty or metric not in sub.columns:
                    continue
                ax.plot(sub["context_length"], sub[metric], marker="o", linewidth=2, label=pos)
            ax.legend(title="Position")

        set_context_axis(ax, data if not data.empty else by_condition)
        if metric == "keyword_accuracy":
            ax.set_ylim(0, 1.05)
        ax.set_xlabel("Context length")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " vs context length")
        ax.grid(alpha=0.3)

    title = "context_light trend summary"
    title += " (run-level comparison)" if multi_run else " (position-level comparison)"
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_accuracy_heatmap(by_condition: pd.DataFrame, overall: pd.DataFrame, out_path: Path) -> None:
    run_names = sorted(by_condition["run_name"].dropna().unique().tolist()) if not by_condition.empty else []
    if len(run_names) <= 1:
        df = by_condition.copy()
        if df.empty:
            return
        piv = df.pivot(index="needle_position", columns="context_length", values="keyword_accuracy").reindex(POSITION_ORDER)
        title = "context_light keyword accuracy heatmap"
        y_label = "Needle position"
    else:
        df = overall.copy()
        if df.empty:
            return
        piv = df.pivot(index="run_name", columns="context_length", values="keyword_accuracy")
        title = "context_light keyword accuracy heatmap by run"
        y_label = "Run"

    vals = piv.values.astype(float)
    fig_h = max(3.8, 0.42 * len(piv.index) + 2.2)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    im = ax.imshow(vals, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(piv.columns)), [str(int(c)) for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [str(i) for i in piv.index])
    ax.set_xlabel("Context length")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            label = "NA" if np.isnan(vals[i, j]) else f"{vals[i, j]:.2f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=10)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Accuracy")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def classify_failure(row: pd.Series) -> str:
    if row.get("effective_match") is True:
        return ""
    pred = str(row.get("prediction") or "")
    gold = str(row.get("groundtruth") or "")
    pl = pred.lower()
    if not pred.strip():
        return "empty"
    if any(t in pl for t in ["not found", "cannot find", "can't find", "no indication", "not mentioned", "there is no", "no specific number"]):
        return "not_found_claim"
    gold_digits = set(re.findall(r"\d+", gold))
    pred_digits = re.findall(r"\d+", pred)
    if pred_digits and gold_digits and not any(g in pred_digits for g in gold_digits):
        return "wrong_number"
    return "missing_answer"


def preview(x: Any, n: int = 300) -> str:
    s = re.sub(r"\s+", " ", str(x or "")).strip()
    return s if len(s) <= n else s[: n - 3] + "..."


def write_examples(pred: pd.DataFrame, path: Path, max_errors_per_condition: int) -> int:
    if pred.empty:
        path.write_text("", encoding="utf-8")
        return 0
    records = []
    work = pred.copy()
    work["failure_type_auto"] = work.apply(classify_failure, axis=1)
    work["needle_position"] = pd.Categorical(work["needle_position"].astype(str), POSITION_ORDER, ordered=True)
    work = work.sort_values(["run_name", "context_length", "needle_position", "sample_id"])

    for (run, ctx, pos), g in work.groupby(["run_name", "context_length", "needle_position"], observed=True, dropna=False):
        records.append({
            "record_type": "condition",
            "run_name": run,
            "context_length": int(ctx) if pd.notna(ctx) else None,
            "needle_position": str(pos),
            "n": int(len(g)),
            "accuracy": float(g["effective_match"].map(lambda x: 1 if x else 0).mean()) if len(g) else None,
        })
        correct = g[g["effective_match"] == True]
        if not correct.empty:
            r = correct.iloc[0]
            records.append({
                "record_type": "correct_example",
                "run_name": run,
                "context_length": int(ctx) if pd.notna(ctx) else None,
                "needle_position": str(pos),
                "sample_id": r.get("sample_id"),
                "groundtruth": r.get("groundtruth"),
                "question": r.get("question"),
                "prediction": r.get("prediction"),
                "prediction_preview": preview(r.get("prediction")),
            })
        errs = g[g["effective_match"] != True]
        if max_errors_per_condition and max_errors_per_condition > 0:
            errs = errs.head(max_errors_per_condition)
        for _, r in errs.iterrows():
            records.append({
                "record_type": "error_example",
                "run_name": run,
                "context_length": int(ctx) if pd.notna(ctx) else None,
                "needle_position": str(pos),
                "sample_id": r.get("sample_id"),
                "groundtruth": r.get("groundtruth"),
                "question": r.get("question"),
                "failure_type": r.get("failure_type_auto"),
                "prediction": r.get("prediction"),
                "prediction_preview": preview(r.get("prediction")),
            })
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


def write_report(out_dir: Path, runs: List[Path], by_condition: pd.DataFrame, overall: pd.DataFrame, examples_n: int) -> None:
    lines = []
    lines.append(SCRIPT_VERSION)
    lines.append("")
    lines.append("Runs:")
    for r in runs:
        lines.append(f"- {r}")
    lines.append("")
    lines.append("Summary by condition:")
    lines.append(table_markdown(by_condition.round(4) if not by_condition.empty else by_condition))
    lines.append("")
    lines.append("Summary overall:")
    lines.append(table_markdown(overall.round(4) if not overall.empty else overall))
    lines.append("")
    lines.append(f"JSONL example records: {examples_n}")
    lines.append("")
    lines.append("Generated figures:")
    lines.append("- context_light_combined.png")
    lines.append("- context_light_accuracy_heatmap.png")
    (out_dir / "context_light_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--root", default="outputs/context_light", help="context_light output dir or parent dir containing multiple runs")
    ap.add_argument("--out", default="visual/context_light", help="output directory")
    ap.add_argument("--max-errors-per-condition", type=int, default=0, help="0 keeps all errors in context_light_examples.jsonl")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred, summary_only, runs = load_all(root)
    by_condition, overall = summarize_predictions(pred)
    by_condition = merge_summary_only(summary_only, by_condition)
    if overall.empty or (not summary_only.empty and pred.empty):
        overall = make_overall_from_condition(by_condition)
    if by_condition.empty:
        raise SystemExit("No usable rows found.")

    by_condition.to_csv(out_dir / "context_light_summary_by_condition.csv", index=False)
    (out_dir / "context_light_summary_by_condition.md").write_text(table_markdown(by_condition.round(4)) + "\n", encoding="utf-8")
    overall.to_csv(out_dir / "context_light_summary_overall.csv", index=False)
    (out_dir / "context_light_summary_overall.md").write_text(table_markdown(overall.round(4)) + "\n", encoding="utf-8")

    examples_n = write_examples(pred, out_dir / "context_light_examples.jsonl", args.max_errors_per_condition)
    plot_combined(by_condition, overall, out_dir / "context_light_combined.png")
    plot_accuracy_heatmap(by_condition, overall, out_dir / "context_light_accuracy_heatmap.png")
    write_report(out_dir, runs, by_condition, overall, examples_n)

    print("\n=== context_light summary by condition ===")
    print(table_markdown(by_condition.round(4)))
    print(f"\nSaved to: {out_dir}")
    for p in sorted(out_dir.iterdir()):
        print(" -", p.name)


if __name__ == "__main__":
    main()
