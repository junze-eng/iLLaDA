#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""native_visual.py

Unified visualizer for native_model.py + native_output.py artifacts.

The goal is to make visualization match the native_output layout:

  native_outputs/<model>/native_output_manifest.csv
  native_outputs/<model>/<task>/<benchmark>/<condition>/metrics.json
  native_outputs/<model>/<task>/<benchmark>/<condition>/scores.csv
  native_outputs/<model>/<task>/<benchmark>/<condition>/sample_traces/...
  native_outputs/_compare/<task>/<benchmark>/<condition>/compare.csv

Modes:
  --m parallel   Non-context task quality/speed plots from native_output manifests.
  --m context    RULER/NIAH double-needle plots from native_output manifests.
  --m arness     Trace-order / ARness plots from sample_traces.
  --m all        Run all three modes.

Examples:
  python native_visual.py --m context --root native_outputs --out visual/native/context
  python native_visual.py --m parallel --root native_outputs --out visual/native/parallel
  python native_visual.py --m arness --root native_outputs --out visual/native/arness
  python native_visual.py --m arness --root model_outputs/iLLaDA/mbpp_s6_l1024/mbpp --out visual/native/arness_mbpp
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONTEXT_BENCHES = {"ruler_niah_double_2", "ruler_niah_order_2", "ruler_niah_double"}
CONTEXT_PAIRS = ["front_middle", "front_back", "middle_back", "front_back_extreme"]
POSITION_ORDER = ["front", "middle", "back"]
MAIN_THRESHOLDS = ["none", "0p6", "0p8"]


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def safe_name(x: Any) -> str:
    s = str(x if x is not None else "none").replace(".", "p")
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "none"


def to_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        s = str(x).strip()
        if s == "" or s.lower() in {"none", "null", "nan"}:
            return default
        v = float(s)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def to_int(x: Any, default: int = 0) -> int:
    v = to_float(x, float("nan"))
    return default if math.isnan(v) else int(v)


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] failed reading {path}: {exc}")
        return pd.DataFrame()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def choose_col(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    if df.empty:
        return None
    lower = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def parse_listish(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    try:
        if pd.isna(x):
            return []
    except Exception:
        pass
    s = str(x).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            val = ast.literal_eval(s)
            return list(val) if isinstance(val, (list, tuple)) else [val]
        except Exception:
            pass
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    return [s]


def canonical_threshold_label(x: Any) -> str:
    s = str(x if x is not None else "none").strip().lower().replace(".", "p")
    if s in {"", "nan", "null", "none"}:
        return "none"
    if s.startswith("thr"):
        s = s[3:]
    return s


def parse_condition_meta(condition: str) -> Dict[str, Any]:
    """Parse prepare_data.condition_name style strings.

    Examples:
      n10_ctx8192_pairfront_middle_l64_b32_st16_seed42
      ctx4096_pairfront_back_l64_w1samgidd_w1st32_seed42
      s6_l1024_b32_st256_thr0p6
    """
    text = str(condition or "")
    meta: Dict[str, Any] = {"condition": text}

    def first_int(patterns: Sequence[str]) -> Optional[int]:
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                return int(m.group(1))
        return None

    meta["sample_idx"] = first_int([r"(?:^|_)s(\d+)(?:_|$)", r"sample[_-]?(\d+)"])
    meta["num_samples"] = first_int([r"(?:^|_)n(\d+)(?:_|$)"])
    meta["context_length"] = first_int([r"(?:^|_)ctx(\d+)(?:_|$)"])
    meta["gen_length"] = first_int([r"(?:^|_)l(\d+)(?:_|$)", r"_len(\d+)"])
    meta["gen_blocksize"] = first_int([r"(?:^|_)b(\d+)(?:_|$)", r"_block(\d+)"])
    meta["gen_steps"] = first_int([r"(?:^|_)st(\d+)(?:_|$)", r"_steps(\d+)"])
    meta["w1_steps"] = first_int([r"(?:^|_)w1st(\d+)(?:_|$)"])
    meta["seed"] = first_int([r"(?:^|_)seed(\d+)(?:_|$)"])

    for pair in CONTEXT_PAIRS:
        if re.search(rf"(?:^|_)pair{re.escape(pair)}(?:_|$)", text):
            meta["needle_pair"] = pair
            break
    for pos in POSITION_ORDER:
        if re.search(rf"(?:^|_)pos{pos}(?:_|$)", text):
            meta["needle_position"] = pos
            break

    m = re.search(r"(?:^|_)thr([A-Za-z0-9p.-]+)(?:_|$)", text)
    meta["threshold_label"] = canonical_threshold_label(m.group(1) if m else None)
    m = re.search(r"(?:^|_)w1sam([^_]+)(?:_|$)", text)
    if m:
        meta["w1_sampler"] = m.group(1)
    m = re.search(r"(?:^|_)w1mode([^_]+)(?:_|$)", text)
    if m:
        meta["w1_decode_mode"] = m.group(1)

    steps = meta.get("gen_steps") or meta.get("w1_steps")
    gen_len = meta.get("gen_length")
    meta["effective_steps"] = steps
    meta["planned_parallelism"] = (float(gen_len) / float(steps)) if gen_len and steps else float("nan")
    return meta


def ensure_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def save_markdown(df: pd.DataFrame, path: Path) -> None:
    try:
        text = df.to_markdown(index=False) + "\n"
    except Exception:
        text = df.to_csv(index=False)
    write_text(path, text)


# -----------------------------------------------------------------------------
# Native manifest loader
# -----------------------------------------------------------------------------

def discover_manifest_paths(root: Path) -> List[Path]:
    root = root.expanduser().resolve()
    paths: List[Path] = []
    if root.is_file() and root.name.endswith(".csv"):
        return [root]
    if (root / "native_output_manifest.csv").exists():
        paths.append(root / "native_output_manifest.csv")
    for p in sorted(root.glob("*/native_output_manifest.csv")):
        if "_manifests" not in p.parts:
            paths.append(p)
    for p in sorted(root.rglob("native_output_manifest.csv")):
        if p not in paths and "_manifests" not in p.parts:
            paths.append(p)
    # Fall back to global manifests if there is no model-scoped manifest.
    if not paths:
        for p in sorted((root / "_manifests").glob("native_output_manifest.csv")):
            paths.append(p)
    return sorted({str(p.resolve()): p for p in paths}.values(), key=lambda p: str(p))


def load_native_manifest(root: Path, models: Optional[Sequence[str]] = None) -> pd.DataFrame:
    paths = discover_manifest_paths(root)
    if not paths:
        raise FileNotFoundError(f"No native_output_manifest.csv found under {root}")
    frames: List[pd.DataFrame] = []
    for p in paths:
        df = read_csv(p)
        if df.empty:
            continue
        df["source_manifest"] = str(p)
        # Model-scoped manifest usually has model already; infer it if missing.
        if "model" not in df.columns or df["model"].isna().all():
            try:
                df["model"] = p.parent.name
            except Exception:
                df["model"] = "model"
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No usable native_output_manifest.csv rows under {root}")
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=[c for c in ["model", "task", "benchmark", "condition", "output_dir"] if c in out.columns])
    if models:
        allowed = set(str(m) for m in models)
        out = out[out["model"].astype(str).isin(allowed)].copy()
    if out.empty:
        raise ValueError("Manifest loaded, but no rows remain after filters.")

    metas = out.get("condition", pd.Series([""] * len(out))).map(parse_condition_meta).tolist()
    meta_df = pd.DataFrame(metas)
    for c in meta_df.columns:
        if c not in out.columns or out[c].isna().all():
            out[c] = meta_df[c]
        else:
            # Fill missing only.
            out[c] = out[c].where(out[c].notna(), meta_df[c])

    # Standardize score/cost names from native_output.py.
    if "score" in out.columns:
        out["score_pct"] = pd.to_numeric(out["score"], errors="coerce")
    elif "accuracy" in out.columns:
        out["score_pct"] = pd.to_numeric(out["accuracy"], errors="coerce")
    else:
        out["score_pct"] = np.nan

    if "avg_latency_sec" not in out.columns and "latency_mean_s" in out.columns:
        out["avg_latency_sec"] = out["latency_mean_s"]
    if "avg_tps" not in out.columns and "tokens_per_second_mean" in out.columns:
        out["avg_tps"] = out["tokens_per_second_mean"]
    if "avg_latency_sec" not in out.columns:
        out["avg_latency_sec"] = np.nan
    if "avg_tps" not in out.columns:
        out["avg_tps"] = np.nan

    numeric_cols = [
        "n", "score", "accuracy", "score_pct", "exact_order_acc", "set_acc", "slot1_acc", "slot2_acc",
        "avg_latency_sec", "avg_tps", "context_length", "gen_length", "gen_steps", "w1_steps",
        "effective_steps", "gen_blocksize", "planned_parallelism", "actual_parallelism", "completion_rate",
    ]
    out = ensure_numeric(out, numeric_cols)
    out["threshold_label"] = out.get("threshold_label", "none").map(canonical_threshold_label)
    return out


# -----------------------------------------------------------------------------
# Parallel visual mode
# -----------------------------------------------------------------------------

def is_context_row(row: pd.Series) -> bool:
    bench = str(row.get("benchmark") or "")
    task = str(row.get("task") or "").lower()
    return bench in CONTEXT_BENCHES or "context" in task or "ruler_niah" in bench


def plot_lines(df: pd.DataFrame, x: str, y: str, group: str, path: Path, title: str, xlabel: str, ylabel: str) -> None:
    if df.empty or x not in df.columns or y not in df.columns:
        return
    work = df.copy()
    work[x] = pd.to_numeric(work[x], errors="coerce")
    work[y] = pd.to_numeric(work[y], errors="coerce")
    work = work.dropna(subset=[x, y])
    if work.empty:
        return
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for name, g in work.groupby(group, dropna=False):
        g = g.sort_values(x)
        ax.plot(g[x], g[y], marker="o", linewidth=1.8, label=str(name))
        for _, r in g.iterrows():
            st = r.get("effective_steps", r.get("gen_steps", ""))
            if pd.notna(st):
                ax.annotate(str(int(st)), (r[x], r[y]), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"[OK] wrote {path}")


def plot_parallel_combined(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    specs = [
        ("score_pct", "Score / accuracy"),
        ("avg_latency_sec", "Latency / sample (s)"),
        ("avg_tps", "Tokens / second"),
        ("actual_parallelism", "Actual parallelism"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.4))
    work = df.copy()
    work["series"] = work["model"].astype(str) + " | " + work["benchmark"].astype(str)
    for ax, (metric, ylabel) in zip(axes.ravel(), specs):
        if metric not in work.columns:
            ax.text(0.5, 0.5, f"No {metric}", ha="center", va="center")
            ax.set_axis_off()
            continue
        plotted = False
        for name, g in work.groupby("series", dropna=False):
            g = g.copy()
            g["planned_parallelism"] = pd.to_numeric(g["planned_parallelism"], errors="coerce")
            g[metric] = pd.to_numeric(g[metric], errors="coerce")
            g = g.dropna(subset=["planned_parallelism", metric]).sort_values("planned_parallelism")
            if g.empty:
                continue
            plotted = True
            ax.plot(g["planned_parallelism"], g[metric], marker="o", linewidth=1.8, label=str(name))
        if not plotted:
            ax.text(0.5, 0.5, f"No numeric {metric}", ha="center", va="center")
        ax.set_xlabel("Planned parallelism = gen_length / steps")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(fontsize=8)
    fig.suptitle("Native parallel summary", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"[OK] wrote {path}")


def run_parallel(root: Path, out_dir: Path, models: Optional[Sequence[str]]) -> pd.DataFrame:
    df = load_native_manifest(root, models=models)
    df = df[~df.apply(is_context_row, axis=1)].copy()
    if df.empty:
        raise SystemExit("[ERROR] no non-context rows found in native_output manifests.")
    df["series"] = df["model"].astype(str) + " | " + df["benchmark"].astype(str)
    keep = [
        "model", "task", "experiment", "benchmark", "condition", "n", "gen_length", "gen_blocksize",
        "gen_steps", "w1_steps", "w1_sampler", "effective_steps", "planned_parallelism", "score_pct",
        "score", "accuracy", "avg_latency_sec", "avg_tps", "actual_parallelism", "completion_rate",
        "error_breakdown", "output_dir", "source_manifest",
    ]
    table = df[[c for c in keep if c in df.columns]].sort_values(["benchmark", "model", "planned_parallelism", "condition"])
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "native_parallel_table.csv", index=False)
    save_markdown(table, out_dir / "native_parallel_table.md")
    plot_parallel_combined(df, out_dir / "parallel_combined.png")
    plot_lines(df, "avg_latency_sec", "score_pct", "series", out_dir / "parallel_speed_quality.png", "Speed-quality tradeoff", "Latency / sample (s)", "Score / accuracy")
    print(f"[DONE] parallel rows: {len(table)} -> {out_dir}")
    return table


# -----------------------------------------------------------------------------
# Context visual mode
# -----------------------------------------------------------------------------

def plot_context_heatmap(df: pd.DataFrame, path: Path, metric: str = "exact_order_acc") -> None:
    if df.empty or metric not in df.columns:
        return
    work = df.copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work["planned_parallelism"] = pd.to_numeric(work["planned_parallelism"], errors="coerce")
    work = work.dropna(subset=[metric, "planned_parallelism"])
    if work.empty:
        return
    work["row"] = (
        work["model"].astype(str)
        + " | ctx=" + work["context_length"].fillna(-1).astype(int).astype(str)
        + " | " + work.get("needle_pair", pd.Series([""] * len(work))).fillna("na").astype(str)
    )
    work["col"] = "p=" + work["planned_parallelism"].map(lambda x: f"{x:g}")
    piv = work.pivot_table(index="row", columns="col", values=metric, aggfunc="mean")
    # Sort columns numerically by the p= value.
    cols = sorted(piv.columns, key=lambda c: to_float(str(c).replace("p=", ""), 999))
    piv = piv[cols]
    vals = piv.values.astype(float)
    fig_h = max(4.2, 0.32 * len(piv.index) + 2.2)
    fig, ax = plt.subplots(figsize=(9.5, fig_h))
    im = ax.imshow(vals, aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(piv.columns)), [str(c) for c in piv.columns], rotation=0)
    ax.set_yticks(range(len(piv.index)), [str(i) for i in piv.index])
    ax.set_xlabel("Planned parallelism")
    ax.set_ylabel("Model / context / needle pair")
    ax.set_title(f"Context double retrieval: {metric}")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            label = "NA" if np.isnan(vals[i, j]) else f"{vals[i, j]:.0f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"[OK] wrote {path}")


def plot_context_combined(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    specs = [
        ("exact_order_acc", "Exact-order accuracy"),
        ("set_acc", "Set accuracy"),
        ("avg_latency_sec", "Latency / sample (s)"),
        ("avg_tps", "Tokens / second"),
    ]
    work = df.copy()
    work["series"] = work["model"].astype(str) + " | ctx=" + work["context_length"].fillna(-1).astype(int).astype(str)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.4))
    for ax, (metric, ylabel) in zip(axes.ravel(), specs):
        if metric not in work.columns:
            ax.text(0.5, 0.5, f"No {metric}", ha="center", va="center")
            ax.set_axis_off()
            continue
        plotted = False
        for name, g in work.groupby("series", dropna=False):
            g = g.copy()
            g["planned_parallelism"] = pd.to_numeric(g["planned_parallelism"], errors="coerce")
            g[metric] = pd.to_numeric(g[metric], errors="coerce")
            # Average across needle pairs for clean presentation.
            g = g.groupby("planned_parallelism", as_index=False)[metric].mean().dropna()
            if g.empty:
                continue
            plotted = True
            g = g.sort_values("planned_parallelism")
            ax.plot(g["planned_parallelism"], g[metric], marker="o", linewidth=1.8, label=str(name))
        if not plotted:
            ax.text(0.5, 0.5, f"No numeric {metric}", ha="center", va="center")
        ax.set_xlabel("Planned parallelism = gen_length / steps")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(fontsize=8)
    fig.suptitle("Native context double-needle summary", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"[OK] wrote {path}")


def collect_examples_from_scores(df: pd.DataFrame, out_path: Path, max_per_condition: int = 3) -> int:
    records = []
    for _, row in df.iterrows():
        out_dir = Path(str(row.get("output_dir") or ""))
        scores = read_csv(out_dir / "scores.csv")
        outputs = read_jsonl(out_dir / "outputs.jsonl")
        if scores.empty:
            continue
        if outputs:
            out_by_sample = {str(o.get("sample_id")): o for o in outputs}
        else:
            out_by_sample = {}
        sub = scores.copy()
        if "correct" in sub.columns:
            # Put failures first for diagnosis, then one correct if available.
            fail = sub[sub["correct"].astype(str).str.lower().isin(["false", "0", "no"])]
            ok = sub[sub["correct"].astype(str).str.lower().isin(["true", "1", "yes"])]
            sub = pd.concat([fail.head(max_per_condition), ok.head(1)], ignore_index=True)
        else:
            sub = sub.head(max_per_condition)
        for _, srow in sub.iterrows():
            sid = str(srow.get("sample_id"))
            out = out_by_sample.get(sid, {})
            pred = out.get("prediction") or out.get("raw_output") or ""
            records.append({
                "model": row.get("model"),
                "task": row.get("task"),
                "benchmark": row.get("benchmark"),
                "condition": row.get("condition"),
                "context_length": row.get("context_length"),
                "needle_pair": row.get("needle_pair"),
                "planned_parallelism": row.get("planned_parallelism"),
                "sample_id": sid,
                "correct": srow.get("correct"),
                "exact_order": srow.get("exact_order"),
                "set_match": srow.get("set_match"),
                "pred_values": srow.get("pred_values"),
                "gold_values": srow.get("gold_values"),
                "prediction_preview": re.sub(r"\s+", " ", str(pred)).strip()[:500],
                "output_dir": str(out_dir),
            })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(records)


def run_context(root: Path, out_dir: Path, models: Optional[Sequence[str]]) -> pd.DataFrame:
    df = load_native_manifest(root, models=models)
    df = df[df.apply(is_context_row, axis=1)].copy()
    if df.empty:
        raise SystemExit("[ERROR] no context/RULER rows found in native_output manifests.")
    keep = [
        "model", "task", "experiment", "benchmark", "condition", "n", "context_length", "needle_pair",
        "gen_length", "gen_blocksize", "gen_steps", "w1_steps", "w1_sampler", "effective_steps",
        "planned_parallelism", "exact_order_acc", "set_acc", "slot1_acc", "slot2_acc", "score_pct",
        "avg_latency_sec", "avg_tps", "output_dir", "source_manifest",
    ]
    table = df[[c for c in keep if c in df.columns]].sort_values(["model", "context_length", "needle_pair", "planned_parallelism", "condition"])
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "native_context_table.csv", index=False)
    save_markdown(table, out_dir / "native_context_table.md")
    overall = table.groupby([c for c in ["model", "task", "benchmark", "context_length", "planned_parallelism"] if c in table.columns], dropna=False).agg({
        "exact_order_acc": "mean",
        "set_acc": "mean",
        "slot1_acc": "mean",
        "slot2_acc": "mean",
        "avg_latency_sec": "mean",
        "avg_tps": "mean",
        "n": "sum",
    }).reset_index()
    overall.to_csv(out_dir / "native_context_overall.csv", index=False)
    save_markdown(overall, out_dir / "native_context_overall.md")
    plot_context_heatmap(table, out_dir / "context_exact_order_heatmap.png", "exact_order_acc")
    plot_context_heatmap(table, out_dir / "context_set_acc_heatmap.png", "set_acc")
    plot_context_combined(table, out_dir / "context_combined.png")
    n_examples = collect_examples_from_scores(table, out_dir / "context_examples.jsonl")
    report = [
        "native_visual context report",
        "",
        f"Rows: {len(table)}",
        f"Example records: {n_examples}",
        "",
        "Files:",
        "- native_context_table.csv/md",
        "- native_context_overall.csv/md",
        "- context_exact_order_heatmap.png",
        "- context_set_acc_heatmap.png",
        "- context_combined.png",
        "- context_examples.jsonl",
    ]
    write_text(out_dir / "context_report.txt", "\n".join(report) + "\n")
    print(f"[DONE] context rows: {len(table)} -> {out_dir}")
    return table


# -----------------------------------------------------------------------------
# ARness / trace visual mode
# -----------------------------------------------------------------------------

def has_trace_artifacts(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "sample_traces").exists():
        return True
    return (path / "trace.jsonl").exists() and ((path / "outputs.jsonl").exists() or (path / "summary.jsonl").exists())


def discover_trace_dirs(root: Path) -> List[Path]:
    root = root.expanduser().resolve()
    if has_trace_artifacts(root):
        return [root]
    out: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        if any(part in {"sample_traces", "visual", "overview", "_commit_orders"} for part in p.parts):
            continue
        if has_trace_artifacts(p):
            out.append(p)
    return sorted({str(p): p for p in out}.values(), key=lambda p: str(p))


def resolve_sample_dir(condition_dir: Path, sample_idx: Optional[int]) -> Optional[Path]:
    st = condition_dir / "sample_traces"
    if not st.exists():
        return None
    sample_dirs = sorted([p for p in st.iterdir() if p.is_dir() and p.name.startswith("sample_")])
    if not sample_dirs:
        return None
    if sample_idx is not None:
        wanted = st / f"sample_{sample_idx:04d}"
        if wanted.exists():
            return wanted
        for d in sample_dirs:
            m = re.search(r"sample[_-]?(\d+)", d.name)
            if m and int(m.group(1)) == sample_idx:
                return d
    return sample_dirs[0]


def parse_complete_cell(x: Any) -> Tuple[float, float, float]:
    m = re.match(r"^(\d+)\s*/\s*(\d+)$", str(x).strip())
    if not m:
        return float("nan"), float("nan"), float("nan")
    visible, total = float(m.group(1)), float(m.group(2))
    return visible, total, visible / total if total else float("nan")


def count_masks_in_state(x: Any) -> int:
    s = str(x)
    total = 0
    for m in re.finditer(r"\[MASK\](?:x(\d+))?", s):
        total += int(m.group(1) or 1)
    return total


def block_completion_matrix(block_timeline: pd.DataFrame) -> Tuple[List[int], List[int], List[List[float]]]:
    if block_timeline.empty:
        return [], [], []
    step_col = choose_col(block_timeline, ["generation_step", "global_step_idx", "step_idx", "step"])
    steps = list(range(len(block_timeline))) if step_col is None else [to_int(v, i) for i, v in enumerate(block_timeline[step_col])]
    complete_cols = [c for c in block_timeline.columns if re.match(r"block_\d+_complete$", str(c))]
    if complete_cols:
        complete_cols = sorted(complete_cols, key=lambda c: int(re.search(r"block_(\d+)_complete", str(c)).group(1)))
        blocks = [int(re.search(r"block_(\d+)_complete", str(c)).group(1)) for c in complete_cols]
        mat = [[parse_complete_cell(v)[2] for v in block_timeline[c].tolist()] for c in complete_cols]
        return blocks, steps, mat
    state_cols = [c for c in block_timeline.columns if re.match(r"block_\d+(?:_tokens)?$", str(c))]
    if state_cols:
        state_cols = sorted(state_cols, key=lambda c: int(re.search(r"block_(\d+)", str(c)).group(1)))
        blocks = [int(re.search(r"block_(\d+)", str(c)).group(1)) for c in state_cols]
        mat = []
        for c in state_cols:
            mask_counts = [count_masks_in_state(v) for v in block_timeline[c].tolist()]
            block_len = max(mask_counts) if mask_counts else 1
            mat.append([max(0, block_len - mc) / block_len for mc in mask_counts])
        return blocks, steps, mat
    return [], [], []


def normalize_step_events(df: pd.DataFrame, metrics: Dict[str, Any]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    step_col = choose_col(out, ["generation_step", "global_step_idx", "step_idx", "step"])
    block_col = choose_col(out, ["active_block_idx", "active_block", "block_idx"])
    local_col = choose_col(out, ["block_local_round", "local_step_idx", "local_round"])
    count_col = choose_col(out, ["selected_count", "actual_transfer_count", "committed_count", "num_selected", "transfer_count"])
    conf_col = choose_col(out, ["mean_selected_confidence", "mean_confidence", "selected_confidence_mean", "avg_confidence", "confidence_mean"])
    completion_col = choose_col(out, ["current_completion_rate", "completion_rate", "overall_completion"])
    pos_col = choose_col(out, ["selected_positions"])

    out["_step"] = pd.to_numeric(out[step_col], errors="coerce").ffill().fillna(0).astype(int) if step_col else range(len(out))
    out["_block"] = pd.to_numeric(out[block_col], errors="coerce").fillna(0).astype(int) if block_col else 0
    out["_local_round"] = pd.to_numeric(out[local_col], errors="coerce").fillna(0).astype(int) if local_col else out.groupby("_block").cumcount()
    if count_col:
        out["_selected_count"] = pd.to_numeric(out[count_col], errors="coerce").fillna(0)
    elif pos_col:
        out["_selected_count"] = out[pos_col].apply(lambda x: len(parse_listish(x)))
    else:
        out["_selected_count"] = 0
    if conf_col:
        out["_mean_confidence"] = pd.to_numeric(out[conf_col], errors="coerce")
    else:
        confs_col = choose_col(out, ["selected_confidences", "selected_confidence", "confidences"])
        if confs_col:
            out["_mean_confidence"] = out[confs_col].apply(lambda x: pd.Series([to_float(v) for v in parse_listish(x)]).dropna().mean())
        else:
            out["_mean_confidence"] = np.nan
    if completion_col:
        out["_completion"] = pd.to_numeric(out[completion_col], errors="coerce")
    else:
        gen_length = to_float(metrics.get("gen_length"), float("nan"))
        denom = gen_length if math.isfinite(gen_length) and gen_length > 0 else max(1.0, float(out["_selected_count"].sum()))
        out["_completion"] = out["_selected_count"].cumsum() / denom
    return out.sort_values(["_step", "_block", "_local_round"]).reset_index(drop=True)


def selected_positions_global(row: pd.Series, block_len: int) -> List[int]:
    positions = [to_int(p, -1) for p in parse_listish(row.get("selected_positions"))]
    positions = [p for p in positions if p >= 0]
    if not positions:
        return []
    block = to_int(row.get("_block", row.get("block_idx", 0)), 0)
    # If all positions look block-local and block > 0, map to global.
    if block_len > 0 and block > 0 and max(positions) < block_len:
        return [block * block_len + p for p in positions]
    return positions


def token_commit_records(step_events: pd.DataFrame, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    if step_events.empty:
        return []
    pos_col = choose_col(step_events, ["selected_positions"])
    tok_col = choose_col(step_events, ["selected_decoded_tokens", "selected_tokens", "decoded_tokens"])
    conf_col = choose_col(step_events, ["selected_confidences", "selected_confidence", "confidences"])
    if not pos_col:
        return []
    block_len = to_int(metrics.get("block_length", metrics.get("gen_blocksize")), 0)
    recs: List[Dict[str, Any]] = []
    for _, r in step_events.iterrows():
        positions = selected_positions_global(r, block_len)
        tokens = parse_listish(r.get(tok_col)) if tok_col else []
        confs = parse_listish(r.get(conf_col)) if conf_col else []
        for j, pos in enumerate(positions):
            recs.append({
                "global_position": pos,
                "position": pos,
                "token": str(tokens[j]) if j < len(tokens) else "",
                "step": to_int(r.get("_step"), 0),
                "block": to_int(r.get("_block"), 0),
                "confidence": to_float(confs[j], float("nan")) if j < len(confs) else float("nan"),
            })
    return [r for r in recs if r["global_position"] >= 0]


def commit_order_dataframe(step_events: pd.DataFrame, metrics: Dict[str, Any]) -> pd.DataFrame:
    recs = token_commit_records(step_events, metrics)
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs)
    for c in ["global_position", "step", "block", "confidence"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values(["global_position", "step"]).drop_duplicates("global_position", keep="first")
    df = df.sort_values("global_position").reset_index(drop=True)
    max_step = max(1.0, float(df["step"].max()))
    df["commit_time_norm"] = df["step"] / max_step
    return df


def kendall_tau_no_x_ties(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float]:
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan")
    concordant = discordant = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if xs[i] == xs[j]:
                continue
            total += 1
            prod = (xs[i] - xs[j]) * (ys[i] - ys[j])
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
    if total <= 0:
        return float("nan"), float("nan")
    return (concordant - discordant) / total, discordant / total


def compute_order_metrics(commit_df: pd.DataFrame, metrics: Dict[str, Any]) -> Dict[str, Any]:
    if commit_df.empty:
        return {}
    xs = [float(x) for x in commit_df["global_position"].tolist()]
    ys = [float(y) for y in commit_df["step"].tolist()]
    tau, inv = kendall_tau_no_x_ties(xs, ys)
    total_len = max(to_int(metrics.get("gen_length"), 0), to_int(commit_df["global_position"].max(), -1) + 1)
    max_step = to_int(commit_df["step"].max(), 0)
    committed: set[int] = set()
    by_step: Dict[int, List[int]] = defaultdict(list)
    for _, r in commit_df.iterrows():
        by_step[to_int(r.get("step"), 0)].append(to_int(r.get("global_position"), -1))
    prefix_gaps: List[float] = []
    for step in range(max_step + 1):
        for pos in by_step.get(step, []):
            if pos >= 0:
                committed.add(pos)
        prefix_len = 0
        while prefix_len in committed:
            prefix_len += 1
        if total_len > 0:
            prefix_gaps.append(len(committed) / total_len - prefix_len / total_len)
    return {
        "order_kendall_tau": tau,
        "inversion_rate": inv,
        "left_to_right_score": (1.0 - inv) if math.isfinite(inv) else float("nan"),
        "mean_prefix_gap": sum(prefix_gaps) / len(prefix_gaps) if prefix_gaps else float("nan"),
        "committed_positions": int(len(commit_df)),
    }


def reconstruct_token_timeline(step_events: pd.DataFrame, metrics: Dict[str, Any]) -> pd.DataFrame:
    if step_events.empty:
        return pd.DataFrame()
    gen_len = to_int(metrics.get("gen_length"), 0)
    block_len = to_int(metrics.get("block_length", metrics.get("gen_blocksize")), 0)
    if gen_len <= 0:
        max_pos = -1
        for _, r in step_events.iterrows():
            ps = selected_positions_global(r, block_len)
            max_pos = max([max_pos] + ps)
        gen_len = max_pos + 1 if max_pos >= 0 else 0
    if block_len <= 0:
        block_len = gen_len or 1
    slots: List[Optional[str]] = [None] * max(gen_len, 1)
    rows: List[Dict[str, Any]] = []
    tok_col = choose_col(step_events, ["selected_decoded_tokens", "selected_tokens", "decoded_tokens"])
    for _, r in step_events.iterrows():
        ps = selected_positions_global(r, block_len)
        toks = parse_listish(r.get(tok_col)) if tok_col else []
        new_parts: List[str] = []
        for j, p in enumerate(ps):
            if 0 <= p < len(slots):
                tok = str(toks[j]) if j < len(toks) else ""
                slots[p] = tok
                new_parts.append(tok)
        row: Dict[str, Any] = {
            "generation_step": to_int(r.get("_step"), 0),
            "active_block": to_int(r.get("_block"), 0),
            "selected_count": len(ps),
            "new_text_fragment": "".join(new_parts).replace("\n", "\\n"),
            "visible_tokens": sum(x is not None for x in slots),
            "current_text_preview": "".join(x if x is not None else "[MASK]" for x in slots[: min(len(slots), 512)]).replace("\n", "\\n"),
        }
        num_blocks = max(1, math.ceil(len(slots) / block_len))
        for b in range(num_blocks):
            chunk = slots[b * block_len : min((b + 1) * block_len, len(slots))]
            row[f"block_{b}_tokens"] = "".join(x if x is not None else "[MASK]" for x in chunk).replace("\n", "\\n")
        rows.append(row)
    return pd.DataFrame(rows)


def load_trace_condition(cond_dir: Path, sample_idx: Optional[int]) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, Optional[Path]]:
    sample_dir = resolve_sample_dir(cond_dir, sample_idx)
    metrics: Dict[str, Any] = {}
    # Path-derived metadata from native_output layout: <model>/<task>/<benchmark>/<condition>
    parts = cond_dir.parts
    if len(parts) >= 4:
        metrics["condition"] = cond_dir.name
        metrics["benchmark"] = cond_dir.parent.name
        metrics["task"] = cond_dir.parent.parent.name
        metrics["model"] = cond_dir.parent.parent.parent.name
    metrics.update(parse_condition_meta(cond_dir.name))
    parent_metrics = read_json(cond_dir / "metrics.json")
    for k, v in parent_metrics.items():
        if k not in metrics or metrics.get(k) in (None, "", float("nan")):
            metrics[k] = v
    run_json = read_json(cond_dir / "run.json")
    for k in ["model", "task", "experiment", "benchmark", "condition"]:
        if k in run_json and not metrics.get(k):
            metrics[k] = run_json[k]
    step_events = pd.DataFrame()
    block_timeline = pd.DataFrame()
    if sample_dir is not None:
        sm = read_json(sample_dir / "sample_metrics.json") or read_json(sample_dir / "metrics.json")
        for k, v in sm.items():
            if k not in metrics or metrics.get(k) in (None, "", float("nan")):
                metrics[k] = v
        step_events = read_csv(sample_dir / "step_events.csv")
        block_timeline = read_csv(sample_dir / "block_timeline.csv")
    # Normalize core fields.
    if metrics.get("gen_steps") is None and metrics.get("w1_steps") is not None:
        metrics["gen_steps"] = metrics.get("w1_steps")
    if metrics.get("gen_steps") is None:
        metrics["gen_steps"] = metrics.get("effective_steps")
    if metrics.get("block_length") is None and metrics.get("gen_blocksize") is not None:
        metrics["block_length"] = metrics.get("gen_blocksize")
    if metrics.get("threshold_label") is None:
        metrics["threshold_label"] = canonical_threshold_label(metrics.get("token_selection_confidence_threshold"))
    metrics["condition_dir"] = str(cond_dir)
    metrics["sample_dir"] = str(sample_dir) if sample_dir is not None else ""
    for k in ["gen_length", "gen_steps", "block_length", "gen_blocksize", "score", "accuracy", "avg_latency_sec", "avg_tps", "actual_parallelism", "completion_rate"]:
        if k in metrics:
            metrics[k] = to_float(metrics[k], float("nan"))
    if math.isfinite(to_float(metrics.get("gen_length"), float("nan"))) and math.isfinite(to_float(metrics.get("gen_steps"), float("nan"))) and to_float(metrics.get("gen_steps")) > 0:
        metrics["planned_parallelism"] = to_float(metrics["gen_length"]) / to_float(metrics["gen_steps"])
    else:
        metrics["planned_parallelism"] = to_float(metrics.get("planned_parallelism"), float("nan"))
    step_events = normalize_step_events(step_events, metrics)
    return metrics, step_events, block_timeline, sample_dir


def plot_generation_chain(step_events: pd.DataFrame, block_timeline: pd.DataFrame, metrics: Dict[str, Any], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.0), gridspec_kw={"height_ratios": [1.55, 1.0]})
    title = (
        f"{metrics.get('model', '')} | {metrics.get('benchmark', '')} | "
        f"len={to_int(metrics.get('gen_length'), 0)}, steps={to_int(metrics.get('gen_steps'), 0)}, "
        f"p={to_float(metrics.get('planned_parallelism'), float('nan')):.2g}"
    )
    fig.suptitle(title, fontsize=12)
    blocks, steps, mat = block_completion_matrix(block_timeline)
    ax = axes[0]
    if blocks and mat:
        im = ax.imshow(mat, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
        ax.set_ylabel("Block index")
        ax.set_xlabel("Generation step")
        ax.set_title("Within-block completion")
        ax.set_yticks(range(len(blocks)))
        ax.set_yticklabels([str(b) for b in blocks])
        if len(steps) <= 25:
            ax.set_xticks(range(len(steps)))
            ax.set_xticklabels([str(s) for s in steps], rotation=90, fontsize=8)
        else:
            ticks = list(range(0, len(steps), max(1, len(steps) // 12)))
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(steps[i]) for i in ticks], rotation=45, fontsize=8)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Completion")
    else:
        ax.text(0.5, 0.5, "No block_timeline.csv", ha="center", va="center")
        ax.set_axis_off()
    ax2 = axes[1]
    if not step_events.empty:
        ax2.plot(step_events["_step"], step_events["_completion"], marker="o", markersize=3, linewidth=1.6, label="completion")
        ax2b = ax2.twinx()
        ax2b.bar(step_events["_step"], step_events["_selected_count"], width=0.8, alpha=0.25, label="tokens / step")
        changes = step_events["_block"].ne(step_events["_block"].shift(1)).fillna(True)
        for s in step_events.loc[changes, "_step"].tolist():
            ax2.axvline(s, linestyle="--", linewidth=0.8, alpha=0.45)
        ax2.set_xlabel("Generation step")
        ax2.set_ylabel("Completion rate")
        ax2.set_ylim(-0.03, 1.03)
        ax2b.set_ylabel("Committed tokens / step")
        ax2.set_title("Overall generation progress")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No step_events.csv", ha="center", va="center")
        ax2.set_axis_off()
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


def plot_commit_order(commit_df: pd.DataFrame, metrics: Dict[str, Any], out_path: Path) -> None:
    if commit_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    x = pd.to_numeric(commit_df["global_position"], errors="coerce")
    y = pd.to_numeric(commit_df["step"], errors="coerce")
    conf = pd.to_numeric(commit_df.get("confidence", pd.Series([np.nan] * len(commit_df))), errors="coerce")
    valid = x.notna() & y.notna()
    sizes = 10 + 24 * conf[valid].fillna(0.5).clip(lower=0, upper=1)
    ax.scatter(x[valid], y[valid], s=sizes, alpha=0.75, linewidths=0)
    block = to_int(metrics.get("block_length", metrics.get("gen_blocksize")), 0)
    gen_len = to_int(metrics.get("gen_length"), 0)
    if block > 0 and gen_len > 0:
        for b in range(block, gen_len, block):
            ax.axvline(b, linestyle="--", linewidth=0.7, alpha=0.35)
    tau = to_float(metrics.get("order_kendall_tau"), float("nan"))
    inv = to_float(metrics.get("inversion_rate"), float("nan"))
    ax.set_title(f"Token position vs first commit step | tau={tau:.2f}, inversion={inv:.2f}")
    ax.set_xlabel("Token position")
    ax.set_ylabel("First commit step")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


def plot_arness_overview(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    specs = [
        ("score_pct", "Score / accuracy"),
        ("avg_latency_sec", "Latency / sample (s)"),
        ("avg_tps", "Tokens / second"),
        ("actual_parallelism", "Actual parallelism"),
        ("order_kendall_tau", "Kendall tau"),
        ("mean_prefix_gap", "Mean prefix gap"),
    ]
    work = df.copy()
    work["series"] = work["model"].astype(str) + " | " + work["benchmark"].astype(str)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4))
    for ax, (metric, ylabel) in zip(axes.ravel(), specs):
        if metric not in work.columns:
            ax.text(0.5, 0.5, f"No {metric}", ha="center", va="center")
            ax.set_axis_off()
            continue
        plotted = False
        for name, g in work.groupby("series", dropna=False):
            g = g.copy()
            g["planned_parallelism"] = pd.to_numeric(g["planned_parallelism"], errors="coerce")
            g[metric] = pd.to_numeric(g[metric], errors="coerce")
            g = g.dropna(subset=["planned_parallelism", metric]).sort_values("planned_parallelism")
            if g.empty:
                continue
            plotted = True
            ax.plot(g["planned_parallelism"], g[metric], marker="o", linewidth=1.8, label=str(name))
        if not plotted:
            ax.text(0.5, 0.5, f"No numeric {metric}", ha="center", va="center")
        ax.set_xlabel("Planned parallelism = gen_length / steps")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(fontsize=8)
    fig.suptitle("Native ARness / trace overview", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


def plot_token_order_grid(rows: pd.DataFrame, out_path: Path, max_panels: int = 18) -> None:
    if rows.empty or "commit_order_csv" not in rows.columns:
        return
    work = rows.copy()
    work = work.sort_values(["benchmark", "model", "planned_parallelism", "condition"])
    if len(work) > max_panels:
        # Prefer diverse planned parallelism/model combinations over dumping dozens of panels.
        work = work.groupby(["model", "benchmark", "planned_parallelism"], dropna=False, group_keys=False).head(1).head(max_panels)
    n = len(work)
    if n == 0:
        return
    cols = min(3, n)
    rows_n = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(5.0 * cols, 3.2 * rows_n), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.set_axis_off()
    for ax, (_, r) in zip(axes.ravel(), work.iterrows()):
        cdf = read_csv(Path(str(r.get("commit_order_csv"))))
        if cdf.empty:
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
            ax.set_axis_off()
            continue
        x = pd.to_numeric(cdf["global_position"], errors="coerce")
        y = pd.to_numeric(cdf["step"], errors="coerce")
        valid = x.notna() & y.notna()
        ax.scatter(x[valid], y[valid], s=10, alpha=0.75, linewidths=0)
        title = f"{r.get('model')} | p={to_float(r.get('planned_parallelism'), float('nan')):.2g}"
        if pd.notna(r.get("w1_sampler", np.nan)) and str(r.get("w1_sampler")) != "nan":
            title += f" | {r.get('w1_sampler')}"
        tau = to_float(r.get("order_kendall_tau"), float("nan"))
        if math.isfinite(tau):
            title += f"\nτ={tau:.2f}"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Token position", fontsize=8)
        ax.set_ylabel("Commit step", fontsize=8)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Token position vs first commit step", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


def run_arness(root: Path, out_dir: Path, models: Optional[Sequence[str]], sample_idx: Optional[int], overview_only: bool) -> pd.DataFrame:
    cond_dirs = discover_trace_dirs(root)
    if not cond_dirs:
        raise SystemExit(f"[ERROR] no trace/sample_traces dirs found under: {root}")
    rows: List[Dict[str, Any]] = []
    model_filter = set(str(m) for m in models) if models else None
    for cond_dir in cond_dirs:
        try:
            metrics, step_events, block_timeline, sample_dir = load_trace_condition(cond_dir, sample_idx)
            if model_filter and str(metrics.get("model")) not in model_filter:
                continue
            commit_df = commit_order_dataframe(step_events, metrics)
            metrics.update(compute_order_metrics(commit_df, metrics))
            # Attach eval/cost metrics from parent metrics.json if available.
            if "score_pct" not in metrics:
                if metrics.get("score") is not None and not isinstance(metrics.get("score"), dict):
                    metrics["score_pct"] = to_float(metrics.get("score"), float("nan"))
                elif metrics.get("accuracy") is not None:
                    metrics["score_pct"] = to_float(metrics.get("accuracy"), float("nan"))
            sample_name = sample_dir.name if sample_dir is not None else "sample_0000"
            rel_name = safe_name(str(metrics.get("model", "model"))) + "_" + safe_name(str(metrics.get("task", cond_dir.parent.parent.name if len(cond_dir.parts) > 2 else "task"))) + "_" + safe_name(cond_dir.name)
            sample_out = out_dir / "traces" / rel_name / sample_name
            sample_out.mkdir(parents=True, exist_ok=True)
            if not commit_df.empty:
                commit_path = sample_out / "commit_order.csv"
                commit_df.to_csv(commit_path, index=False)
                metrics["commit_order_csv"] = str(commit_path)
            token_timeline = reconstruct_token_timeline(step_events, metrics)
            if not token_timeline.empty:
                token_path = sample_out / "token_timeline_reconstructed.csv"
                token_timeline.to_csv(token_path, index=False)
                metrics["token_timeline_csv"] = str(token_path)
            if not overview_only:
                plot_generation_chain(step_events, block_timeline, metrics, sample_out / "generation_chain.png")
                plot_commit_order(commit_df, metrics, sample_out / "token_commit_order.png")
            write_json(sample_out / "sample_overview.json", metrics)
            metrics["visual_output_dir"] = str(sample_out)
            rows.append(metrics)
        except Exception as exc:
            print(f"[WARN] skip {cond_dir}: {exc}")
    if not rows:
        raise SystemExit("[ERROR] no usable trace rows after filtering.")
    df = pd.DataFrame(rows)
    if "benchmark" not in df.columns:
        df["benchmark"] = "unknown"
    if "model" not in df.columns:
        df["model"] = "model"
    df = ensure_numeric(df, ["gen_length", "gen_steps", "planned_parallelism", "score_pct", "avg_latency_sec", "avg_tps", "actual_parallelism", "order_kendall_tau", "inversion_rate", "mean_prefix_gap", "committed_positions"])
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "arness_trace_overview.csv", index=False)
    save_markdown(df[[c for c in ["model", "task", "benchmark", "condition", "sample_idx", "gen_length", "gen_steps", "w1_sampler", "planned_parallelism", "score_pct", "actual_parallelism", "order_kendall_tau", "inversion_rate", "mean_prefix_gap", "visual_output_dir"] if c in df.columns]], out_dir / "arness_trace_overview.md")
    plot_arness_overview(df, out_dir / "arness_overview.png")
    plot_token_order_grid(df, out_dir / "token_order_grid.png")
    print(f"[DONE] arness trace rows: {len(df)} -> {out_dir}")
    return df


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--m", choices=["parallel", "context", "arness", "all"], required=True, help="Visualization mode.")
    ap.add_argument("--root", default="native_outputs", help="native_outputs root for parallel/context; native_outputs or model_outputs subtree for arness.")
    ap.add_argument("--out", default="visual/native", help="Output visual directory.")
    ap.add_argument("--models", nargs="*", default=None, help="Optional model aliases to include, e.g. iLLaDA w1_4b.")
    ap.add_argument("--sample-idx", type=int, default=None, help="ARness mode: sample index to visualize when multiple sample traces exist.")
    ap.add_argument("--overview-only", action="store_true", help="ARness mode: build overview/order CSVs but skip per-condition figures.")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    modes = ["parallel", "context", "arness"] if args.m == "all" else [args.m]
    for mode in modes:
        mode_out = out / mode if args.m == "all" else out
        if mode == "parallel":
            run_parallel(root, mode_out, args.models)
        elif mode == "context":
            run_context(root, mode_out, args.models)
        elif mode == "arness":
            run_arness(root, mode_out, args.models, args.sample_idx, args.overview_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
