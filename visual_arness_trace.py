#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visual_arness_trace.py

Simplified ARness visualizer focused on:
1. One generation figure per sample/condition.
2. Compact overview comparison plots.
3. Two-step-length x five-threshold comparison plots.
4. Kendall-tau based AR-like order score.

Default:
  input  = outputs/arness
  output = visual/arness

Expected canonical input:
  outputs/arness/<experiment>/<condition>/
    summary.jsonl / aggregate.csv / sample_traces/
    sample_traces/sample_XXXX/
      step_events.csv
      block_timeline.csv
      block_metrics.csv
      sample_metrics.json

Outputs:
  visual/arness/<experiment>/<condition>/sample_XXXX/generation_chain.png
  visual/arness/overview/<experiment>/overall_comparison.png
  visual/arness/overview/<experiment>/top2_steps_thresholds.png
  visual/arness/overview/<experiment>/overview_metrics.csv
  visual/arness/overview/all_conditions_overview.csv
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
import pandas as pd

MAIN_THRESHOLDS = {"none", "0p6", "0p8"}
ALL_THRESHOLDS_ORDER = ["none", "0p6", "0p7", "0p8", "0p9"]

# -----------------------------
# Basic IO / parsing helpers
# -----------------------------

def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
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
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] failed reading {path}: {exc}")
        return pd.DataFrame()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def choose_col(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    if df.empty:
        return None
    lower_to_real = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        real = lower_to_real.get(name.lower())
        if real is not None:
            return real
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


def threshold_sort_key(x: Any) -> Tuple[int, float, str]:
    s = canonical_threshold_label(x)
    if s in ALL_THRESHOLDS_ORDER:
        return (ALL_THRESHOLDS_ORDER.index(s), 0.0, s)
    return (99, to_float(s.replace("p", "."), 999.0), s)


def safe_name(s: Any) -> str:
    s = str(s if s is not None else "none")
    s = s.replace(".", "p")
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "none"


# -----------------------------
# Condition discovery
# -----------------------------

COND_RE = re.compile(
    r"^(?:arness_trace_(?P<pb>gsm8k|mbpp)_sample(?P<ps>\d+)_)?"
    r"(?P<benchmark>gsm8k|mbpp)_sample(?P<sample>\d+)_len(?P<gen_length>\d+)"
    r"_block(?P<block>\d+)_steps(?P<steps>\d+)_thr(?P<thr>[A-Za-z0-9p.-]+)$",
    re.IGNORECASE,
)


def parse_condition_from_name(name: str) -> Optional[Dict[str, Any]]:
    m = COND_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    benchmark = str(d["benchmark"]).lower()
    sample = int(d["sample"])
    thr = canonical_threshold_label(d["thr"])
    threshold = None if thr == "none" else to_float(thr.replace("p", "."), float("nan"))
    if isinstance(threshold, float) and math.isnan(threshold):
        threshold = None
    return {
        "benchmark": benchmark,
        "sample_idx": sample,
        "gen_length": int(d["gen_length"]),
        "gen_steps": int(d["steps"]),
        "steps": int(d["steps"]),
        "gen_blocksize": int(d["block"]),
        "block_length": int(d["block"]),
        "threshold_label": thr,
        "token_selection_confidence_threshold": threshold,
        "experiment": f"arness_trace_{benchmark}_sample{sample}",
        "condition_key": f"{benchmark}_sample{sample}_len{int(d['gen_length'])}_block{int(d['block'])}_steps{int(d['steps'])}_thr{thr}",
    }


def is_condition_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if parse_condition_from_name(path.name) is None:
        return False
    return (path / "summary.jsonl").exists() or (path / "trace.jsonl").exists() or (path / "sample_traces").exists() or (path / "aggregate.csv").exists()


def find_condition_dirs(root: Path) -> List[Path]:
    root = root.expanduser().resolve()
    if is_condition_dir(root):
        return [root]
    out: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        if any(x in p.parts for x in ["sample_traces", "visual_trace", "visual", "overview"]):
            continue
        if is_condition_dir(p):
            out.append(p)
    return sorted({str(x): x for x in out}.values(), key=lambda p: str(p))


def infer_sample_idx_from_name(path: Path) -> Optional[int]:
    m = re.search(r"sample[_-]?(\d+)", str(path).lower())
    return int(m.group(1)) if m else None


def resolve_sample_dir(condition_dir: Path, sample_idx: Optional[int]) -> Optional[Path]:
    st = condition_dir / "sample_traces"
    if not st.exists():
        return None
    sample_dirs = sorted([x for x in st.iterdir() if x.is_dir() and x.name.startswith("sample_")])
    if not sample_dirs:
        return None
    if sample_idx is not None:
        wanted = st / f"sample_{sample_idx:04d}"
        if wanted.exists():
            return wanted
        for d in sample_dirs:
            if infer_sample_idx_from_name(d) == sample_idx:
                return d
    return sample_dirs[0]


# -----------------------------
# Step/block normalization
# -----------------------------

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

    if step_col is None:
        out["_step"] = range(len(out))
    else:
        out["_step"] = pd.to_numeric(out[step_col], errors="coerce").ffill().fillna(0).astype(int)
    out["_block"] = pd.to_numeric(out[block_col], errors="coerce").fillna(0).astype(int) if block_col else 0
    out["_local_round"] = pd.to_numeric(out[local_col], errors="coerce").fillna(0).astype(int) if local_col else out.groupby("_block").cumcount()

    if count_col:
        out["_selected_count"] = pd.to_numeric(out[count_col], errors="coerce").fillna(0)
    else:
        pos_col = choose_col(out, ["selected_positions"])
        out["_selected_count"] = out[pos_col].apply(lambda x: len(parse_listish(x))) if pos_col else 0

    if conf_col:
        out["_mean_confidence"] = pd.to_numeric(out[conf_col], errors="coerce")
    else:
        confs_col = choose_col(out, ["selected_confidences", "selected_confidence", "confidences"])
        if confs_col:
            out["_mean_confidence"] = out[confs_col].apply(lambda x: pd.Series([to_float(v) for v in parse_listish(x)]).dropna().mean())
        else:
            out["_mean_confidence"] = float("nan")

    if completion_col:
        out["_completion"] = pd.to_numeric(out[completion_col], errors="coerce")
    else:
        gen_length = to_float(metrics.get("gen_length"), float("nan"))
        if math.isfinite(gen_length) and gen_length > 0:
            out["_completion"] = out["_selected_count"].cumsum() / gen_length
        else:
            total = max(1.0, float(out["_selected_count"].sum()))
            out["_completion"] = out["_selected_count"].cumsum() / total

    return out.sort_values(["_step", "_block", "_local_round"]).reset_index(drop=True)


def parse_complete_cell(x: Any) -> Tuple[float, float, float]:
    s = str(x).strip()
    m = re.match(r"^(\d+)\s*/\s*(\d+)$", s)
    if not m:
        return float("nan"), float("nan"), float("nan")
    visible = float(m.group(1))
    total = float(m.group(2))
    return visible, total, (visible / total if total else float("nan"))


def count_masks_in_state(text: Any) -> int:
    s = str(text)
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
        complete_cols = sorted(complete_cols, key=lambda c: int(re.search(r"block_(\d+)_complete", c).group(1)))
        blocks = [int(re.search(r"block_(\d+)_complete", c).group(1)) for c in complete_cols]
        mat = []
        for c in complete_cols:
            mat.append([parse_complete_cell(v)[2] for v in block_timeline[c].tolist()])
        return blocks, steps, mat

    state_cols = [c for c in block_timeline.columns if re.match(r"block_\d+$", str(c))]
    if state_cols:
        state_cols = sorted(state_cols, key=lambda c: int(re.search(r"block_(\d+)$", c).group(1)))
        blocks = [int(re.search(r"block_(\d+)$", c).group(1)) for c in state_cols]
        mat = []
        for c in state_cols:
            mask_counts = [count_masks_in_state(v) for v in block_timeline[c].tolist()]
            block_len = max(mask_counts) if mask_counts else 1
            rates = [max(0, block_len - mc) / block_len for mc in mask_counts]
            mat.append(rates)
        return blocks, steps, mat

    return [], [], []


# -----------------------------
# Kendall-tau style AR-like order scoring
# -----------------------------

def token_commit_records(step_events: pd.DataFrame) -> List[Dict[str, Any]]:
    if step_events.empty:
        return []
    pos_col = choose_col(step_events, ["selected_positions"])
    tok_col = choose_col(step_events, ["selected_decoded_tokens", "selected_tokens", "decoded_tokens"])
    conf_col = choose_col(step_events, ["selected_confidences", "selected_confidence", "confidences"])
    recs: List[Dict[str, Any]] = []
    if not pos_col:
        return recs
    for _, r in step_events.iterrows():
        positions = parse_listish(r.get(pos_col))
        tokens = parse_listish(r.get(tok_col)) if tok_col else []
        confs = parse_listish(r.get(conf_col)) if conf_col else []
        for j, pos in enumerate(positions):
            recs.append({
                "position": to_int(pos, -1),
                "token": str(tokens[j]) if j < len(tokens) else "",
                "step": to_int(r.get("_step"), 0),
                "block": to_int(r.get("_block"), 0),
                "confidence": to_float(confs[j], float("nan")) if j < len(confs) else float("nan"),
            })
    return [r for r in recs if r["position"] >= 0]


def commit_order_dataframe(step_events: pd.DataFrame, metrics: Dict[str, Any]) -> pd.DataFrame:
    recs = token_commit_records(step_events)
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs)
    for col in ["position", "step", "block", "confidence"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    block_len = to_int(metrics.get("block_length", metrics.get("gen_blocksize")), 0)
    max_block = int(df["block"].max()) if df["block"].notna().any() else 0
    max_pos = int(df["position"].max()) if df["position"].notna().any() else 0
    if block_len > 0 and max_block > 0 and max_pos < block_len:
        df["global_position"] = df["block"].fillna(0).astype(int) * block_len + df["position"].fillna(0).astype(int)
    else:
        df["global_position"] = df["position"].fillna(0).astype(int)

    df = df.sort_values(["global_position", "step"]).drop_duplicates("global_position", keep="first")
    df = df.sort_values("global_position").reset_index(drop=True)
    max_step = max(1.0, float(pd.to_numeric(df["step"], errors="coerce").max()))
    df["commit_time_norm"] = pd.to_numeric(df["step"], errors="coerce") / max_step
    return df


def kendall_tau_no_x_ties(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float]:
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan")
    concordant = 0
    discordant = 0
    total = 0
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
    tau = (concordant - discordant) / total
    inversion_rate = discordant / total
    return tau, inversion_rate


def compute_order_metrics(step_events: pd.DataFrame, metrics: Dict[str, Any]) -> Dict[str, Any]:
    commit_df = commit_order_dataframe(step_events, metrics)
    if commit_df.empty:
        return {}

    xs = [float(x) for x in commit_df["global_position"].tolist()]
    ys = [float(y) for y in commit_df["step"].tolist()]
    tau, inversion_rate = kendall_tau_no_x_ties(xs, ys)

    total_len = max(to_int(metrics.get("gen_length"), 0), to_int(commit_df["global_position"].max(), -1) + 1)
    max_step = to_int(commit_df["step"].max(), 0)
    committed: set[int] = set()
    by_step: Dict[int, List[int]] = defaultdict(list)
    for _, r in commit_df.iterrows():
        by_step[to_int(r["step"], 0)].append(to_int(r["global_position"], -1))
    prefix_gaps: List[float] = []
    for step in range(max_step + 1):
        for pos in by_step.get(step, []):
            if pos >= 0:
                committed.add(pos)
        prefix_len = 0
        while prefix_len in committed:
            prefix_len += 1
        if total_len > 0:
            overall = len(committed) / total_len
            prefix = prefix_len / total_len
            prefix_gaps.append(overall - prefix)

    return {
        "order_kendall_tau": tau,
        "inversion_rate": inversion_rate,
        "left_to_right_score": (1.0 - inversion_rate) if math.isfinite(inversion_rate) else float("nan"),
        "mean_prefix_gap": sum(prefix_gaps) / len(prefix_gaps) if prefix_gaps else float("nan"),
        "commit_order_csv": None,  # filled later if exported
    }


# -----------------------------
# Per-sample: keep only one generation figure
# -----------------------------

def plot_generation_chain(step_events: pd.DataFrame, block_timeline: pd.DataFrame, metrics: Dict[str, Any], out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.0), gridspec_kw={"height_ratios": [1.55, 1.0]})
    fig.suptitle(title, fontsize=12)

    blocks, steps, mat = block_completion_matrix(block_timeline)
    ax = axes[0]
    if blocks and mat:
        im = ax.imshow(mat, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
        ax.set_ylabel("Block index")
        ax.set_xlabel("Generation step")
        ax.set_title("Generation chain by block")
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
        cbar.set_label("Within-block completion")
    else:
        ax.text(0.5, 0.5, "No block_timeline.csv / block completion data", ha="center", va="center")
        ax.set_axis_off()

    ax2 = axes[1]
    if not step_events.empty:
        ax2.plot(step_events["_step"], step_events["_completion"], marker="o", markersize=3, linewidth=1.6, label="overall completion")
        ax2_t = ax2.twinx()
        ax2_t.bar(step_events["_step"], step_events["_selected_count"], width=0.8, alpha=0.25, label="committed tokens / step")
        block_changes = step_events["_block"].ne(step_events["_block"].shift(1)).fillna(True)
        change_steps = step_events.loc[block_changes, "_step"].tolist()
        for s in change_steps:
            ax2.axvline(s, linestyle="--", linewidth=0.8, alpha=0.45)
        ax2.set_xlabel("Generation step")
        ax2.set_ylabel("Completion rate")
        ax2.set_ylim(-0.03, 1.03)
        ax2_t.set_ylabel("Committed tokens / step")
        ax2.set_title("Overall generation progress (dashed lines = block switches)")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No step_events.csv", ha="center", va="center")
        ax2.set_axis_off()

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


# -----------------------------
# Load condition metrics and create overview rows
# -----------------------------

def load_condition(condition_dir: Path, sample_idx: Optional[int]) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, Optional[Path]]:
    parsed = parse_condition_from_name(condition_dir.name) or {}
    metrics: Dict[str, Any] = dict(parsed)

    sample_dir = resolve_sample_dir(condition_dir, sample_idx)
    step_events = pd.DataFrame()
    block_timeline = pd.DataFrame()

    if sample_dir is not None:
        metrics.update(read_json(sample_dir / "sample_metrics.json") or read_json(sample_dir / "metrics.json"))
        step_events = read_csv(sample_dir / "step_events.csv")
        block_timeline = read_csv(sample_dir / "block_timeline.csv")

    for csv_name in ["aggregate.csv", "run_summary.csv"]:
        df = read_csv(condition_dir / csv_name)
        if not df.empty:
            row = df.iloc[0].to_dict()
            for k, v in row.items():
                if k not in metrics or metrics.get(k) in (None, ""):
                    metrics[k] = v

    if not metrics:
        summary_rows = read_jsonl(condition_dir / "summary.jsonl")
        if summary_rows:
            metrics.update(summary_rows[0])

    # Normalize names
    if "gen_length" not in metrics and "param_gen_length" in metrics:
        metrics["gen_length"] = metrics["param_gen_length"]
    if "gen_steps" not in metrics and "param_gen_steps" in metrics:
        metrics["gen_steps"] = metrics["param_gen_steps"]
    if "gen_blocksize" not in metrics and "param_gen_blocksize" in metrics:
        metrics["gen_blocksize"] = metrics["param_gen_blocksize"]
    if "block_length" not in metrics and "gen_blocksize" in metrics:
        metrics["block_length"] = metrics["gen_blocksize"]
    if "primary_metric_value" not in metrics and "metric_value" in metrics:
        metrics["primary_metric_value"] = metrics["metric_value"]
    if "actual_arness" not in metrics and "actual_arness_mean" in metrics:
        metrics["actual_arness"] = metrics["actual_arness_mean"]
    if "threshold_pass_rate" not in metrics and "threshold_pass_rate_mean" in metrics:
        metrics["threshold_pass_rate"] = metrics["threshold_pass_rate_mean"]
    if "fallback_rate" not in metrics and "fallback_rate_mean" in metrics:
        metrics["fallback_rate"] = metrics["fallback_rate_mean"]
    if "tokens_per_second" not in metrics and "tokens_per_second_mean" in metrics:
        metrics["tokens_per_second"] = metrics["tokens_per_second_mean"]
    if "elapsed_seconds" not in metrics and "latency_mean_s" in metrics:
        metrics["elapsed_seconds"] = metrics["latency_mean_s"]
    if "actual_parallelism" not in metrics and "effective_parallelism_mean" in metrics:
        metrics["actual_parallelism"] = metrics["effective_parallelism_mean"]
    if "threshold_label" not in metrics:
        metrics["threshold_label"] = parsed.get("threshold_label", "none")
    metrics["threshold_label"] = canonical_threshold_label(metrics.get("threshold_label"))
    metrics["benchmark"] = metrics.get("benchmark") or parsed.get("benchmark", "unknown")
    metrics["sample_idx"] = metrics.get("sample_idx", parsed.get("sample_idx"))
    metrics["experiment"] = metrics.get("experiment", parsed.get("experiment", condition_dir.parent.name))
    metrics["condition_key"] = metrics.get("condition_key", parsed.get("condition_key", condition_dir.name))
    metrics["condition_dir"] = str(condition_dir)

    # Cast important numeric fields
    for key in ["gen_length", "gen_steps", "gen_blocksize", "block_length", "primary_metric_value", "completion_rate", "actual_parallelism", "actual_arness", "threshold_pass_rate", "fallback_rate", "elapsed_seconds", "tokens_per_second"]:
        if key in metrics:
            metrics[key] = to_float(metrics[key], metrics[key] if isinstance(metrics[key], (int, float)) else float("nan"))
    if math.isfinite(to_float(metrics.get("gen_length"), float("nan"))) and math.isfinite(to_float(metrics.get("gen_steps"), float("nan"))) and to_float(metrics.get("gen_steps")) > 0:
        metrics["planned_parallelism"] = to_float(metrics["gen_length"]) / to_float(metrics["gen_steps"])
    else:
        metrics["planned_parallelism"] = float("nan")

    step_events = normalize_step_events(step_events, metrics)
    order_metrics = compute_order_metrics(step_events, metrics)
    metrics.update(order_metrics)

    return metrics, step_events, block_timeline, sample_dir


# -----------------------------
# Overview plots (compact only)
# -----------------------------

def _plot_metric_lines_by_parallelism(ax, df: pd.DataFrame, col: str, ylabel: str, thresholds: Sequence[str]) -> None:
    plot_df = df.copy()
    plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df["planned_parallelism"] = pd.to_numeric(plot_df["planned_parallelism"], errors="coerce")
    plot_df = plot_df.dropna(subset=["planned_parallelism", col])
    for thr in thresholds:
        g = plot_df[plot_df["threshold_label"] == thr].sort_values("planned_parallelism")
        if g.empty:
            continue
        ax.plot(g["planned_parallelism"], g[col], marker="o", linewidth=1.8, markersize=5, label=f"thr={thr}")
        for _, r in g.iterrows():
            ax.annotate(str(to_int(r.get("gen_steps"), 0)), (r["planned_parallelism"], r[col]), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("Planned parallelism = gen_length / gen_steps")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def plot_overall_comparison(exp_df: pd.DataFrame, out_path: Path, title: str) -> None:
    """Big comparison for main thresholds.

    This plot is intentionally compact: it makes speed benefit visible, while still
    showing score and AR-like order behavior.
    """
    df = exp_df[exp_df["threshold_label"].isin(MAIN_THRESHOLDS)].copy()
    if df.empty:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4))
    specs = [
        ("primary_metric_value", "Score / accuracy"),
        ("elapsed_seconds", "Latency / sample (s)"),
        ("tokens_per_second", "Tokens / second"),
        ("actual_parallelism", "Actual parallelism"),
        ("order_kendall_tau", "Kendall tau"),
        ("fallback_rate", "Fallback rate"),
    ]

    for ax, (col, ylabel) in zip(axes.ravel(), specs):
        _plot_metric_lines_by_parallelism(ax, df, col, ylabel, ["none", "0p6", "0p8"])

    fig.suptitle(title + " | main comparison: quality, speed, and AR-like order", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


def _plot_metric_lines_by_threshold(ax, df: pd.DataFrame, col: str, ylabel: str, steps: Sequence[int]) -> None:
    plot_df = df.copy()
    plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df = plot_df.dropna(subset=[col])
    order_map = {thr: i for i, thr in enumerate(ALL_THRESHOLDS_ORDER)}
    for step in steps:
        g = plot_df[plot_df["gen_steps"] == step].copy()
        if g.empty:
            continue
        g["thr_order"] = g["threshold_label"].map(lambda x: order_map.get(canonical_threshold_label(x), 999))
        g = g.sort_values("thr_order")
        xs = [order_map.get(canonical_threshold_label(x), 999) for x in g["threshold_label"]]
        ax.plot(xs, g[col], marker="o", linewidth=1.8, markersize=5, label=f"steps={step}")
    ax.set_xlabel("Threshold")
    ax.set_xticks(list(range(len(ALL_THRESHOLDS_ORDER))))
    ax.set_xticklabels(ALL_THRESHOLDS_ORDER)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def plot_top2_steps_thresholds(exp_df: pd.DataFrame, out_path: Path, title: str) -> None:
    """For the two longest step budgets, show all five thresholds."""
    if exp_df.empty:
        return
    steps_sorted = sorted([int(x) for x in pd.to_numeric(exp_df["gen_steps"], errors="coerce").dropna().unique()], reverse=True)
    top2 = steps_sorted[:2]
    if not top2:
        return
    df = exp_df[exp_df["gen_steps"].isin(top2)].copy()
    if df.empty:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4))
    specs = [
        ("primary_metric_value", "Score / accuracy"),
        ("elapsed_seconds", "Latency / sample (s)"),
        ("tokens_per_second", "Tokens / second"),
        ("actual_parallelism", "Actual parallelism"),
        ("order_kendall_tau", "Kendall tau"),
        ("fallback_rate", "Fallback rate"),
    ]

    for ax, (col, ylabel) in zip(axes.ravel(), specs):
        _plot_metric_lines_by_threshold(ax, df, col, ylabel, top2)

    fig.suptitle(title + f" | two longest step budgets, five thresholds ({', '.join(map(str, top2))})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


def _read_commit_order(path_value: Any) -> pd.DataFrame:
    if path_value is None:
        return pd.DataFrame()
    s = str(path_value)
    if not s or s.lower() in {"nan", "none"}:
        return pd.DataFrame()
    return read_csv(Path(s))


def _draw_commit_order_panel(ax, cdf: pd.DataFrame, row: pd.Series, title: str) -> None:
    if cdf.empty or "global_position" not in cdf.columns or "step" not in cdf.columns:
        ax.text(0.5, 0.5, "missing\ncommit_order", ha="center", va="center", fontsize=9)
        ax.set_axis_off()
        return
    x = pd.to_numeric(cdf["global_position"], errors="coerce")
    y = pd.to_numeric(cdf["step"], errors="coerce")
    conf = pd.to_numeric(cdf.get("confidence", pd.Series([float("nan")] * len(cdf))), errors="coerce")
    valid = x.notna() & y.notna()
    x = x[valid]
    y = y[valid]
    conf = conf[valid]
    if x.empty:
        ax.text(0.5, 0.5, "empty\ncommit_order", ha="center", va="center", fontsize=9)
        ax.set_axis_off()
        return

    # confidence as alpha-like size; keep color default-free enough and interpretable.
    sizes = 10 + 24 * conf.fillna(0.5).clip(lower=0, upper=1)
    ax.scatter(x, y, s=sizes, alpha=0.75, linewidths=0)

    block = to_int(row.get("block_length", row.get("gen_blocksize")), 0)
    gen_len = to_int(row.get("gen_length"), 0)
    if block > 0 and gen_len > 0:
        for b in range(block, gen_len, block):
            ax.axvline(b, linestyle="--", linewidth=0.7, alpha=0.35)

    tau = to_float(row.get("order_kendall_tau"), float("nan"))
    lat = to_float(row.get("elapsed_seconds"), float("nan"))
    subtitle = title
    if math.isfinite(tau):
        subtitle += f"\nτ={tau:.2f}"
    if math.isfinite(lat):
        subtitle += f", {lat:.1f}s"
    ax.set_title(subtitle, fontsize=9)
    ax.set_xlabel("Token position", fontsize=8)
    ax.set_ylabel("Commit step", fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, alpha=0.25)


def plot_token_order_grid_main(exp_df: pd.DataFrame, out_path: Path, title: str) -> None:
    """Direct token-position comparison: rows=gen_steps, cols=null/0.6/0.8."""
    df = exp_df[exp_df["threshold_label"].isin(MAIN_THRESHOLDS)].copy()
    if df.empty or "commit_order_csv" not in df.columns:
        return
    steps = sorted([int(x) for x in pd.to_numeric(df["gen_steps"], errors="coerce").dropna().unique()], reverse=True)
    thrs = ["none", "0p6", "0p8"]
    if not steps:
        return

    fig, axes = plt.subplots(len(steps), len(thrs), figsize=(5.0 * len(thrs), max(3.0, 2.6 * len(steps))), squeeze=False)
    for i, step in enumerate(steps):
        for j, thr in enumerate(thrs):
            ax = axes[i][j]
            sub = df[(df["gen_steps"] == step) & (df["threshold_label"] == thr)]
            if sub.empty:
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
                ax.set_axis_off()
                continue
            row = sub.iloc[0]
            cdf = _read_commit_order(row.get("commit_order_csv"))
            _draw_commit_order_panel(ax, cdf, row, f"steps={step}, thr={thr}")

    fig.suptitle(title + " | token position vs first commit step (main thresholds)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


def plot_token_order_grid_top2_thresholds(exp_df: pd.DataFrame, out_path: Path, title: str) -> None:
    """Direct token-position comparison: rows=top2 gen_steps, cols=all five thresholds."""
    if exp_df.empty or "commit_order_csv" not in exp_df.columns:
        return
    steps = sorted([int(x) for x in pd.to_numeric(exp_df["gen_steps"], errors="coerce").dropna().unique()], reverse=True)[:2]
    thrs = ALL_THRESHOLDS_ORDER
    if not steps:
        return

    fig, axes = plt.subplots(len(steps), len(thrs), figsize=(4.25 * len(thrs), max(3.0, 2.8 * len(steps))), squeeze=False)
    for i, step in enumerate(steps):
        for j, thr in enumerate(thrs):
            ax = axes[i][j]
            sub = exp_df[(exp_df["gen_steps"] == step) & (exp_df["threshold_label"] == thr)]
            if sub.empty:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=9)
                ax.set_axis_off()
                continue
            row = sub.iloc[0]
            cdf = _read_commit_order(row.get("commit_order_csv"))
            _draw_commit_order_panel(ax, cdf, row, f"steps={step}, thr={thr}")

    fig.suptitle(title + " | token order comparison for two longest step budgets", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[OK] wrote {out_path}")




# -----------------------------
# Main driver
# -----------------------------

def visualize_condition(condition_dir: Path, sample_idx: Optional[int], out_root: Path) -> Dict[str, Any]:
    metrics, step_events, block_timeline, sample_dir = load_condition(condition_dir, sample_idx)
    sample_name = sample_dir.name if sample_dir is not None else f"sample_{int(metrics.get('sample_idx') or 0):04d}"
    title = (
        f"{metrics.get('benchmark')} sample {metrics.get('sample_idx')} | "
        f"len={to_int(metrics.get('gen_length'), 0)}, block={to_int(metrics.get('block_length'), 0)}, "
        f"steps={to_int(metrics.get('gen_steps'), 0)}, thr={metrics.get('threshold_label')}"
    )
    out_dir = out_root / str(metrics.get("experiment")) / condition_dir.name / sample_name
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_generation_chain(step_events, block_timeline, metrics, out_dir / "generation_chain.png", title)

    commit_df = commit_order_dataframe(step_events, metrics)
    if not commit_df.empty:
        commit_path = out_dir / "commit_order.csv"
        commit_df.to_csv(commit_path, index=False)
        metrics["commit_order_csv"] = str(commit_path)

    metrics_path = out_dir / "sample_overview.json"
    write_text(metrics_path, json.dumps(metrics, ensure_ascii=False, indent=2))
    metrics["visual_output_dir"] = str(out_dir)
    print(f"[DONE] visual trace output: {out_dir}")
    return metrics


def write_overview(all_metrics: List[Dict[str, Any]], out_root: Path) -> None:
    if not all_metrics:
        return
    out_root.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_metrics)
    df["threshold_label"] = df["threshold_label"].apply(canonical_threshold_label)
    df.to_csv(out_root / "all_conditions_overview.csv", index=False)
    print(f"[OK] wrote {out_root / 'all_conditions_overview.csv'}")

    for exp, exp_df in df.groupby("experiment", dropna=False):
        exp_dir = out_root / safe_name(exp)
        exp_dir.mkdir(parents=True, exist_ok=True)
        exp_df = exp_df.copy().sort_values(["gen_steps", "threshold_label"], ascending=[False, True])
        keep_cols = [
            "benchmark", "sample_idx", "gen_length", "gen_steps", "block_length", "threshold_label",
            "planned_parallelism", "primary_metric_value", "completion_rate", "actual_parallelism",
            "actual_arness", "threshold_pass_rate", "fallback_rate", "elapsed_seconds",
            "tokens_per_second", "order_kendall_tau", "inversion_rate", "mean_prefix_gap",
            "condition_dir", "visual_output_dir",
        ]
        exp_df[[c for c in keep_cols if c in exp_df.columns]].to_csv(exp_dir / "overview_metrics.csv", index=False)
        print(f"[OK] wrote {exp_dir / 'overview_metrics.csv'}")
        plot_overall_comparison(exp_df, exp_dir / "overall_comparison.png", str(exp))
        plot_top2_steps_thresholds(exp_df, exp_dir / "top2_steps_thresholds.png", str(exp))
        plot_token_order_grid_main(exp_df, exp_dir / "token_order_grid_main.png", str(exp))
        plot_token_order_grid_top2_thresholds(exp_df, exp_dir / "token_order_grid_top2_thresholds.png", str(exp))


def main() -> None:
    parser = argparse.ArgumentParser(description="Simplified iLLaDA ARness visualizer.")
    parser.add_argument("path", nargs="?", default="outputs/arness", help="ARness root / experiment / condition. Default: outputs/arness")
    parser.add_argument("--sample-idx", type=int, default=None, help="Sample index to use when multiple sample traces exist.")
    parser.add_argument("--out", default="visual/arness", help="Output root. Default: visual/arness")
    parser.add_argument("--overview-only", action="store_true", help="Only rebuild overview plots/CSVs, skip per-condition generation figures.")
    args = parser.parse_args()

    input_root = Path(args.path).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()
    condition_dirs = find_condition_dirs(input_root)
    if not condition_dirs:
        raise SystemExit(f"[ERROR] no ARness condition dirs found under: {input_root}")

    metrics_rows: List[Dict[str, Any]] = []
    for cond_dir in condition_dirs:
        try:
            if args.overview_only:
                metrics, _, _, _ = load_condition(cond_dir, args.sample_idx)
            else:
                metrics = visualize_condition(cond_dir, args.sample_idx, out_root)
            metrics_rows.append(metrics)
        except Exception as exc:
            print(f"[WARN] skip {cond_dir}: {exc}")

    write_overview(metrics_rows, out_root / "overview")
    print(f"[DONE] conditions visualized/indexed: {len(metrics_rows)}")
    print(f"[DONE] visual root: {out_root}")


if __name__ == "__main__":
    main()
