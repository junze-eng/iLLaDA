#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arness_visual_trace.py

Single-run / single-sample visualizer for iLLaDA ARness trace case studies.

Expected canonical input after running trace.py / export_trace.py:

  outputs/arness/<experiment>/<condition>/
    summary.jsonl
    trace.jsonl
    sample_traces/
      sample_0007/
        block_timeline.csv
        step_events.csv
        block_metrics.csv
        sample_metrics.json
        final_prediction.txt
        problem_groundtruth_prediction.txt

It also works on a condition directory that only contains trace.jsonl + summary.jsonl,
although block-level timeline figures will be skipped.

Examples:

  python arness_visual_trace.py \
    "C:\\Users\\whaletech002\\Desktop\\whaletech\\iLLaDA\\outputs\\arness\\arness_trace_gsm8k_sample7\\gsm8k_sample7_len256_block64_steps64_thr0p6"

  python arness_visual_trace.py outputs/arness/arness_trace_gsm8k_sample7/gsm8k_sample7_len256_block64_steps64_thr0p6 --sample-idx 7

Outputs are written to:
  <condition_dir>/visual_trace/sample_0007/

Figures:
  - trace_dashboard.png
  - block_completion_heatmap.png
  - block_metrics.png
  - commit_confidence_hist.png
  - token_commit_timeline.html
  - trace_summary.md
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd


# -----------------------------
# IO helpers
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


def safe_name(s: Any) -> str:
    s = str(s if s is not None else "none")
    s = s.replace(".", "p")
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "none"


def parse_listish(x: Any) -> List[Any]:
    """Parse JSON/python-list-like strings from CSV cells."""
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
    # Fallback: comma separated.
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    return [s]


# -----------------------------
# Artifact discovery
# -----------------------------


def infer_sample_idx_from_name(path: Path) -> Optional[int]:
    m = re.search(r"sample[_-]?(\d+)", str(path).lower())
    return int(m.group(1)) if m else None


def resolve_sample_dir(input_path: Path, sample_idx: Optional[int]) -> Tuple[Path, Optional[Path]]:
    """Return (condition_dir, sample_dir_or_None)."""
    p = input_path.expanduser().resolve()

    # Direct sample directory.
    if (p / "sample_metrics.json").exists() or (p / "step_events.csv").exists():
        return p.parent.parent if p.parent.name == "sample_traces" else p, p

    # Condition directory with sample_traces.
    st = p / "sample_traces"
    if st.exists():
        sample_dirs = sorted([x for x in st.iterdir() if x.is_dir() and x.name.startswith("sample_")])
        if not sample_dirs:
            return p, None
        if sample_idx is not None:
            wanted = st / f"sample_{sample_idx:04d}"
            if wanted.exists():
                return p, wanted
            # allow non-padded folder names
            for d in sample_dirs:
                if infer_sample_idx_from_name(d) == sample_idx:
                    return p, d
            print(f"[WARN] sample_idx={sample_idx} not found under {st}; using {sample_dirs[0].name}")
        return p, sample_dirs[0]

    # Experiment directory: try to find condition/sample underneath.
    sample_dirs = sorted(p.glob("**/sample_traces/sample_*"))
    if sample_dirs:
        if sample_idx is not None:
            for d in sample_dirs:
                if infer_sample_idx_from_name(d) == sample_idx:
                    return d.parent.parent, d
        return sample_dirs[0].parent.parent, sample_dirs[0]

    # No exported sample artifacts. Maybe raw condition dir with trace.jsonl.
    if (p / "trace.jsonl").exists():
        return p, None

    raise SystemExit(
        f"[ERROR] Cannot find sample trace artifacts under: {p}\n"
        "Expected either a condition dir with sample_traces/sample_XXXX, "
        "a sample_XXXX dir, or a dir containing trace.jsonl."
    )


# -----------------------------
# Raw trace fallback
# -----------------------------


def load_raw_trace_as_step_events(condition_dir: Path, sample_idx: Optional[int]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = read_jsonl(condition_dir / "trace.jsonl")
    if not rows:
        return pd.DataFrame(), {}
    if sample_idx is None:
        sample_idx = infer_sample_idx_from_name(condition_dir) or to_int(rows[0].get("sample_idx", rows[0].get("sample_id")), 0)
    rows = [r for r in rows if to_int(r.get("sample_idx", r.get("sample_id")), -1) == sample_idx]
    rows.sort(key=lambda r: (to_int(r.get("step_idx"), 0), to_int(r.get("block_idx"), 0)))
    summary_rows = read_jsonl(condition_dir / "summary.jsonl")
    summary = {}
    for r in summary_rows:
        if to_int(r.get("sample_idx", r.get("sample_id")), -1) == sample_idx:
            summary = r
            break
    if not summary and summary_rows:
        summary = summary_rows[0]
    return pd.DataFrame(rows), summary


# -----------------------------
# Data normalization
# -----------------------------


def normalize_step_events(df: pd.DataFrame, metrics: Dict[str, Any]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()

    step_col = choose_col(out, ["generation_step", "global_step_idx", "step_idx", "step"])
    block_col = choose_col(out, ["active_block_idx", "active_block", "block_idx"])
    local_col = choose_col(out, ["block_local_round", "local_step_idx", "local_round"])
    count_col = choose_col(out, ["selected_count", "actual_transfer_count", "committed_count", "num_selected", "transfer_count"])
    conf_col = choose_col(out, ["mean_confidence", "selected_confidence_mean", "avg_confidence", "confidence_mean"])
    completion_col = choose_col(out, ["current_completion_rate", "completion_rate", "overall_completion"])
    mask_before_col = choose_col(out, ["mask_count_before"])
    mask_after_col = choose_col(out, ["mask_count_after", "final_mask_count"])
    reason_col = choose_col(out, ["transfer_reason", "reason"])

    if step_col is None:
        out["_step"] = range(len(out))
    else:
        out["_step"] = pd.to_numeric(out[step_col], errors="coerce")
        if out["_step"].isna().all():
            out["_step"] = range(len(out))
        out["_step"] = out["_step"].fillna(method="ffill").fillna(0).astype(int)

    out["_block"] = pd.to_numeric(out[block_col], errors="coerce").fillna(0).astype(int) if block_col else 0
    out["_local_round"] = pd.to_numeric(out[local_col], errors="coerce").fillna(0).astype(int) if local_col else out.groupby("_block").cumcount()

    if count_col:
        out["_selected_count"] = pd.to_numeric(out[count_col], errors="coerce").fillna(0)
    else:
        pos_col = choose_col(out, ["selected_positions"])
        if pos_col:
            out["_selected_count"] = out[pos_col].apply(lambda x: len(parse_listish(x)))
        else:
            out["_selected_count"] = 0

    if conf_col:
        out["_mean_confidence"] = pd.to_numeric(out[conf_col], errors="coerce")
    else:
        confs_col = choose_col(out, ["selected_confidences", "confidences"])
        if confs_col:
            out["_mean_confidence"] = out[confs_col].apply(
                lambda x: pd.Series([to_float(v) for v in parse_listish(x)]).dropna().mean()
            )
        else:
            out["_mean_confidence"] = float("nan")

    if completion_col:
        out["_completion"] = pd.to_numeric(out[completion_col], errors="coerce")
    else:
        gen_length = to_float(metrics.get("gen_length"), float("nan"))
        if not math.isnan(gen_length) and gen_length > 0:
            out["_completion"] = out["_selected_count"].cumsum() / gen_length
        elif mask_after_col and mask_before_col:
            initial_mask = to_float(out[mask_before_col].iloc[0], float("nan"))
            out["_completion"] = 1 - pd.to_numeric(out[mask_after_col], errors="coerce") / initial_mask
        else:
            total = max(1.0, float(out["_selected_count"].sum()))
            out["_completion"] = out["_selected_count"].cumsum() / total

    out["_mask_before"] = pd.to_numeric(out[mask_before_col], errors="coerce") if mask_before_col else float("nan")
    out["_mask_after"] = pd.to_numeric(out[mask_after_col], errors="coerce") if mask_after_col else float("nan")
    out["_reason"] = out[reason_col].astype(str) if reason_col else ""

    out = out.sort_values(["_step", "_block", "_local_round"]).reset_index(drop=True)
    return out


def parse_complete_cell(x: Any) -> Tuple[float, float, float]:
    """Return visible, total, rate for cells like '12/64'."""
    s = str(x).strip()
    m = re.match(r"^(\d+)\s*/\s*(\d+)$", s)
    if not m:
        return float("nan"), float("nan"), float("nan")
    visible = float(m.group(1))
    total = float(m.group(2))
    rate = visible / total if total else float("nan")
    return visible, total, rate


def block_completion_matrix(block_timeline: pd.DataFrame) -> Tuple[List[int], List[int], List[List[float]]]:
    if block_timeline.empty:
        return [], [], []
    complete_cols = [c for c in block_timeline.columns if re.match(r"block_\d+_complete$", str(c))]
    if not complete_cols:
        # Some older chain files may use block_00_final_complete only in metrics, not timeline.
        return [], [], []
    def block_id(c: str) -> int:
        m = re.search(r"block_(\d+)_complete", c)
        return int(m.group(1)) if m else 0
    complete_cols = sorted(complete_cols, key=block_id)
    blocks = [block_id(c) for c in complete_cols]
    step_col = choose_col(block_timeline, ["generation_step", "step_idx", "global_step_idx", "step"])
    steps = list(range(len(block_timeline))) if step_col is None else [to_int(v, i) for i, v in enumerate(block_timeline[step_col])]
    mat: List[List[float]] = []
    for c in complete_cols:
        mat.append([parse_complete_cell(v)[2] for v in block_timeline[c].tolist()])
    return blocks, steps, mat


# -----------------------------
# Plotting
# -----------------------------


def condition_title(condition_dir: Path, sample_dir: Optional[Path], metrics: Dict[str, Any]) -> str:
    bench = metrics.get("benchmark") or "unknown"
    sample = metrics.get("sample_idx") or infer_sample_idx_from_name(sample_dir or condition_dir) or "?"
    gl = metrics.get("gen_length", "?")
    bl = metrics.get("block_length", metrics.get("gen_blocksize", "?"))
    gs = metrics.get("gen_steps", metrics.get("steps", "?"))
    th = metrics.get("threshold_label", metrics.get("threshold", "none"))
    return f"{bench} sample {sample} | len={gl}, block={bl}, steps={gs}, thr={th}"


def plot_dashboard(step_events: pd.DataFrame, metrics: Dict[str, Any], out_path: Path, title: str) -> None:
    if step_events.empty:
        print("[WARN] skip dashboard: no step events")
        return
    df = step_events
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2))
    fig.suptitle(title, fontsize=12)

    ax = axes[0, 0]
    ax.plot(df["_step"], df["_completion"], marker="o", linewidth=1.6, markersize=3)
    ax.set_xlabel("Generation step")
    ax.set_ylabel("Completion rate")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(df["_step"], df["_selected_count"], marker="o", linewidth=1.4, markersize=3)
    ax.set_xlabel("Generation step")
    ax.set_ylabel("Committed tokens / step")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if df["_mean_confidence"].notna().any():
        ax.plot(df["_step"], df["_mean_confidence"], marker="o", linewidth=1.4, markersize=3)
    ax.set_xlabel("Generation step")
    ax.set_ylabel("Mean selected confidence")
    ax.set_ylim(0, 1.03)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    # Active block as a step function; jitter-free and useful for block boundary reading.
    ax.step(df["_step"], df["_block"], where="post", linewidth=1.8)
    ax.scatter(df["_step"], df["_block"], s=(df["_selected_count"].clip(lower=0) + 1) * 10, alpha=0.7)
    ax.set_xlabel("Generation step")
    ax.set_ylabel("Active block index")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


def plot_block_completion(block_timeline: pd.DataFrame, out_path: Path, title: str) -> None:
    blocks, steps, mat = block_completion_matrix(block_timeline)
    if not blocks or not mat:
        print("[WARN] skip block completion heatmap: no block_*_complete columns")
        return
    fig, ax = plt.subplots(figsize=(14, max(3.2, 0.42 * len(blocks))))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title(title + " | block completion heatmap")
    ax.set_xlabel("Timeline row / generation order")
    ax.set_ylabel("Block index")
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
    cbar.set_label("Completion rate within block")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


def plot_block_metrics(block_metrics: pd.DataFrame, out_path: Path, title: str) -> None:
    if block_metrics.empty:
        print("[WARN] skip block metrics: no block_metrics.csv")
        return
    df = block_metrics.copy()
    block_col = choose_col(df, ["block_idx", "active_block_idx", "block"])
    if block_col is None:
        df["_block"] = range(len(df))
    else:
        df["_block"] = pd.to_numeric(df[block_col], errors="coerce").fillna(0).astype(int)
    total_col = choose_col(df, ["selected_tokens_total", "total_selected", "committed_tokens_total"])
    conf_col = choose_col(df, ["mean_confidence", "selected_confidence_mean", "avg_confidence"])
    rounds_col = choose_col(df, ["local_rounds", "block_steps", "rounds"])
    fallback_col = choose_col(df, ["fallback_steps", "fallback_count"])

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.4))
    fig.suptitle(title + " | block metrics", fontsize=12)

    ax = axes[0, 0]
    if total_col:
        ax.bar(df["_block"], pd.to_numeric(df[total_col], errors="coerce"))
    ax.set_xlabel("Block")
    ax.set_ylabel("Selected tokens total")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[0, 1]
    if rounds_col:
        ax.bar(df["_block"], pd.to_numeric(df[rounds_col], errors="coerce"))
    ax.set_xlabel("Block")
    ax.set_ylabel("Local rounds")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 0]
    if conf_col:
        ax.bar(df["_block"], pd.to_numeric(df[conf_col], errors="coerce"))
    ax.set_xlabel("Block")
    ax.set_ylabel("Mean confidence")
    ax.set_ylim(0, 1.03)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 1]
    if fallback_col:
        ax.bar(df["_block"], pd.to_numeric(df[fallback_col], errors="coerce"))
    ax.set_xlabel("Block")
    ax.set_ylabel("Fallback steps")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


def plot_confidence_hist(step_events: pd.DataFrame, out_path: Path, title: str) -> None:
    if step_events.empty:
        return
    conf_col = choose_col(step_events, ["selected_confidences", "confidences"])
    values: List[float] = []
    if conf_col:
        for x in step_events[conf_col].tolist():
            values.extend([to_float(v) for v in parse_listish(x)])
    else:
        values.extend([to_float(v) for v in step_events.get("_mean_confidence", pd.Series(dtype=float)).tolist()])
    values = [v for v in values if math.isfinite(v)]
    if not values:
        print("[WARN] skip confidence hist: no confidence values")
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.hist(values, bins=20)
    ax.set_title(title + " | selected token confidence")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 1.03)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


# -----------------------------
# HTML token timeline
# -----------------------------


def token_commit_records(step_events: pd.DataFrame) -> List[Dict[str, Any]]:
    if step_events.empty:
        return []
    pos_col = choose_col(step_events, ["selected_positions"])
    tok_col = choose_col(step_events, ["selected_decoded_tokens", "selected_tokens", "decoded_tokens"])
    conf_col = choose_col(step_events, ["selected_confidences", "confidences"])
    recs: List[Dict[str, Any]] = []
    if pos_col and tok_col:
        for _, r in step_events.iterrows():
            positions = parse_listish(r.get(pos_col))
            tokens = parse_listish(r.get(tok_col))
            confs = parse_listish(r.get(conf_col)) if conf_col else []
            for j, pos in enumerate(positions):
                tok = tokens[j] if j < len(tokens) else ""
                conf = to_float(confs[j], float("nan")) if j < len(confs) else float("nan")
                recs.append({
                    "position": to_int(pos, -1),
                    "token": str(tok),
                    "step": to_int(r.get("_step"), 0),
                    "block": to_int(r.get("_block"), 0),
                    "confidence": conf,
                })
    else:
        # Fall back to aggregate per-step tokens without positions.
        tok_col = choose_col(step_events, ["selected_decoded_tokens", "selected_tokens", "decoded_tokens"])
        if tok_col:
            pos = 0
            for _, r in step_events.iterrows():
                for tok in parse_listish(r.get(tok_col)):
                    recs.append({
                        "position": pos,
                        "token": str(tok),
                        "step": to_int(r.get("_step"), 0),
                        "block": to_int(r.get("_block"), 0),
                        "confidence": float("nan"),
                    })
                    pos += 1
    recs = [r for r in recs if r["position"] >= 0]
    recs.sort(key=lambda r: r["position"])
    return recs


def confidence_to_alpha(conf: float) -> float:
    if not math.isfinite(conf):
        return 0.35
    return max(0.25, min(1.0, conf))


def step_to_bg(step: int, max_step: int) -> str:
    # Light for early, dark for late. No matplotlib colormap dependency for HTML.
    if max_step <= 0:
        q = 0.0
    else:
        q = step / max_step
    # Blue-ish: late = darker.
    base = int(245 - 145 * q)
    g = int(250 - 120 * q)
    b = int(255 - 50 * q)
    return f"rgb({base},{g},{b})"


def write_token_timeline_html(step_events: pd.DataFrame, metrics: Dict[str, Any], out_path: Path, title: str) -> None:
    recs = token_commit_records(step_events)
    if not recs:
        print("[WARN] skip token html: no selected token records")
        return
    max_step = max(r["step"] for r in recs)
    by_pos: Dict[int, Dict[str, Any]] = {}
    # Keep first commit per position.
    for r in recs:
        by_pos.setdefault(r["position"], r)
    ordered = [by_pos[p] for p in sorted(by_pos)]
    spans = []
    for r in ordered:
        bg = step_to_bg(r["step"], max_step)
        alpha = confidence_to_alpha(r["confidence"])
        tok = html.escape(r["token"])
        tooltip = html.escape(f"pos={r['position']} step={r['step']} block={r['block']} conf={r['confidence']:.4f}" if math.isfinite(r["confidence"]) else f"pos={r['position']} step={r['step']} block={r['block']}")
        spans.append(
            f'<span class="tok" title="{tooltip}" style="background:{bg}; opacity:{alpha:.2f}">{tok}</span>'
        )
    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 24px; }
    .meta { color: #444; margin-bottom: 16px; }
    .tok { display: inline-block; padding: 2px 4px; margin: 1px; border-radius: 4px; white-space: pre-wrap; }
    .legend { margin: 12px 0; font-size: 14px; }
    .box { max-width: 1100px; line-height: 1.9; }
    code { background: #f2f2f2; padding: 2px 4px; border-radius: 3px; }
    """
    metric_lines = []
    for k in ["metric_value", "completion_rate", "actual_parallelism", "actual_arness", "threshold_pass_rate", "fallback_rate"]:
        if k in metrics and metrics.get(k) not in (None, ""):
            metric_lines.append(f"<code>{html.escape(k)}={html.escape(str(metrics.get(k)))}</code>")
    body = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title><style>{css}</style></head>
<body>
<h1>{html.escape(title)}</h1>
<div class="meta">{' '.join(metric_lines)}</div>
<div class="legend">Token background: lighter = committed earlier, darker = committed later. Opacity roughly follows confidence.</div>
<div class="box">{''.join(spans)}</div>
</body></html>
"""
    write_text(out_path, body)
    print(f"[OK] wrote {out_path}")


# -----------------------------
# Summary markdown
# -----------------------------


def summarize_repeated_tokens(step_events: pd.DataFrame, top_k: int = 12) -> List[Tuple[str, int]]:
    tok_col = choose_col(step_events, ["selected_decoded_tokens", "selected_tokens", "decoded_tokens"])
    if not tok_col:
        return []
    cnt: Counter[str] = Counter()
    for x in step_events[tok_col].tolist():
        for tok in parse_listish(x):
            s = str(tok)
            if s.strip():
                cnt[s] += 1
    return cnt.most_common(top_k)


def write_summary_md(
    condition_dir: Path,
    sample_dir: Optional[Path],
    metrics: Dict[str, Any],
    step_events: pd.DataFrame,
    block_metrics: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    lines: List[str] = []
    lines.append(f"# ARness visual trace summary\n")
    lines.append(f"Condition: `{condition_dir}`  \n")
    if sample_dir:
        lines.append(f"Sample dir: `{sample_dir}`  \n")
    lines.append(f"\n## Config / metrics\n")
    keys = [
        "benchmark", "sample_idx", "gen_length", "block_length", "gen_steps", "threshold", "threshold_label",
        "metric_value", "completion_rate", "actual_parallelism", "actual_arness", "threshold_pass_rate",
        "fallback_rate", "final_mask_count", "elapsed_seconds", "tokens_per_second", "actual_commit_tps",
    ]
    for k in keys:
        if k in metrics and metrics.get(k) not in (None, ""):
            lines.append(f"- **{k}**: `{metrics.get(k)}`\n")

    if not step_events.empty:
        total_committed = float(step_events["_selected_count"].sum())
        total_steps = len(step_events)
        avg_commit = total_committed / max(1, total_steps)
        final_completion = step_events["_completion"].dropna().iloc[-1] if step_events["_completion"].notna().any() else float("nan")
        lines.append("\n## Derived trace stats\n")
        lines.append(f"- **trace steps**: `{total_steps}`\n")
        lines.append(f"- **total committed tokens in trace**: `{total_committed:.0f}`\n")
        lines.append(f"- **avg committed tokens / step**: `{avg_commit:.3f}`\n")
        if math.isfinite(float(final_completion)):
            lines.append(f"- **final observed completion**: `{float(final_completion):.4f}`\n")
        if step_events["_mean_confidence"].notna().any():
            lines.append(f"- **mean selected confidence**: `{step_events['_mean_confidence'].mean():.4f}`\n")
        if "_reason" in step_events.columns:
            reason_counts = step_events["_reason"].value_counts().to_dict()
            lines.append(f"- **transfer reasons**: `{reason_counts}`\n")

    repeated = summarize_repeated_tokens(step_events)
    if repeated:
        lines.append("\n## Most repeated committed tokens\n")
        for tok, n in repeated:
            lines.append(f"- `{tok}`: {n}\n")

    if not block_metrics.empty:
        lines.append("\n## Block metrics preview\n")
        preview = block_metrics.head(20).to_markdown(index=False)
        lines.append("\n" + preview + "\n")

    lines.append("\n## Generated files\n")
    for name in [
        "trace_dashboard.png",
        "block_completion_heatmap.png",
        "block_metrics.png",
        "commit_confidence_hist.png",
        "token_commit_timeline.html",
    ]:
        lines.append(f"- `{name}`\n")

    write_text(out_path, "".join(lines))
    print(f"[OK] wrote {out_path}")


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize one iLLaDA ARness trace condition/sample.")
    parser.add_argument("path", help="Condition dir, experiment dir, or sample_XXXX dir.")
    parser.add_argument("--sample-idx", type=int, default=None, help="Sample index to visualize, e.g. 7.")
    parser.add_argument("--out", default=None, help="Output directory. Default: <condition>/visual_trace/sample_XXXX")
    parser.add_argument("--no-html", action="store_true", help="Skip token timeline HTML.")
    args = parser.parse_args()

    condition_dir, sample_dir = resolve_sample_dir(Path(args.path), args.sample_idx)
    sample_idx = args.sample_idx
    if sample_idx is None:
        sample_idx = infer_sample_idx_from_name(sample_dir or condition_dir)

    if sample_dir is not None:
        metrics = read_json(sample_dir / "sample_metrics.json") or read_json(sample_dir / "metrics.json")
        step_events = read_csv(sample_dir / "step_events.csv")
        block_timeline = read_csv(sample_dir / "block_timeline.csv")
        block_metrics = read_csv(sample_dir / "block_metrics.csv")
        if not metrics:
            # fallback to condition summary
            summary_rows = read_jsonl(condition_dir / "summary.jsonl")
            if sample_idx is not None:
                for r in summary_rows:
                    if to_int(r.get("sample_idx", r.get("sample_id")), -1) == sample_idx:
                        metrics = r
                        break
            if not metrics and summary_rows:
                metrics = summary_rows[0]
    else:
        print("[WARN] no sample_traces/sample_XXXX found; falling back to raw trace.jsonl only.")
        print("       For full block timeline plots, run:")
        print(f"       python trace.py --runs-root {condition_dir.parent} --overwrite --compress-masks --write-task-index")
        step_events, metrics = load_raw_trace_as_step_events(condition_dir, sample_idx)
        block_timeline = pd.DataFrame()
        block_metrics = pd.DataFrame()

    step_events = normalize_step_events(step_events, metrics)

    # Fill missing essential metadata from path.
    if "sample_idx" not in metrics or metrics.get("sample_idx") in (None, ""):
        metrics["sample_idx"] = sample_idx
    if "benchmark" not in metrics or not metrics.get("benchmark"):
        lname = str(condition_dir).lower()
        metrics["benchmark"] = "gsm8k" if "gsm8k" in lname else "mbpp" if "mbpp" in lname else "unknown"
    if "threshold_label" not in metrics:
        m = re.search(r"thr([^\\/]+)$", condition_dir.name)
        metrics["threshold_label"] = m.group(1).replace("p", ".") if m else metrics.get("threshold", "none")

    title = condition_title(condition_dir, sample_dir, metrics)

    if args.out:
        out_dir = Path(args.out).expanduser().resolve()
    else:
        sample_name = sample_dir.name if sample_dir is not None else f"sample_{int(sample_idx or 0):04d}"
        out_dir = condition_dir / "visual_trace" / sample_name
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_dashboard(step_events, metrics, out_dir / "trace_dashboard.png", title)
    plot_block_completion(block_timeline, out_dir / "block_completion_heatmap.png", title)
    plot_block_metrics(block_metrics, out_dir / "block_metrics.png", title)
    plot_confidence_hist(step_events, out_dir / "commit_confidence_hist.png", title)
    if not args.no_html:
        write_token_timeline_html(step_events, metrics, out_dir / "token_commit_timeline.html", title)
    write_summary_md(condition_dir, sample_dir, metrics, step_events, block_metrics, out_dir / "trace_summary.md", title)

    # Also save normalized step events for later report plots.
    if not step_events.empty:
        step_events.to_csv(out_dir / "normalized_step_events.csv", index=False)
        print(f"[OK] wrote {out_dir / 'normalized_step_events.csv'}")

    print(f"[DONE] visual trace output: {out_dir}")


if __name__ == "__main__":
    main()
