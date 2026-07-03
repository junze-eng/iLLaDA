#!/usr/bin/env python3
"""Visualize iLLaDA context benchmark and record failure cases.

Default outputs are intentionally small:
  - context_combined.png
  - context_score_heatmap.png
  - context_table.csv / .md
  - context_errors.csv / .md
  - context_report.txt

It supports either:
  python visual_context.py --root outputs/context_short --out outputs/context_figs
  python visual_context.py --zip context.zip --out outputs/context_figs
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_VERSION = "visual_context_v3_errors"
POSITION_ORDER = ["front", "middle", "back"]

CTX_POS_RE = re.compile(r"ctx(?P<ctx>\d+)_pos(?P<pos>front|middle|back)|ctx(?P<ctx2>\d+)_(?P<pos2>front|middle|back)")


def parse_ctx_pos(path: str) -> Tuple[Optional[int], Optional[str]]:
    m = CTX_POS_RE.search(path.replace("\\", "/"))
    if not m:
        return None, None
    ctx = m.group("ctx") or m.group("ctx2")
    pos = m.group("pos") or m.group("pos2")
    return int(ctx), pos


def read_csv_from_zip(zf: zipfile.ZipFile, candidates: List[str]):
    names = set(zf.namelist())
    for name in candidates:
        if name in names:
            return pd.read_csv(io.BytesIO(zf.read(name))), name
    base_candidates = {Path(c).name for c in candidates}
    for name in zf.namelist():
        if Path(name).name in base_candidates:
            return pd.read_csv(io.BytesIO(zf.read(name))), name
    raise FileNotFoundError(f"Missing any of {candidates}")


def load_summary(root: Optional[str], zip_path: Optional[str]) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], str]:
    if zip_path:
        with zipfile.ZipFile(zip_path) as zf:
            aggregate, agg_name = read_csv_from_zip(zf, ["context/aggregate.csv", "aggregate.csv"])
            try:
                summary, _ = read_csv_from_zip(zf, ["context/summary_all.csv", "summary_all.csv"])
            except Exception:
                summary = None
        return aggregate, summary, f"{zip_path}:{agg_name}"

    root_path = Path(root or "outputs/context_short")
    agg = root_path / "aggregate.csv"
    summ = root_path / "summary_all.csv"
    if not agg.exists():
        raise FileNotFoundError(f"Cannot find {agg}. Pass --root or --zip.")
    aggregate = pd.read_csv(agg)
    summary = pd.read_csv(summ) if summ.exists() else None
    return aggregate, summary, str(agg)


def normalize_summary(aggregate: pd.DataFrame, summary: Optional[pd.DataFrame]) -> pd.DataFrame:
    df = aggregate.copy()

    if summary is not None and "run_name" in df.columns and "run_name" in summary.columns:
        keep = [
            "run_name",
            "latency_mean_s",
            "tokens_per_second_mean",
            "peak_vram",
            "completion_rate",
            "actual_parallelism",
            "actual_arness_mean",
            "gpu_util_mean",
        ]
        keep = [c for c in keep if c in summary.columns]
        if keep:
            sup = summary[keep].drop_duplicates(subset=["run_name"])
            df = df.merge(sup, on="run_name", how="left", suffixes=("", "_summary"))

    if "primary_metric_value" in df.columns:
        df["score"] = pd.to_numeric(df["primary_metric_value"], errors="coerce")
    elif "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
    else:
        raise ValueError("No score / primary_metric_value column found")

    if "latency_mean" not in df.columns:
        df["latency_mean"] = df.get("latency_mean_s", np.nan)
    df["latency_mean"] = pd.to_numeric(df["latency_mean"], errors="coerce")

    if "tokens_per_second_mean" not in df.columns:
        df["tokens_per_second_mean"] = np.nan
    df["tokens_per_second_mean"] = pd.to_numeric(df["tokens_per_second_mean"], errors="coerce")

    if "peak_vram" not in df.columns:
        df["peak_vram"] = df.get("peak_vram_summary", np.nan)
    df["peak_vram"] = pd.to_numeric(df["peak_vram"], errors="coerce")
    df["peak_vram_gb"] = df["peak_vram"].map(lambda x: x / 1024.0 if pd.notna(x) and x > 100 else x)

    if "completion_rate" not in df.columns:
        df["completion_rate"] = df.get("completion_rate_summary", np.nan)
    df["completion_rate"] = pd.to_numeric(df["completion_rate"], errors="coerce")

    df["context_length"] = pd.to_numeric(df["context_length"], errors="coerce").astype("Int64")
    df["needle_position"] = df["needle_position"].astype(str)
    df["needle_position"] = pd.Categorical(df["needle_position"], POSITION_ORDER, ordered=True)
    return df.sort_values(["context_length", "needle_position"]).reset_index(drop=True)


def make_table(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "context": df["context_length"].astype("Int64"),
        "position": df["needle_position"].astype(str),
        "score": df["score"].round(1),
        "avg_latency_s": df["latency_mean"].round(2),
        "tps": df["tokens_per_second_mean"].round(2),
        "peak_vram_gb": df["peak_vram_gb"].round(1),
        "completion_rate": df["completion_rate"].round(4),
    })
    if "num_samples" in df.columns:
        out["num_samples"] = pd.to_numeric(df["num_samples"], errors="coerce").astype("Int64")
    return out


def norm_text(x: str) -> str:
    return re.sub(r"\s+", " ", str(x).lower()).strip()


def parse_references(ref_obj) -> List[str]:
    if ref_obj is None:
        return []
    if isinstance(ref_obj, list):
        return [str(x) for x in ref_obj]
    if isinstance(ref_obj, (int, float)):
        return [str(ref_obj)]
    s = str(ref_obj)
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
        return [str(parsed)]
    except Exception:
        return [s]


def classify_failure(pred: str, refs: List[str]) -> str:
    p = str(pred or "")
    pl = p.lower()
    if not p.strip():
        return "empty"
    not_found_terms = [
        "no indication", "not found", "cannot find", "can't find", "could not find",
        "does not contain", "doesn't contain", "no specific number", "no number",
        "not mentioned", "there is no", "since there",
    ]
    if any(t in pl for t in not_found_terms):
        return "not_found_claim"
    gold_digits = set()
    for r in refs:
        gold_digits.update(re.findall(r"\d+", str(r)))
    pred_digits = re.findall(r"\d+", p)
    if pred_digits and gold_digits and not any(g in pred_digits for g in gold_digits):
        return "wrong_number"
    if len(p) >= 100 and ("<think>" in pl) and not any(norm_text(r) in norm_text(p) for r in refs):
        return "reasoning_no_answer"
    return "missing_answer"


def preview(text: str, max_chars: int = 240) -> str:
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def iter_result_json_from_zip(zip_path: str) -> Iterable[Tuple[str, dict]]:
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if "/results/" in name and name.endswith(".json") and Path(name).name == "ruler_niah_single_1.json":
                try:
                    yield name, json.loads(zf.read(name))
                except Exception:
                    continue


def iter_result_json_from_root(root: str) -> Iterable[Tuple[str, dict]]:
    root_path = Path(root)
    for p in root_path.rglob("ruler_niah_single_1.json"):
        if "results" not in p.parts:
            continue
        try:
            yield str(p), json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue


def collect_errors(root: Optional[str], zip_path: Optional[str], max_per_condition: int) -> pd.DataFrame:
    rows = []
    iterator = iter_result_json_from_zip(zip_path) if zip_path else iter_result_json_from_root(root or "outputs/context_short")

    for path, obj in iterator:
        ctx, pos = parse_ctx_pos(path)
        if ctx is None or pos is None:
            continue

        details = obj.get("details", {}) if isinstance(obj, dict) else {}
        for key, item in details.items():
            if key == "type" or not isinstance(item, dict):
                continue
            refs = parse_references(item.get("references"))
            pred = item.get("predictions", item.get("prediction", item.get("origin_prediction", "")))
            raw_prompt = item.get("prompt", item.get("input", ""))
            correct = any(norm_text(r) in norm_text(pred) for r in refs if r)
            if correct:
                continue

            rows.append({
                "context_length": ctx,
                "needle_position": pos,
                "sample_id": key,
                "gold": "; ".join(refs),
                "prediction_preview": preview(pred, 360),
                "failure_type": classify_failure(pred, refs),
                "source_result": path,
                "prompt_tail": preview(raw_prompt[-600:] if raw_prompt else "", 260),
            })

    if not rows:
        return pd.DataFrame(columns=[
            "context_length", "needle_position", "sample_id", "gold",
            "prediction_preview", "failure_type", "source_result", "prompt_tail"
        ])

    df = pd.DataFrame(rows)
    df["needle_position"] = pd.Categorical(df["needle_position"], POSITION_ORDER, ordered=True)
    df = df.sort_values(["context_length", "needle_position", "sample_id"]).reset_index(drop=True)

    if max_per_condition and max_per_condition > 0:
        df = df.groupby(["context_length", "needle_position"], observed=True, group_keys=False).head(max_per_condition).reset_index(drop=True)

    return df


def set_x_ticks(ax, df: pd.DataFrame):
    xs = sorted([int(x) for x in df["context_length"].dropna().unique()])
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs, [str(x) for x in xs])


def plot_heatmap(df: pd.DataFrame, path: Path):
    piv = df.pivot(index="needle_position", columns="context_length", values="score").reindex(POSITION_ORDER)
    piv = piv.dropna(how="all")
    vals = piv.values.astype(float)

    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    im = ax.imshow(vals, aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(piv.columns)), [str(int(c)) for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [str(i) for i in piv.index])
    ax.set_xlabel("Context length")
    ax.set_ylabel("Needle position")
    ax.set_title("RULER NIAH single_1 score heatmap")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, "NA" if np.isnan(vals[i, j]) else f"{vals[i, j]:.0f}", ha="center", va="center", fontsize=11)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Score")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_combined(df: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    specs = [
        ("score", "Score", "Retrieval score"),
        ("latency_mean", "Avg latency (s)", "Latency"),
        ("tokens_per_second_mean", "Output TPS", "Output throughput"),
        ("peak_vram_gb", "Peak VRAM (GB)", "Peak VRAM"),
    ]

    for ax, (metric, ylabel, title) in zip(axes.ravel(), specs):
        for pos in POSITION_ORDER:
            sub = df[df["needle_position"].astype(str) == pos]
            if sub.empty:
                continue
            ax.plot(sub["context_length"], sub[metric], marker="o", linewidth=2, label=pos)
        set_x_ticks(ax, df)
        if metric == "score":
            ax.set_ylim(0, 105)
        ax.set_xlabel("Context length")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)

    axes[0, 0].legend(title="Needle position")
    fig.suptitle("Context benchmark: quality and cost", fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def write_report(table: pd.DataFrame, errors: pd.DataFrame, source: str, out_dir: Path):
    lines = []
    lines.append(SCRIPT_VERSION)
    lines.append(f"Source: {source}")
    lines.append("")
    lines.append("Summary:")
    lines.append(table.to_markdown(index=False))
    lines.append("")
    lines.append("Failure counts by condition:")
    if errors.empty:
        lines.append("No failures found in result details.")
    else:
        cnt = errors.groupby(["context_length", "needle_position", "failure_type"], observed=True).size().reset_index(name="n")
        lines.append(cnt.to_markdown(index=False))
    lines.append("")
    lines.append("Generated figures:")
    lines.append("- context_combined.png")
    lines.append("- context_score_heatmap.png")
    lines.append("")
    lines.append("Generated error files:")
    lines.append("- context_errors.csv")
    lines.append("- context_errors.md")
    (out_dir / "context_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--root", default="outputs/context_short", help="Directory containing aggregate.csv and result json files")
    ap.add_argument("--zip", dest="zip_path", default=None, help="Optional context.zip path")
    ap.add_argument("--out", default="outputs/context_figs", help="Output directory")
    ap.add_argument("--max-errors-per-condition", type=int, default=3, help="Rows to keep per condition in context_errors.md/csv; 0 keeps all")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    aggregate, summary, source = load_summary(args.root, args.zip_path)
    df = normalize_summary(aggregate, summary)
    table = make_table(df)

    errors = collect_errors(args.root, args.zip_path, args.max_errors_per_condition)

    table.to_csv(out_dir / "context_table.csv", index=False)
    (out_dir / "context_table.md").write_text(table.to_markdown(index=False) + "\n", encoding="utf-8")

    errors.to_csv(out_dir / "context_errors.csv", index=False)
    if errors.empty:
        (out_dir / "context_errors.md").write_text("No failures found.\n", encoding="utf-8")
    else:
        md_cols = ["context_length", "needle_position", "sample_id", "gold", "failure_type", "prediction_preview"]
        (out_dir / "context_errors.md").write_text(errors[md_cols].to_markdown(index=False) + "\n", encoding="utf-8")

    plot_heatmap(df, out_dir / "context_score_heatmap.png")
    plot_combined(df, out_dir / "context_combined.png")
    write_report(table, errors, source, out_dir)

    print("\n=== Context summary ===")
    print(table.to_markdown(index=False))
    print("\n=== Failure examples ===")
    if errors.empty:
        print("No failures found.")
    else:
        show_cols = ["context_length", "needle_position", "sample_id", "gold", "failure_type", "prediction_preview"]
        print(errors[show_cols].to_markdown(index=False))
    print(f"\nSaved to: {out_dir}")
    for p in sorted(out_dir.iterdir()):
        print(" -", p.name)


if __name__ == "__main__":
    main()
