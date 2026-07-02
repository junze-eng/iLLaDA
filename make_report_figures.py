"""Generate report-ready figures from run_test.py output.

Usage:
    python make_report_figures.py --summary-csv outputs/task2_context_full/summary_all.csv \
        --sample-jsonl outputs/task2_context_full/task1_gsm8k_ar_like_gsm8k_1/summary.jsonl \
        --sample-idx 7 --output-dir outputs/report_figures

Produces:
    speed_accuracy_tradeoff.png  -- accuracy/score vs ARness, one panel per benchmark, one
                                    line per condition group (parallel sweep, threshold=X, ...)
    sample_<idx>_<failure_type>.png -- annotated before/after view of one generated sample,
                                    highlighting where the model should have stopped
"""
import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch


def load_summary(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    for col in ("primary_metric_value", "tokens_per_second_mean", "param_arness",
                "param_token_selection_confidence_threshold"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('%', ''), errors='coerce')
    return df


def condition_groups(df: pd.DataFrame, benchmark: str):
    """Split a benchmark's rows into series by confidence threshold: one series for rows with
    no threshold set (the plain parallel/ARness sweep), one per distinct threshold value."""
    sub = df[df['benchmark'] == benchmark].copy()
    if 'param_arness' not in sub.columns:
        return {}
    sub = sub.dropna(subset=['param_arness', 'primary_metric_value'])
    groups = {}
    thr_col = 'param_token_selection_confidence_threshold'
    has_thr = thr_col in sub.columns
    no_thr = sub[sub[thr_col].isna()] if has_thr else sub
    if not no_thr.empty:
        agg = no_thr.groupby('param_arness', as_index=False)['primary_metric_value'].mean()
        groups['no threshold'] = agg.sort_values('param_arness')
    if has_thr:
        for thr, rows in sub.dropna(subset=[thr_col]).groupby(thr_col):
            agg = rows.groupby('param_arness', as_index=False)['primary_metric_value'].mean()
            groups[f'threshold = {thr:g}'] = agg.sort_values('param_arness')
    return groups


def plot_speed_accuracy_tradeoff(df: pd.DataFrame, output_dir: Path):
    benchmarks = sorted(b for b in df['benchmark'].dropna().unique() if b)
    if not benchmarks:
        print("[-] No benchmarks found in summary CSV. Skipping tradeoff figure.")
        return

    fig, axes = plt.subplots(1, len(benchmarks), figsize=(6.6 * len(benchmarks), 5.4), squeeze=False)
    axes = axes[0]
    fig.suptitle("Accuracy vs. ARness (Decoding Parallelism)", fontsize=13.5, fontweight='bold')
    styles = [
        dict(color="#1f6f43", marker='o', linestyle='-'),
        dict(color="#2f6fb2", marker='^', linestyle='--'),
        dict(color="#b35806", marker='s', linestyle=':'),
        dict(color="#7b3294", marker='D', linestyle='-.'),
    ]

    any_series = False
    for ax, bench in zip(axes, benchmarks):
        groups = condition_groups(df, bench)
        for (label, agg), style in zip(groups.items(), styles):
            if agg.empty:
                continue
            any_series = True
            ax.plot(agg['param_arness'], agg['primary_metric_value'], label=label,
                    linewidth=2.3, markersize=8.5, **style)
        ax.set_xscale('log', base=2)
        xticks = sorted({v for g in groups.values() for v in g['param_arness']})
        if xticks:
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(int(v)) for v in xticks])
        ax.set_xlabel("ARness (steps per token, higher = more parallel)", fontsize=10.5)
        ax.set_ylabel("Accuracy / Score", fontsize=10.5)
        ax.set_title(bench.upper(), fontsize=12.5, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        if groups:
            ax.legend(fontsize=8.8, loc='upper right')

    if not any_series:
        print("[-] No (benchmark, arness) series with data found. Skipping tradeoff figure.")
        plt.close(fig)
        return

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_path = output_dir / "speed_accuracy_tradeoff.png"
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"[+] Saved {save_path}")


def plot_sample_annotated(sample_jsonl: Path, sample_idx: int, output_dir: Path):
    if not sample_jsonl.exists():
        print(f"[-] {sample_jsonl} does not exist. Skipping sample figure.")
        return
    record = None
    with sample_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("sample_idx") == sample_idx:
                record = row
                break
    if record is None:
        print(f"[-] sample_idx={sample_idx} not found in {sample_jsonl}. Skipping sample figure.")
        return

    question = record.get("input", "")
    prediction = record.get("prediction", "")
    failure_type = record.get("failure_type", "unknown")
    decoding_config = record.get("decoding_config_name", "")

    # Best-effort split: show the last ~600 chars of input as "the question" and the full
    # prediction as-is; if the prediction looks like it ran past a natural stopping point
    # (contains a second "user"/question marker), split there for the two-tone highlight.
    question_tail = question[-600:]
    split_markers = ["user Question:", "user question:", "<[BOS]>user"]
    split_at = None
    for marker in split_markers:
        idx = prediction.find(marker)
        if idx > 0:
            split_at = idx
            break
    correct_part = prediction if split_at is None else prediction[:split_at]
    rambling_part = "" if split_at is None else prediction[split_at:]

    fig, ax = plt.subplots(figsize=(9, 8.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    fig.suptitle(f"{decoding_config or sample_jsonl.parent.name} — sample #{sample_idx} "
                 f"(failure_type = {failure_type})", fontsize=12.5, fontweight='bold', y=0.975)

    q_wrapped = "\n".join(textwrap.wrap(question_tail, width=78)[-8:])
    ax.add_patch(FancyBboxPatch((0.3, 8.05), 9.4, 1.55, boxstyle="round,pad=0.08",
                                 linewidth=1.2, edgecolor="#555555", facecolor="#f2f2f2"))
    ax.text(0.55, 9.35, "Prompt (tail)", fontsize=9.5, fontweight='bold', color="#333333", va='top')
    ax.text(0.55, 9.0, q_wrapped, fontsize=8.5, va='top', family='monospace', color="#222222")

    c_wrapped = "\n".join(textwrap.wrap(correct_part.strip(), width=78)[:26])
    body_top, body_height = (3.55, 4.15) if rambling_part else (0.35, 7.35)
    ax.add_patch(FancyBboxPatch((0.3, body_top), 9.4, body_height, boxstyle="round,pad=0.08",
                                 linewidth=1.5, edgecolor="#2e7d32", facecolor="#e8f5e9"))
    ax.text(0.55, body_top + body_height - 0.35, "Model output", fontsize=9.5,
            fontweight='bold', color="#2e7d32", va='top')
    ax.text(0.55, body_top + body_height - 0.7, c_wrapped, fontsize=8.2, va='top',
            family='monospace', color="#1b5e20")

    legend_handles = [mpatches.Patch(facecolor="#e8f5e9", edgecolor="#2e7d32", label="Model output")]

    if rambling_part:
        ax.annotate(
            "generation continues past this point instead of stopping ↓",
            xy=(5, body_top), xytext=(5, body_top - 0.4),
            fontsize=8.8, fontweight='bold', color="#b71c1c", ha='center',
            arrowprops=dict(arrowstyle='-|>', color="#b71c1c", lw=1.6),
        )
        r_wrapped = "\n".join(textwrap.wrap(rambling_part.strip(), width=78)[:14])
        ax.add_patch(FancyBboxPatch((0.3, 0.35), 9.4, 2.55, boxstyle="round,pad=0.08",
                                     linewidth=1.5, edgecolor="#b71c1c", facecolor="#ffebee"))
        ax.text(0.55, 2.65, "Wasted tokens after the answer", fontsize=9.5,
                fontweight='bold', color="#b71c1c", va='top')
        ax.text(0.55, 2.3, r_wrapped, fontsize=8.2, va='top', family='monospace', color="#7f0000")
        legend_handles.append(mpatches.Patch(facecolor="#ffebee", edgecolor="#b71c1c",
                                              label="Wasted tokens after the answer"))

    ax.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.02),
              ncol=len(legend_handles), fontsize=8.5, frameon=False)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    save_path = output_dir / f"sample_{sample_idx}_{failure_type}.png"
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"[+] Saved {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate report figures from run_test.py output.")
    parser.add_argument("--summary-csv", default="outputs/illada_runs/summary_all.csv",
                         help="Path to summary_all.csv for the speed/accuracy tradeoff figure.")
    parser.add_argument("--sample-jsonl", default=None,
                         help="Path to a per-experiment summary.jsonl to pull one annotated sample from.")
    parser.add_argument("--sample-idx", type=int, default=None,
                         help="sample_idx to annotate from --sample-jsonl.")
    parser.add_argument("--output-dir", default="outputs/report_figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = Path(args.summary_csv)
    if summary_path.exists():
        plot_speed_accuracy_tradeoff(load_summary(summary_path), output_dir)
    else:
        print(f"[-] {summary_path} does not exist. Skipping tradeoff figure.")

    if args.sample_jsonl and args.sample_idx is not None:
        plot_sample_annotated(Path(args.sample_jsonl), args.sample_idx, output_dir)


if __name__ == "__main__":
    main()
