"""Inspect the step-by-step diffusion trace for one sample, to see exactly which
step(s) a generation started degenerating (e.g. collapsing into a repeated token).

Usage:
    python inspect_trace.py outputs/task2_context_full/task1_gsm8k_strong_parallel_gsm8k_1/trace.jsonl \
        --sample-idx 7 --plot
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_steps(trace_jsonl, sample_idx):
    rows = []
    with open(trace_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("sample_idx") == sample_idx:
                rows.append(row)
    rows.sort(key=lambda r: (r.get("block_idx", 0), r.get("step_idx", 0)))
    return rows


def print_steps(rows, limit):
    token_counts = defaultdict(int)
    first_repeat_step = None

    for row in rows[:limit]:
        tokens = row.get("selected_decoded_tokens") or []
        confidences = row.get("selected_confidences") or []
        step_label = f"block {row.get('block_idx')} step {row.get('step_idx')}"
        pairs = ", ".join(f"{repr(t)}@{c:.3f}" if isinstance(c, (int, float)) else f"{repr(t)}"
                           for t, c in zip(tokens, confidences)) or "(no tokens committed this step)"
        print(f"[{step_label}] mask {row.get('mask_count_before')}->{row.get('mask_count_after')}  "
              f"completion={row.get('current_completion_rate')}  committed: {pairs}")

        for t in tokens:
            token_counts[t] += 1
            if token_counts[t] >= 3 and first_repeat_step is None:
                first_repeat_step = step_label
                print(f"    ^^^ token {repr(t)} has now been committed {token_counts[t]}x -- "
                      f"this is where degeneration likely starts")

    if first_repeat_step:
        print(f"\nFirst sign of a token repeating 3+ times: {first_repeat_step}")
    print("\nMost repeated tokens overall:")
    for tok, count in sorted(token_counts.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {repr(tok)}: committed {count} time(s)")

    return token_counts


def write_narrative(rows, dominant_token, token_counts, sample_idx, trace_jsonl, output_path):
    """Write a report-ready markdown walkthrough: every step's committed tokens in a table,
    plus an auto-generated narrative paragraph pointing at the specific steps where the
    dominant token first appears, first repeats, and locks in (confidence crosses 0.9)."""
    first_step_label = None
    first_repeat_label = None
    lock_in_label = None
    running_count = 0
    for row in rows:
        tokens = row.get("selected_decoded_tokens") or []
        confidences = row.get("selected_confidences") or []
        step_label = f"block {row.get('block_idx')} step {row.get('step_idx')}"
        for t, c in zip(tokens, confidences):
            if t != dominant_token:
                continue
            if first_step_label is None:
                first_step_label = (step_label, c)
            running_count += 1
            if running_count == 3 and first_repeat_label is None:
                first_repeat_label = (step_label, c)
            if lock_in_label is None and isinstance(c, (int, float)) and c >= 0.9:
                lock_in_label = (step_label, c)

    total_committed = sum(token_counts.values()) or 1
    dominant_total = token_counts.get(dominant_token, 0)
    dominant_pct = 100.0 * dominant_total / total_committed

    lines = []
    lines.append(f"# Diffusion trace walkthrough — {Path(trace_jsonl).parent.name}, sample #{sample_idx}\n")
    lines.append(f"Source: `{trace_jsonl}`  \nSteps recorded: {len(rows)}  \n"
                 f"Dominant repeated token: `{dominant_token!r}` "
                 f"({dominant_total}/{total_committed} = {dominant_pct:.1f}% of all committed tokens)\n")

    lines.append("## Narrative\n")
    if first_step_label:
        step, conf = first_step_label
        conf_str = f"{conf:.3f}" if isinstance(conf, (int, float)) else str(conf)
        lines.append(f"At **{step}**, the token `{dominant_token!r}` is first committed, with confidence "
                     f"{conf_str} — at this point in decoding there is little to no surrounding generated "
                     f"context to condition on, so this is close to a guess among the model's top candidates.\n")
    if first_repeat_label:
        step, conf = first_repeat_label
        conf_str = f"{conf:.3f}" if isinstance(conf, (int, float)) else str(conf)
        lines.append(f"By **{step}**, `{dominant_token!r}` has been committed 3 times, now with confidence "
                     f"{conf_str}. The model's attention over its own (still mostly incomplete/incorrect) "
                     f"partial output is starting to reinforce the repeated pattern rather than correct it.\n")
    if lock_in_label:
        step, conf = lock_in_label
        conf_str = f"{conf:.3f}" if isinstance(conf, (int, float)) else str(conf)
        lines.append(f"From **{step}** onward, confidence for `{dominant_token!r}` crosses 0.9 and the token "
                     f"is selected in nearly every remaining step — the pattern is now locked in. "
                     f"By the end of decoding, `{dominant_token!r}` accounts for {dominant_total} of the "
                     f"{total_committed} committed tokens ({dominant_pct:.1f}%).\n")
    else:
        lines.append(f"Confidence for `{dominant_token!r}` never crosses 0.9 in this trace, but it still ends up "
                     f"as the single most-repeated token, accounting for {dominant_pct:.1f}% of all committed "
                     f"tokens.\n")

    lines.append("## Step-by-step record\n")
    lines.append("| Step | Mask before→after | Completion | Tokens committed (token@confidence) |")
    lines.append("|---|---|---|---|")
    for row in rows:
        tokens = row.get("selected_decoded_tokens") or []
        confidences = row.get("selected_confidences") or []
        step_label = f"block {row.get('block_idx')} step {row.get('step_idx')}"
        pairs = ", ".join(
            f"`{t!r}`@{c:.3f}" if isinstance(c, (int, float)) else f"`{t!r}`"
            for t, c in zip(tokens, confidences)
        ) or "(none)"
        lines.append(f"| {step_label} | {row.get('mask_count_before')}→{row.get('mask_count_after')} | "
                     f"{row.get('current_completion_rate')} | {pairs} |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[+] Saved narrative to {output_path}")


def write_csv(rows, sample_idx, output_path):
    """One row per diffusion step: how mask count/completion progressed, and exactly which
    tokens (with confidence) got committed that step -- the raw iteration-by-iteration table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "block_idx", "step_idx", "mask_count_before", "mask_count_after",
            "completion_rate", "num_tokens_committed", "tokens_committed", "confidences_committed",
        ])
        for row in rows:
            tokens = row.get("selected_decoded_tokens") or []
            confidences = row.get("selected_confidences") or []
            writer.writerow([
                row.get("block_idx"),
                row.get("step_idx"),
                row.get("mask_count_before"),
                row.get("mask_count_after"),
                row.get("current_completion_rate"),
                len(tokens),
                ";".join(repr(t) for t in tokens),
                ";".join(f"{c:.4f}" if isinstance(c, (int, float)) else str(c) for c in confidences),
            ])
    print(f"[+] Saved CSV to {output_path}")


def plot_collapse_curve(rows, dominant_token, sample_idx, trace_jsonl, output_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"[-] matplotlib not available ({exc}); skipping --plot.")
        return

    step_x, mean_conf, dominant_frac = [], [], []
    for row in rows:
        tokens = row.get("selected_decoded_tokens") or []
        confidences = row.get("selected_confidences") or []
        if not tokens:
            continue
        matched = [c for t, c in zip(tokens, confidences) if t == dominant_token and isinstance(c, (int, float))]
        step_x.append(row.get("step_idx"))
        mean_conf.append(sum(matched) / len(matched) if matched else 0.0)
        dominant_frac.append(100.0 * sum(1 for t in tokens if t == dominant_token) / len(tokens))

    fig, ax1 = plt.subplots(figsize=(9.5, 5.4))
    ax2 = ax1.twinx()
    l1, = ax1.plot(step_x, mean_conf, marker='o', linewidth=2.3, markersize=6.5, color="#b71c1c",
                    label=f"Mean confidence of {dominant_token!r} tokens committed this step")
    l2, = ax2.plot(step_x, dominant_frac, marker='s', linewidth=1.8, markersize=5.5, linestyle='--',
                    color="#555555", label=f"% of this step's tokens that are {dominant_token!r}")
    ax1.set_xlabel("Diffusion step", fontsize=10.5)
    ax1.set_ylabel(f"Mean confidence of {dominant_token!r}", fontsize=10.5, color="#b71c1c")
    ax2.set_ylabel(f"% of step's tokens = {dominant_token!r}", fontsize=10.5, color="#555555")
    ax1.tick_params(axis='y', labelcolor="#b71c1c")
    ax2.tick_params(axis='y', labelcolor="#555555")
    ax1.set_ylim(0, 1.05)
    ax2.set_ylim(0, 105)
    ax1.set_title(f"Token collapse trace — {Path(trace_jsonl).parent.name}, sample #{sample_idx}",
                  fontsize=12, fontweight='bold')
    ax1.legend(handles=[l1, l2], loc='center right', fontsize=8.5)
    ax1.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"\n[+] Saved collapse curve to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace_jsonl")
    parser.add_argument("--sample-idx", type=int, required=True)
    parser.add_argument("--limit", type=int, default=1000, help="max step rows to print")
    parser.add_argument("--plot", action="store_true",
                         help="Also save a confidence-vs-step chart for the most-repeated token.")
    parser.add_argument("--output", default=None,
                         help="Path for --plot output (default: outputs/report_figures/trace_<sample_idx>.png)")
    parser.add_argument("--narrative", action="store_true",
                         help="Also write a report-ready markdown walkthrough (every step, plus an "
                              "auto-generated explanation of where the collapse starts/locks in).")
    parser.add_argument("--narrative-output", default=None,
                         help="Path for --narrative output "
                              "(default: outputs/report_figures/trace_<sample_idx>_narrative.md)")
    parser.add_argument("--csv", action="store_true",
                         help="Also write a CSV: one row per step, with mask progress, completion "
                              "rate, and every token+confidence committed that step.")
    parser.add_argument("--csv-output", default=None,
                         help="Path for --csv output (default: outputs/report_figures/trace_<sample_idx>.csv)")
    args = parser.parse_args()

    rows = load_steps(args.trace_jsonl, args.sample_idx)
    if not rows:
        print(f"No trace rows found for sample_idx={args.sample_idx} in {args.trace_jsonl}")
        return

    print(f"{len(rows)} step(s) for sample_idx={args.sample_idx}\n")
    token_counts = print_steps(rows, args.limit)

    if args.csv:
        csv_path = (Path(args.csv_output) if args.csv_output
                    else Path("outputs/report_figures") / f"trace_{args.sample_idx}.csv")
        write_csv(rows, args.sample_idx, csv_path)

    if args.plot or args.narrative:
        if not token_counts:
            print("[-] No committed tokens found; skipping --plot/--narrative.")
            return
        dominant_token = max(token_counts.items(), key=lambda kv: kv[1])[0]

        if args.plot:
            output_path = Path(args.output) if args.output else Path("outputs/report_figures") / f"trace_{args.sample_idx}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plot_collapse_curve(rows, dominant_token, args.sample_idx, args.trace_jsonl, output_path)

        if args.narrative:
            narrative_path = (Path(args.narrative_output) if args.narrative_output
                              else Path("outputs/report_figures") / f"trace_{args.sample_idx}_narrative.md")
            write_narrative(rows, dominant_token, token_counts, args.sample_idx, args.trace_jsonl, narrative_path)


if __name__ == "__main__":
    main()
