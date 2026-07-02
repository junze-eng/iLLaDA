"""Inspect the step-by-step diffusion trace for one sample, to see exactly which
step(s) a generation started degenerating (e.g. collapsing into a repeated token).

Usage:
    python inspect_trace.py outputs/task2_context_full/task1_gsm8k_strong_parallel_gsm8k_1/trace.jsonl \
        --sample-idx 7 --plot
"""
import argparse
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
    args = parser.parse_args()

    rows = load_steps(args.trace_jsonl, args.sample_idx)
    if not rows:
        print(f"No trace rows found for sample_idx={args.sample_idx} in {args.trace_jsonl}")
        return

    print(f"{len(rows)} step(s) for sample_idx={args.sample_idx}\n")
    token_counts = print_steps(rows, args.limit)

    if args.plot:
        if not token_counts:
            print("[-] No committed tokens found; skipping --plot.")
            return
        dominant_token = max(token_counts.items(), key=lambda kv: kv[1])[0]
        output_path = Path(args.output) if args.output else Path("outputs/report_figures") / f"trace_{args.sample_idx}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plot_collapse_curve(rows, dominant_token, args.sample_idx, args.trace_jsonl, output_path)


if __name__ == "__main__":
    main()
