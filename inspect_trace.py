"""Inspect the step-by-step diffusion trace for one sample, to see exactly which
step(s) a generation started degenerating (e.g. collapsing into a repeated token).

Usage:
    python inspect_trace.py outputs/task2_context_full/task1_gsm8k_strong_parallel_gsm8k_1/trace.jsonl --sample-idx 7
"""
import argparse
import json
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace_jsonl")
    parser.add_argument("--sample-idx", type=int, required=True)
    parser.add_argument("--limit", type=int, default=1000, help="max step rows to print")
    args = parser.parse_args()

    rows = []
    with open(args.trace_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("sample_idx") == args.sample_idx:
                rows.append(row)

    if not rows:
        print(f"No trace rows found for sample_idx={args.sample_idx} in {args.trace_jsonl}")
        return

    rows.sort(key=lambda r: (r.get("block_idx", 0), r.get("step_idx", 0)))
    print(f"{len(rows)} step(s) for sample_idx={args.sample_idx}\n")

    # Track a running tally of decoded tokens to spot when repetition kicks in.
    token_counts = defaultdict(int)
    repeat_streak = 0
    first_repeat_step = None

    for row in rows[: args.limit]:
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


if __name__ == "__main__":
    main()
