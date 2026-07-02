"""Find a GSM8K/MBPP sample that experiment A answered correctly but experiment B got wrong,
to use as a concrete before/after example alongside the aggregate accuracy numbers.

Usage:
    python find_contrast_sample.py \
        --results-a outputs/task2_context_full/task1_gsm8k_ar_like_gsm8k_1/20260701_132447/results/illada_8b_instruct_task1_gsm8k_ar_like_gsm8k_1/gsm8k.json \
        --results-b outputs/task2_context_full/task1_gsm8k_strong_parallel_gsm8k_1/20260701_145133/results/illada_8b_instruct_task1_gsm8k_strong_parallel_gsm8k_1/gsm8k.json \
        --sample-a outputs/task2_context_full/task1_gsm8k_ar_like_gsm8k_1/summary.jsonl \
        --sample-b outputs/task2_context_full/task1_gsm8k_strong_parallel_gsm8k_1/summary.jsonl
"""
import argparse
import json
import re


def _first(value):
    return value[0] if isinstance(value, list) else value


def load_correctness(results_path):
    """Handles both schemas seen in OpenCompass results/*.json:
    - GSM8K-style: {"pred": [...], "answer": [...], "correct": [bool]}
    - MBPP-style:  {"programs": [...], "result": [str], "is_correct": [bool]}
    """
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    by_idx = {}
    for item in data.get("details", []):
        match = re.search(r"(\d+)$", item.get("example_abbr", ""))
        if not match:
            continue
        idx = int(match.group(1))
        if "correct" in item:
            by_idx[idx] = {
                "correct": bool(_first(item.get("correct"))),
                "pred": item.get("pred"),
                "answer": item.get("answer"),
            }
        else:
            by_idx[idx] = {
                "correct": bool(_first(item.get("is_correct"))),
                "pred": _first(item.get("result")),
                "answer": "pass",
            }
    return by_idx


def load_samples(jsonl_path):
    by_idx = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            by_idx[row.get("sample_idx")] = row
    return by_idx


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-a", required=True, help="OpenCompass results/*.json for the 'correct' condition")
    parser.add_argument("--results-b", required=True, help="OpenCompass results/*.json for the 'wrong' condition")
    parser.add_argument("--sample-a", required=True, help="summary.jsonl for the 'correct' condition")
    parser.add_argument("--sample-b", required=True, help="summary.jsonl for the 'wrong' condition")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    corr_a = load_correctness(args.results_a)
    corr_b = load_correctness(args.results_b)
    samp_a = load_samples(args.sample_a)
    samp_b = load_samples(args.sample_b)

    candidates = []
    for idx, a in corr_a.items():
        b = corr_b.get(idx)
        if b is None:
            continue
        if a["correct"] and not b["correct"]:
            candidates.append(idx)

    print(f"Found {len(candidates)} candidate sample_idx where A was correct and B was wrong: {candidates}\n")
    for idx in candidates[: args.limit]:
        a_row, b_row = samp_a.get(idx, {}), samp_b.get(idx, {})
        print(f"=== sample_idx {idx} ===")
        print(f"question (tail): ...{a_row.get('input', '')[-400:]}\n")
        print(f"[A] pred={corr_a[idx]['pred']} answer={corr_a[idx]['answer']} correct={corr_a[idx]['correct']}")
        print(f"[A] prediction text:\n{a_row.get('prediction', '(not found in summary.jsonl)')}\n")
        print(f"[B] pred={corr_b[idx]['pred']} answer={corr_b[idx]['answer']} correct={corr_b[idx]['correct']}")
        print(f"[B] prediction text:\n{b_row.get('prediction', '(not found in summary.jsonl)')}\n")
        print("-" * 80)


if __name__ == "__main__":
    main()
