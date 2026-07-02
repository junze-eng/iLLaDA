"""Lightweight prepared RULER NIAH single_1 context probe.

This script is a quick sanity-check tool, not a replacement for the
OpenCompass RULER benchmark. It reuses already prepared RULER NIAH data,
runs a small number of direct iLLaDA generations, and writes predictions plus
simple keyword-match summaries for manual review.
"""

import argparse
import csv
import json
import random
import re
import string
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


PROMPT_KEYS = ("prompt", "input", "query", "text")
ANSWER_KEYS = ("answer", "needle", "target", "value", "gold", "answers")
CONTEXT_KEYS = ("context_length", "requested_context_length", "max_seq_length")
POSITION_KEYS = ("needle_position", "position", "depth", "depth_percent")
QUESTION_KEYS = ("question", "query")
POSITION_ALIASES = {
    "front": "front",
    "beginning": "front",
    "start": "front",
    "0": "front",
    "0.0": "front",
    "middle": "middle",
    "mid": "middle",
    "50": "middle",
    "50.0": "middle",
    "back": "back",
    "end": "back",
    "100": "back",
    "100.0": "back",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight prepared RULER NIAH context probe.")
    parser.add_argument("--prepared-root", default="data/prepared/ruler_niah_single_1")
    parser.add_argument("--output-dir", default="outputs/prepared_context_probe")
    parser.add_argument("--model-path", default="GSAI-ML/iLLaDA-8B-Instruct")
    parser.add_argument("--context-lengths", nargs="+", type=int, default=[1024, 2048, 4096, 8192])
    parser.add_argument("--needle-positions", nargs="+", default=["front", "middle", "back"])
    parser.add_argument("--num-samples-per-condition", type=int, default=1)
    parser.add_argument("--sample-selection", choices=["first", "index", "random"], default="first")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument("--gen-steps", type=int, default=64)
    parser.add_argument("--gen-blocksize", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg", type=float, default=0.0)
    parser.add_argument("--token-selection-confidence-threshold", type=float, default=None)
    parser.add_argument("--return-trace", action="store_true")
    parser.add_argument("--trace-token-snapshots", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--mask-id", type=int, default=5)
    parser.add_argument("--remasking", choices=["low_confidence", "random"], default="low_confidence")
    parser.add_argument("--apply-chat-template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--add-special-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def first_present(item: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in item and item[key] not in (None, ""):
            return item[key]
    return None


def normalize_position(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    text = str(value).strip().lower()
    if text.startswith("depth"):
        text = text[len("depth"):]
    text = text.strip("_- %[]")
    return POSITION_ALIASES.get(text)


def parse_context_from_path(path: Path) -> Optional[int]:
    text = path.as_posix().lower()
    patterns = [
        r"(?:ctx|context_length|len|max_seq_length)[_-]?(\d{3,5})",
        r"(\d{3,5})[_-]?(?:front|middle|back)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    for candidate in (1024, 2048, 4096, 8192):
        if str(candidate) in text:
            return candidate
    return None


def parse_position_from_path(path: Path) -> Optional[str]:
    text = path.as_posix().lower()
    for pos in ("front", "middle", "back"):
        if re.search(rf"(^|[_\-/]){pos}($|[_\-.\\/])", text):
            return pos
    depth = re.search(r"depth[_-]?(0|50|100)", text)
    if depth:
        return normalize_position(depth.group(1))
    position = re.search(r"(?:pos|position)[_-]?(front|middle|back|0|50|100)", text)
    if position:
        return normalize_position(position.group(1))
    return None


def coerce_answer(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return "" if value is None else str(value)


def iter_records(path: Path) -> Iterable[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                if line.strip():
                    yield line_no, json.loads(line)
        return
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        for idx, item in enumerate(data):
            yield idx, item
    elif isinstance(data, dict):
        records = data.get("data") or data.get("records") or data.get("test") or []
        if isinstance(records, list):
            for idx, item in enumerate(records):
                yield idx, item


def load_prepared_samples(prepared_root: Path) -> List[Dict[str, Any]]:
    if not prepared_root.exists():
        raise SystemExit(f"Prepared root does not exist: {prepared_root}")
    files = sorted([p for p in prepared_root.rglob("*") if p.suffix.lower() in (".jsonl", ".json")])
    if not files:
        raise SystemExit(f"No .jsonl/.json prepared files found under: {prepared_root}")

    samples = []
    for path in files:
        path_ctx = parse_context_from_path(path)
        path_pos = parse_position_from_path(path)
        for source_line, item in iter_records(path):
            if not isinstance(item, dict):
                continue
            prompt = first_present(item, PROMPT_KEYS)
            answer = first_present(item, ANSWER_KEYS)
            if prompt is None or answer is None:
                continue
            ctx = first_present(item, CONTEXT_KEYS)
            pos = first_present(item, POSITION_KEYS)
            try:
                ctx = int(ctx) if ctx is not None else path_ctx
            except (TypeError, ValueError):
                ctx = path_ctx
            pos = normalize_position(pos) or path_pos
            if ctx is None or pos is None:
                continue
            rel = path.relative_to(prepared_root).as_posix()
            samples.append({
                "sample_id": f"{ctx}_{pos}_{rel}:{source_line}",
                "source_file": str(path),
                "source_line": int(source_line),
                "context_length": int(ctx),
                "needle_position": pos,
                "prompt": str(prompt),
                "question": first_present(item, QUESTION_KEYS),
                "groundtruth": coerce_answer(answer),
            })
    return samples


def select_samples(
    samples: List[Dict[str, Any]],
    context_lengths: List[int],
    positions: List[str],
    num_per_condition: int,
    selection: str,
    sample_index: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    by_condition: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_condition[(sample["context_length"], sample["needle_position"])].append(sample)

    selected = []
    for ctx in context_lengths:
        for pos in positions:
            candidates = by_condition.get((ctx, pos), [])
            if not candidates:
                print(f"[prepared_context_probe] warning: no prepared samples for ctx={ctx} pos={pos}", flush=True)
                continue
            if selection == "first":
                chosen = candidates[:num_per_condition]
            elif selection == "index":
                chosen = candidates[sample_index:sample_index + num_per_condition]
            else:
                chosen = rng.sample(candidates, k=min(num_per_condition, len(candidates)))
            selected.extend(chosen)
    if not selected:
        raise SystemExit("No samples selected. Check prepared-root, context lengths, and needle positions.")
    return selected


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    punctuation = string.punctuation.replace("-", "")
    text = text.translate(str.maketrans({char: " " for char in punctuation}))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def score_prediction(prediction: str, groundtruth: str) -> Dict[str, bool]:
    exact = groundtruth in prediction
    normalized = normalize_text(groundtruth) in normalize_text(prediction)
    return {
        "exact_match": bool(exact),
        "normalized_match": bool(normalized),
        "keyword_match": bool(exact or normalized),
    }


def read_done_ids(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get("sample_id"):
                    done.add(str(item["sample_id"]))
    return done


def load_prediction_rows(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summaries(output_dir: Path):
    rows = load_prediction_rows(output_dir / "predictions.jsonl")
    if not rows:
        return

    def aggregate(group_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(group_rows)
        return {
            "n": n,
            "keyword_accuracy": sum(1 for r in group_rows if r.get("keyword_match")) / n,
            "exact_match_rate": sum(1 for r in group_rows if r.get("exact_match")) / n,
            "normalized_match_rate": sum(1 for r in group_rows if r.get("normalized_match")) / n,
            "mean_latency_sec": mean([float(r["latency_sec"]) for r in group_rows if r.get("latency_sec") is not None]),
            "mean_tokens_per_second": mean([float(r["tokens_per_second"]) for r in group_rows if r.get("tokens_per_second") is not None]),
            "mean_prompt_tokens": mean([float(r["prompt_tokens"]) for r in group_rows if r.get("prompt_tokens") is not None]),
            "mean_max_memory_gb": mean([float(r["max_memory_gb"]) for r in group_rows if r.get("max_memory_gb") is not None]),
        }

    condition_rows = []
    by_condition: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    by_context: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[(row.get("context_length"), row.get("needle_position"))].append(row)
        by_context[row.get("context_length")].append(row)
    for (ctx, pos), group_rows in sorted(by_condition.items()):
        condition_rows.append({"context_length": ctx, "needle_position": pos, **aggregate(group_rows)})
    write_csv(
        output_dir / "summary_by_condition.csv",
        condition_rows,
        ["context_length", "needle_position", "n", "keyword_accuracy", "exact_match_rate",
         "normalized_match_rate", "mean_latency_sec", "mean_tokens_per_second",
         "mean_prompt_tokens", "mean_max_memory_gb"],
    )

    overall_rows = []
    for ctx, group_rows in sorted(by_context.items()):
        overall_rows.append({"context_length": ctx, "group": f"ctx{ctx}", **aggregate(group_rows)})
    overall_rows.append({"context_length": "all", "group": "all", **aggregate(rows)})
    write_csv(
        output_dir / "summary_overall.csv",
        overall_rows,
        ["group", "context_length", "n", "keyword_accuracy", "exact_match_rate",
         "normalized_match_rate", "mean_latency_sec", "mean_tokens_per_second",
         "mean_prompt_tokens", "mean_max_memory_gb"],
    )

    review_rows = [{
        "context_length": row.get("context_length"),
        "needle_position": row.get("needle_position"),
        "sample_id": row.get("sample_id"),
        "groundtruth": row.get("groundtruth"),
        "prediction": row.get("prediction"),
        "keyword_match": row.get("keyword_match"),
        "manual_judgment": row.get("manual_judgment", ""),
        "failure_type": row.get("failure_type", ""),
    } for row in rows]
    write_csv(
        output_dir / "manual_review.csv",
        review_rows,
        ["context_length", "needle_position", "sample_id", "groundtruth", "prediction",
         "keyword_match", "manual_judgment", "failure_type"],
    )


def git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def torch_dtype(name: str):
    import torch
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def build_prompt(tokenizer, prompt: str, apply_chat_template: bool) -> str:
    if not apply_chat_template:
        return prompt
    message = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
    except Exception as exc:
        print(f"[prepared_context_probe] warning: failed to apply chat template ({exc}); using raw prompt.", flush=True)
        return prompt


def main() -> int:
    args = parse_args()
    prepared_root = resolve_path(args.prepared_root)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "predictions.jsonl"
    prompts_path = output_dir / "prompts.jsonl"
    if args.overwrite:
        for path in (predictions_path, prompts_path):
            if path.exists():
                path.unlink()

    samples = load_prepared_samples(prepared_root)
    positions = [normalize_position(pos) or pos for pos in args.needle_positions]
    selected = select_samples(
        samples=samples,
        context_lengths=args.context_lengths,
        positions=positions,
        num_per_condition=args.num_samples_per_condition,
        selection=args.sample_selection,
        sample_index=args.sample_index,
        seed=args.seed,
    )
    done_ids = read_done_ids(predictions_path)
    pending = [sample for sample in selected if sample["sample_id"] not in done_ids]

    run_config = {
        "argv": sys.argv,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "model_path": args.model_path,
        "prepared_root": str(prepared_root),
        "output_dir": str(output_dir),
        "num_selected": len(selected),
        "num_pending": len(pending),
        "args": vars(args),
        "note": (
            "prepared_context_probe is a lightweight context sanity check using "
            "prepared RULER NIAH single_1 data. It does not replace the "
            "OpenCompass RULER benchmark."
        ),
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    if not pending:
        print("[prepared_context_probe] nothing to do; all selected samples already exist.")
        write_summaries(output_dir)
        return 0

    import torch
    from transformers import AutoModel, AutoTokenizer
    from generate import generate as LLaDA_generate

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype(args.torch_dtype) if device.type == "cuda" else torch.float32,
    ).to(device).eval()

    with predictions_path.open("a", encoding="utf-8") as pred_f, prompts_path.open("a", encoding="utf-8") as prompt_f:
        for sample in pending:
            full_prompt = build_prompt(tokenizer, sample["prompt"], args.apply_chat_template)
            encoded = tokenizer(full_prompt, add_special_tokens=args.add_special_tokens, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.inference_mode():
                generated = LLaDA_generate(
                    model=model,
                    prompt=input_ids,
                    attention_mask=attention_mask,
                    steps=args.gen_steps,
                    gen_length=args.gen_length,
                    block_length=args.gen_blocksize,
                    temperature=args.temperature,
                    cfg_scale=args.cfg,
                    remasking=args.remasking,
                    mask_id=args.mask_id,
                    token_selection_confidence_threshold=args.token_selection_confidence_threshold,
                    return_trace=args.return_trace,
                    trace_token_snapshots=args.trace_token_snapshots,
                    tokenizer=tokenizer,
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latency = time.perf_counter() - started
            if args.return_trace:
                generated_tokens, _trace = generated
            else:
                generated_tokens = generated
            prediction = tokenizer.decode(
                generated_tokens[0, input_ids.shape[1]:],
                skip_special_tokens=True,
            )
            max_memory_gb = None
            if device.type == "cuda":
                max_memory_gb = torch.cuda.max_memory_allocated(device) / 1024 ** 3

            scores = score_prediction(prediction, sample["groundtruth"])
            record = {
                "sample_id": sample["sample_id"],
                "source_file": sample["source_file"],
                "source_line": sample["source_line"],
                "context_length": sample["context_length"],
                "needle_position": sample["needle_position"],
                "question": sample.get("question"),
                "groundtruth": sample["groundtruth"],
                "prediction": prediction,
                **scores,
                "manual_judgment": "",
                "failure_type": "",
                "prompt_tokens": int(input_ids.shape[1]),
                "prompt_chars": len(full_prompt),
                "gen_length": args.gen_length,
                "gen_steps": args.gen_steps,
                "gen_blocksize": args.gen_blocksize,
                "latency_sec": round(latency, 6),
                "tokens_per_second": round(args.gen_length / latency, 6) if latency > 0 else None,
                "max_memory_gb": round(max_memory_gb, 6) if max_memory_gb is not None else None,
            }
            prompt_record = {
                "sample_id": sample["sample_id"],
                "context_length": sample["context_length"],
                "needle_position": sample["needle_position"],
                "groundtruth": sample["groundtruth"],
                "prompt": full_prompt,
            }
            pred_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            pred_f.flush()
            prompt_f.write(json.dumps(prompt_record, ensure_ascii=False) + "\n")
            prompt_f.flush()

            short_pred = prediction.replace("\n", " ")[:120]
            print(
                "[prepared_context_probe] "
                f"ctx={sample['context_length']} pos={sample['needle_position']} "
                f"sample={sample['source_line']} match={record['keyword_match']} "
                f"latency={latency:.2f}s tps={record['tokens_per_second']} "
                f"pred={short_pred!r}",
                flush=True,
            )

    write_summaries(output_dir)
    rows = load_prediction_rows(predictions_path)
    for ctx in args.context_lengths:
        ctx_rows = [row for row in rows if row.get("context_length") == ctx]
        if not ctx_rows:
            continue
        passed = sum(1 for row in ctx_rows if row.get("keyword_match"))
        lat = mean([float(row["latency_sec"]) for row in ctx_rows if row.get("latency_sec") is not None])
        print(f"[prepared_context_probe] ctx={ctx} acc={passed}/{len(ctx_rows)} mean_latency={lat}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
