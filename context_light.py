#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight prepared RULER NIAH context probe.

Direct replacement version with clean progress output.

It reuses prepared RULER NIAH single_1 data, bypasses OpenCompass, runs direct
iLLaDA generation, and writes keyword-match summaries plus manual-review files.

Progress output is fixed:
- Interactive terminal: one-line spinner with line clearing.
- nohup/log file: no carriage-return spam; prints a throttled running line.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import random
import re
import string
import subprocess
import sys
import time
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCRIPT_VERSION = "context_light_v4_oc_aligned_clean_progress"


def find_repo_root(start: Path) -> Path:
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "generate.py").exists() and (candidate / "test_config.yaml").exists():
            return candidate
    return start.resolve()


ROOT = find_repo_root(Path(__file__).resolve().parent)
sys.path.insert(0, str(ROOT))

PROMPT_KEYS = ("prompt", "input", "query", "text")
ANSWER_KEYS = ("answer", "needle", "target", "value", "gold", "answers")
CONTEXT_KEYS = ("context_length", "requested_context_length", "max_seq_length")
POSITION_KEYS = ("needle_position", "position", "depth", "depth_percent")
QUESTION_KEYS = ("question", "query")

POSITION_ALIASES = {
    "front": "front", "beginning": "front", "begin": "front", "start": "front",
    "0": "front", "0.0": "front",
    "middle": "middle", "mid": "middle", "50": "middle", "50.0": "middle",
    "back": "back", "end": "back", "100": "back", "100.0": "back",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight probe on prepared RULER NIAH data.")
    parser.add_argument("--prepared-root", default="data/prepared/ruler_niah_single_1")
    parser.add_argument("--output-dir", default="outputs/context_light")
    parser.add_argument("--model-path", default="GSAI-ML/iLLaDA-8B-Instruct")
    parser.add_argument("--context-lengths", nargs="+", type=int, default=[1024, 2048, 4096, 8192])
    parser.add_argument("--needle-positions", nargs="+", default=["front", "middle", "back"])
    parser.add_argument("--num-samples-per-condition", type=int, default=1)
    parser.add_argument("--sample-selection", choices=["first", "index", "random"], default="first")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--gen-steps", type=int, default=128)
    parser.add_argument("--gen-blocksize", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg", type=float, default=0.0)
    parser.add_argument("--token-selection-confidence-threshold", type=float, default=None)
    parser.add_argument("--return-trace", action="store_true")
    parser.add_argument("--trace-token-snapshots", action="store_true")
    parser.add_argument("--remasking", choices=["low_confidence", "random"], default="low_confidence")
    parser.add_argument("--diff-confidence-eos-eot-inf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diff-logits-eos-inf", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--mask-id", type=int, default=5)
    parser.add_argument("--apply-chat-template", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--add-special-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--progress-log-interval", type=float, default=10.0, help="Seconds between running progress lines when stdout is not a TTY.")
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
        r"(?:ctx|context_length|context|len|max_seq_length|seq)[_-]?(\d{3,5})",
        r"(\d{3,5})[_-]?(?:front|middle|back|depth0|depth50|depth100)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    for candidate in (500, 512, 1000, 1024, 2000, 2048, 4000, 4096, 8000, 8192):
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


def iter_records(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"[context_light] warning: bad jsonl line {path}:{line_no}: {exc}", flush=True)
                    continue
                if isinstance(item, dict):
                    yield line_no, item
        return

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                yield idx, item
    elif isinstance(data, dict):
        records = data.get("data") or data.get("records") or data.get("samples") or data.get("test") or []
        if isinstance(records, list):
            for idx, item in enumerate(records):
                if isinstance(item, dict):
                    yield idx, item


def load_prepared_samples(prepared_root: Path) -> List[Dict[str, Any]]:
    if not prepared_root.exists():
        raise SystemExit(f"Prepared root does not exist: {prepared_root}")
    files = sorted(p for p in prepared_root.rglob("*") if p.suffix.lower() in (".jsonl", ".json"))
    if not files:
        raise SystemExit(f"No .jsonl/.json prepared files found under: {prepared_root}")

    samples: List[Dict[str, Any]] = []
    for path in files:
        path_ctx = parse_context_from_path(path)
        path_pos = parse_position_from_path(path)
        for source_line, item in iter_records(path):
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
                "sample_id": f"ctx{ctx}_{pos}_{rel}:{source_line}",
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
    context_lengths: Sequence[int],
    positions: Sequence[str],
    num_per_condition: int,
    selection: str,
    sample_index: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    by_condition: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_condition[(sample["context_length"], sample["needle_position"])].append(sample)

    selected: List[Dict[str, Any]] = []
    for ctx in context_lengths:
        for pos_raw in positions:
            pos = normalize_position(pos_raw) or str(pos_raw)
            candidates = by_condition.get((ctx, pos), [])
            if not candidates:
                print(f"[context_light] warning: no prepared samples for ctx={ctx} pos={pos}", flush=True)
                continue
            if selection == "first":
                chosen = candidates[:num_per_condition]
            elif selection == "index":
                chosen = candidates[sample_index:sample_index + num_per_condition]
            else:
                chosen = rng.sample(candidates, k=min(num_per_condition, len(candidates)))
            selected.extend(chosen)
    if not selected:
        raise SystemExit("No samples selected. Check --prepared-root, lengths, and positions.")
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


def read_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("sample_id"):
                done.add(str(item["sample_id"]))
    return done


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "--"
    seconds = max(0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{sec:02d}s"


def progress_bar(done: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + "-" * width + "]"
    done = max(0, min(done, total))
    filled = int(round(width * done / total))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def truncate_text(text: Any, max_len: int = 120) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\n", " ").strip()
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def print_run_header(args: argparse.Namespace, selected: List[Dict[str, Any]], pending: List[Dict[str, Any]], output_dir: Path) -> None:
    print("\n" + "=" * 88)
    print(f"context_light: prepared RULER NIAH lightweight probe ({SCRIPT_VERSION})")
    print("=" * 88)
    print(f"Model        : {args.model_path}")
    print(f"Output       : {output_dir}")
    print(f"Contexts     : {' / '.join(str(x) for x in args.context_lengths)}")
    print(f"Positions    : {' / '.join(args.needle_positions)}")
    print(f"Samples      : selected={len(selected)} pending={len(pending)} already_done={len(selected) - len(pending)}")
    print(f"Generation   : len={args.gen_length}, steps={args.gen_steps}, block={args.gen_blocksize}, temp={args.temperature}, cfg={args.cfg}")
    print(f"EOS/EOT cfg  : diff_confidence_eos_eot_inf={args.diff_confidence_eos_eot_inf}, diff_logits_eos_inf={args.diff_logits_eos_inf}")
    print(f"Mask ID      : {args.mask_id}  (aligned with project OpenCompass context config)")
    print("=" * 88 + "\n")


def print_selected_table(selected: List[Dict[str, Any]], done_ids: set[str]) -> None:
    print("Selected samples:")
    print(f"{'#':>3}  {'status':<7} {'ctx':>5}  {'pos':<6} {'line':>5}  gold")
    print("-" * 88)
    for idx, sample in enumerate(selected, start=1):
        status = "done" if sample["sample_id"] in done_ids else "pending"
        print(
            f"{idx:>3}  {status:<7} {sample['context_length']:>5}  "
            f"{sample['needle_position']:<6} {sample['source_line']:>5}  "
            f"{truncate_text(sample['groundtruth'], 42)}"
        )
    print("-" * 88 + "\n")


def print_progress_line(
    done_now: int,
    total_pending: int,
    started_all: float,
    last_latency: Optional[float],
    match: Optional[bool],
    sample: Dict[str, Any],
    prediction: str,
) -> None:
    elapsed = time.perf_counter() - started_all
    avg = elapsed / done_now if done_now else None
    remain = (total_pending - done_now) * avg if avg is not None else None
    pct = (100.0 * done_now / total_pending) if total_pending else 100.0
    print(
        f"{progress_bar(done_now, total_pending)} "
        f"{done_now:>2}/{total_pending:<2} {pct:5.1f}% | "
        f"elapsed {format_duration(elapsed)} | eta {format_duration(remain)} | "
        f"last {format_duration(last_latency)} | "
        f"ctx={sample['context_length']} pos={sample['needle_position']:<6} "
        f"match={str(match):<5} | pred={truncate_text(prediction, 90)!r}",
        flush=True,
    )


class LiveSpinner:
    """Tiny dependency-free spinner with clean terminal/log output."""

    def __init__(self, message_fn, interval: float = 0.5, log_interval: float = 10.0):
        self.message_fn = message_fn
        self.interval = interval
        self.log_interval = log_interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._is_tty = sys.stdout.isatty()
        self._last_log_time = 0.0

    def __enter__(self):
        def _run() -> None:
            frames = "|/-\\"
            i = 0
            while not self._stop.is_set():
                msg = self.message_fn()
                now = time.perf_counter()
                if self._is_tty:
                    sys.stdout.write("\r\033[K" + frames[i % len(frames)] + " " + msg)
                    sys.stdout.flush()
                else:
                    if now - self._last_log_time >= self.log_interval:
                        sys.stdout.write(frames[i % len(frames)] + " " + msg + "\n")
                        sys.stdout.flush()
                        self._last_log_time = now
                i += 1
                self._stop.wait(self.interval)

            if self._is_tty:
                sys.stdout.write("\r\033[K")
                sys.stdout.flush()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._is_tty:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()


def make_running_message(
    progress_idx: int,
    total_pending: int,
    sample: Dict[str, Any],
    overall_started: float,
    sample_started: float,
    completed_latencies: Sequence[float],
) -> str:
    completed = progress_idx - 1
    elapsed_all = time.perf_counter() - overall_started
    elapsed_this = time.perf_counter() - sample_started
    avg_done = mean([float(x) for x in completed_latencies])
    eta = None
    if avg_done is not None:
        eta = (total_pending - completed - 1) * avg_done + elapsed_this
    pct = 100.0 * completed / total_pending if total_pending else 100.0
    return (
        f"{progress_bar(completed, total_pending)} {completed:>2}/{total_pending:<2} {pct:5.1f}% | "
        f"running ctx={sample['context_length']} pos={sample['needle_position']:<6} "
        f"sample_elapsed={format_duration(elapsed_this)} | total_elapsed={format_duration(elapsed_all)} | eta~{format_duration(eta)}"
    )


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    return {
        "n": n,
        "keyword_accuracy": sum(1 for r in rows if r.get("keyword_match")) / n if n else None,
        "exact_match_rate": sum(1 for r in rows if r.get("exact_match")) / n if n else None,
        "normalized_match_rate": sum(1 for r in rows if r.get("normalized_match")) / n if n else None,
        "mean_latency_sec": mean([float(r["latency_sec"]) for r in rows if r.get("latency_sec") is not None]),
        "mean_tokens_per_second": mean([float(r["tokens_per_second"]) for r in rows if r.get("tokens_per_second") is not None]),
        "mean_prompt_tokens": mean([float(r["prompt_tokens"]) for r in rows if r.get("prompt_tokens") is not None]),
        "mean_max_memory_gb": mean([float(r["max_memory_gb"]) for r in rows if r.get("max_memory_gb") is not None]),
    }


def write_summaries(output_dir: Path) -> None:
    rows = load_jsonl(output_dir / "predictions.jsonl")
    if not rows:
        return

    by_condition: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    by_context: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[(row.get("context_length"), row.get("needle_position"))].append(row)
        by_context[row.get("context_length")].append(row)

    condition_rows = []
    for (ctx, pos), group in sorted(by_condition.items(), key=lambda x: (int(x[0][0]), str(x[0][1]))):
        condition_rows.append({"context_length": ctx, "needle_position": pos, **aggregate_rows(group)})
    write_csv(
        output_dir / "summary_by_condition.csv",
        condition_rows,
        [
            "context_length", "needle_position", "n", "keyword_accuracy", "exact_match_rate",
            "normalized_match_rate", "mean_latency_sec", "mean_tokens_per_second",
            "mean_prompt_tokens", "mean_max_memory_gb",
        ],
    )

    overall_rows = []
    for ctx, group in sorted(by_context.items(), key=lambda x: int(x[0])):
        overall_rows.append({"group": f"ctx{ctx}", "context_length": ctx, **aggregate_rows(group)})
    overall_rows.append({"group": "all", "context_length": "all", **aggregate_rows(rows)})
    write_csv(
        output_dir / "summary_overall.csv",
        overall_rows,
        [
            "group", "context_length", "n", "keyword_accuracy", "exact_match_rate",
            "normalized_match_rate", "mean_latency_sec", "mean_tokens_per_second",
            "mean_prompt_tokens", "mean_max_memory_gb",
        ],
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
        [
            "context_length", "needle_position", "sample_id", "groundtruth", "prediction",
            "keyword_match", "manual_judgment", "failure_type",
        ],
    )


def git_commit() -> Optional[str]:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True, check=False)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def get_torch_dtype(name: str):
    import torch
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def build_prompt(tokenizer: Any, prompt: str, apply_chat_template: bool) -> str:
    if not apply_chat_template:
        return prompt
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    except Exception as exc:
        print(f"[context_light] warning: failed to apply chat template ({exc}); using raw prompt.", flush=True)
        return prompt


def call_llada_generate(generate_fn: Any, **kwargs: Any) -> Any:
    sig = inspect.signature(generate_fn)
    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return generate_fn(**accepted)


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

    all_samples = load_prepared_samples(prepared_root)
    positions = [normalize_position(pos) or str(pos) for pos in args.needle_positions]
    selected = select_samples(
        all_samples,
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
        "script_version": SCRIPT_VERSION,
        "argv": sys.argv,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "model_path": args.model_path,
        "prepared_root": str(prepared_root),
        "output_dir": str(output_dir),
        "num_selected": len(selected),
        "num_pending": len(pending),
        "args": vars(args),
        "note": "context_light is a lightweight context sanity check using prepared RULER NIAH data.",
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    print_run_header(args, selected, pending, output_dir)
    print(f"Prepared data : {prepared_root}")
    print_selected_table(selected, done_ids)

    if args.dry_run:
        print("Dry-run only; no model loaded.")
        return 0

    if not pending:
        print("[context_light] nothing to do; all selected samples already exist.")
        write_summaries(output_dir)
        return 0

    import torch
    from transformers import AutoModel, AutoTokenizer
    from generate import generate as LLaDA_generate

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Loading tokenizer/model on {device} ...", flush=True)
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=get_torch_dtype(args.torch_dtype) if device.type == "cuda" else torch.float32,
    ).to(device).eval()
    print(f"Model loaded in {format_duration(time.perf_counter() - load_started)}. Starting {len(pending)} sample(s).\n", flush=True)

    total_pending = len(pending)
    overall_started = time.perf_counter()
    completed_latencies: List[float] = []

    with predictions_path.open("a", encoding="utf-8") as pred_f, prompts_path.open("a", encoding="utf-8") as prompt_f:
        for progress_idx, sample in enumerate(pending, start=1):
            print(
                f"Running {progress_idx}/{total_pending}: ctx={sample['context_length']} "
                f"pos={sample['needle_position']} line={sample['source_line']} "
                f"gold={truncate_text(sample['groundtruth'], 60)!r}",
                flush=True,
            )

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
            with LiveSpinner(
                lambda idx=progress_idx, sample=sample, started=started: make_running_message(
                    idx, total_pending, sample, overall_started, started, completed_latencies
                ),
                log_interval=args.progress_log_interval,
            ):
                with torch.inference_mode():
                    generated = call_llada_generate(
                        LLaDA_generate,
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
                        diff_confidence_eos_eot_inf=args.diff_confidence_eos_eot_inf,
                        diff_logits_eos_inf=args.diff_logits_eos_inf,
                        return_trace=args.return_trace,
                        trace_token_snapshots=args.trace_token_snapshots,
                        tokenizer=tokenizer,
                    )

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latency = time.perf_counter() - started
            completed_latencies.append(latency)

            if args.return_trace and isinstance(generated, tuple):
                generated_tokens, _trace = generated
            else:
                generated_tokens = generated

            prediction = tokenizer.decode(generated_tokens[0, input_ids.shape[1]:], skip_special_tokens=True)
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
                "mask_id": args.mask_id,
                "diff_confidence_eos_eot_inf": args.diff_confidence_eos_eot_inf,
                "diff_logits_eos_inf": args.diff_logits_eos_inf,
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

            print_progress_line(progress_idx, total_pending, overall_started, latency, record["keyword_match"], sample, prediction)

    write_summaries(output_dir)
    rows = load_jsonl(predictions_path)
    print("\n" + "=" * 88)
    print("Finished. Summary by context length")
    print("=" * 88)
    print(f"{'ctx':>6}  {'n':>3}  {'pass':>5}  {'acc':>7}  {'mean latency':>13}  {'mean tps':>10}")
    print("-" * 88)
    for ctx in args.context_lengths:
        ctx_rows = [row for row in rows if row.get("context_length") == ctx]
        if not ctx_rows:
            continue
        passed = sum(1 for row in ctx_rows if row.get("keyword_match"))
        lat = mean([float(row["latency_sec"]) for row in ctx_rows if row.get("latency_sec") is not None])
        tps = mean([float(row["tokens_per_second"]) for row in ctx_rows if row.get("tokens_per_second") is not None])
        acc = passed / len(ctx_rows) if ctx_rows else 0.0
        print(f"{ctx:>6}  {len(ctx_rows):>3}  {passed:>5}  {acc:>7.2%}  {format_duration(lat):>13}  {tps if tps is not None else '--':>10}")
    print("-" * 88)
    print(f"Outputs written to: {output_dir}")
    print("Review file       : manual_review.csv")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
