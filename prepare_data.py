#!/usr/bin/env python3
"""Config-driven native data preparation.

This file keeps the original LLaDA data-preparation layout.

Only local RULER-style context data is materialized under:

    data/prepared/<ruler_benchmark>/<condition_id>.jsonl

GSM8K / MBPP / custom_math continue to use their original data sources
(HuggingFace dataset cache or data/custom_math) and are not duplicated here.

The only added context variant is:

    ruler_niah_double_2

It sits next to ruler_niah_single_1 and inserts two needle records, requiring
ordered retrieval of the two values.  `ruler_niah_order_2` is accepted as a
backward-compatible alias.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from run_test import ROOT, as_list, collect_experiments, deep_merge, expand_matrix, load_yaml, safe_name
except Exception:  # pragma: no cover - keeps the file importable in minimal environments
    ROOT = Path(__file__).resolve().parent

    def as_list(value: Any) -> List[Any]:
        if value is None:
            return []
        return value if isinstance(value, list) else [value]

    def safe_name(value: str) -> str:
        return re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", str(value).lower())).strip("_")

    def load_yaml(path: Path) -> Dict[str, Any]:
        import yaml
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        out = deepcopy(base)
        for k, v in (override or {}).items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = deep_merge(out[k], v)
            else:
                out[k] = v
        return out

    def expand_matrix(experiment: Dict[str, Any]):
        import itertools
        params = experiment.get("params", {}) or {}
        sweep = experiment.get("sweep", {}) or {}
        if not sweep:
            yield params
            return
        keys = list(sweep.keys())
        vals = [as_list(sweep[k]) for k in keys]
        for combo in itertools.product(*vals):
            item = deepcopy(params)
            for k, v in zip(keys, combo):
                item[k] = v
            yield item

    def collect_experiments(config: Dict[str, Any], selected: Set[str]):
        tasks = config.get("tasks", {}) or {}
        out = []
        for task_name, task_def in tasks.items():
            for exp in task_def.get("experiments", []) or []:
                if selected and task_name not in selected and exp.get("name") not in selected and "all" not in selected:
                    continue
                item = deepcopy(exp)
                item.setdefault("task", task_name)
                out.append(item)
        return out


BENCH_ALIASES = {
    "gsm8k": "gsm8k",
    "mbpp": "mbpp",
    "custom_math": "cmath",
    "ruler_niah_single_1": "niah1",
    "ruler_niah_order_2": "niah2",
    "ruler_niah_double_2": "niah2",
}

DEPTH_BY_POSITION = {"front": [15], "middle": [50], "back": [85], "end": [95]}
DEPTH_BY_PAIR = {
    "front_middle": [15, 55],
    "front_back": [15, 85],
    "middle_back": [45, 85],
    "front_back_extreme": [5, 95],
}

PARAM_ALIASES = {
    "sample_indices": "s",
    "sample_limit": "n",
    "num_samples": "n",
    "context_length": "ctx",
    "needle_position": "pos",
    "needle_pair": "pair",
    "gen_length": "l",
    "gen_blocksize": "b",
    "gen_steps": "st",
    "token_selection_confidence_threshold": "thr",
    "threshold_schedule_label": "thrs",
    "speed_schedule_label": "spd",
    "speed_schedule_name": "spd",
    "decode_order": "ord",
    "w1_sampler": "w1sam",
    "w1_steps": "w1st",
    "w1_parallel_tokens": "w1pt",
    "w1_decode_mode": "w1mode",
    "w1_decode_order": "w1ord",
    "w1_confidence_threshold": "w1thr",
    "seed": "seed",
}

PREFERRED_KEYS = [
    "sample_indices",
    "sample_limit",
    "num_samples",
    "context_length",
    "needle_position",
    "needle_pair",
    "gen_length",
    "gen_blocksize",
    "gen_steps",
    "token_selection_confidence_threshold",
    "threshold_schedule_label",
    "speed_schedule_label",
    "speed_schedule_name",
    "decode_order",
    "w1_sampler",
    "w1_steps",
    "w1_parallel_tokens",
    "w1_decode_mode",
    "w1_decode_order",
    "w1_confidence_threshold",
    "seed",
]

NAMELESS_KEYS = {
    "return_trace",
    "trace_token_snapshots",
    "trace_decode_snapshots",
    "trace_step0_full_confidence",
    "min_transfer_tokens",
    "temperature",
    "cfg",
    "remasking",
    "max_seq_len",
    "decoding_config_name",
    "arness",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["status"]
        rows = [{"status": "empty"}]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def bench_alias(benchmark: str) -> str:
    return BENCH_ALIASES.get(str(benchmark), safe_name(str(benchmark))[:24] or "bench")


def value_label(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return ("%.6g" % value).replace(".", "p").replace("-", "m")
    if isinstance(value, (list, tuple)):
        return "-".join(value_label(v) for v in value)
    return safe_name(str(value).replace(".", "p"))


def condition_name(params: Dict[str, Any]) -> str:
    pieces = []
    # W1 public backend uses w1_steps / w1_sampler instead of iLLaDA
    # gen_steps / gen_blocksize.  Skip irrelevant default iLLaDA knobs so paths
    # stay short and semantically meaningful.
    dynamic_skip = set(NAMELESS_KEYS)
    if params.get("w1_sampler") is not None or params.get("w1_steps") is not None:
        dynamic_skip.update({"gen_steps", "gen_blocksize", "token_selection_confidence_threshold"})
    for key in PREFERRED_KEYS:
        if key in params and key not in dynamic_skip and params.get(key) is not None:
            pieces.append(f"{PARAM_ALIASES.get(key, key)}{value_label(params.get(key))}")
    if not pieces:
        pieces.append("default")
    text = "_".join(pieces)
    if len(text) > 120:
        import hashlib
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
        text = text[:111].rstrip("_") + "_" + digest
    return text


def ruler_prepared_condition_id(benchmark: str, params: Dict[str, Any]) -> str:
    """Return the original LLaDA-style prepared filename stem for RULER data.

    Keep RULER data under data/prepared/<benchmark>/<stem>.jsonl so the
    native path can reuse the same data split as the OpenCompass/LLaDA path.
    """
    seed = int(params.get("seed", 42) or 42)
    gen_length = int(params.get("gen_length", 128) or 128)
    num_samples = int(params.get("num_samples", 20) or 20)
    if benchmark == "ruler_niah_single_1":
        return safe_name(
            "ruler_niah_single_1_"
            f"ctx{params.get('context_length')}_"
            f"pos{params.get('needle_position')}_"
            f"gen{gen_length}_"
            f"samples{num_samples}_"
            f"seed{seed}"
        )
    if benchmark in {"ruler_niah_order_2", "ruler_niah_double_2"}:
        return safe_name(
            f"{benchmark}_"
            f"ctx{params.get('context_length')}_"
            f"pair{params.get('needle_pair')}_"
            f"gen{gen_length}_"
            f"samples{num_samples}_"
            f"seed{seed}"
        )
    return condition_name(params)


def prepared_file_for(root: Path, benchmark: str, params: Dict[str, Any]) -> Path:
    if benchmark in {"ruler_niah_single_1", "ruler_niah_order_2", "ruler_niah_double_2"}:
        return root / benchmark / f"{ruler_prepared_condition_id(benchmark, params)}.jsonl"
    return root / benchmark / f"{condition_name(params)}.jsonl"


def prepared_dir_for(root: Path, task: str, benchmark: str, params: Dict[str, Any]) -> Path:
    # Backward-compatible helper retained for older native_model imports.
    # New RULER data is stored as data/prepared/<benchmark>/<file>.jsonl, not
    # under task/native subdirectories.
    return root / benchmark / condition_name(params)


def iter_conditions(config: Dict[str, Any], selected: Set[str]) -> Iterable[Dict[str, Any]]:
    defaults = config.get("defaults", {}) or {}
    for experiment in collect_experiments(config, selected):
        task = experiment.get("task") or "runs"
        for benchmark in as_list(experiment.get("benchmark")):
            for idx, params in enumerate(expand_matrix(experiment), start=1):
                merged = deep_merge(defaults, params)
                yield {
                    "task": task,
                    "experiment": experiment.get("name"),
                    "benchmark": str(benchmark),
                    "params": merged,
                    "condition": condition_name(merged),
                    "condition_index": idx,
                }


def selected_sample_ids(params: Dict[str, Any], total: Optional[int] = None, default_limit: Optional[int] = None) -> List[int]:
    if params.get("sample_indices") is not None:
        return [int(x) for x in as_list(params.get("sample_indices"))]
    limit = params.get("sample_limit")
    if limit is None:
        limit = default_limit if default_limit is not None else total
    if limit is None:
        raise SystemExit("sample_limit is required when dataset size is unknown.")
    return list(range(min(int(limit), int(total)))) if total is not None else list(range(int(limit)))


# -------------------------- standard task loaders --------------------------


def load_hf_dataset_any(candidates: Sequence[Tuple[str, Optional[str], str]]):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing dependency `datasets`. Install it or prepare the data on a machine that has dataset caches.") from exc
    errors = []
    for name, subset, split in candidates:
        try:
            if subset is None:
                return load_dataset(name, split=split)
            return load_dataset(name, subset, split=split)
        except Exception as exc:  # keep trying common mirrors/names
            errors.append(f"{name}/{subset}/{split}: {repr(exc)}")
    raise SystemExit("Could not load any dataset candidate:\n" + "\n".join(errors))


def prepare_gsm8k(condition: Dict[str, Any]) -> List[Dict[str, Any]]:
    ds = load_hf_dataset_any([
        ("gsm8k", "main", "test"),
        ("openai/gsm8k", "main", "test"),
    ])
    params = condition["params"]
    ids = selected_sample_ids(params, total=len(ds))
    rows: List[Dict[str, Any]] = []
    for sid in ids:
        item = ds[int(sid)]
        question = str(item.get("question") or item.get("prompt") or "")
        answer = str(item.get("answer") or item.get("target") or "")
        prompt = (
            "Solve the following grade-school math problem. "
            "Give the final answer clearly.\n\n"
            f"Question: {question}\nAnswer:"
        )
        rows.append({
            "task": condition["task"],
            "experiment": condition["experiment"],
            "benchmark": "gsm8k",
            "sample_id": int(sid),
            "prompt": prompt,
            "answer": answer,
            "metadata": {"question": question, "source": "gsm8k/test"},
        })
    return rows


def prepare_mbpp(condition: Dict[str, Any]) -> List[Dict[str, Any]]:
    ds = load_hf_dataset_any([
        ("google-research-datasets/mbpp", "sanitized", "test"),
        ("mbpp", "sanitized", "test"),
        ("google-research-datasets/mbpp", None, "test"),
        ("mbpp", None, "test"),
    ])
    params = condition["params"]
    ids = selected_sample_ids(params, total=len(ds))
    rows: List[Dict[str, Any]] = []
    for sid in ids:
        item = ds[int(sid)]
        text = str(item.get("text") or item.get("prompt") or item.get("description") or "")
        test_list = item.get("test_list") or item.get("tests") or []
        if isinstance(test_list, str):
            try:
                test_list = json.loads(test_list)
            except Exception:
                test_list = [test_list]
        prompt = (
            "You are an expert Python programmer. Write Python code that solves the following problem.\n"
            "Return only the code, without explanations.\n\n"
            f"Problem: {text}\n"
        )
        rows.append({
            "task": condition["task"],
            "experiment": condition["experiment"],
            "benchmark": "mbpp",
            "sample_id": int(item.get("task_id", sid)),
            "dataset_index": int(sid),
            "prompt": prompt,
            "answer": item.get("code") or "",
            "metadata": {
                "text": text,
                "test_list": test_list,
                "test_setup_code": item.get("test_setup_code") or "",
                "entry_point": item.get("entry_point") or None,
                "source": "mbpp/test",
            },
        })
    return rows


def prepare_custom_math(condition: Dict[str, Any], data_root: Path) -> List[Dict[str, Any]]:
    candidates = []
    custom_root = data_root / "custom_math"
    if custom_root.is_dir():
        candidates.extend(sorted(custom_root.glob("*.jsonl")))
        candidates.extend(sorted(custom_root.glob("*.json")))
    if not candidates:
        raise SystemExit(f"custom_math data not found under {custom_root}")
    raw: List[Dict[str, Any]] = []
    for path in candidates:
        if path.suffix == ".jsonl":
            raw.extend(read_jsonl(path))
        else:
            obj = json.loads(path.read_text(encoding="utf-8"))
            raw.extend(obj if isinstance(obj, list) else obj.get("data", []))
    params = condition["params"]
    ids = selected_sample_ids(params, total=len(raw))
    rows = []
    for sid in ids:
        item = raw[int(sid)]
        prompt = item.get("prompt") or item.get("question") or item.get("input")
        answer = item.get("answer") or item.get("target") or item.get("reference")
        rows.append({
            "task": condition["task"],
            "experiment": condition["experiment"],
            "benchmark": "custom_math",
            "sample_id": int(sid),
            "prompt": str(prompt),
            "answer": str(answer),
            "metadata": {k: v for k, v in item.items() if k not in {"prompt", "question", "input", "answer", "target", "reference"}},
        })
    return rows


def source_check(condition: Dict[str, Any], data_root: Path) -> Dict[str, Any]:
    """Validate non-prepared data sources without materializing them.

    GSM8K and MBPP stay in HuggingFace datasets cache.  This function only
    tries to load the split and select the requested sample ids.  It writes no
    per-sample data under data/prepared.  custom_math stays under data/custom_math.
    """
    benchmark = str(condition.get("benchmark"))
    params = condition.get("params", {}) or {}
    base = {
        "task": condition.get("task"),
        "experiment": condition.get("experiment"),
        "benchmark": benchmark,
        "condition": condition.get("condition"),
        "params": params,
    }

    try:
        if benchmark == "gsm8k":
            ds = load_hf_dataset_any([
                ("gsm8k", "main", "test"),
                ("openai/gsm8k", "main", "test"),
            ])
            ids = selected_sample_ids(params, total=len(ds))
            # Touch selected rows to catch index/schema/cache problems early.
            for sid in ids:
                _ = ds[int(sid)]
            return {
                **base,
                "status": "ok",
                "source_type": "huggingface_cache",
                "source": "gsm8k/main/test",
                "total_size": len(ds),
                "selected_samples": len(ids),
                "selected_indices": json.dumps(ids, ensure_ascii=False),
                "prepared_file": "",
            }

        if benchmark == "mbpp":
            ds = load_hf_dataset_any([
                ("google-research-datasets/mbpp", "sanitized", "test"),
                ("mbpp", "sanitized", "test"),
                ("google-research-datasets/mbpp", None, "test"),
                ("mbpp", None, "test"),
            ])
            ids = selected_sample_ids(params, total=len(ds))
            for sid in ids:
                _ = ds[int(sid)]
            return {
                **base,
                "status": "ok",
                "source_type": "huggingface_cache",
                "source": "mbpp/test",
                "total_size": len(ds),
                "selected_samples": len(ids),
                "selected_indices": json.dumps(ids, ensure_ascii=False),
                "prepared_file": "",
            }

        if benchmark == "custom_math":
            custom_root = data_root / "custom_math"
            files = []
            if custom_root.is_dir():
                files.extend(sorted(custom_root.glob("*.jsonl")))
                files.extend(sorted(custom_root.glob("*.json")))
            if not files:
                raise FileNotFoundError(f"custom_math data not found under {custom_root}")
            # Reuse loader for index validation but do not write data/prepared.
            rows = prepare_custom_math(condition, data_root)
            return {
                **base,
                "status": "ok",
                "source_type": "local_original",
                "source": str(custom_root),
                "total_size": "",
                "selected_samples": len(rows),
                "selected_indices": json.dumps([r.get("sample_id") for r in rows], ensure_ascii=False),
                "prepared_file": "",
            }

        return {**base, "status": "skipped", "source_type": "unknown", "source": "", "prepared_file": ""}
    except Exception as exc:
        return {
            **base,
            "status": "failed",
            "source_type": "huggingface_cache" if benchmark in {"gsm8k", "mbpp"} else "local_original",
            "source": benchmark,
            "error": repr(exc),
            "prepared_file": "",
        }


# -------------------------- RULER-style generators -------------------------


def load_haystack_text(haystack_path: Path) -> str:
    if not haystack_path.exists():
        raise SystemExit(f"Missing RULER haystack file: {haystack_path}")
    chunks: List[str] = []
    with haystack_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text") or obj.get("content") or ""
            except json.JSONDecodeError:
                text = line
            if text:
                chunks.append(str(text).strip())
    text = " ".join(chunks)
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def make_key(rng: random.Random) -> str:
    adjs = ["amber", "brisk", "crimson", "distant", "emerald", "frozen", "golden", "hidden", "ivory", "jade", "quiet", "velvet"]
    nouns = ["otter", "llama", "falcon", "panda", "tiger", "orbit", "harbor", "comet", "meadow", "canyon", "lantern", "river"]
    return f"{rng.choice(adjs)}-{rng.choice(nouns)}-{rng.randint(100, 999)}"


def make_number(rng: random.Random, used: Set[str]) -> str:
    while True:
        value = str(rng.randint(1_000_000, 9_999_999))
        if value not in used:
            used.add(value)
            return value


def token_count(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text.split()))


def insert_at_depths(sentences: List[str], needles: List[str], depths: List[int]) -> str:
    output = list(sentences)
    n = max(1, len(output))
    for depth, needle in sorted(zip(depths, needles), key=lambda x: int(x[0]), reverse=True):
        pos = int(round(n * max(0, min(100, int(depth))) / 100.0))
        pos = max(0, min(len(output), pos))
        output.insert(pos, needle)
    return " ".join(output)


def fit_context_prompt(
    *,
    base_sentences: List[str],
    needles: List[str],
    depths: List[int],
    context_length: int,
    gen_length: int,
    prompt_builder,
) -> Tuple[str, int, bool]:
    target = max(1, int(context_length) - int(gen_length))
    lo, hi = 1, len(base_sentences)
    best_context = insert_at_depths(base_sentences[:1], needles, depths)
    best_prompt = prompt_builder(best_context)
    best_tokens = token_count(best_prompt)
    if best_tokens > target:
        return best_prompt, best_tokens, False
    while lo <= hi:
        mid = (lo + hi) // 2
        context = insert_at_depths(base_sentences[:mid], needles, depths)
        prompt = prompt_builder(context)
        n_tokens = token_count(prompt)
        if n_tokens <= target:
            best_prompt, best_tokens = prompt, n_tokens
            lo = mid + 1
        else:
            hi = mid - 1
    return best_prompt, best_tokens, best_tokens < 0.85 * target and context_length >= 4096


def build_single_prompt(context: str, key: str) -> str:
    return (
        "A needle record is hidden within the following passage.\n"
        "Read the passage carefully.\n\n"
        f"{context}\n\n"
        f"Question: What is the 7-digit magic number for {key}?\n"
        "Output only the 7-digit number."
    )


def build_order2_prompt(context: str) -> str:
    return (
        "Some needle records are hidden within the following passage.\n"
        "Each needle record contains a unique key and one 7-digit magic number.\n"
        "Read the passage carefully and keep the order of the records.\n\n"
        f"{context}\n\n"
        "Question: There are exactly two needle records in the passage. "
        "Return the two 7-digit magic numbers in the exact order their needle records appear in the passage.\n"
        "Output JSON only, for example: [\"1234567\",\"7654321\"]."
    )


def prepare_ruler_single(condition: Dict[str, Any], haystack_path: Path) -> List[Dict[str, Any]]:
    params = condition["params"]
    context_length = int(params["context_length"])
    gen_length = int(params.get("gen_length", 128))
    num_samples = int(params.get("num_samples", 20))
    seed = int(params.get("seed", 42) or 42)
    position = str(params.get("needle_position", "middle"))
    if position not in DEPTH_BY_POSITION:
        raise SystemExit(f"Unsupported needle_position `{position}`. Available: {', '.join(DEPTH_BY_POSITION)}")
    base_sentences = split_sentences(load_haystack_text(haystack_path))
    rows: List[Dict[str, Any]] = []
    for i in range(num_samples):
        rng = random.Random(seed + i * 7919)
        used: Set[str] = set()
        key = make_key(rng)
        value = make_number(rng, used)
        needle = f"Needle record: The magic number for {key} is {value}."
        prompt, prompt_tokens, underfilled = fit_context_prompt(
            base_sentences=base_sentences,
            needles=[needle],
            depths=DEPTH_BY_POSITION[position],
            context_length=context_length,
            gen_length=gen_length,
            prompt_builder=lambda ctx, k=key: build_single_prompt(ctx, k),
        )
        rows.append({
            "task": condition["task"],
            "experiment": condition["experiment"],
            "benchmark": "ruler_niah_single_1",
            "sample_id": i,
            "prompt": prompt,
            "answer": value,
            "metadata": {
                "context_length": context_length,
                "needle_position": position,
                "needle_depths": DEPTH_BY_POSITION[position],
                "needle_key": key,
                "needle_value": value,
                "nominal_prompt_tokens": prompt_tokens,
                "underfilled": underfilled,
            },
        })
    return rows


def prepare_ruler_order2(condition: Dict[str, Any], haystack_path: Path) -> List[Dict[str, Any]]:
    params = condition["params"]
    context_length = int(params["context_length"])
    gen_length = int(params.get("gen_length", 64))
    num_samples = int(params.get("num_samples", 20))
    seed = int(params.get("seed", 42) or 42)
    pair = str(params.get("needle_pair", "front_back"))
    if pair not in DEPTH_BY_PAIR:
        raise SystemExit(f"Unsupported needle_pair `{pair}`. Available: {', '.join(DEPTH_BY_PAIR)}")
    depths = DEPTH_BY_PAIR[pair]
    base_sentences = split_sentences(load_haystack_text(haystack_path))
    rows: List[Dict[str, Any]] = []
    for i in range(num_samples):
        rng = random.Random(seed + i * 7919)
        used: Set[str] = set()
        key1, key2 = make_key(rng), make_key(rng)
        while key2 == key1:
            key2 = make_key(rng)
        value1, value2 = make_number(rng, used), make_number(rng, used)
        needles = [
            f"Needle record: The magic number for {key1} is {value1}.",
            f"Needle record: The magic number for {key2} is {value2}.",
        ]
        prompt, prompt_tokens, underfilled = fit_context_prompt(
            base_sentences=base_sentences,
            needles=needles,
            depths=depths,
            context_length=context_length,
            gen_length=gen_length,
            prompt_builder=build_order2_prompt,
        )
        rows.append({
            "task": condition["task"],
            "experiment": condition["experiment"],
            "benchmark": "ruler_niah_double_2",
            "sample_id": i,
            "prompt": prompt,
            "answer": [value1, value2],
            "metadata": {
                "context_length": context_length,
                "needle_pair": pair,
                "needle_depths": depths,
                "needle_keys_in_order": [key1, key2],
                "needle_values_in_order": [value1, value2],
                "nominal_prompt_tokens": prompt_tokens,
                "underfilled": underfilled,
            },
        })
    return rows


def prepare_rows(condition: Dict[str, Any], config: Dict[str, Any], data_root: Path) -> Optional[List[Dict[str, Any]]]:
    """Prepare only synthetic RULER data.

    GSM8K / MBPP / custom_math already have existing data loaders and should
    not be duplicated into data/prepared.  native_model.py will read those
    original sources directly and only copy the exact selected inputs into
    model_outputs/<model>/... for provenance.
    """
    benchmark = condition["benchmark"]
    data_cfg = config.get("data", {}) or {}

    if benchmark in {"gsm8k", "mbpp", "custom_math"}:
        return None

    if benchmark in {"ruler_niah_single_1", "ruler_niah_order_2", "ruler_niah_double_2"}:
        haystack = Path(data_cfg.get("ruler_haystack_path", "data/ruler/paul_graham_essay.jsonl"))
        if not haystack.is_absolute():
            haystack = ROOT / haystack
        if benchmark == "ruler_niah_single_1":
            return prepare_ruler_single(condition, haystack)
        return prepare_ruler_order2(condition, haystack)

    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare native input jsonl files from test_config.yaml.")
    parser.add_argument("--config", default="test_config.yaml")
    parser.add_argument("--only", nargs="*", default=[], help="Task or experiment names to prepare.")
    parser.add_argument("--output-root", default=None, help="Default: data.prepared_dir or data/prepared.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_yaml(config_path)
    selected = set(args.only or [])
    data_cfg = config.get("data", {}) or {}
    base_prepared = Path(args.output_root or Path(data_cfg.get("prepared_dir", "data/prepared")))
    if not base_prepared.is_absolute():
        base_prepared = ROOT / base_prepared

    data_root = ROOT / "data"
    conditions = list(iter_conditions(config, selected))
    if not conditions:
        raise SystemExit("No experiments matched. Use --only <task-or-experiment> or set run.tasks in config.")

    manifest_rows: List[Dict[str, Any]] = []
    log(f"Preparing {len(conditions)} condition(s). Output root: {base_prepared}")

    for condition in conditions:
        input_path = prepared_file_for(base_prepared, condition["benchmark"], condition["params"])
        meta_path = input_path.with_suffix(".json")
        rows = prepare_rows(condition, config, data_root)
        if rows is None:
            log(
                f"- {condition['experiment']} / {condition['benchmark']} / {condition['condition']} "
                "-> original dataset source, no prepared file needed"
            )
            if args.dry_run:
                continue
            if args.skip_source_check:
                check = {
                    "task": condition["task"],
                    "experiment": condition["experiment"],
                    "benchmark": condition["benchmark"],
                    "condition": condition["condition"],
                    "params": condition["params"],
                    "input_path": "original_source",
                    "num_samples": None,
                    "status": "skipped_source_check",
                    "source_type": "original",
                    "prepared_file": "",
                }
            else:
                check = source_check(condition, data_root)
                check["input_path"] = "original_source"
                check["num_samples"] = check.get("selected_samples")
            if check.get("status") == "ok":
                log(f"  [OK] {check.get('source_type')}: {check.get('source')} selected={check.get('selected_samples')}")
            elif check.get("status") == "failed":
                log(f"  [FAILED] {condition['benchmark']}: {check.get('error')}")
            manifest_rows.append(check)
            continue

        exists = input_path.exists()
        log(f"- {condition['experiment']} / {condition['benchmark']} / {condition['condition']} -> {input_path} ({'exists' if exists else 'new'})")
        if args.dry_run:
            continue
        if exists and not args.force:
            rows = read_jsonl(input_path)
            status = "exists"
        else:
            write_jsonl(input_path, rows)
            status = "prepared"
        meta = {
            "task": condition["task"],
            "experiment": condition["experiment"],
            "benchmark": condition["benchmark"],
            "condition": condition["condition"],
            "params": condition["params"],
            "input_path": str(input_path),
            "num_samples": len(rows),
            "status": status,
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
        manifest_rows.append(meta)

    if not args.dry_run:
        write_jsonl(base_prepared / "prepared_manifest.jsonl", manifest_rows)
        write_csv(base_prepared / "prepared_manifest.csv", manifest_rows)
        log(f"Wrote manifest: {base_prepared / 'prepared_manifest.jsonl'}")
        failed = [row for row in manifest_rows if row.get("status") == "failed"]
        if failed:
            log(f"Data source check failed for {len(failed)} condition(s).")
            return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
