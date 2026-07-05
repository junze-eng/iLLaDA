#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Config-driven native data preparation for the iLLaDA experiments.

Default behavior:
  - Expands all experiments selected by test_config.yaml/run.tasks or --only.
  - Checks every condition by default.
  - Materializes local RULER/NIAH prepared jsonl files under data/prepared.
  - For GSM8K / MBPP / custom_math, does not duplicate data, but validates that the
    requested source data and selected sample ids are available.

Supported RULER benchmarks:
  - ruler_niah_single_1
  - ruler_niah_double
  - ruler_niah_double_2
  - ruler_niah_order_2

The double variants insert two needle records and require the model to return the two
7-digit values in the exact order in which the records appear in the context.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parent


# -----------------------------------------------------------------------------
# Optional reuse of helpers from run_test.py
# -----------------------------------------------------------------------------

def _fallback_as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _fallback_safe_name(value: Any) -> str:
    text = str(value).replace(".", "p")
    text = re.sub(r"[^0-9a-zA-Z_-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "none"


def _fallback_load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Missing dependency `pyyaml`. Install it with: pip install pyyaml") from exc
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return obj or {}


def _fallback_deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base or {})
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _fallback_deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _fallback_expand_matrix(experiment: Dict[str, Any]):
    params = experiment.get("params", {}) or {}
    sweep = experiment.get("sweep", {}) or {}
    if not sweep:
        yield deepcopy(params)
        return
    keys = list(sweep.keys())
    values = [_fallback_as_list(sweep[k]) for k in keys]
    for combo in itertools.product(*values):
        item = deepcopy(params)
        for k, v in zip(keys, combo):
            item[k] = v
        yield item


def _fallback_collect_experiments(config: Dict[str, Any], selected: Set[str]):
    tasks = config.get("tasks", {}) or {}
    run_cfg = config.get("run", {}) or {}
    run_tasks = set(_fallback_as_list(run_cfg.get("tasks")))
    run_exps = set(_fallback_as_list(run_cfg.get("experiments")))

    out = []
    for task_name, task_def in tasks.items():
        if run_tasks and task_name not in run_tasks and "all" not in run_tasks:
            continue
        for exp in task_def.get("experiments", []) or []:
            exp_name = exp.get("name")
            if run_exps and exp_name not in run_exps and "all" not in run_exps:
                continue
            if selected and "all" not in selected and task_name not in selected and exp_name not in selected:
                continue
            item = deepcopy(exp)
            item.setdefault("task", task_name)
            item.setdefault("task_output_path", task_def.get("output_path"))
            out.append(item)
    return out


try:  # Prefer project helpers if they exist and are compatible.
    from run_test import (  # type: ignore
        as_list as _rt_as_list,
        collect_experiments as _rt_collect_experiments,
        deep_merge as _rt_deep_merge,
        expand_matrix as _rt_expand_matrix,
        load_yaml as _rt_load_yaml,
        safe_name as _rt_safe_name,
    )

    as_list = _rt_as_list
    safe_name = _rt_safe_name
    load_yaml = _rt_load_yaml
    deep_merge = _rt_deep_merge
    expand_matrix = _rt_expand_matrix

    def collect_experiments(config: Dict[str, Any], selected: Set[str]):
        try:
            return _rt_collect_experiments(config, selected)
        except TypeError:
            return _fallback_collect_experiments(config, selected)

except Exception:  # pragma: no cover
    as_list = _fallback_as_list
    safe_name = _fallback_safe_name
    load_yaml = _fallback_load_yaml
    deep_merge = _fallback_deep_merge
    expand_matrix = _fallback_expand_matrix
    collect_experiments = _fallback_collect_experiments


# -----------------------------------------------------------------------------
# Naming / config expansion
# -----------------------------------------------------------------------------

SINGLE_BENCHMARKS = {"ruler_niah_single_1"}
DOUBLE_BENCHMARKS = {"ruler_niah_double", "ruler_niah_double_2", "ruler_niah_order_2"}
RULER_BENCHMARKS = SINGLE_BENCHMARKS | DOUBLE_BENCHMARKS
STANDARD_BENCHMARKS = {"gsm8k", "mbpp", "custom_math"}


def bench_alias(benchmark: str) -> str:
    """Canonical benchmark key used by native_model.py for output/prepared paths.

    `ruler_niah_double` is kept as a backwards-compatible alias of
    `ruler_niah_double_2`; every other name is filesystem-sanitized only.
    """
    text = str(benchmark)
    if text == "ruler_niah_double":
        text = "ruler_niah_double_2"
    try:
        return safe_name(text)
    except Exception:
        return _fallback_safe_name(text)

DEPTH_BY_POSITION = {
    "front": [15],
    "middle": [50],
    "back": [85],
    "end": [95],
}

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
    "needle_positions": "pos2",
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
    "needle_positions",
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
    "context_prefix_tokens",
    "diff_confidence_eos_eot_inf",
    "diff_logits_eos_inf",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
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
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
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


def value_label(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return ("%.6g" % value).replace(".", "p").replace("-", "m")
    if isinstance(value, (list, tuple)):
        return "-".join(value_label(v) for v in value)
    return _fallback_safe_name(str(value).replace(".", "p"))


def normalize_needle_pair(params: Dict[str, Any]) -> str:
    pair = params.get("needle_pair")
    if pair:
        pair_s = str(pair).strip().lower().replace("-", "_")
        if pair_s in DEPTH_BY_PAIR:
            return pair_s
        raise SystemExit(f"Unsupported needle_pair `{pair}`. Available: {', '.join(DEPTH_BY_PAIR)}")

    positions = params.get("needle_positions")
    if positions is None:
        # Backward-compatible fallback.
        p = params.get("needle_position")
        if isinstance(p, (list, tuple)) and len(p) == 2:
            positions = p
        else:
            return "front_back"

    pos_list = [str(x).strip().lower() for x in as_list(positions)]
    if len(pos_list) != 2:
        raise SystemExit(f"needle_positions must contain exactly two positions, got: {positions}")
    pair_s = f"{pos_list[0]}_{pos_list[1]}".replace("-", "_")
    if pair_s not in DEPTH_BY_PAIR:
        raise SystemExit(f"Unsupported needle_positions `{positions}` -> `{pair_s}`. Available: {', '.join(DEPTH_BY_PAIR)}")
    return pair_s


def condition_name(params: Dict[str, Any]) -> str:
    params = dict(params or {})
    if params.get("needle_pair") is None and params.get("needle_positions") is not None:
        try:
            params["needle_pair"] = normalize_needle_pair(params)
        except SystemExit:
            pass

    pieces: List[str] = []
    dynamic_skip = set(NAMELESS_KEYS)
    if params.get("w1_sampler") is not None or params.get("w1_steps") is not None:
        dynamic_skip.update({"gen_steps", "gen_blocksize", "token_selection_confidence_threshold"})

    for key in PREFERRED_KEYS:
        if key in params and key not in dynamic_skip and params.get(key) is not None:
            # Do not include both needle_positions and its normalized pair.
            if key == "needle_positions" and params.get("needle_pair") is not None:
                continue
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
    seed = int(params.get("seed", 42) or 42)
    gen_length = int(params.get("gen_length", 128) or 128)
    num_samples = int(params.get("num_samples", params.get("sample_limit", 20)) or 20)
    ctx = params.get("context_length")

    if benchmark in SINGLE_BENCHMARKS:
        pos = str(params.get("needle_position", "middle"))
        return _fallback_safe_name(
            f"ruler_niah_single_1_ctx{ctx}_pos{pos}_gen{gen_length}_samples{num_samples}_seed{seed}"
        )

    if benchmark in DOUBLE_BENCHMARKS:
        pair = normalize_needle_pair(params)
        return _fallback_safe_name(
            f"{benchmark}_ctx{ctx}_pair{pair}_gen{gen_length}_samples{num_samples}_seed{seed}"
        )

    return condition_name(params)


def prepared_file_for(root: Path, benchmark: str, params: Dict[str, Any]) -> Path:
    if benchmark in RULER_BENCHMARKS:
        return root / benchmark / f"{ruler_prepared_condition_id(benchmark, params)}.jsonl"
    return root / benchmark / f"{condition_name(params)}.jsonl"


def prepared_dir_for(root: Path, task: str, benchmark: str, params: Dict[str, Any]) -> Path:
    return root / benchmark / condition_name(params)


def iter_conditions(config: Dict[str, Any], selected: Set[str]) -> Iterable[Dict[str, Any]]:
    defaults = config.get("defaults", {}) or {}
    for experiment in collect_experiments(config, selected):
        task = experiment.get("task") or "runs"
        benchmarks = as_list(experiment.get("benchmark"))
        for benchmark in benchmarks:
            for idx, params in enumerate(expand_matrix(experiment), start=1):
                merged = deep_merge(defaults, params)
                if str(benchmark) in DOUBLE_BENCHMARKS and merged.get("needle_pair") is None:
                    merged["needle_pair"] = normalize_needle_pair(merged)
                yield {
                    "task": task,
                    "experiment": experiment.get("name"),
                    "benchmark": str(benchmark),
                    "params": merged,
                    "condition": condition_name(merged),
                    "condition_index": idx,
                    "output_path": experiment.get("output_path") or experiment.get("task_output_path"),
                }


def selected_sample_ids(params: Dict[str, Any], total: Optional[int] = None, default_limit: Optional[int] = None) -> List[int]:
    if params.get("sample_indices") is not None:
        return [int(x) for x in as_list(params.get("sample_indices"))]

    limit = params.get("sample_limit")
    if limit is None:
        limit = params.get("num_samples")
    if limit is None:
        limit = default_limit if default_limit is not None else total
    if limit is None:
        raise SystemExit("sample_limit or num_samples is required when dataset size is unknown.")

    if total is None:
        return list(range(int(limit)))
    return list(range(min(int(limit), int(total))))


# -----------------------------------------------------------------------------
# Standard task source checks
# -----------------------------------------------------------------------------

def load_hf_dataset_any(candidates: Sequence[Tuple[str, Optional[str], str]]):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Missing dependency `datasets`. Install it with: pip install datasets") from exc

    errors: List[str] = []
    for name, subset, split in candidates:
        try:
            if subset is None:
                return load_dataset(name, split=split)
            return load_dataset(name, subset, split=split)
        except Exception as exc:
            errors.append(f"{name}/{subset}/{split}: {repr(exc)}")
    raise RuntimeError("Could not load any dataset candidate:\n" + "\n".join(errors))


def prepare_gsm8k(condition: Dict[str, Any]) -> List[Dict[str, Any]]:
    ds = load_hf_dataset_any([("gsm8k", "main", "test"), ("openai/gsm8k", "main", "test")])
    ids = selected_sample_ids(condition["params"], total=len(ds))
    rows: List[Dict[str, Any]] = []
    for sid in ids:
        item = ds[int(sid)]
        question = str(item.get("question") or item.get("prompt") or "")
        answer = str(item.get("answer") or item.get("target") or "")
        rows.append(
            {
                "task": condition["task"],
                "experiment": condition["experiment"],
                "benchmark": "gsm8k",
                "sample_id": int(sid),
                "prompt": "Solve the following grade-school math problem. Give the final answer clearly.\n\n"
                f"Question: {question}\nAnswer:",
                "answer": answer,
                "metadata": {"question": question, "source": "gsm8k/test"},
            }
        )
    return rows


def prepare_mbpp(condition: Dict[str, Any]) -> List[Dict[str, Any]]:
    ds = load_hf_dataset_any(
        [
            ("google-research-datasets/mbpp", "sanitized", "test"),
            ("mbpp", "sanitized", "test"),
            ("google-research-datasets/mbpp", None, "test"),
            ("mbpp", None, "test"),
        ]
    )
    ids = selected_sample_ids(condition["params"], total=len(ds))
    rows: List[Dict[str, Any]] = []
    for sid in ids:
        item = ds[int(sid)]
        text = str(item.get("text") or item.get("prompt") or item.get("description") or "")
        tests = item.get("test_list") or item.get("tests") or []
        if isinstance(tests, str):
            try:
                tests = json.loads(tests)
            except Exception:
                tests = [tests]
        rows.append(
            {
                "task": condition["task"],
                "experiment": condition["experiment"],
                "benchmark": "mbpp",
                "sample_id": int(item.get("task_id", sid)),
                "dataset_index": int(sid),
                "prompt": "You are an expert Python programmer. Write Python code that solves the following problem.\n"
                "Return only the code, without explanations.\n\n"
                f"Problem: {text}\n",
                "answer": item.get("code") or "",
                "metadata": {
                    "text": text,
                    "test_list": tests,
                    "test_setup_code": item.get("test_setup_code") or "",
                    "entry_point": item.get("entry_point") or None,
                    "source": "mbpp/test",
                },
            }
        )
    return rows


def prepare_custom_math(condition: Dict[str, Any], data_root: Path) -> List[Dict[str, Any]]:
    custom_root = data_root / "custom_math"
    candidates: List[Path] = []
    if custom_root.is_dir():
        candidates.extend(sorted(custom_root.glob("*.jsonl")))
        candidates.extend(sorted(custom_root.glob("*.json")))
    if not candidates:
        raise RuntimeError(f"custom_math data not found under {custom_root}")

    raw: List[Dict[str, Any]] = []
    for path in candidates:
        if path.suffix == ".jsonl":
            raw.extend(read_jsonl(path))
        else:
            obj = json.loads(path.read_text(encoding="utf-8"))
            raw.extend(obj if isinstance(obj, list) else obj.get("data", []))

    ids = selected_sample_ids(condition["params"], total=len(raw))
    rows: List[Dict[str, Any]] = []
    for sid in ids:
        item = raw[int(sid)]
        prompt = item.get("prompt") or item.get("question") or item.get("input")
        answer = item.get("answer") or item.get("target") or item.get("reference")
        rows.append(
            {
                "task": condition["task"],
                "experiment": condition["experiment"],
                "benchmark": "custom_math",
                "sample_id": int(sid),
                "prompt": str(prompt),
                "answer": str(answer),
                "metadata": {
                    k: v
                    for k, v in item.items()
                    if k not in {"prompt", "question", "input", "answer", "target", "reference"}
                },
            }
        )
    return rows


def source_check(condition: Dict[str, Any], data_root: Path) -> Dict[str, Any]:
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
            ds = load_hf_dataset_any([("gsm8k", "main", "test"), ("openai/gsm8k", "main", "test")])
            ids = selected_sample_ids(params, total=len(ds))
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
            ds = load_hf_dataset_any(
                [
                    ("google-research-datasets/mbpp", "sanitized", "test"),
                    ("mbpp", "sanitized", "test"),
                    ("google-research-datasets/mbpp", None, "test"),
                    ("mbpp", None, "test"),
                ]
            )
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
            rows = prepare_custom_math(condition, data_root)
            return {
                **base,
                "status": "ok",
                "source_type": "local_original",
                "source": str(data_root / "custom_math"),
                "total_size": "",
                "selected_samples": len(rows),
                "selected_indices": json.dumps([r.get("sample_id") for r in rows], ensure_ascii=False),
                "prepared_file": "",
            }

        return {
            **base,
            "status": "failed",
            "source_type": "unknown",
            "source": benchmark,
            "error": f"Unknown benchmark `{benchmark}` in prepare_data.py",
            "prepared_file": "",
        }
    except Exception as exc:
        return {
            **base,
            "status": "failed",
            "source_type": "huggingface_cache" if benchmark in {"gsm8k", "mbpp"} else "local_original",
            "source": benchmark,
            "error": repr(exc),
            "prepared_file": "",
        }


# -----------------------------------------------------------------------------
# RULER-style synthetic NIAH data
# -----------------------------------------------------------------------------

def load_haystack_text(haystack_path: Path) -> str:
    if not haystack_path.exists():
        raise RuntimeError(f"Missing RULER haystack file: {haystack_path}")
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
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise RuntimeError(f"RULER haystack file is empty or unreadable: {haystack_path}")
    return text


def split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences or [text]


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
    # Insert from deeper to shallower so positions remain close to target depths.
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

    underfilled = bool(best_tokens < 0.85 * target and context_length >= 4096)
    return best_prompt, best_tokens, underfilled


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
        "Output JSON only, for example: [\"1234567\", \"7654321\"]."
    )


def prepare_ruler_single(condition: Dict[str, Any], haystack_path: Path) -> List[Dict[str, Any]]:
    params = condition["params"]
    context_length = int(params["context_length"])
    gen_length = int(params.get("gen_length", 128) or 128)
    num_samples = int(params.get("num_samples", params.get("sample_limit", 20)) or 20)
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
        rows.append(
            {
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
            }
        )
    return rows


def prepare_ruler_double(condition: Dict[str, Any], haystack_path: Path) -> List[Dict[str, Any]]:
    params = condition["params"]
    context_length = int(params["context_length"])
    gen_length = int(params.get("gen_length", 128) or 128)
    num_samples = int(params.get("num_samples", params.get("sample_limit", 20)) or 20)
    seed = int(params.get("seed", 42) or 42)
    pair = normalize_needle_pair(params)
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
        rows.append(
            {
                "task": condition["task"],
                "experiment": condition["experiment"],
                "benchmark": condition["benchmark"],
                "sample_id": i,
                "prompt": prompt,
                "answer": [value1, value2],
                "metadata": {
                    "context_length": context_length,
                    "needle_pair": pair,
                    "needle_positions": pair.split("_"),
                    "needle_depths": depths,
                    "needle_keys_in_order": [key1, key2],
                    "needle_values_in_order": [value1, value2],
                    "nominal_prompt_tokens": prompt_tokens,
                    "underfilled": underfilled,
                },
            }
        )
    return rows


def prepare_ruler_order2(condition: Dict[str, Any], haystack_path: Path) -> List[Dict[str, Any]]:
    """Compatibility alias expected by native_model.py for double-needle/order-2 NIAH."""
    return prepare_ruler_double(condition, haystack_path)


def prepare_ruler_double_2(condition: Dict[str, Any], haystack_path: Path) -> List[Dict[str, Any]]:
    """Compatibility alias for explicit double-needle benchmark naming."""
    return prepare_ruler_double(condition, haystack_path)


def prepare_rows(condition: Dict[str, Any], config: Dict[str, Any], data_root: Path) -> Optional[List[Dict[str, Any]]]:
    benchmark = str(condition["benchmark"])
    data_cfg = config.get("data", {}) or {}

    if benchmark in STANDARD_BENCHMARKS:
        return None

    if benchmark in RULER_BENCHMARKS:
        haystack = Path(data_cfg.get("ruler_haystack_path", "data/ruler/paul_graham_essay.jsonl"))
        if not haystack.is_absolute():
            haystack = ROOT / haystack
        if benchmark in SINGLE_BENCHMARKS:
            return prepare_ruler_single(condition, haystack)
        return prepare_ruler_double(condition, haystack)

    return None


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare/check input data from test_config.yaml.")
    parser.add_argument("--config", default="test_config.yaml")
    parser.add_argument("--only", nargs="*", default=[], help="Task or experiment names to prepare/check.")
    parser.add_argument("--output-root", default=None, help="Default: data.prepared_dir or data/prepared.")
    parser.add_argument("--force", action="store_true", help="Rewrite prepared RULER jsonl files.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned work but do not write files.")
    parser.add_argument(
        "--skip-source-check",
        action="store_true",
        help="Skip source checks for non-prepared datasets. By default all sources are checked.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_yaml(config_path)

    selected = set(args.only or [])
    data_cfg = config.get("data", {}) or {}
    base_prepared = Path(args.output_root or data_cfg.get("prepared_dir", "data/prepared"))
    if not base_prepared.is_absolute():
        base_prepared = ROOT / base_prepared
    data_root = ROOT / "data"

    conditions = list(iter_conditions(config, selected))
    if not conditions:
        raise SystemExit("No experiments matched. Use --only <task_or_experiment> or set run.tasks in config.")

    manifest_rows: List[Dict[str, Any]] = []
    log(f"Preparing/checking {len(conditions)} condition(s). Output root: {base_prepared}")

    for condition in conditions:
        benchmark = condition["benchmark"]
        params = condition["params"]
        input_path = prepared_file_for(base_prepared, benchmark, params)
        meta_path = input_path.with_suffix(".json")

        try:
            rows = prepare_rows(condition, config, data_root)
        except Exception as exc:
            row = {
                "task": condition["task"],
                "experiment": condition["experiment"],
                "benchmark": benchmark,
                "condition": condition["condition"],
                "params": params,
                "input_path": str(input_path),
                "status": "failed",
                "error": repr(exc),
            }
            manifest_rows.append(row)
            log(f"- {condition['experiment']} / {benchmark} / {condition['condition']} -> [FAILED] {exc}")
            continue

        if rows is None:
            log(
                f"- {condition['experiment']} / {benchmark} / {condition['condition']} "
                "-> original dataset source, checking availability"
            )
            if args.dry_run:
                continue
            if args.skip_source_check:
                check = {
                    "task": condition["task"],
                    "experiment": condition["experiment"],
                    "benchmark": benchmark,
                    "condition": condition["condition"],
                    "params": params,
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
                    log(f"  [FAILED] {benchmark}: {check.get('error')}")
            manifest_rows.append(check)
            continue

        exists = input_path.exists()
        status = "exists"
        log(
            f"- {condition['experiment']} / {benchmark} / {condition['condition']} "
            f"-> {input_path} ({'exists' if exists else 'new'}, samples={len(rows)})"
        )
        if args.dry_run:
            continue
        if exists and not args.force:
            try:
                rows = read_jsonl(input_path)
            except Exception:
                # Recreate corrupt file even without --force.
                write_jsonl(input_path, rows)
                status = "prepared"
        else:
            write_jsonl(input_path, rows)
            status = "prepared"

        meta = {
            "task": condition["task"],
            "experiment": condition["experiment"],
            "benchmark": benchmark,
            "condition": condition["condition"],
            "params": params,
            "input_path": str(input_path),
            "prepared_file": str(input_path),
            "num_samples": len(rows),
            "status": status,
        }
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
        manifest_rows.append(meta)

    if not args.dry_run:
        write_jsonl(base_prepared / "prepared_manifest.jsonl", manifest_rows)
        write_csv(base_prepared / "prepared_manifest.csv", manifest_rows)
        log(f"Wrote manifest: {base_prepared / 'prepared_manifest.jsonl'}")

    failed = [row for row in manifest_rows if row.get("status") == "failed"]
    if failed:
        log(f"Data check failed for {len(failed)} condition(s). See prepared_manifest.csv/jsonl.")
        return 4

    log("[DONE] data preparation/source check finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
