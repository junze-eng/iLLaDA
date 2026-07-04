#!/usr/bin/env python3
"""
Compact full runner for iLLaDA/OpenCompass experiments.

This keeps the original one-shot run_test.py workflow, but shortens both the
experiment folder and the per-condition folder while preserving the artifacts
needed by the ARness visualizer:

  outputs/<task>/<compact_exp>/<param_label>/
    oc_config.py
    run.json
    outputs.jsonl
    summary.jsonl
    trace.jsonl
    gpu.csv
    visual_command.txt
    <opencompass_timestamp>/...

Example:
  outputs/arness/mbpp_s6/mbpp_sample6_len512_block16_steps512_thr0p6/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

ROOT = Path(__file__).resolve().parent
OPENCOMPASS_DIR = ROOT / "opencompass"

BENCHMARKS = {
    "gsm8k": {
        "module": "opencompass.configs.datasets.gsm8k.gsm8k_gen",
        "var": "gsm8k_datasets",
    },
    "mbpp": {
        "module": "opencompass.configs.datasets.mbpp.mbpp_gen",
        "var": "mbpp_datasets",
    },
    "ruler_niah_single_1": {
        "module": "opencompass.configs.datasets.ruler.ruler_niah_single_1",
        "var": "ruler_niah_single_1_datasets",
    },
    "custom_math": {
        "module": "opencompass.configs.datasets.custom_math.custom_math_gen",
        "var": "custom_math_datasets",
    },
}

CUSTOM_BENCHMARKS: Set[str] = set()
DATASET_PARAM_KEYS = {"context_length", "needle_position", "num_samples", "depth_percents"}
EXPERIMENT_ONLY_KEYS = {"sample_limit", "sample_indices", "seed", "speed_schedule_label"} | DATASET_PARAM_KEYS
MODEL_TYPES = {
    "instruct": "LLaDAModel",
    "base": "LLaDABaseModel",
}

CONTROL_ARGS_WITH_VALUE = {"-m", "--mode", "-w", "--work-dir", "-r", "--reuse"}
CONTROL_ARGS_NO_VALUE = {"--dry-run"}

BENCH_ALIASES = {
    "gsm8k": "gsm8k",
    "mbpp": "mbpp",
    "humaneval": "he",
    "mmlu_pro": "mmlup",
    "mmlu": "mmlu",
    "arc_c": "arcc",
    "arc": "arc",
    "hellaswag": "hs",
    "piqa": "piqa",
    "math": "math",
    "ruler_niah_single_1": "niah1",
    "custom_math": "cmath",
}

PARAM_ALIASES = {
    "sample_indices": "s",
    "sample_limit": "n",
    "context_length": "ctx",
    "needle_position": "pos",
    "num_samples": "n",
    "gen_length": "l",
    "gen_blocksize": "b",
    "gen_steps": "st",
    "min_transfer_tokens": "mt",
    "token_selection_confidence_threshold": "thr",
    "token_selection_confidence_threshold_schedule": "thrs",
    "threshold_schedule_label": "thrs",
    "steps_per_block_schedule": "sch",
    "speed_schedule_name": "spd",
    "speed_schedule_label": "spd",
    "seed": "seed",
}

NAMELESS_KEYS = {
    "return_trace",
    "trace_token_snapshots",
    "trace_decode_snapshots",
    "benchmark",
    "profile_sample_indices",
}

PREFERRED_NAME_KEYS = [
    "sample_indices",
    "context_length",
    "needle_position",
    "num_samples",
    "sample_limit",
    "gen_length",
    "gen_blocksize",
    "gen_steps",
    "token_selection_confidence_threshold",
    "speed_schedule_name",
    "speed_schedule_label",
    "steps_per_block_schedule",
    "token_selection_confidence_threshold_schedule",
    "threshold_schedule_label",
    "seed",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        return load_simple_yaml(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"Config must be a YAML mapping: {path}")
    return data


def _strip_yaml_comment(line: str) -> str:
    in_single = False
    in_double = False
    for idx, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:idx]
    return line


def _split_inline_list(value: str) -> List[str]:
    items: List[str] = []
    current: List[str] = []
    in_single = False
    in_double = False
    for char in value:
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        if char == "," and not in_single and not in_double:
            items.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    tail = "".join(current).strip()
    if tail:
        items.append(tail)
    return items


def _parse_yaml_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_yaml_scalar(item) for item in _split_inline_list(inner)]
    lowered = value.lower()
    if lowered in ("null", "none", "~"):
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_simple_yaml(path: Path) -> Dict[str, Any]:
    raw_lines: List[tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            clean = _strip_yaml_comment(line).rstrip()
            if clean.strip():
                raw_lines.append((len(clean) - len(clean.lstrip(" ")), clean.strip()))

    def parse_block(index: int, indent: int):
        if index >= len(raw_lines) or raw_lines[index][0] < indent:
            return {}, index
        is_list = raw_lines[index][0] == indent and raw_lines[index][1].startswith("- ")
        if is_list:
            result = []
            while index < len(raw_lines) and raw_lines[index][0] == indent and raw_lines[index][1].startswith("- "):
                item_text = raw_lines[index][1][2:].strip()
                index += 1
                if item_text == "":
                    item, index = parse_block(index, indent + 2)
                    result.append(item)
                    continue
                if ":" in item_text:
                    key, value = item_text.split(":", 1)
                    item = {key.strip(): _parse_yaml_scalar(value)}
                    if index < len(raw_lines) and raw_lines[index][0] > indent:
                        child, index = parse_block(index, indent + 2)
                        if isinstance(child, dict):
                            item.update(child)
                    result.append(item)
                else:
                    result.append(_parse_yaml_scalar(item_text))
            return result, index

        result: Dict[str, Any] = {}
        while index < len(raw_lines) and raw_lines[index][0] == indent and not raw_lines[index][1].startswith("- "):
            text = raw_lines[index][1]
            if ":" not in text:
                raise SystemExit(f"Unsupported YAML line in {path}: {text}")
            key, value = text.split(":", 1)
            key = key.strip()
            value = value.strip()
            index += 1
            if value:
                result[key] = _parse_yaml_scalar(value)
            else:
                child, index = parse_block(index, indent + 2)
                result[key] = child
        return result, index

    data, index = parse_block(0, 0)
    if index != len(raw_lines) or not isinstance(data, dict):
        raise SystemExit(f"Config must be a YAML mapping: {path}")
    return data


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def python_literal(value: Any) -> str:
    return repr(value)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def expand_matrix(experiment: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    params = experiment.get("params", {}) or {}
    sweep = experiment.get("sweep", {}) or {}
    if not sweep:
        yield params
        return
    keys = list(sweep.keys())
    values = [as_list(sweep[key]) for key in keys]
    for combo in itertools.product(*values):
        item = deepcopy(params)
        for key, value in zip(keys, combo):
            item[key] = value
        yield item


def collect_experiments(config: Dict[str, Any], selected: Set[str]) -> List[Dict[str, Any]]:
    tasks_cfg = config.get("tasks")
    if not tasks_cfg:
        experiments = config.get("experiments", []) or []
        if selected:
            return [exp for exp in experiments if exp.get("name") in selected]
        return [exp for exp in experiments if exp.get("enabled", True) is not False]

    run_cfg = config.get("run", {}) or {}
    configured_tasks = set(as_list(run_cfg.get("tasks")))
    configured_experiments = set(as_list(run_cfg.get("experiments")))
    if not selected and not configured_tasks and not configured_experiments:
        raise SystemExit("No task selected. Set `run.tasks` in config or pass `--only`.")

    selected_all = "all" in selected or "all" in configured_tasks
    active: List[Dict[str, Any]] = []
    for task_name, task_def in tasks_cfg.items():
        task_experiments = task_def.get("experiments", []) or []
        task_selected = selected_all or task_name in selected or (not selected and task_name in configured_tasks)
        for experiment in task_experiments:
            exp_name = experiment.get("name")
            exp_selected = selected_all or exp_name in selected or (not selected and exp_name in configured_experiments)
            if not task_selected and not exp_selected:
                continue
            if experiment.get("enabled", True) is False and not selected and not configured_experiments:
                continue
            item = deepcopy(experiment)
            item.setdefault("task", task_name)
            if task_def.get("output_path") is not None:
                item.setdefault("_task_output_path", task_def.get("output_path"))
            active.append(item)
    return active


def safe_name(value: str) -> str:
    keep: List[str] = []
    for char in str(value).lower():
        keep.append(char if char.isalnum() else "_")
    return "_".join("".join(keep).split("_"))


def short_slug(text: str, max_len: int = 60) -> str:
    slug = safe_name(str(text)).strip("_") or "run"
    if len(slug) <= max_len:
        return slug
    digest = hashlib.sha1(slug.encode("utf-8")).hexdigest()[:8]
    head = slug[: max_len - len(digest) - 1].rstrip("_-")
    return f"{head}_{digest}"


def compact_task_name(task: Any) -> str:
    return short_slug(str(task or "runs"), max_len=24)


def _bench_alias(name: str) -> str:
    lowered = safe_name(name).lower()
    if lowered in BENCH_ALIASES:
        return BENCH_ALIASES[lowered]
    for old, new in BENCH_ALIASES.items():
        if old in lowered:
            return new
    return short_slug(lowered or name, max_len=16)


def _first_benchmark(value: Any) -> str:
    items = as_list(value)
    return str(items[0]) if items else ""


def compact_experiment_name(experiment: Dict[str, Any], benchmark: Optional[str] = None) -> str:
    raw = str(experiment.get("name") or "experiment")
    lower = safe_name(raw).lower()
    sample_match = re.search(
        r"(?P<bench>ruler_niah_single_1|gsm8k|mbpp|humaneval|mmlu_pro|mmlu|arc_c|arc|hellaswag|piqa|math|custom_math)[_-]*sample(?P<idx>\d+)",
        lower,
    )
    if sample_match:
        bench = _bench_alias(sample_match.group("bench"))
        return short_slug(f"{bench}_s{sample_match.group('idx')}", max_len=32)

    bench = _bench_alias(str(benchmark or _first_benchmark(experiment.get("benchmark")) or ""))
    compact = lower
    compact = re.sub(r"^task\d+[_-]*", "", compact)
    compact = re.sub(r"^(arness[_-]*)?trace[_-]*", "", compact)
    compact = re.sub(r"^(context[_-]*)?trace[_-]*", "", compact)
    compact = re.sub(r"[_-]+", "_", compact).strip("_")
    if bench and bench not in compact:
        compact = f"{bench}_{compact}" if compact else bench
    return short_slug(compact or raw, max_len=48)


def experiment_output_dir(base_output_dir: Path, experiment: Dict[str, Any]) -> Path:
    raw = experiment.get("output_path")
    if raw is None and experiment.get("_task_output_path") is not None:
        raw = Path(str(experiment.get("_task_output_path"))) / str(experiment.get("name"))
    if raw is None:
        task = safe_name(str(experiment.get("task") or "runs"))
        return base_output_dir / task
    path = Path(raw)
    path = path if path.is_absolute() else ROOT / path
    return path.parent


def compact_experiment_output_dir(base_output_dir: Path, experiment: Dict[str, Any], compact_exp_name: str) -> Path:
    return experiment_output_dir(base_output_dir, experiment) / compact_exp_name


def ruler_prepared_condition_id(params: Dict[str, Any]) -> str:
    seed = int(params.get("seed", 42) or 42)
    return safe_name(
        "ruler_niah_single_1_"
        f"ctx{params.get('context_length')}_"
        f"pos{params.get('needle_position')}_"
        f"gen{params.get('gen_length', 128)}_"
        f"samples{params.get('num_samples', 20)}_"
        f"seed{seed}"
    )


def ruler_prepared_path(data_cfg: Dict[str, Any], params: Dict[str, Any]) -> Path:
    prepared_dir = Path(data_cfg.get("prepared_dir", "data/prepared"))
    if not prepared_dir.is_absolute():
        prepared_dir = ROOT / prepared_dir
    return prepared_dir / "ruler_niah_single_1" / f"{ruler_prepared_condition_id(params)}.jsonl"


def schedule_label(value: Any) -> str:
    items = as_list(value)
    if not items:
        return ""
    return "schedule" + "_".join(str(item) for item in items)


def threshold_schedule_label(value: Any) -> str:
    items = as_list(value)
    if not items:
        return ""
    labels = []
    for item in items:
        labels.append("none" if item is None else str(item).replace(".", "p"))
    return "thrsch" + "_".join(labels)


def condition_run_name(exp_name: str, benchmark: str, params: Dict[str, Any], idx: int) -> str:
    parts = [exp_name, benchmark]
    if params.get("sample_indices") is not None:
        sample_label = "_".join(str(item) for item in as_list(params.get("sample_indices")))
        parts.append(f"sample{sample_label}")
    elif params.get("sample_limit") is not None:
        parts.append(f"n{params.get('sample_limit')}")
    if params.get("context_length") is not None:
        parts.append(f"ctx{params.get('context_length')}")
    if params.get("needle_position") is not None:
        parts.append(f"pos{params.get('needle_position')}")
    for key, label in (("gen_length", "len"), ("gen_blocksize", "block"), ("gen_steps", "steps")):
        if params.get(key) is not None:
            parts.append(f"{label}{params.get(key)}")
    if params.get("speed_schedule_name") is not None:
        parts.append(f"speed{params.get('speed_schedule_name')}")
    if params.get("speed_schedule_label") is not None:
        parts.append(f"speed{params.get('speed_schedule_label')}")
    if params.get("steps_per_block_schedule") is not None and params.get("speed_schedule_label") is None:
        parts.append(schedule_label(params.get("steps_per_block_schedule")))
    if params.get("threshold_schedule_label") is not None:
        parts.append(f"thr{params.get('threshold_schedule_label')}")
    elif params.get("token_selection_confidence_threshold_schedule") is not None:
        parts.append(threshold_schedule_label(params.get("token_selection_confidence_threshold_schedule")))
    if "token_selection_confidence_threshold" in params:
        threshold = params.get("token_selection_confidence_threshold")
        threshold_text = "none" if threshold is None else str(threshold).replace(".", "p")
        if params.get("threshold_schedule_label") is None and params.get("token_selection_confidence_threshold_schedule") is None:
            parts.append(f"thr{threshold_text}")
    if len(parts) <= 2:
        parts.append(str(idx))
    return safe_name("_".join(str(part) for part in parts))


def build_model_cfg(global_model: Dict[str, Any], params: Dict[str, Any], benchmark: str, run_name: str) -> Dict[str, Any]:
    model_cfg = deepcopy(global_model)
    model_type = model_cfg.pop("type", "instruct")
    if model_type not in MODEL_TYPES:
        raise SystemExit(f"Unsupported model.type `{model_type}`. Choose one of: {', '.join(MODEL_TYPES)}")
    model_cfg.setdefault("abbr", f"{Path(str(model_cfg.get('path', 'model'))).name}-{benchmark}")
    model_cfg["abbr"] = safe_name(f"{model_cfg['abbr']}_{run_name}")
    model_cfg["type"] = MODEL_TYPES[model_type]
    model_cfg.update({key: value for key, value in params.items() if key not in EXPERIMENT_ONLY_KEYS})
    model_cfg["benchmark"] = benchmark
    if "sample_indices" in params and params.get("sample_indices") is not None:
        model_cfg["profile_sample_indices"] = [int(item) for item in as_list(params.get("sample_indices"))]
    if "context_length" in params:
        model_cfg["context_length"] = params.get("context_length")
    if "needle_position" in params:
        model_cfg["needle_position"] = params.get("needle_position")
    model_cfg.setdefault("decoding_config_name", run_name)
    return model_cfg


def render_opencompass_config(
    benchmark: str,
    model_cfg: Dict[str, Any],
    runner_cfg: Dict[str, Any],
    sample_limit: Any = None,
    sample_indices: Any = None,
    experiment_params: Optional[Dict[str, Any]] = None,
    data_cfg: Optional[Dict[str, Any]] = None,
) -> str:
    if benchmark not in BENCHMARKS:
        raise SystemExit(f"Unknown benchmark `{benchmark}`. Available: {', '.join(sorted(BENCHMARKS))}")
    bench = BENCHMARKS[benchmark]
    model_type = model_cfg.pop("type")
    lines: List[str] = []
    lines.append("from mmengine.config import read_base")
    lines.append("")
    lines.append("with read_base():")
    lines.append(f"    from {bench['module']} import {bench['var']}")
    if "summary_module" in bench:
        lines.append(f"    from {bench['summary_module']} import {bench['summary_var']}")
    lines.append("")
    lines.append(f"from opencompass.models import {model_type}")
    lines.append("from opencompass.partitioners import NumWorkerPartitioner")
    lines.append("from opencompass.runners import LocalRunner")
    lines.append("from opencompass.tasks import OpenICLInferTask")
    lines.append("")
    lines.append(f"datasets = {bench['var']}")

    if sample_indices is not None:
        indices = [int(item) for item in as_list(sample_indices)]
        if not indices:
            raise SystemExit("sample_indices cannot be empty.")
        sorted_indices = sorted(indices)
        expected = list(range(sorted_indices[0], sorted_indices[-1] + 1))
        if sorted_indices == expected:
            test_range = f"[{sorted_indices[0]}:{sorted_indices[-1] + 1}]"
        else:
            test_range = "indices:" + ",".join(str(index) for index in indices)
        lines.append(f"_sample_test_range = {python_literal(test_range)}")
        lines.append("for _dataset in datasets:")
        lines.append("    _dataset.setdefault('reader_cfg', {})['test_range'] = _sample_test_range")
    elif sample_limit is not None:
        test_range = f"[:{sample_limit}]" if isinstance(sample_limit, int) else sample_limit
        lines.append(f"_sample_test_range = {python_literal(test_range)}")
        lines.append("for _dataset in datasets:")
        lines.append("    _dataset.setdefault('reader_cfg', {})['test_range'] = _sample_test_range")

    if "summary_var" in bench:
        lines.append(f"summarizer = dict(summary_groups={bench['summary_var']})")

    if benchmark == "custom_math":
        custom_math_path = ROOT / "data" / "custom_math"
        lines.append("for _dataset in datasets:")
        lines.append(f"    _dataset['path'] = {python_literal(str(custom_math_path))}")

    if benchmark == "ruler_niah_single_1":
        experiment_params = experiment_params or {}
        context_length = experiment_params.get("context_length")
        num_samples = experiment_params.get("num_samples")
        needle_position = experiment_params.get("needle_position")
        tokens_to_generate = model_cfg.get("gen_length")
        prepared_path = ruler_prepared_path(data_cfg or {}, experiment_params)
        depth_by_position = {"front": [0], "middle": [50], "back": [100], "end": [100]}
        lines.append("for _dataset in datasets:")
        lines.append(f"    _dataset['prepared_file_path'] = {python_literal(str(prepared_path))}")
        if context_length is not None:
            lines.append(f"    _dataset['max_seq_length'] = {python_literal(int(context_length))}")
        if tokens_to_generate is not None:
            lines.append(f"    _dataset['tokens_to_generate'] = {python_literal(int(tokens_to_generate))}")
            lines.append("    _dataset.setdefault('infer_cfg', {}).setdefault('inferencer', {})['max_out_len'] = _dataset['tokens_to_generate']")
        if num_samples is not None:
            lines.append(f"    _dataset['num_samples'] = {python_literal(int(num_samples))}")
        if needle_position is not None:
            depths = depth_by_position.get(str(needle_position))
            if depths is None:
                raise SystemExit(f"Unsupported needle_position `{needle_position}` for ruler_niah_single_1.")
            lines.append(f"    _dataset['depth_percents'] = {python_literal(depths)}")

    model_entries = ",\n        ".join(f"{key}={python_literal(value)}" for key, value in model_cfg.items())
    lines.append("models = [")
    lines.append("    dict(")
    lines.append(f"        type={model_type},")
    if model_entries:
        lines.append(f"        {model_entries},")
    lines.append("    )")
    lines.append("]")
    lines.append("")

    partitioner = runner_cfg.get("partitioner", {}) or {}
    runner = runner_cfg.get("runner", {}) or {}
    num_worker = int(partitioner.get("num_worker", 1))
    num_split = partitioner.get("num_split", None)
    min_task_size = int(partitioner.get("min_task_size", 16))
    max_num_workers = int(runner.get("max_num_workers", max(1, num_worker)))
    retry = int(runner.get("retry", 1))
    lines.append("infer = dict(")
    lines.append("    partitioner=dict(")
    lines.append("        type=NumWorkerPartitioner,")
    lines.append(f"        num_worker={num_worker},")
    lines.append(f"        num_split={python_literal(num_split)},")
    lines.append(f"        min_task_size={min_task_size},")
    lines.append("    ),")
    lines.append("    runner=dict(")
    lines.append("        type=LocalRunner,")
    lines.append(f"        max_num_workers={max_num_workers},")
    lines.append("        task=dict(type=OpenICLInferTask),")
    lines.append(f"        retry={retry},")
    lines.append("    ),")
    lines.append(")")
    lines.append("")
    return "\n".join(lines)


def is_custom_benchmark(benchmark: str) -> bool:
    return benchmark in CUSTOM_BENCHMARKS


def canonical_value(value: Any) -> Any:
    if isinstance(value, list):
        return [canonical_value(v) for v in value]
    if isinstance(value, tuple):
        return [canonical_value(v) for v in value]
    if isinstance(value, dict):
        return {k: canonical_value(value[k]) for k in sorted(value)}
    return value


def varying_param_keys(param_list: List[Dict[str, Any]]) -> List[str]:
    all_keys: List[str] = []
    for params in param_list:
        for key in params:
            if key not in all_keys:
                all_keys.append(key)
    varying: List[str] = []
    for key in all_keys:
        if key in NAMELESS_KEYS:
            continue
        values = {json.dumps(canonical_value(params.get(key)), sort_keys=True, ensure_ascii=False) for params in param_list}
        if len(values) > 1:
            varying.append(key)
    return varying


def value_to_label(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return ("%.6g" % value).replace(".", "p").replace("-", "m")
    if isinstance(value, (list, tuple)):
        return "-".join(value_to_label(v) for v in value)
    if isinstance(value, dict):
        return "-".join(f"{short_slug(str(k), 8)}{value_to_label(v)}" for k, v in sorted(value.items()))
    return short_slug(str(value).replace(".", "p"), max_len=32)


def param_piece(key: str, value: Any) -> str:
    return f"{PARAM_ALIASES.get(key, short_slug(key, max_len=10))}{value_to_label(value)}"


def sample_already_in_exp_name(exp_dir: str) -> bool:
    return re.search(r"(^|_)s\d+($|_)", exp_dir) is not None


def compact_run_label(params: Dict[str, Any], include_keys: List[str], compact_exp_dir: str, idx: int) -> str:
    keys: List[str] = []
    for key in PREFERRED_NAME_KEYS + include_keys:
        if key in params and key not in NAMELESS_KEYS and key not in keys:
            keys.append(key)
    pieces: List[str] = []
    for key in keys:
        if key == "sample_indices" and sample_already_in_exp_name(compact_exp_dir):
            continue
        pieces.append(param_piece(key, params.get(key)))
    if not pieces:
        pieces.append(f"r{idx:03d}")
    return short_slug("_".join(pieces), max_len=96)


def arness_visual_condition_label(benchmark: Any, params: Dict[str, Any]) -> Optional[str]:
    """Return the condition directory name expected by visual_arness_trace.py.

    The ARness visualizer keys conditions by benchmark/sample/length/block/steps/
    threshold, so keep that name stable across the one-shot and split runners.
    """
    bench = safe_name(str(benchmark)).lower()
    sample_indices = as_list(params.get("sample_indices"))
    if bench not in {"gsm8k", "mbpp"} or len(sample_indices) != 1:
        return None
    required = ["gen_length", "gen_blocksize", "gen_steps"]
    if any(params.get(key) is None for key in required):
        return None
    threshold = params.get("token_selection_confidence_threshold")
    return safe_name(
        f"{bench}_sample{int(sample_indices[0])}_"
        f"len{int(params['gen_length'])}_"
        f"block{int(params['gen_blocksize'])}_"
        f"steps{int(params['gen_steps'])}_"
        f"thr{value_to_label(threshold)}"
    )


def unique_label(label: str, used: Set[str], idx: int) -> str:
    if label not in used:
        used.add(label)
        return label
    candidate = short_slug(f"{label}_r{idx:03d}", max_len=100)
    salt = 1
    while candidate in used:
        salt += 1
        candidate = short_slug(f"{label}_r{idx:03d}_{salt}", max_len=100)
    used.add(candidate)
    return candidate


def resolve_under_root(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def rel_to(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def rel_to_root(path: Path) -> str:
    return rel_to(path, ROOT)


def strip_opencompass_control_args(args: Sequence[Any] | None) -> List[str]:
    cleaned: List[str] = []
    items = [str(x) for x in (args or [])]
    i = 0
    while i < len(items):
        item = items[i]
        if item in CONTROL_ARGS_WITH_VALUE:
            i += 2
            continue
        if any(item.startswith(flag + "=") for flag in CONTROL_ARGS_WITH_VALUE):
            i += 1
            continue
        if item in CONTROL_ARGS_NO_VALUE:
            i += 1
            continue
        cleaned.append(item)
        i += 1
    return cleaned


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    fieldnames = list(row.keys())
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_command(command: Sequence[str], cwd: Path, env: Dict[str, str]) -> int:
    print("$ " + " ".join(command), flush=True)
    proc = subprocess.Popen(
        list(command),
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    return proc.wait()


def build_env() -> Dict[str, str]:
    env = os.environ.copy()
    pythonpath = [str(ROOT), str(OPENCOMPASS_DIR)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return env


def current_gpu_snapshot() -> Dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return {"available": False}
    if result.returncode != 0:
        return {"available": False, "error": result.stderr.strip()}
    rows: List[Dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 6:
            rows.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "memory_used_mb": parts[2],
                    "memory_total_mb": parts[3],
                    "utilization_gpu_percent": parts[4],
                    "power_draw_w": parts[5],
                }
            )
    return {"available": True, "gpus": rows}


class GpuTelemetry:
    def __init__(self, output_path: Path, interval_seconds: float = 1.0):
        self.output_path = output_path
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 2))

    def _run(self) -> None:
        fields = [
            "timestamp_utc",
            "gpu_index",
            "gpu_name",
            "utilization_gpu_percent",
            "memory_used_mb",
            "memory_total_mb",
            "power_draw_w",
            "temperature_gpu_c",
        ]
        query = [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        with self.output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            while not self._stop.is_set():
                timestamp = utc_now()
                try:
                    result = subprocess.run(query, capture_output=True, text=True, check=False, timeout=10)
                    if result.returncode == 0:
                        for line in result.stdout.splitlines():
                            parts = [part.strip() for part in line.split(",")]
                            if len(parts) == 7:
                                writer.writerow(
                                    {
                                        "timestamp_utc": timestamp,
                                        "gpu_index": parts[0],
                                        "gpu_name": parts[1],
                                        "utilization_gpu_percent": parts[2],
                                        "memory_used_mb": parts[3],
                                        "memory_total_mb": parts[4],
                                        "power_draw_w": parts[5],
                                        "temperature_gpu_c": parts[6],
                                    }
                                )
                    else:
                        writer.writerow({"timestamp_utc": timestamp})
                    f.flush()
                except (OSError, subprocess.TimeoutExpired):
                    writer.writerow({"timestamp_utc": timestamp})
                    f.flush()
                self._stop.wait(self.interval_seconds)


def opencompass_timestamp_dirs(work_dir: Path) -> Set[str]:
    if not work_dir.exists():
        return set()
    return {child.name for child in work_dir.iterdir() if child.is_dir() and (child / "configs").exists()}


def latest_opencompass_timestamp(work_dir: Path) -> Optional[str]:
    names = sorted(opencompass_timestamp_dirs(work_dir))
    return names[-1] if names else None


def read_jsonl_count(path: Path) -> int:
    if not path.exists() or path.is_dir():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def copy_aliases_for_visualizer(outputs_jsonl: Path, summary_jsonl: Path) -> None:
    if outputs_jsonl.exists() and not summary_jsonl.exists():
        shutil.copy2(outputs_jsonl, summary_jsonl)
    if summary_jsonl.exists() and not outputs_jsonl.exists():
        shutil.copy2(summary_jsonl, outputs_jsonl)


def write_visual_command(path: Path, run_dir: Path, params: Dict[str, Any]) -> None:
    sample_idx = None
    indices = params.get("sample_indices")
    if isinstance(indices, list) and len(indices) == 1:
        sample_idx = indices[0]
    elif isinstance(indices, int):
        sample_idx = indices
    command = f'python visual_arness_trace.py "{rel_to_root(run_dir)}"'
    if sample_idx is not None:
        command += f" --sample-idx {sample_idx}"
    path.write_text(command + "\n", encoding="utf-8")


def mode_commands(mode: str) -> List[Optional[str]]:
    if mode == "all":
        return [None]
    if mode == "both":
        return ["infer", "eval"]
    return [mode]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run iLLaDA/OpenCompass experiments with compact output paths.")
    parser.add_argument("--config", default="test_config.yaml")
    parser.add_argument("--only", nargs="*", default=None, help="Task section or experiment name.")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--mode", choices=["all", "infer", "eval", "viz", "both"], default=None)
    parser.add_argument("--legacy-names", action="store_true", help="Use old long output folders.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = resolve_under_root(args.config)
    config = load_yaml(config_path)
    execution_cfg = config.get("execution", {}) or {}
    data_cfg = config.get("data", {}) or {}
    global_model = config.get("model", {}) or {}
    runner_cfg = config.get("runner", {}) or {}
    default_params = config.get("defaults", {}) or {}

    selected = set(args.only or [])
    experiments = collect_experiments(config, selected)
    if not experiments:
        raise SystemExit("No experiments matched.")

    output_root = resolve_under_root(args.output_root)
    dry_run = args.dry_run or bool(execution_cfg.get("dry_run", False))
    run_mode = args.mode or str(execution_cfg.get("mode", "all"))
    extra_opencompass_args = strip_opencompass_control_args(execution_cfg.get("opencompass_args", []) or [])

    telemetry_cfg = execution_cfg.get("gpu_telemetry", {}) or {}
    telemetry_enabled = bool(telemetry_cfg.get("enabled", True))
    telemetry_interval = float(telemetry_cfg.get("interval_seconds", 1.0))

    env = build_env()
    planned = 0

    for experiment in experiments:
        exp_name = experiment.get("name")
        if not exp_name:
            raise SystemExit("Every experiment needs a `name`.")
        benchmark_list = as_list(experiment.get("benchmark"))
        first_benchmark = str(benchmark_list[0]) if benchmark_list else None
        compact_exp_dir = compact_experiment_name(experiment, first_benchmark)

        if args.legacy_names:
            exp_root = resolve_under_root(experiment.get("output_path") or (output_root / safe_name(str(experiment.get("task") or "runs")) / safe_name(str(exp_name))))
        else:
            exp_root = compact_experiment_output_dir(output_root, experiment, compact_exp_dir)
        exp_root.mkdir(parents=True, exist_ok=True)

        run_manifest = exp_root / ("run_manifest.jsonl" if args.legacy_names else "runs.jsonl")
        run_csv = exp_root / ("run_manifest.csv" if args.legacy_names else "runs.csv")

        expanded_params = [deep_merge(default_params, p) for p in expand_matrix(experiment)]
        include_keys = varying_param_keys(expanded_params)
        used_labels: Set[str] = set()

        for benchmark in benchmark_list:
            for idx, params in enumerate(expand_matrix(experiment), start=1):
                planned += 1
                merged_params = deep_merge(default_params, params)
                full_run_name = condition_run_name(str(exp_name), str(benchmark), merged_params, idx)
                if args.legacy_names:
                    run_label = full_run_name
                    run_dir = exp_root / full_run_name
                    config_path_out = exp_root / "generated_configs" / f"{full_run_name}.py"
                    run_json = run_dir / "config.json"
                    outputs_jsonl = run_dir / "summary.jsonl"
                    summary_jsonl = run_dir / "summary.jsonl"
                    trace_jsonl = run_dir / "trace.jsonl"
                    gpu_csv = run_dir / "gpu_telemetry.csv"
                    oc_run_name = full_run_name
                else:
                    visual_label = arness_visual_condition_label(benchmark, merged_params)
                    run_label = unique_label(
                        visual_label or compact_run_label(merged_params, include_keys, compact_exp_dir, idx),
                        used_labels,
                        idx,
                    )
                    run_dir = exp_root / run_label
                    config_path_out = run_dir / "oc_config.py"
                    run_json = run_dir / "run.json"
                    outputs_jsonl = run_dir / "outputs.jsonl"
                    summary_jsonl = run_dir / "summary.jsonl"
                    trace_jsonl = run_dir / "trace.jsonl"
                    gpu_csv = run_dir / "gpu.csv"
                    oc_run_name = run_label

                run_dir.mkdir(parents=True, exist_ok=True)
                config_path_out.parent.mkdir(parents=True, exist_ok=True)

                model_cfg = build_model_cfg(deepcopy(global_model), merged_params, str(benchmark), oc_run_name)
                model_cfg["task_id"] = experiment.get("task")
                model_cfg.setdefault("per_sample_output", str(outputs_jsonl))
                model_cfg.setdefault("metrics_output", str(summary_jsonl))
                if model_cfg.get("return_trace") or model_cfg.get("trace_token_snapshots") or model_cfg.get("trace_decode_snapshots"):
                    model_cfg.setdefault("step_trace_output", str(trace_jsonl))
                    model_cfg["arness_trace_output"] = str(run_dir / "sample_traces")

                config_text = render_opencompass_config(
                    benchmark=str(benchmark),
                    model_cfg=deepcopy(model_cfg),
                    runner_cfg=runner_cfg,
                    sample_limit=merged_params.get("sample_limit"),
                    sample_indices=merged_params.get("sample_indices"),
                    experiment_params=merged_params,
                    data_cfg=data_cfg,
                )
                config_path_out.write_text(config_text, encoding="utf-8")

                run_config = {
                    "mode": run_mode,
                    "created_at": utc_now(),
                    "source_config": str(config_path),
                    "source_config_rel": rel_to_root(config_path),
                    "task": experiment.get("task"),
                    "experiment": exp_name,
                    "compact_experiment": compact_exp_dir,
                    "benchmark": benchmark,
                    "run_label": run_label,
                    "run_name": full_run_name,
                    "opencompass_run_name": oc_run_name,
                    "params": merged_params,
                    "model": model_cfg,
                    "runner": runner_cfg,
                    "layout": "legacy" if args.legacy_names else "compact_param",
                    "output_files": {
                        "outputs_jsonl": str(outputs_jsonl),
                        "summary_jsonl": str(summary_jsonl),
                        "trace_jsonl": str(trace_jsonl),
                        "gpu_csv": str(gpu_csv),
                    },
                }
                write_json(run_json, run_config)
                write_visual_command(run_dir / "visual_command.txt", run_dir, merged_params)

                before_timestamps = opencompass_timestamp_dirs(run_dir)
                returncode: Optional[int] = None
                elapsed_total = 0.0
                commands: List[List[str]] = []

                for mode_item in mode_commands(run_mode):
                    command = [sys.executable, str(OPENCOMPASS_DIR / "run.py"), str(config_path_out), "-w", str(run_dir)]
                    if mode_item is not None:
                        command.extend(["-m", mode_item])
                    command.extend(extra_opencompass_args)
                    commands.append(command)

                record: Dict[str, Any] = {
                    "created_at": utc_now(),
                    "mode": run_mode,
                    "dry_run": dry_run,
                    "task": experiment.get("task"),
                    "experiment": exp_name,
                    "compact_experiment": compact_exp_dir,
                    "benchmark": benchmark,
                    "run_label": run_label,
                    "run_name": full_run_name,
                    "config": str(config_path_out),
                    "config_rel": rel_to(config_path_out, exp_root),
                    "work_dir": str(run_dir),
                    "work_dir_rel": rel_to(run_dir, exp_root),
                    "outputs_jsonl": str(outputs_jsonl),
                    "summary_jsonl": str(summary_jsonl),
                    "trace_jsonl": str(trace_jsonl),
                    "gpu_csv": str(gpu_csv),
                    "visual_command": (run_dir / "visual_command.txt").read_text(encoding="utf-8").strip(),
                    "params": merged_params,
                    "commands": commands,
                    "gpu_before": current_gpu_snapshot(),
                }

                print(f"\n[run:{run_label}] run_name: {full_run_name}")
                print(f"[run:{run_label}] dir: {run_dir}")
                print(f"[run:{run_label}] visual: {record['visual_command']}")

                if dry_run:
                    for command in commands:
                        print("[dry-run] " + " ".join(command))
                    record.update({"returncode": None, "elapsed_seconds": None})
                else:
                    telemetry: Optional[GpuTelemetry] = None
                    if telemetry_enabled:
                        telemetry = GpuTelemetry(gpu_csv, interval_seconds=telemetry_interval)
                        telemetry.start()
                    started = time.perf_counter()
                    try:
                        for command in commands:
                            returncode = run_command(command, OPENCOMPASS_DIR, env)
                            if returncode != 0:
                                break
                    finally:
                        if telemetry is not None:
                            telemetry.stop()
                    elapsed_total = round(time.perf_counter() - started, 3)
                    copy_aliases_for_visualizer(outputs_jsonl, summary_jsonl)
                    after_timestamps = opencompass_timestamp_dirs(run_dir)
                    new_timestamps = sorted(after_timestamps - before_timestamps)
                    reuse_timestamp = new_timestamps[-1] if new_timestamps else latest_opencompass_timestamp(run_dir)
                    record.update(
                        {
                            "returncode": returncode,
                            "elapsed_seconds": elapsed_total,
                            "opencompass_reuse_timestamp": reuse_timestamp,
                            "gpu_after": current_gpu_snapshot(),
                            "num_output_records": read_jsonl_count(outputs_jsonl),
                            "num_summary_records": read_jsonl_count(summary_jsonl),
                            "trace_exists": trace_jsonl.exists(),
                        }
                    )
                    if returncode != 0 and execution_cfg.get("stop_on_error", True):
                        append_jsonl(run_manifest, record)
                        return int(returncode or 1)

                append_jsonl(run_manifest, record)
                append_csv(
                    run_csv,
                    {
                        "created_at": record["created_at"],
                        "mode": run_mode,
                        "run_label": run_label,
                        "run_name": full_run_name,
                        "benchmark": benchmark,
                        "returncode": "" if record.get("returncode") is None else record.get("returncode"),
                        "elapsed_seconds": "" if record.get("elapsed_seconds") is None else record.get("elapsed_seconds"),
                        "work_dir": str(run_dir),
                        "visual_command": record["visual_command"],
                    },
                )

    print(f"\nPlanned {planned} run(s).")
    print(f"Outputs root: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
