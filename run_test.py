import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import threading
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


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

CUSTOM_BENCHMARKS = set()
DATASET_PARAM_KEYS = {"context_length", "needle_position", "num_samples", "depth_percents"}
EXPERIMENT_ONLY_KEYS = {"sample_limit", "sample_indices", "seed", "speed_schedule_label"} | DATASET_PARAM_KEYS

MODEL_TYPES = {
    "instruct": "LLaDAModel",
    "base": "LLaDABaseModel",
}


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
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
    items = []
    current = []
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
    raw_lines = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            clean = _strip_yaml_comment(line).rstrip()
            if clean.strip():
                raw_lines.append((len(clean) - len(clean.lstrip(" ")), clean.strip()))

    def parse_block(index: int, indent: int):
        if index >= len(raw_lines):
            return {}, index
        if raw_lines[index][0] < indent:
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

        result = {}
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


def collect_experiments(config: Dict[str, Any], selected: set) -> List[Dict[str, Any]]:
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
        raise SystemExit("No task selected. Set `run.tasks` in config or pass `--only <task_or_experiment>`.")

    selected_all = "all" in selected or "all" in configured_tasks
    active = []
    for task_name, task_def in tasks_cfg.items():
        task_experiments = task_def.get("experiments", []) or []
        task_selected = (
            selected_all
            or task_name in selected
            or (not selected and task_name in configured_tasks)
        )
        for experiment in task_experiments:
            exp_name = experiment.get("name")
            exp_selected = (
                selected_all
                or exp_name in selected
                or (not selected and exp_name in configured_experiments)
            )
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
    keep = []
    for char in value.lower():
        keep.append(char if char.isalnum() else "_")
    return "_".join("".join(keep).split("_"))


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
        if item is None:
            labels.append("none")
        else:
            labels.append(str(item).replace(".", "p"))
    return "thrsch" + "_".join(labels)


def condition_run_name(exp_name: str, benchmark: str, params: Dict[str, Any], idx: int) -> str:
    """Name a run by the actual config values, with idx only as a fallback."""
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
    for key, label in (
        ("gen_length", "len"),
        ("gen_blocksize", "block"),
        ("gen_steps", "steps"),
    ):
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
        threshold_label = "none" if threshold is None else str(threshold).replace(".", "p")
        if params.get("threshold_schedule_label") is None and params.get("token_selection_confidence_threshold_schedule") is None:
            parts.append(f"thr{threshold_label}")
    if len(parts) <= 2:
        parts.append(str(idx))
    return safe_name("_".join(str(part) for part in parts))


def experiment_output_dir(base_output_dir: Path, experiment: Dict[str, Any]) -> Path:
    """Resolve root output dir for an experiment from exp/task/base config."""
    raw = experiment.get("output_path")
    if raw is None and experiment.get("_task_output_path") is not None:
        raw = Path(str(experiment.get("_task_output_path"))) / str(experiment.get("name"))
    if raw is None:
        task = safe_name(str(experiment.get("task") or "runs"))
        return base_output_dir / task
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def ruler_prepared_path(data_cfg: Dict[str, Any], params: Dict[str, Any]) -> Path:
    prepared_dir = Path(data_cfg.get("prepared_dir", "data/prepared"))
    if not prepared_dir.is_absolute():
        prepared_dir = ROOT / prepared_dir
    return prepared_dir / "ruler_niah_single_1" / f"{ruler_prepared_condition_id(params)}.jsonl"


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
    imports = ["from mmengine.config import read_base", ""]
    imports.append("with read_base():")
    imports.append(f"    from {bench['module']} import {bench['var']}")
    if "summary_module" in bench:
        imports.append(f"    from {bench['summary_module']} import {bench['summary_var']}")
    imports.append("")
    imports.append(f"from opencompass.models import {model_type}")
    imports.append("from opencompass.partitioners import NumWorkerPartitioner")
    imports.append("from opencompass.runners import LocalRunner")
    imports.append("from opencompass.tasks import OpenICLInferTask")
    imports.append("")
    imports.append(f"datasets = {bench['var']}")
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
        imports.append(f"_sample_test_range = {python_literal(test_range)}")
        imports.append("for _dataset in datasets:")
        imports.append("    _dataset.setdefault('reader_cfg', {})['test_range'] = _sample_test_range")
    elif sample_limit is not None:
        if isinstance(sample_limit, int):
            test_range = f"[:{sample_limit}]"
        else:
            test_range = sample_limit
        imports.append(f"_sample_test_range = {python_literal(test_range)}")
        imports.append("for _dataset in datasets:")
        imports.append("    _dataset.setdefault('reader_cfg', {})['test_range'] = _sample_test_range")
    if "summary_var" in bench:
        imports.append(f"summarizer = dict(summary_groups={bench['summary_var']})")
    if benchmark == "custom_math":
        custom_math_path = ROOT / "data" / "custom_math"
        imports.append("for _dataset in datasets:")
        imports.append(f"    _dataset['path'] = {python_literal(str(custom_math_path))}")
    if benchmark == "ruler_niah_single_1":
        experiment_params = experiment_params or {}
        context_length = experiment_params.get("context_length")
        num_samples = experiment_params.get("num_samples")
        needle_position = experiment_params.get("needle_position")
        tokens_to_generate = model_cfg.get("gen_length")
        prepared_path = ruler_prepared_path(data_cfg or {}, experiment_params)
        depth_by_position = {"front": [0], "middle": [50], "back": [100], "end": [100]}
        imports.append("for _dataset in datasets:")
        imports.append(f"    _dataset['prepared_file_path'] = {python_literal(str(prepared_path))}")
        if context_length is not None:
            imports.append(f"    _dataset['max_seq_length'] = {python_literal(int(context_length))}")
        if tokens_to_generate is not None:
            imports.append(f"    _dataset['tokens_to_generate'] = {python_literal(int(tokens_to_generate))}")
            imports.append("    _dataset.setdefault('infer_cfg', {}).setdefault('inferencer', {})['max_out_len'] = _dataset['tokens_to_generate']")
        if num_samples is not None:
            imports.append(f"    _dataset['num_samples'] = {python_literal(int(num_samples))}")
        if needle_position is not None:
            depths = depth_by_position.get(str(needle_position), None)
            if depths is None:
                raise SystemExit(f"Unsupported needle_position `{needle_position}` for ruler_niah_single_1.")
            imports.append(f"    _dataset['depth_percents'] = {python_literal(depths)}")

    model_entries = ",\n        ".join(
        f"{key}={python_literal(value)}" for key, value in model_cfg.items()
    )
    imports.append("models = [")
    imports.append("    dict(")
    imports.append(f"        type={model_type},")
    if model_entries:
        imports.append(f"        {model_entries},")
    imports.append("    )")
    imports.append("]")
    imports.append("")

    partitioner = runner_cfg.get("partitioner", {}) or {}
    runner = runner_cfg.get("runner", {}) or {}
    num_worker = int(partitioner.get("num_worker", 1))
    num_split = partitioner.get("num_split", None)
    min_task_size = int(partitioner.get("min_task_size", 16))
    max_num_workers = int(runner.get("max_num_workers", max(1, num_worker)))
    retry = int(runner.get("retry", 1))
    imports.append("infer = dict(")
    imports.append("    partitioner=dict(")
    imports.append("        type=NumWorkerPartitioner,")
    imports.append(f"        num_worker={num_worker},")
    imports.append(f"        num_split={python_literal(num_split)},")
    imports.append(f"        min_task_size={min_task_size},")
    imports.append("    ),")
    imports.append("    runner=dict(")
    imports.append("        type=LocalRunner,")
    imports.append(f"        max_num_workers={max_num_workers},")
    imports.append("        task=dict(type=OpenICLInferTask),")
    imports.append(f"        retry={retry},")
    imports.append("    ),")
    imports.append(")")
    imports.append("")
    return "\n".join(imports)


def is_custom_benchmark(benchmark: str) -> bool:
    return benchmark in CUSTOM_BENCHMARKS


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
    rows = []
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

    def start(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 2))

    def _run(self):
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
                timestamp = datetime.now(timezone.utc).isoformat()
                try:
                    result = subprocess.run(query, capture_output=True, text=True, check=False, timeout=10)
                    if result.returncode == 0:
                        for line in result.stdout.splitlines():
                            parts = [part.strip() for part in line.split(",")]
                            if len(parts) == 7:
                                writer.writerow({
                                    "timestamp_utc": timestamp,
                                    "gpu_index": parts[0],
                                    "gpu_name": parts[1],
                                    "utilization_gpu_percent": parts[2],
                                    "memory_used_mb": parts[3],
                                    "memory_total_mb": parts[4],
                                    "power_draw_w": parts[5],
                                    "temperature_gpu_c": parts[6],
                                })
                                f.flush()
                    else:
                        writer.writerow({"timestamp_utc": timestamp})
                        f.flush()
                except (OSError, subprocess.TimeoutExpired):
                    writer.writerow({"timestamp_utc": timestamp})
                    f.flush()
                self._stop.wait(self.interval_seconds)


def read_jsonl(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None or not path.exists() or path.is_dir():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def numeric_values(records: List[Dict[str, Any]], key: str) -> List[float]:
    values = []
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * p))
    return ordered[idx]


def telemetry_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    def col(name: str) -> List[float]:
        values = []
        for row in rows:
            try:
                if row.get(name) not in (None, ""):
                    values.append(float(row[name]))
            except ValueError:
                pass
        return values

    util = col("utilization_gpu_percent")
    mem = col("memory_used_mb")
    power = col("power_draw_w")
    temp = col("temperature_gpu_c")
    return {
        "gpu_samples": len(rows),
        "gpu_util_mean": mean(util),
        "gpu_util_max": max(util) if util else None,
        "gpu_memory_used_max_mb": max(mem) if mem else None,
        "gpu_power_mean_w": mean(power),
        "gpu_temperature_max_c": max(temp) if temp else None,
    }


def find_latest_opencompass_summary(work_dir: Path) -> Dict[str, Any]:
    # OpenCompass writes into work_dir/<timestamp>/summary/*.csv, not work_dir/summary directly.
    csv_files = sorted(work_dir.glob("*/summary/*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csv_files:
        return {}
    latest = csv_files[0]
    return {"opencompass_summary_csv": str(latest)}


PRIMARY_METRIC_CANDIDATES = [
    "accuracy",
    "acc",
    "pass@1",
    "exact_match",
    "em",
    "score",
]


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("%"):
        text = text[:-1]
    try:
        return float(text)
    except ValueError:
        return None


def parse_opencompass_primary_metric(work_dir: Path, benchmark: str) -> Dict[str, Any]:
    summary = find_latest_opencompass_summary(work_dir)
    path_text = summary.get("opencompass_summary_csv")
    if not path_text:
        return summary
    path = Path(path_text)
    rows = []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return summary
    if not rows:
        return summary

    def row_matches(row: Dict[str, Any]) -> bool:
        haystack = " ".join(str(value).lower() for value in row.values() if value is not None)
        return benchmark.lower() in haystack

    candidate_rows = [row for row in rows if row_matches(row)] or rows

    for metric_name in PRIMARY_METRIC_CANDIDATES:
        for row in candidate_rows:
            for key, value in row.items():
                key_norm = (key or "").strip().lower()
                if key_norm == metric_name:
                    parsed = _to_float(value)
                    if parsed is not None:
                        return {
                            **summary,
                            "primary_metric_name": key,
                            "primary_metric_value": parsed,
                        }

    for row in candidate_rows:
        metric_label = None
        for label_key in ("metric", "metrics", "name", "dataset"):
            label = row.get(label_key)
            if label and str(label).strip().lower() in PRIMARY_METRIC_CANDIDATES:
                metric_label = str(label).strip()
                break
        if not metric_label:
            continue
        for key, value in row.items():
            if key in ("metric", "metrics", "name", "dataset", "version", "mode"):
                continue
            parsed = _to_float(value)
            if parsed is not None:
                return {
                    **summary,
                    "primary_metric_name": metric_label,
                    "primary_metric_value": parsed,
                }

    for metric_name in PRIMARY_METRIC_CANDIDATES:
        for row in candidate_rows:
            for key, value in row.items():
                if metric_name in (str(key).lower() + " " + str(value).lower()):
                    parsed = _to_float(value)
                    if parsed is not None:
                        return {
                            **summary,
                            "primary_metric_name": key,
                            "primary_metric_value": parsed,
                        }
    return summary


def upsert_csv_row(output_path: Path, row: Dict[str, Any]):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    fieldnames: List[str] = []
    if output_path.exists():
        with output_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
    for key in row.keys():
        if key not in fieldnames:
            fieldnames.append(key)
    rows.append(row)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for existing in rows:
            writer.writerow(existing)


def write_run_summary(
    output_path: Path,
    run_name: str,
    experiment: str,
    benchmark: str,
    params: Dict[str, Any],
    per_sample_path: Optional[Path],
    telemetry_path: Path,
    work_dir: Path,
    returncode: Optional[int],
    elapsed_seconds: Optional[float],
) -> Dict[str, Any]:
    samples = read_jsonl(per_sample_path)
    latencies = numeric_values(samples, "elapsed_seconds")
    tps = numeric_values(samples, "tokens_per_second")
    peak_alloc = numeric_values(samples, "cuda_max_memory_allocated_mb")
    peak_reserved = numeric_values(samples, "cuda_max_memory_reserved_mb")
    completion_rates = numeric_values(samples, "completion_rate")
    actual_parallelism = numeric_values(samples, "actual_parallelism")
    actual_arness = numeric_values(samples, "actual_arness")
    planned_steps = numeric_values(samples, "planned_steps")
    planned_parallelism = numeric_values(samples, "planned_parallelism")
    threshold_pass_rates = numeric_values(samples, "threshold_pass_rate")
    fallback_rates = numeric_values(samples, "fallback_rate")
    effective_parallelism = numeric_values(samples, "effective_parallelism")
    arness = numeric_values(samples, "arness")
    peak_vram = max(peak_reserved or peak_alloc) if (peak_reserved or peak_alloc) else None
    failure_types = [str(sample.get("failure_type")) for sample in samples if sample.get("failure_type")]

    def failure_rate(name: str) -> Optional[float]:
        if not samples:
            return None
        return sum(1 for item in failure_types if item == name) / len(samples)

    row = {
        "run_name": run_name,
        "experiment": experiment,
        "benchmark": benchmark,
        "decoding_config_name": samples[0].get("decoding_config_name") if samples else run_name,
        "returncode": returncode,
        "elapsed_seconds": elapsed_seconds,
        "num_samples": len(samples),
        "format_failure_rate": failure_rate("format_failure"),
        "retrieval_failure_rate": failure_rate("retrieval_failure"),
        "task_failure_rate": failure_rate("task_failure"),
        "unfinished_generation_rate": failure_rate("unfinished_generation"),
        "truncation_failure_rate": failure_rate("truncation_failure"),
        "empty_output_rate": failure_rate("empty_output"),
        "latency_mean_s": mean(latencies),
        "latency_p50_s": percentile(latencies, 0.5),
        "latency_p95_s": percentile(latencies, 0.95),
        "tokens_per_second_mean": mean(tps),
        "tokens_per_second_p50": percentile(tps, 0.5),
        "tokens_per_second_p95": percentile(tps, 0.95),
        "peak_vram": peak_vram,
        "cuda_max_memory_allocated_mb": max(peak_alloc) if peak_alloc else None,
        "cuda_max_memory_reserved_mb": max(peak_reserved) if peak_reserved else None,
        "completion_rate": mean(completion_rates),
        "actual_parallelism": mean(actual_parallelism),
        "actual_arness_mean": mean(actual_arness),
        "planned_steps": mean(planned_steps),
        "planned_parallelism_mean": mean(planned_parallelism),
        "steps_per_block_schedule": samples[0].get("steps_per_block_schedule") if samples else None,
        "speed_schedule_name": samples[0].get("speed_schedule_name") if samples else params.get("speed_schedule_name"),
        "threshold_schedule_label": samples[0].get("threshold_schedule_label") if samples else params.get("threshold_schedule_label"),
        "token_selection_confidence_threshold_schedule": samples[0].get("token_selection_confidence_threshold_schedule") if samples else params.get("token_selection_confidence_threshold_schedule"),
        "visible_tokens_by_block": samples[0].get("visible_tokens_by_block") if samples else None,
        "completion_rate_by_block": samples[0].get("completion_rate_by_block") if samples else None,
        "threshold_pass_rate_mean": mean(threshold_pass_rates),
        "fallback_rate_mean": mean(fallback_rates),
        "effective_parallelism_mean": mean(effective_parallelism),
        "arness_mean": mean(arness),
        **telemetry_summary(telemetry_path),
        **parse_opencompass_primary_metric(work_dir, benchmark),
    }
    for key, value in params.items():
        row[f"param_{key}"] = value

    upsert_csv_row(output_path, row)
    return row


def aggregate_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Compact table used for cross-run experiment reading."""
    return {
        "run_name": row.get("run_name"),
        "experiment": row.get("experiment"),
        "benchmark": row.get("benchmark"),
        "decoding_config_name": row.get("decoding_config_name"),
        "primary_metric_name": row.get("primary_metric_name"),
        "primary_metric_value": row.get("primary_metric_value"),
        "latency_mean": row.get("latency_mean_s"),
        "tokens_per_second_mean": row.get("tokens_per_second_mean"),
        "peak_vram": row.get("peak_vram"),
        "actual_parallelism": row.get("actual_parallelism"),
        "completion_rate": row.get("completion_rate"),
        "gen_length": row.get("param_gen_length"),
        "gen_steps": row.get("param_gen_steps"),
        "gen_blocksize": row.get("param_gen_blocksize"),
        "speed_schedule_label": row.get("param_speed_schedule_label"),
        "speed_schedule_name": row.get("speed_schedule_name") or row.get("param_speed_schedule_name"),
        "steps_per_block_schedule": row.get("steps_per_block_schedule") or row.get("param_steps_per_block_schedule"),
        "planned_steps": row.get("planned_steps"),
        "planned_parallelism": row.get("planned_parallelism_mean"),
        "effective_parallelism": row.get("effective_parallelism_mean"),
        "arness_mean": row.get("arness_mean"),
        "actual_arness_mean": row.get("actual_arness_mean"),
        "token_selection_confidence_threshold": row.get("param_token_selection_confidence_threshold"),
        "threshold_schedule_label": row.get("threshold_schedule_label") or row.get("param_threshold_schedule_label"),
        "token_selection_confidence_threshold_schedule": row.get("token_selection_confidence_threshold_schedule") or row.get("param_token_selection_confidence_threshold_schedule"),
        "visible_tokens_by_block": row.get("visible_tokens_by_block"),
        "completion_rate_by_block": row.get("completion_rate_by_block"),
        "min_transfer_tokens": row.get("param_min_transfer_tokens"),
        "context_length": row.get("param_context_length"),
        "needle_position": row.get("param_needle_position"),
        "num_samples": row.get("num_samples"),
        "returncode": row.get("returncode"),
    }


def run_command(command: List[str], cwd: Path, env: Dict[str, str]) -> int:
    print("$ " + " ".join(command), flush=True)
    process = subprocess.Popen(command, cwd=str(cwd), env=env)
    return process.wait()


def cuda_stats_before(device):
    try:
        import torch
    except ImportError:
        return
    if getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def cuda_stats_after(device) -> Dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {}
    if getattr(device, "type", None) != "cuda":
        return {}
    torch.cuda.synchronize(device)
    return {
        "cuda_max_memory_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 3),
        "cuda_max_memory_reserved_mb": round(torch.cuda.max_memory_reserved(device) / 1024 ** 2, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run iLLaDA/LLaDA benchmark experiments from test_config.yaml.")
    parser.add_argument("--config", default="test_config.yaml", help="Path to the YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Generate configs and commands without running OpenCompass.")
    parser.add_argument("--only", nargs="*", help="Run only these task sections or experiment names.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_yaml(config_path)

    execution_cfg = config.get("execution", {}) or {}
    base_output_dir = Path(execution_cfg.get("output_dir", "outputs/illada_runs"))
    if not base_output_dir.is_absolute():
        base_output_dir = ROOT / base_output_dir

    dry_run = args.dry_run or bool(execution_cfg.get("dry_run", False))
    telemetry_enabled = bool(execution_cfg.get("gpu_telemetry", {}).get("enabled", True))
    telemetry_interval = float(execution_cfg.get("gpu_telemetry", {}).get("interval_seconds", 1.0))
    data_cfg = config.get("data", {}) or {}
    global_model = config.get("model", {}) or {}
    runner_cfg = config.get("runner", {}) or {}
    default_params = config.get("defaults", {}) or {}
    selected = set(args.only or [])
    experiments = collect_experiments(config, selected)
    if not experiments:
        raise SystemExit("No experiments matched the selected task or experiment names.")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(OPENCOMPASS_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    manifest_paths = set()
    planned = 0

    for experiment in experiments:
        exp_name = experiment.get("name")
        if not exp_name:
            raise SystemExit("Every experiment needs a `name`.")
        output_dir = experiment_output_dir(base_output_dir, experiment)
        generated_dir = output_dir / "generated_configs"
        generated_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "run_manifest.jsonl"
        manifest_paths.add(str(manifest_path))

        for benchmark in as_list(experiment.get("benchmark")):
            if not benchmark:
                raise SystemExit(f"Experiment `{exp_name}` is missing `benchmark`.")
            for idx, params in enumerate(expand_matrix(experiment), start=1):
                planned += 1
                merged_params = deep_merge(default_params, params)
                run_name = condition_run_name(exp_name, benchmark, merged_params, idx)
                work_dir = output_dir / run_name
                model_cfg = build_model_cfg(global_model, merged_params, benchmark, run_name)
                model_cfg["task_id"] = experiment.get("task")
                if model_cfg.get("arness_trace_output"):
                    model_cfg["arness_trace_output"] = str(work_dir / "sample_traces")
                if execution_cfg.get("collect_metrics", True) and not model_cfg.get("metrics_output"):
                    model_cfg["per_sample_output"] = str(work_dir / "summary.jsonl")
                    if model_cfg.get("return_trace") or model_cfg.get("trace_token_snapshots") or model_cfg.get("trace_decode_snapshots"):
                        model_cfg["step_trace_output"] = str(work_dir / "trace.jsonl")
                    model_cfg["metrics_output"] = str(work_dir / "summary.jsonl")
                generated_config = generated_dir / f"{run_name}.py"
                config_text = render_opencompass_config(
                    benchmark,
                    deepcopy(model_cfg),
                    runner_cfg,
                    sample_limit=merged_params.get("sample_limit"),
                    sample_indices=merged_params.get("sample_indices"),
                    experiment_params=merged_params,
                    data_cfg=data_cfg,
                )
                generated_config.write_text(config_text, encoding="utf-8")
                work_dir.mkdir(parents=True, exist_ok=True)
                run_config = {
                    "run_name": run_name,
                    "task": experiment.get("task"),
                    "experiment": exp_name,
                    "benchmark": benchmark,
                    "params": merged_params,
                    "model": model_cfg,
                    "runner": runner_cfg,
                    "source_config": str(config_path),
                    "generated_opencompass_config": str(generated_config),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                (work_dir / "config.json").write_text(
                    json.dumps(run_config, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                command = [
                    sys.executable,
                    "run.py",
                    str(generated_config),
                    "-w",
                    str(work_dir),
                ]
                extra_args = execution_cfg.get("opencompass_args", []) or []
                command.extend(str(item) for item in extra_args)

                manifest = {
                    "run_name": run_name,
                    "task": experiment.get("task"),
                    "experiment": exp_name,
                    "benchmark": benchmark,
                    "params": merged_params,
                    "model": model_cfg,
                    "config": str(generated_config),
                    "work_dir": str(work_dir),
                    "command": command,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "dry_run": dry_run,
                    "gpu_before": current_gpu_snapshot(),
                    "artifacts": {
                        "config_json": str(work_dir / "config.json"),
                        "summary_jsonl": model_cfg.get("per_sample_output") or model_cfg.get("metrics_output"),
                        "trace_jsonl": model_cfg.get("step_trace_output"),
                        "gpu_telemetry_csv": str(work_dir / "gpu_telemetry.csv"),
                        "aggregate_csv": str(work_dir / "aggregate.csv"),
                    },
                }
                print(f"[{run_name}] config: {generated_config}")
                print(f"[{run_name}] work_dir: {work_dir}")
                if dry_run:
                    print(f"[{run_name}] dry-run: {' '.join(command)}")
                    manifest["returncode"] = None
                else:
                    telemetry = None
                    if telemetry_enabled:
                        telemetry = GpuTelemetry(work_dir / "gpu_telemetry.csv", interval_seconds=telemetry_interval)
                        telemetry.start()
                    start = time.perf_counter()
                    try:
                        returncode = run_command(command, OPENCOMPASS_DIR, env)
                    finally:
                        if telemetry is not None:
                            telemetry.stop()
                    elapsed_seconds = round(time.perf_counter() - start, 3)
                    manifest["elapsed_seconds"] = elapsed_seconds
                    manifest["returncode"] = returncode
                    manifest["gpu_after"] = current_gpu_snapshot()
                    run_summary = write_run_summary(
                        output_path=work_dir / "run_summary.csv",
                        run_name=run_name,
                        experiment=exp_name,
                        benchmark=benchmark,
                        params=merged_params,
                        per_sample_path=Path(model_cfg["per_sample_output"]) if model_cfg.get("per_sample_output") else None,
                        telemetry_path=work_dir / "gpu_telemetry.csv",
                        work_dir=work_dir,
                        returncode=returncode,
                        elapsed_seconds=elapsed_seconds,
                    )
                    compact_summary = aggregate_row(run_summary)
                    upsert_csv_row(work_dir / "aggregate.csv", compact_summary)
                    upsert_csv_row(output_dir / "aggregate.csv", compact_summary)
                    upsert_csv_row(output_dir / "summary_all.csv", run_summary)
                    with manifest_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(manifest, ensure_ascii=False) + "\n")
                    if returncode != 0 and execution_cfg.get("stop_on_error", True):
                        print(f"[{run_name}] failed with return code {returncode}", file=sys.stderr)
                        return returncode
                if dry_run:
                    with manifest_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(manifest, ensure_ascii=False) + "\n")

    if planned == 0:
        raise SystemExit("No enabled experiments matched the selection.")
    manifests = ", ".join(sorted(manifest_paths))
    print(f"Planned {planned} run(s). Manifest(s): {manifests}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
