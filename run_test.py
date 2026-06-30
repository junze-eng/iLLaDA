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
    "math": {
        "module": "opencompass.configs.datasets.math.math_gen",
        "var": "math_datasets",
    },
    "gpqa": {
        "module": "opencompass.configs.datasets.gpqa.gpqa_gen",
        "var": "gpqa_datasets",
    },
    "mmlu": {
        "module": "opencompass.configs.datasets.mmlu.mmlu_gen_a484b3",
        "var": "mmlu_datasets",
        "summary_module": "opencompass.configs.summarizers.groups.mmlu",
        "summary_var": "mmlu_summary_groups",
    },
    "mmlu_pro": {
        "module": "opencompass.configs.datasets.mmlu_pro.mmlu_pro_gen",
        "var": "mmlu_pro_datasets",
    },
    "hellaswag": {
        "module": "opencompass.configs.datasets.hellaswag.hellaswag_gen",
        "var": "hellaswag_datasets",
    },
    "arc_c": {
        "module": "opencompass.configs.datasets.ARC_c.ARC_c_gen",
        "var": "ARC_c_datasets",
    },
    "humaneval": {
        "module": "opencompass.configs.datasets.humaneval.humaneval_gen",
        "var": "humaneval_datasets",
    },
    "mbpp": {
        "module": "opencompass.configs.datasets.mbpp.mbpp_gen",
        "var": "mbpp_datasets",
    },
    "ifeval": {
        "module": "opencompass.configs.datasets.IFEval.IFEval_gen",
        "var": "ifeval_datasets",
    },
}

CUSTOM_BENCHMARKS = {"needle_passkey"}

MODEL_TYPES = {
    "instruct": "LLaDAModel",
    "base": "LLaDABaseModel",
}


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "PyYAML is required to read test_config.yaml. Install it with `pip install pyyaml`."
        ) from exc

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
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


def safe_name(value: str) -> str:
    keep = []
    for char in value.lower():
        keep.append(char if char.isalnum() else "_")
    return "_".join("".join(keep).split("_"))


def build_model_cfg(global_model: Dict[str, Any], params: Dict[str, Any], benchmark: str, run_name: str) -> Dict[str, Any]:
    model_cfg = deepcopy(global_model)
    model_type = model_cfg.pop("type", "instruct")
    if model_type not in MODEL_TYPES:
        raise SystemExit(f"Unsupported model.type `{model_type}`. Choose one of: {', '.join(MODEL_TYPES)}")

    model_cfg.setdefault("abbr", f"{Path(str(model_cfg.get('path', 'model'))).name}-{benchmark}")
    model_cfg["abbr"] = safe_name(f"{model_cfg['abbr']}_{run_name}")
    model_cfg["type"] = MODEL_TYPES[model_type]
    model_cfg.update(params)
    return model_cfg


def render_opencompass_config(
    benchmark: str,
    model_cfg: Dict[str, Any],
    runner_cfg: Dict[str, Any],
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
    if "summary_var" in bench:
        imports.append(f"summarizer = dict(summary_groups={bench['summary_var']})")

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
    summary_dir = work_dir / "summary"
    if not summary_dir.exists():
        return {}
    csv_files = sorted(summary_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csv_files:
        return {}
    latest = csv_files[0]
    return {"opencompass_summary_csv": str(latest)}


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
    final_masks = numeric_values(samples, "final_mask_count")
    fallback_counts = numeric_values(samples, "fallback_transfer_count")
    correctness = numeric_values(samples, "correctness")
    scores = numeric_values(samples, "score")
    effective_parallelism = numeric_values(samples, "effective_parallelism")
    arness = numeric_values(samples, "arness")

    row = {
        "run_name": run_name,
        "experiment": experiment,
        "benchmark": benchmark,
        "returncode": returncode,
        "elapsed_seconds": elapsed_seconds,
        "num_samples": len(samples),
        "latency_mean_s": mean(latencies),
        "latency_p50_s": percentile(latencies, 0.5),
        "latency_p95_s": percentile(latencies, 0.95),
        "tokens_per_second_mean": mean(tps),
        "tokens_per_second_p50": percentile(tps, 0.5),
        "tokens_per_second_p95": percentile(tps, 0.95),
        "cuda_max_memory_allocated_mb": max(peak_alloc) if peak_alloc else None,
        "cuda_max_memory_reserved_mb": max(peak_reserved) if peak_reserved else None,
        "final_mask_count_sum": sum(final_masks) if final_masks else None,
        "fallback_transfer_count_sum": sum(fallback_counts) if fallback_counts else None,
        "accuracy_or_pass_rate": mean(correctness) if correctness else None,
        "score_mean": mean(scores) if scores else None,
        "effective_parallelism_mean": mean(effective_parallelism),
        "arness_mean": mean(arness),
        **telemetry_summary(telemetry_path),
        **find_latest_opencompass_summary(work_dir),
    }
    for key, value in params.items():
        row[f"param_{key}"] = value

    upsert_csv_row(output_path, row)
    return row


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


def build_needle_prompt(tokenizer, context_length: int, needle_position: str, secret: str, seed: int) -> str:
    filler_unit = (
        f"Record {seed}: ocean logistics, warehouse timing, and invoice notes are unrelated. "
        "The evaluation should ignore these details. "
    )
    filler = filler_unit
    while len(tokenizer.encode(filler, add_special_tokens=False)) < max(1, context_length):
        filler += filler_unit
    filler_ids = tokenizer.encode(filler, add_special_tokens=False)[:max(1, context_length)]
    needle = f"The secret code is {secret}."
    needle_ids = tokenizer.encode(" " + needle + " ", add_special_tokens=False)
    if len(needle_ids) >= len(filler_ids):
        filler_ids = []

    available = max(0, context_length - len(needle_ids))
    filler_ids = filler_ids[:available]
    if needle_position == "front":
        insert_at = 0
    elif needle_position == "middle":
        insert_at = len(filler_ids) // 2
    elif needle_position == "end":
        insert_at = len(filler_ids)
    else:
        raise ValueError(f"Unknown needle_position `{needle_position}`.")
    context_ids = filler_ids[:insert_at] + needle_ids + filler_ids[insert_at:]
    context = tokenizer.decode(context_ids, skip_special_tokens=True)
    return (
        "You will receive a long context. Find the exact secret code in it.\n\n"
        f"{context}\n\n"
        "Question: What is the secret code? Reply with only the digits."
    )


def run_needle_passkey(
    model_cfg: Dict[str, Any],
    params: Dict[str, Any],
    work_dir: Path,
) -> int:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
        from generate import generate
    except ImportError as exc:
        print(f"Missing dependency for needle_passkey: {exc}", file=sys.stderr)
        return 1

    model_path = model_cfg["path"]
    dtype_name = (model_cfg.get("model_kwargs", {}) or {}).get("torch_dtype", "torch.bfloat16")
    torch_dtype = {
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.float32": torch.float32,
        "torch.float": torch.float32,
    }.get(dtype_name, torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"

    num_samples = int(params.get("num_samples", 20))
    context_length = int(params.get("context_length", params.get("context_prefix_tokens", 2048)))
    needle_position = params.get("needle_position", "middle")
    seed = int(params.get("seed", 1234))
    gen_length = int(params.get("gen_length", 32))
    gen_steps = int(params.get("gen_steps", gen_length))
    gen_blocksize = int(params.get("gen_blocksize", gen_length))
    mask_id = int(model_cfg.get("mask_id", 5))
    per_sample_path = Path(model_cfg["per_sample_output"])
    step_trace_path = Path(model_cfg["step_trace_output"]) if model_cfg.get("step_trace_output") else None
    per_sample_path.parent.mkdir(parents=True, exist_ok=True)

    for sample_idx in range(num_samples):
        sample_seed = seed + sample_idx
        secret = f"{(sample_seed * 7919) % 1000000:06d}"
        prompt = build_needle_prompt(tokenizer, context_length, needle_position, secret, sample_seed)
        encoded = tokenizer([prompt], add_special_tokens=False, padding=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        cuda_stats_before(model.device)
        started = time.perf_counter()
        generated, trace = generate(
            model=model,
            prompt=input_ids,
            attention_mask=attention_mask,
            steps=gen_steps,
            gen_length=gen_length,
            block_length=gen_blocksize,
            temperature=float(model_cfg.get("temperature", 0.0)),
            cfg_scale=float(model_cfg.get("cfg", 0.0)),
            remasking=model_cfg.get("remasking", "low_confidence"),
            mask_id=mask_id,
            confidence_eos_eot_inf=bool(params.get("diff_confidence_eos_eot_inf", False)),
            logits_eos_inf=bool(params.get("diff_logits_eos_inf", False)),
            token_selection_confidence_threshold=params.get("token_selection_confidence_threshold"),
            min_transfer_tokens=int(params.get("min_transfer_tokens", 1)),
            return_trace=True,
        )
        elapsed = time.perf_counter() - started
        cuda_stats = cuda_stats_after(model.device)
        prediction = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        correct = 1 if secret in prediction else 0
        record = {
            "sample_idx": sample_idx,
            "seed": sample_seed,
            "benchmark": "needle_passkey",
            "input": prompt,
            "prediction": prediction,
            "target": secret,
            "correctness": correct,
            "score": correct,
            "evaluator_status": "exact_secret_substring",
            "context_length": context_length,
            "needle_position": needle_position,
            "prompt_tokens": int(input_ids.shape[1]),
            "generated_tokens": gen_length,
            "elapsed_seconds": round(elapsed, 6),
            "tokens_per_second": round(gen_length / elapsed, 6) if elapsed > 0 else None,
            "steps": gen_steps,
            "gen_length": gen_length,
            "block_length": gen_blocksize,
            "forward_passes": gen_steps,
            "effective_parallelism": float(gen_length / gen_steps) if gen_steps else None,
            "arness": float(gen_steps / gen_length) if gen_length else None,
            "mask_id": mask_id,
            "token_selection_confidence_threshold": params.get("token_selection_confidence_threshold"),
            "min_transfer_tokens": int(params.get("min_transfer_tokens", 1)),
            "final_mask_count": trace.get("final_mask_count"),
            "fallback_transfer_count": trace.get("fallback_transfer_count"),
            **cuda_stats,
        }
        with per_sample_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if step_trace_path is not None:
            with step_trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"sample_idx": sample_idx, "trace": trace}, ensure_ascii=False) + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run iLLaDA/LLaDA benchmark experiments from test_config.yaml.")
    parser.add_argument("--config", default="test_config.yaml", help="Path to the YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Generate configs and commands without running OpenCompass.")
    parser.add_argument("--only", nargs="*", help="Run only experiments with these names.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_yaml(config_path)

    execution_cfg = config.get("execution", {}) or {}
    output_dir = Path(execution_cfg.get("output_dir", "outputs/illada_runs"))
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    generated_dir = output_dir / "generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)

    dry_run = args.dry_run or bool(execution_cfg.get("dry_run", False))
    telemetry_enabled = bool(execution_cfg.get("gpu_telemetry", {}).get("enabled", True))
    telemetry_interval = float(execution_cfg.get("gpu_telemetry", {}).get("interval_seconds", 1.0))
    global_model = config.get("model", {}) or {}
    runner_cfg = config.get("runner", {}) or {}
    default_params = config.get("defaults", {}) or {}
    experiments = config.get("experiments", []) or []
    if not experiments:
        raise SystemExit("No experiments found in config.")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(OPENCOMPASS_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    manifest_path = output_dir / "run_manifest.jsonl"
    selected = set(args.only or [])
    planned = 0

    for experiment in experiments:
        exp_name = experiment.get("name")
        if not exp_name:
            raise SystemExit("Every experiment needs a `name`.")
        if selected and exp_name not in selected:
            continue
        if not selected and experiment.get("enabled", True) is False:
            continue

        for benchmark in as_list(experiment.get("benchmark")):
            if not benchmark:
                raise SystemExit(f"Experiment `{exp_name}` is missing `benchmark`.")
            for idx, params in enumerate(expand_matrix(experiment), start=1):
                planned += 1
                merged_params = deep_merge(default_params, params)
                run_name = safe_name(f"{exp_name}_{benchmark}_{idx}")
                work_dir = output_dir / run_name
                model_cfg = build_model_cfg(global_model, merged_params, benchmark, run_name)
                if execution_cfg.get("collect_metrics", True) and not model_cfg.get("metrics_output"):
                    model_cfg["per_sample_output"] = str(work_dir / "per_sample.jsonl")
                    if model_cfg.get("return_trace") or is_custom_benchmark(benchmark):
                        model_cfg["step_trace_output"] = str(work_dir / "step_trace.jsonl")
                    model_cfg["metrics_output"] = str(work_dir / "per_sample.jsonl")
                generated_config = generated_dir / f"{run_name}.py"
                if is_custom_benchmark(benchmark):
                    generated_config.write_text(
                        "# Custom diagnostic benchmark; executed directly by run_test.py.\n",
                        encoding="utf-8",
                    )
                else:
                    config_text = render_opencompass_config(benchmark, deepcopy(model_cfg), runner_cfg)
                    generated_config.write_text(config_text, encoding="utf-8")
                work_dir.mkdir(parents=True, exist_ok=True)
                run_config = {
                    "run_name": run_name,
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

                if is_custom_benchmark(benchmark):
                    command = [sys.executable, "run_test.py", "--custom-benchmark", benchmark, "--run-name", run_name]
                else:
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
                        "per_sample_jsonl": model_cfg.get("per_sample_output") or model_cfg.get("metrics_output"),
                        "step_trace_jsonl": model_cfg.get("step_trace_output"),
                        "gpu_telemetry_csv": str(work_dir / "gpu_telemetry.csv"),
                        "summary_csv": str(work_dir / "summary.csv"),
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
                        if is_custom_benchmark(benchmark):
                            print(f"$ run custom benchmark {benchmark} ({run_name})", flush=True)
                            returncode = run_needle_passkey(model_cfg, merged_params, work_dir)
                        else:
                            returncode = run_command(command, OPENCOMPASS_DIR, env)
                    finally:
                        if telemetry is not None:
                            telemetry.stop()
                    elapsed_seconds = round(time.perf_counter() - start, 3)
                    manifest["elapsed_seconds"] = elapsed_seconds
                    manifest["returncode"] = returncode
                    manifest["gpu_after"] = current_gpu_snapshot()
                    run_summary = write_run_summary(
                        output_path=work_dir / "summary.csv",
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
    print(f"Planned {planned} run(s). Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
