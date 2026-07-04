#!/usr/bin/env python3
"""Native offline evaluator for model_outputs produced by native_output.py.

This script reads `test_config.yaml` and evaluates existing model outputs without
loading any model and without invoking OpenCompass CLI.  The scoring follows the
same practical conventions used by OpenCompass-style generation benchmarks:

  - GSM8K: final numeric answer extraction and exact match.
  - MBPP: extract Python code and run the provided unit tests.
  - RULER single needle: normalized substring recall.
  - RULER order-2: exact ordered pair + unordered set + slot accuracies.

It writes final artifacts under:

    outputs/<task>/<benchmark_alias>/<condition>/<model_alias>/

For ARness trace runs, trace.jsonl / summary.jsonl / sample_traces are copied so
`visual_arness_trace.py` can be pointed directly at the final output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from run_test import ROOT, as_list, collect_experiments, deep_merge, expand_matrix, load_yaml, safe_name
except Exception:  # pragma: no cover
    ROOT = Path(__file__).resolve().parent
    from prepare_data import as_list, deep_merge, load_yaml, safe_name, collect_experiments, expand_matrix  # type: ignore

from prepare_data import bench_alias, condition_name, read_jsonl, write_jsonl, json_default
from native_output import model_alias, normalize_models, output_dir_for


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


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


def iter_conditions(config: Dict[str, Any], selected: Set[str]) -> Iterable[Dict[str, Any]]:
    defaults = config.get("defaults", {}) or {}
    for experiment in collect_experiments(config, selected):
        task = experiment.get("task") or "runs"
        exp_models = set(str(x) for x in as_list(experiment.get("models"))) if experiment.get("models") is not None else None
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
                    "experiment_models": exp_models,
                }


def final_dir_for(root: Path, model_name: str, condition: Dict[str, Any]) -> Path:
    return root / safe_name(condition["task"] or "runs") / bench_alias(condition["benchmark"]) / condition["condition"] / safe_name(model_name)


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


# ------------------------------- GSM8K -------------------------------------


def normalize_number(text: str) -> Optional[str]:
    if text is None:
        return None
    text = str(text).replace(",", "")
    nums = re.findall(r"[-+]?\d*\.?\d+(?:/[1-9]\d*)?", text)
    if not nums:
        return None
    val = nums[-1]
    if "/" in val:
        try:
            a, b = val.split("/", 1)
            return str(float(a) / float(b)).rstrip("0").rstrip(".")
        except Exception:
            return val
    try:
        f = float(val)
        if abs(f - int(f)) < 1e-9:
            return str(int(f))
        return ("%.10f" % f).rstrip("0").rstrip(".")
    except Exception:
        return val


def gsm8k_gold(answer: Any) -> Optional[str]:
    text = str(answer or "")
    if "####" in text:
        text = text.split("####")[-1]
    return normalize_number(text)


def score_gsm8k(row: Dict[str, Any]) -> Dict[str, Any]:
    pred = normalize_number(row.get("prediction") or row.get("raw_output") or "")
    gold = gsm8k_gold(row.get("answer"))
    correct = pred is not None and gold is not None and pred == gold
    return {"correct": bool(correct), "prediction_answer": pred, "gold_answer": gold, "score": 1.0 if correct else 0.0}


# -------------------------------- MBPP --------------------------------------


def extract_code(raw: str) -> str:
    raw = str(raw or "")
    m = re.search(r"```(?:python)?\s*(.*?)```", raw, flags=re.S | re.I)
    if m:
        return m.group(1).strip()
    lines = raw.splitlines()
    start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("def ") or stripped.startswith("import ") or stripped.startswith("from ") or stripped.startswith("class "):
            start = i
            break
    return "\n".join(lines[start:]).strip() if start is not None else raw.strip()


def run_python_tests(code: str, tests: Sequence[str], setup_code: str = "", timeout: int = 8) -> Dict[str, Any]:
    program = "\n".join([setup_code or "", code, "", "\n".join(str(t) for t in tests)])
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "candidate.py"
        path.write_text(program, encoding="utf-8")
        try:
            proc = subprocess.run(
                ["python", str(path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"passed": False, "error_type": "timeout", "stderr": "TimeoutExpired"}
    if proc.returncode == 0:
        return {"passed": True, "error_type": "pass", "stderr": ""}
    stderr = proc.stderr or ""
    if "SyntaxError" in stderr:
        etype = "syntax_error"
    elif "NameError" in stderr:
        etype = "name_error_or_missing_function"
    elif "AssertionError" in stderr:
        etype = "wrong_answer"
    else:
        etype = "runtime_error"
    return {"passed": False, "error_type": etype, "stderr": stderr[-1200:]}


def score_mbpp(row: Dict[str, Any], timeout: int = 8) -> Dict[str, Any]:
    metadata = row.get("metadata") or {}
    tests = metadata.get("test_list") or metadata.get("tests") or []
    if isinstance(tests, str):
        try:
            tests = json.loads(tests)
        except Exception:
            tests = [tests]
    code = extract_code(row.get("prediction") or row.get("raw_output") or "")
    if not tests:
        return {"correct": None, "score": None, "error_type": "no_tests", "extracted_code": code}
    result = run_python_tests(code, tests, setup_code=str(metadata.get("test_setup_code") or ""), timeout=timeout)
    return {
        "correct": bool(result["passed"]),
        "score": 1.0 if result["passed"] else 0.0,
        "error_type": result["error_type"],
        "stderr": result.get("stderr", ""),
        "extracted_code": code,
    }


# ------------------------------- RULER --------------------------------------


def score_ruler_single(row: Dict[str, Any]) -> Dict[str, Any]:
    pred = normalize_text(row.get("prediction") or row.get("raw_output") or "")
    gold = normalize_text(row.get("answer"))
    correct = bool(gold and gold in pred)
    return {"correct": correct, "score": 1.0 if correct else 0.0, "gold_answer": row.get("answer")}


def extract_order_values(text: str) -> List[str]:
    text = str(text or "")
    # Prefer JSON array if the model followed instructions.
    m = re.search(r"\[[^\]]*\]", text, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, list):
                vals = [str(x) for x in obj]
                nums = []
                for val in vals:
                    found = re.findall(r"\b\d{7}\b", val)
                    nums.extend(found if found else [val])
                return nums
        except Exception:
            pass
    return re.findall(r"\b\d{7}\b", text)


def score_ruler_order2(row: Dict[str, Any]) -> Dict[str, Any]:
    pred_vals = extract_order_values(row.get("prediction") or row.get("raw_output") or "")
    answer = row.get("answer")
    if not isinstance(answer, list):
        metadata = row.get("metadata") or {}
        answer = metadata.get("needle_values_in_order") or []
    ref = [str(x) for x in answer][:2]
    first_two = pred_vals[:2]
    exact = len(first_two) >= 2 and first_two == ref
    unordered = len(first_two) >= 2 and set(first_two) == set(ref)
    slot1 = len(first_two) >= 1 and len(ref) >= 1 and first_two[0] == ref[0]
    slot2 = len(first_two) >= 2 and len(ref) >= 2 and first_two[1] == ref[1]
    return {
        "correct": bool(exact),
        "score": 1.0 if exact else 0.0,
        "exact_order": bool(exact),
        "set_match": bool(unordered),
        "slot1": bool(slot1),
        "slot2": bool(slot2),
        "pred_values": pred_vals,
        "gold_values": ref,
    }


# ------------------------------- generic ------------------------------------


def score_generic(row: Dict[str, Any]) -> Dict[str, Any]:
    pred = normalize_text(row.get("prediction") or row.get("raw_output") or "")
    gold = normalize_text(row.get("answer"))
    correct = bool(gold and gold in pred)
    return {"correct": correct, "score": 1.0 if correct else 0.0}


def score_row(row: Dict[str, Any], timeout: int = 8) -> Dict[str, Any]:
    bench = str(row.get("benchmark"))
    if bench == "gsm8k":
        return score_gsm8k(row)
    if bench == "mbpp":
        return score_mbpp(row, timeout=timeout)
    if bench == "ruler_niah_single_1":
        return score_ruler_single(row)
    if bench == "ruler_niah_order_2":
        return score_ruler_order2(row)
    return score_generic(row)


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def aggregate_scores(rows: List[Dict[str, Any]], outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    score = mean([r.get("score") for r in rows])
    metrics: Dict[str, Any] = {
        "n": n,
        "score": round(score * 100, 4) if score is not None else None,
        "accuracy": round(score * 100, 4) if score is not None else None,
    }
    # Optional task-specific metrics.
    for key, out_key in [
        ("exact_order", "exact_order_acc"),
        ("set_match", "set_acc"),
        ("slot1", "slot1_acc"),
        ("slot2", "slot2_acc"),
    ]:
        if any(key in r for r in rows):
            vals = [1.0 if r.get(key) else 0.0 for r in rows]
            metrics[out_key] = round(sum(vals) / max(len(vals), 1) * 100, 4)
    # Error breakdown for code tasks.
    errors: Dict[str, int] = {}
    for row in rows:
        et = row.get("error_type")
        if et:
            errors[str(et)] = errors.get(str(et), 0) + 1
    if errors:
        metrics["error_breakdown"] = errors
    # Timing from outputs.
    latencies = []
    tps = []
    for row in outputs:
        timing = row.get("timing") or {}
        if timing.get("elapsed_seconds") is not None:
            latencies.append(timing.get("elapsed_seconds"))
        if timing.get("tokens_per_second") is not None:
            tps.append(timing.get("tokens_per_second"))
    metrics["avg_latency_sec"] = round(mean(latencies) or 0.0, 6) if latencies else None
    metrics["avg_tps"] = round(mean(tps) or 0.0, 6) if tps else None
    return metrics


def copy_artifacts(src: Path, dst: Path, sample_idx: Optional[int] = None) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in ["inputs.jsonl", "outputs.jsonl", "summary.jsonl", "trace.jsonl", "gpu.csv", "run.json", "output_manifest.json"]:
        sp = src / name
        if sp.exists():
            shutil.copy2(sp, dst / name)
    for name in ["sample_traces"]:
        sp = src / name
        dp = dst / name
        if sp.exists():
            if dp.exists():
                shutil.rmtree(dp)
            shutil.copytree(sp, dp)
    if (dst / "trace.jsonl").exists():
        idx = 0 if sample_idx is None else sample_idx
        (dst / "visual_command.txt").write_text(f'python visual_arness_trace.py "{dst}" --sample-idx {idx}\n', encoding="utf-8")


def evaluate_run(model_output_dir: Path, final_dir: Path, timeout: int = 8) -> Dict[str, Any]:
    outputs_path = model_output_dir / "outputs.jsonl"
    if not outputs_path.exists():
        raise FileNotFoundError(f"Missing outputs.jsonl: {outputs_path}")
    outputs = read_jsonl(outputs_path)
    scores: List[Dict[str, Any]] = []
    for row in outputs:
        score = score_row(row, timeout=timeout)
        scores.append({
            "model": row.get("model"),
            "task": row.get("task"),
            "experiment": row.get("experiment"),
            "benchmark": row.get("benchmark"),
            "condition": row.get("condition"),
            "sample_id": row.get("sample_id"),
            **score,
        })
    metrics = aggregate_scores(scores, outputs)
    final_dir.mkdir(parents=True, exist_ok=True)
    copy_artifacts(model_output_dir, final_dir, sample_idx=outputs[0].get("sample_id") if outputs else None)
    write_jsonl(final_dir / "scores.jsonl", scores)
    write_json(final_dir / "metrics.json", metrics)
    write_csv(final_dir / "scores.csv", scores)
    return metrics


def model_output_dir_for(root: Path, model_name: str, condition: Dict[str, Any]) -> Path:
    return output_dir_for(root, model_name, condition)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate native model_outputs and materialize final outputs.")
    parser.add_argument("--config", default="test_config.yaml")
    parser.add_argument("--only", nargs="*", default=[], help="Task or experiment names to evaluate.")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--model-output-root", default="model_outputs")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--timeout", type=int, default=8, help="MBPP subprocess timeout per sample.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_yaml(config_path)
    selected = set(args.only or [])
    conditions = list(iter_conditions(config, selected))
    models = normalize_models(config)
    explicit = set(args.models or []) if args.models else None

    model_root = Path(args.model_output_root)
    if not model_root.is_absolute():
        model_root = ROOT / model_root
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root

    plan: List[Tuple[str, Dict[str, Any], Path, Path]] = []
    for condition in conditions:
        exp_models = condition.get("experiment_models")
        for model_cfg in models:
            alias = model_alias(model_cfg)
            if explicit and alias not in explicit and str(model_cfg.get("name")) not in explicit:
                continue
            if exp_models and alias not in exp_models and str(model_cfg.get("name")) not in exp_models:
                continue
            src = model_output_dir_for(model_root, alias, condition)
            dst = final_dir_for(output_root, alias, condition)
            plan.append((alias, condition, src, dst))

    print(f"Native eval plan: {len(plan)} run(s).")
    for alias, condition, src, dst in plan:
        exists = (src / "outputs.jsonl").exists()
        print(f"- {alias} | {condition['experiment']} | {condition['condition']} | src={src} ({'ok' if exists else 'missing'}) | dst={dst}")
    if args.dry_run:
        return 0

    manifest: List[Dict[str, Any]] = []
    compare_by_condition: Dict[Path, List[Dict[str, Any]]] = {}
    for alias, condition, src, dst in plan:
        if not (src / "outputs.jsonl").exists():
            print(f"[SKIP] missing outputs: {src}")
            continue
        print(f"[EVAL] {alias} | {condition['experiment']} | {condition['condition']}")
        metrics = evaluate_run(src, dst, timeout=args.timeout)
        row = {
            "model": alias,
            "task": condition["task"],
            "experiment": condition["experiment"],
            "benchmark": condition["benchmark"],
            "condition": condition["condition"],
            "output_dir": str(dst),
            **{k: v for k, v in metrics.items() if not isinstance(v, (dict, list))},
        }
        manifest.append(row)
        compare_dir = dst.parent
        compare_by_condition.setdefault(compare_dir, []).append(row)

    write_jsonl(output_root / "native_eval_manifest.jsonl", manifest)
    write_csv(output_root / "native_eval_manifest.csv", manifest)
    for compare_dir, rows in compare_by_condition.items():
        write_csv(compare_dir / "compare.csv", rows)
    print(f"Wrote eval manifest: {output_root / 'native_eval_manifest.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
