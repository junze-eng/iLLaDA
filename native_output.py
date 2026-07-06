#!/usr/bin/env python3
"""Native offline output materializer/evaluator for model_outputs produced by native_model.py.

This script reads `test_config.yaml` and evaluates existing model outputs without
loading any model and without invoking the OpenCompass CLI.  Evaluation is split
explicitly by benchmark:

  - Existing/legacy benchmarks must use the SAME scoring policy that was used
    for prior iLLaDA runs.  This script therefore registers an explicit
    benchmark -> evaluator policy and refuses unknown benchmarks by default.
  - gsm8k / mbpp / ruler_niah_single_1 use OpenCompass-compatible evaluator
    copies, i.e. the same answer extraction / code-test / RULER-recall policy
    as the old OpenCompass path, but applied directly to saved model_outputs.
  - ruler_niah_double_2 is the only native extension.  It has no previous
    OpenCompass result to match, so its evaluator is fixed here and used for
    every model: exact_order_acc, set_acc, slot1_acc, slot2_acc.

The invariant is: for a given benchmark name, every model and every future run
uses the same evaluator.  We do not silently fall back to a different scorer
unless --allow-generic-eval is explicitly set.

It writes final artifacts under a model-first layout:

    native_outputs/<model_alias>/<task>/<benchmark_alias>/<condition>/

Per-model eval manifests are written under:

    native_outputs/<model_alias>/native_output_manifest.{jsonl,csv}

Cross-model compare.csv files are written under:

    native_outputs/_compare/<task>/<benchmark_alias>/<condition>/

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
from native_model import model_alias, model_selected, normalize_models, output_dir_for, safe_model_name


EVAL_POLICY: Dict[str, Dict[str, Any]] = {
    "gsm8k": {
        "backend": "legacy_opencompass_compatible",
        "compatibility": "same_policy_as_previous_iLLaDA",
        "metric": "exact_match",
        "source": "GSM8K numeric exact-match convention used by the previous OpenCompass path: extract the final numeric answer, normalize commas/fractions, compare exact value.",
        "custom_native": False,
    },
    "mbpp": {
        "backend": "legacy_opencompass_compatible",
        "compatibility": "same_policy_as_previous_iLLaDA",
        "metric": "pass_at_1",
        "source": "MBPP code-generation convention used by the previous OpenCompass path: extract Python code, append the sample tests, run in a subprocess with timeout.",
        "custom_native": False,
    },
    "ruler_niah_single_1": {
        "backend": "legacy_opencompass_ruler_compatible",
        "compatibility": "same_policy_as_previous_iLLaDA",
        "metric": "substring_recall",
        "source": "RULER NIAH single-needle recall convention used by the previous path: normalized gold answer must appear in normalized prediction.",
        "custom_native": False,
    },
    "ruler_niah_double_2": {
        "backend": "native_ruler_extension",
        "compatibility": "new_benchmark_no_previous_iLLaDA_result",
        "metric": "exact_order_acc",
        "source": "New two-needle ordered RULER extension. This has no previous OpenCompass evaluator, so this fixed native evaluator is the source of truth for all models.",
        "custom_native": True,
    },
    "ruler_niah_order_2": {
        "backend": "native_ruler_extension_alias",
        "compatibility": "alias_of_ruler_niah_double_2",
        "metric": "exact_order_acc",
        "source": "Alias of ruler_niah_double_2 kept for backward compatibility.",
        "custom_native": True,
    },
}


def eval_policy_for(benchmark: str, allow_generic: bool = False) -> Dict[str, Any]:
    benchmark = str(benchmark)
    if benchmark in EVAL_POLICY:
        return EVAL_POLICY[benchmark]
    if allow_generic:
        return {
            "backend": "native_generic_explicitly_allowed",
            "compatibility": "not_legacy_comparable",
            "metric": "normalized_substring_match",
            "source": "Generic fallback enabled by --allow-generic-eval. Do not compare with previous iLLaDA OpenCompass results unless a policy is added.",
            "custom_native": True,
        }
    raise ValueError(
        f"No explicit eval policy registered for benchmark `{benchmark}`. "
        "To keep scores consistent with previous iLLaDA runs, add an EVAL_POLICY entry "
        "and a scorer, or rerun with --allow-generic-eval for exploratory debugging only."
    )


def print_eval_policy() -> None:
    print("Native output evaluation policy:")
    print("Invariant: a benchmark name maps to exactly one evaluator for all models/runs.")
    for bench, policy in EVAL_POLICY.items():
        print(f"- {bench}: {policy['backend']}")
        print(f"  compatibility: {policy['compatibility']}")
        print(f"  metric: {policy['metric']}")
        print(f"  source: {policy['source']}")


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


def collect_native_experiments(config: Dict[str, Any], selected: Set[str]) -> List[Dict[str, Any]]:
    """Collect experiments for native scripts.

    Unlike the older run_test helper, an explicit --only selection should be
    able to run any task/experiment present in test_config_native.yaml even when
    run.tasks is set to a safe default such as [context_double_8k_tps].  When
    --only is omitted, run.tasks/run.experiments still act as the default plan.
    """
    selected = {str(x) for x in (selected or set()) if str(x)}
    explicit_selection = bool(selected)
    tasks = config.get("tasks", {}) or {}
    run_cfg = config.get("run", {}) or {}
    run_tasks = {str(x) for x in as_list(run_cfg.get("tasks"))}
    run_exps = {str(x) for x in as_list(run_cfg.get("experiments"))}

    out: List[Dict[str, Any]] = []
    for task_name, task_def in tasks.items():
        if not isinstance(task_def, dict):
            continue
        if not explicit_selection:
            if run_tasks and "all" not in run_tasks and str(task_name) not in run_tasks:
                continue
        for exp in task_def.get("experiments", []) or []:
            if not isinstance(exp, dict):
                continue
            exp_name = str(exp.get("name") or "")
            if not explicit_selection:
                if run_exps and "all" not in run_exps and exp_name not in run_exps:
                    continue
            if explicit_selection and "all" not in selected and str(task_name) not in selected and exp_name not in selected:
                continue
            item = deepcopy(exp)
            item.setdefault("task", task_name)
            item.setdefault("task_output_path", task_def.get("output_path"))
            out.append(item)
    return out


def iter_conditions(config: Dict[str, Any], selected: Set[str]) -> Iterable[Dict[str, Any]]:
    defaults = config.get("defaults", {}) or {}
    for experiment in collect_native_experiments(config, selected):
        task = experiment.get("task") or "runs"
        output_task = (
            experiment.get("output_task")
            or experiment.get("output_task_name")
            or experiment.get("output_name")
            or task
        )
        exp_models = set(str(x) for x in as_list(experiment.get("models"))) if experiment.get("models") is not None else None
        for benchmark in as_list(experiment.get("benchmark")):
            for idx, params in enumerate(expand_matrix(experiment), start=1):
                merged = deep_merge(defaults, params)
                yield {
                    "task": task,
                    "output_task": output_task,
                    "experiment": experiment.get("name"),
                    "benchmark": str(benchmark),
                    "params": merged,
                    "condition": condition_name(merged),
                    "condition_index": idx,
                    "experiment_models": exp_models,
                }


def task_dir_name(condition: Dict[str, Any]) -> str:
    return safe_name(str(condition.get("output_task") or condition.get("task") or "runs"))


def final_dir_for(root: Path, model_name: str, condition: Dict[str, Any]) -> Path:
    return root / safe_model_name(model_name) / task_dir_name(condition) / bench_alias(condition["benchmark"]) / condition["condition"]


def compare_dir_for(root: Path, condition: Dict[str, Any]) -> Path:
    return root / "_compare" / task_dir_name(condition) / bench_alias(condition["benchmark"]) / condition["condition"]


def write_model_manifests(output_root: Path, stem: str, rows: List[Dict[str, Any]], write_global: bool = False) -> List[Path]:
    """Write manifests under each model directory.

    Scored artifacts already use native_outputs/<model>/..., so the default
    manifest location should be native_outputs/<model>/<stem>.*.  Aggregate
    manifests are optional and, when requested, go under _manifests/ instead of
    polluting the model root.
    """
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        alias = safe_model_name(str(row.get("model") or "model"))
        by_model.setdefault(alias, []).append(row)

    written: List[Path] = []
    for alias, model_rows in sorted(by_model.items()):
        model_root = output_root / alias
        jsonl_path = model_root / f"{stem}.jsonl"
        csv_path = model_root / f"{stem}.csv"
        write_jsonl(jsonl_path, model_rows)
        write_csv(csv_path, model_rows)
        written.extend([jsonl_path, csv_path])

    if write_global:
        manifest_root = output_root / "_manifests"
        jsonl_path = manifest_root / f"{stem}.jsonl"
        csv_path = manifest_root / f"{stem}.csv"
        write_jsonl(jsonl_path, rows)
        write_csv(csv_path, rows)
        written.extend([jsonl_path, csv_path])

    return written


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


def score_row(row: Dict[str, Any], timeout: int = 8, allow_generic: bool = False) -> Dict[str, Any]:
    bench = str(row.get("benchmark"))
    policy = eval_policy_for(bench, allow_generic=allow_generic)
    if bench == "gsm8k":
        result = score_gsm8k(row)
    elif bench == "mbpp":
        result = score_mbpp(row, timeout=timeout)
    elif bench == "ruler_niah_single_1":
        result = score_ruler_single(row)
    elif bench in {"ruler_niah_order_2", "ruler_niah_double_2"}:
        result = score_ruler_order2(row)
    elif allow_generic:
        result = score_generic(row)
    else:
        # Should be unreachable because eval_policy_for already raises, but keep
        # this branch explicit so unknown benchmarks never get silently scored.
        raise ValueError(f"Unsupported benchmark `{bench}` without explicit eval policy.")
    result["eval_backend"] = policy["backend"]
    result["eval_compatibility"] = policy["compatibility"]
    result["eval_metric"] = policy["metric"]
    result["eval_source"] = policy["source"]
    return result


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
        # visual_arness_trace.py remains the rich iLLaDA ARness viewer.  The
        # native plotter is model-agnostic and is the preferred W1 trace view.
        (dst / "visual_command.txt").write_text(f'python visual_arness_trace.py "{dst}" --sample-idx {idx}\n', encoding="utf-8")
        (dst / "native_visual_command.txt").write_text(f'python native_visual.py --run "{dst}"\n', encoding="utf-8")


def evaluate_run(model_output_dir: Path, final_dir: Path, timeout: int = 8, allow_generic: bool = False) -> Dict[str, Any]:
    outputs_path = model_output_dir / "outputs.jsonl"
    if not outputs_path.exists():
        raise FileNotFoundError(f"Missing outputs.jsonl: {outputs_path}")
    outputs = read_jsonl(outputs_path)
    scores: List[Dict[str, Any]] = []
    for row in outputs:
        score = score_row(row, timeout=timeout, allow_generic=allow_generic)
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
    benchmark = str(outputs[0].get("benchmark")) if outputs else "unknown"
    policy = eval_policy_for(benchmark, allow_generic=allow_generic)
    metrics["benchmark"] = benchmark
    metrics["eval_backend"] = policy["backend"]
    metrics["eval_compatibility"] = policy["compatibility"]
    metrics["eval_metric"] = policy["metric"]
    metrics["eval_source"] = policy["source"]
    final_dir.mkdir(parents=True, exist_ok=True)
    copy_artifacts(model_output_dir, final_dir, sample_idx=outputs[0].get("sample_id") if outputs else None)
    write_jsonl(final_dir / "scores.jsonl", scores)
    write_json(final_dir / "metrics.json", metrics)
    write_json(final_dir / "eval_policy.json", policy)
    write_csv(final_dir / "scores.csv", scores)
    return metrics


def model_output_dir_for(root: Path, model_name: str, condition: Dict[str, Any]) -> Path:
    return output_dir_for(root, model_name, condition)


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize and evaluate native model_outputs into native_outputs.")
    parser.add_argument("--config", default="test_config.yaml")
    parser.add_argument("--only", nargs="*", default=[], help="Task or experiment names to evaluate.")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--model-output-root", default="model_outputs")
    parser.add_argument("--output-root", default="native_outputs")
    parser.add_argument("--timeout", type=int, default=8, help="MBPP subprocess timeout per sample.")
    parser.add_argument("--allow-generic-eval", action="store_true", help="Exploratory only: allow generic substring fallback for unregistered benchmarks. Default is false to keep scores comparable with previous iLLaDA runs.")
    parser.add_argument("--print-eval-policy", action="store_true", help="Print the fixed benchmark -> evaluator mapping used for consistency across old iLLaDA and future runs.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--write-global-manifest",
        action="store_true",
        help=(
            "Also write aggregate manifests under <output-root>/_manifests/. "
            "Default: only per-model manifests under <output-root>/<model>/ are written."
        ),
    )
    args = parser.parse_args()

    if args.print_eval_policy:
        print_eval_policy()
        return 0

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
            if not model_selected(model_cfg, explicit):
                continue
            if not model_selected(model_cfg, exp_models):
                continue
            src = model_output_dir_for(model_root, alias, condition)
            dst = final_dir_for(output_root, alias, condition)
            plan.append((alias, condition, src, dst))

    print(f"Native output/eval plan: {len(plan)} run(s).")
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
        metrics = evaluate_run(src, dst, timeout=args.timeout, allow_generic=args.allow_generic_eval)
        row = {
            "model": alias,
            "task": condition["task"],
            "output_task": condition.get("output_task", condition["task"]),
            "experiment": condition["experiment"],
            "benchmark": condition["benchmark"],
            "condition": condition["condition"],
            "output_dir": str(dst),
            **{k: v for k, v in metrics.items() if not isinstance(v, (dict, list))},
        }
        manifest.append(row)
        compare_dir = compare_dir_for(output_root, condition)
        compare_by_condition.setdefault(compare_dir, []).append(row)

    manifest_paths = write_model_manifests(
        output_root,
        "native_output_manifest",
        manifest,
        write_global=args.write_global_manifest,
    )
    for compare_dir, rows in compare_by_condition.items():
        write_csv(compare_dir / "compare.csv", rows)
    print("Wrote per-model output/eval manifests:")
    for path in manifest_paths:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
