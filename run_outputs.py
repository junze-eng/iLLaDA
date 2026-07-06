#!/usr/bin/env python3
"""Evaluate existing model outputs with OpenCompass and materialize final outputs.

This is the CPU/local half of the split workflow.  It reads infer_manifest.jsonl
created by run_model.py, then runs OpenCompass in eval/viz mode with -r to reuse
already generated predictions.

Input example:
    model_outputs/iLLaDA/arness/...

Final scored artifacts are written back to the old report/visualization layout:
    outputs/arness/...

Typical usage:
    python run_outputs.py --config test_config.yaml --only arness
    python run_outputs.py --root model_outputs/iLLaDA/arness/mbpp_s6
    python run_outputs.py --root iLLaDA/model_outputs/iLLaDA/arness/mbpp_s6
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from run_test import (
    OPENCOMPASS_DIR,
    ROOT,
    as_list,
    collect_experiments,
    compact_experiment_name,
    copy_aliases_for_visualizer,
    load_yaml,
    safe_name,
)

CONTROL_ARGS_WITH_VALUE = {
    "-m",
    "--mode",
    "-w",
    "--work-dir",
    "-r",
    "--reuse",
}
CONTROL_ARGS_NO_VALUE = {
    "--dry-run",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_under_root(path_like: str | Path) -> Path:
    """Resolve paths from either repo root or its parent directory.

    This accepts both ``model_outputs/iLLaDA/...`` when running inside the
    repository and ``iLLaDA/model_outputs/iLLaDA/...`` when running from the
    parent directory.
    """
    path = Path(os.path.expanduser(str(path_like)))
    if path.is_absolute():
        return path

    candidates: List[Path] = []
    if path.parts and path.parts[0] == ROOT.name:
        candidates.append(ROOT.parent / path)
    candidates.extend([Path.cwd() / path, ROOT / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else ROOT / path


def safe_path_segment(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or "model"


def model_alias(model_cfg: Dict[str, Any], override: Optional[str] = None) -> str:
    if override:
        return safe_path_segment(override)
    for key in ("output_name", "output_alias", "model_alias", "name"):
        value = model_cfg.get(key)
        if value:
            return safe_path_segment(value)
    raw = str(
        model_cfg.get("abbr")
        or model_cfg.get("path")
        or model_cfg.get("model_path")
        or model_cfg.get("backend")
        or "model"
    )
    lowered = raw.lower()
    if "w1-4b" in lowered or "w1_4b" in lowered:
        return "W1-4B"
    if "illada" in lowered:
        return "iLLaDA"
    if "llada" in lowered:
        return "LLaDA"
    return safe_path_segment(Path(raw).name if "/" in raw else raw)


def contains_model_outputs_alias(path: Path, model_name: str) -> bool:
    parts = tuple(path.parts)
    for i in range(len(parts) - 1):
        if parts[i] == "model_outputs" and parts[i + 1] == model_name:
            return True
    return False


def append_model_alias_once(raw_root: Path, model_name: str) -> Path:
    """Match run_model.py default without doubling model alias segments."""
    if raw_root.name == model_name or contains_model_outputs_alias(raw_root, model_name):
        return raw_root
    return raw_root / model_name


def experiment_root(output_root: Path, experiment: Dict[str, Any], compact_exp_name: str) -> Path:
    """Return the experiment folder without duplicating task/experiment suffixes.

    This lets both run_model.py and run_outputs.py understand the current
    project layout directly:
        model_outputs/iLLaDA/arness/mbpp_s6
    while still accepting generic roots such as model_outputs or outputs.
    """
    task = safe_name(str(experiment.get("task") or "runs"))
    compact = safe_name(str(compact_exp_name))
    if output_root.name == compact:
        return output_root
    if output_root.name == task:
        return output_root / compact
    if len(output_root.parts) >= 2 and output_root.parts[-2:] == (task, compact):
        return output_root
    return output_root / task / compact


def configured_output_root(config: Optional[Dict[str, Any]], cli_value: Optional[str]) -> Path:
    execution_cfg = (config or {}).get("execution", {}) or {}
    value = cli_value or execution_cfg.get("output_dir") or "outputs"
    return resolve_under_root(str(value))


def configured_model_output_root(config: Dict[str, Any], cli_value: Optional[str]) -> Path:
    execution_cfg = config.get("execution", {}) or {}
    value = (
        cli_value
        or execution_cfg.get("model_output_dir")
        or execution_cfg.get("model_outputs_dir")
        or execution_cfg.get("model_output_root")
        or "model_outputs"
    )
    return resolve_under_root(str(value))


def manifest_search_roots_from_config(
    config_path: Path,
    selected: Set[str],
    model_alias_override: Optional[str],
    root_override: Optional[str],
) -> Tuple[List[Path], Dict[str, Any]]:
    """Build the exact model_outputs roots described by the YAML config.

    Example for the current arness/MBPP trace run:
        model_outputs/iLLaDA/arness/mbpp_s6
    """
    config = load_yaml(config_path)
    model_name = model_alias(config.get("model", {}) or {}, model_alias_override)
    raw_root = configured_model_output_root(config, root_override)
    model_root = raw_root if root_override else append_model_alias_once(raw_root, model_name)

    experiments = collect_experiments(config, selected or set())
    roots: List[Path] = []
    for experiment in experiments:
        benchmark_list = as_list(experiment.get("benchmark"))
        first_benchmark = str(benchmark_list[0]) if benchmark_list else None
        compact_exp_dir = compact_experiment_name(experiment, first_benchmark)
        roots.append(experiment_root(model_root, experiment, compact_exp_dir))

    # Stable de-duplication while preserving config order.
    deduped: List[Path] = []
    seen: Set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            deduped.append(root)
    return deduped, config


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


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_jsonl_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return list(read_jsonl(path))


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\n")


def append_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    fieldnames = list(row.keys())
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


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


def json_default(obj: Any) -> Any:
    try:
        return obj.tolist()
    except Exception:
        return str(obj)


def find_manifests(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob("infer_manifest.jsonl"))


def build_env() -> Dict[str, str]:
    env = os.environ.copy()
    pythonpath = [str(ROOT), str(OPENCOMPASS_DIR)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return env


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





def _find_opencompass_configs_dir() -> Optional[Path]:
    """Return the real OpenCompass configs directory if this checkout has one.

    Do not assume that ``mbpp_gen.py`` exists. Some OpenCompass versions use
    hashed/deprecated MBPP config names, and some pip/editable installs do not
    ship the configs tree at all. This helper only returns a configs root when
    it can prove that an MBPP dataset config is available under it.
    """
    candidates = [
        OPENCOMPASS_DIR / "configs",
        OPENCOMPASS_DIR / "opencompass" / "configs",
        ROOT / "configs",
    ]
    probes = [
        Path("datasets") / "mbpp" / "mbpp_gen.py",
        Path("datasets") / "mbpp" / "mbpp_gen_1e1056.py",
        Path("datasets") / "mbpp" / "deprecated_mbpp_gen_1e1056.py",
        Path("datasets") / "mbpp" / "deprecated_mbpp_passk_gen_1e1056.py",
        Path("datasets") / "mbpp" / "deprecated_mbpp_repeat10_gen_1e1056.py",
    ]
    for candidate in candidates:
        if any((candidate / p).exists() for p in probes):
            return candidate.resolve()

    try:
        for hit in OPENCOMPASS_DIR.rglob("*mbpp*gen*.py"):
            parts = list(hit.resolve().parts)
            for i, part in enumerate(parts):
                if part == "configs" and "datasets" in parts[i + 1 :] and "mbpp" in parts[i + 1 :]:
                    return Path(*parts[: i + 1]).resolve()
    except Exception:
        pass
    return None


def _mbpp_gen_fallback_text() -> str:
    """A minimal OpenCompass MBPP config compatible with run_test imports."""
    return """
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import MBPPDataset, MBPPEvaluator

mbpp_reader_cfg = dict(input_columns=['text', 'test_list'], output_column='test_list_2')

mbpp_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='You are an expert Python programmer, and here is your task: Write a function to find the similar elements from the given two tuple lists.\\nYour code should pass these tests:\\n\\n assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)\\n assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4) \\n assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14) \\n'),
                dict(
                    role='BOT',
                    prompt=\"[BEGIN]\\n'def similar_elements(test_tup1, test_tup2):\\\\n    res = tuple(set(test_tup1) & set(test_tup2))\\\\n    return (res)'\\n[DONE]\\n\\n\"),
                dict(
                    role='HUMAN',
                    prompt='You are an expert Python programmer, and here is your task: Write a python function to identify non-prime numbers.\\nYour code should pass these tests:\\n\\n assert is_not_prime(2) == False \\n assert is_not_prime(10) == True \\n assert is_not_prime(35) == True \\n'),
                dict(
                    role='BOT',
                    prompt=\"[BEGIN]\\n'import math\\\\ndef is_not_prime(n):\\\\n    result = False\\\\n    for i in range(2,int(math.sqrt(n)) + 1):\\\\n        if n % i == 0:\\\\n            result = True\\\\n    return result'\\n[DONE]\\n\\n\"),
                dict(
                    role='HUMAN',
                    prompt='You are an expert Python programmer, and here is your task: Write a function to find the largest integers from a given list of numbers using heap queue algorithm.\\nYour code should pass these tests:\\n\\n assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] \\n assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] \\n assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35] \\n'),
                dict(
                    role='BOT',
                    prompt=\"[BEGIN]\\n'import heapq as hq\\\\ndef heap_queue_largest(nums,n):\\\\n    largest_nums = hq.nlargest(n, nums)\\\\n    return largest_nums'\\n[DONE]\\n\\n\"),
                dict(
                    role='HUMAN',
                    prompt='You are an expert Python programmer, and here is your task: {text} Your code should pass these tests:\\n\\n {test_list} \\n'),
                dict(role='BOT', prompt='[BEGIN]\\n'),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512),
)

mbpp_eval_cfg = dict(evaluator=dict(type=MBPPEvaluator), pred_role='BOT')

mbpp_datasets = [
    dict(
        type=MBPPDataset,
        abbr='mbpp',
        path='./data/mbpp/mbpp.jsonl',
        reader_cfg=mbpp_reader_cfg,
        infer_cfg=mbpp_infer_cfg,
        eval_cfg=mbpp_eval_cfg,
    )
]
""".lstrip()


def _write_mbpp_config_fallback(configs_root: Path) -> None:
    mbpp_dir = configs_root / "datasets" / "mbpp"
    mbpp_dir.mkdir(parents=True, exist_ok=True)
    for pkg_dir in [configs_root, configs_root / "datasets", mbpp_dir]:
        init = pkg_dir / "__init__.py"
        if not init.exists():
            init.write_text("", encoding="utf-8")
    target = mbpp_dir / "mbpp_gen.py"
    if not target.exists():
        target.write_text(_mbpp_gen_fallback_text(), encoding="utf-8")


def ensure_opencompass_configs_available() -> None:
    """Make ``opencompass.configs.datasets.mbpp.mbpp_gen`` resolvable.

    Keep the run_test config style (``with read_base()`` and
    ``from opencompass.configs...``). If the local/pip OpenCompass checkout does
    not ship configs, create the minimal MBPP config at the package locations
    MMEngine actually checks.
    """
    probe_rel = Path("datasets") / "mbpp" / "mbpp_gen.py"
    src = _find_opencompass_configs_dir()

    candidates: Set[Path] = set()
    candidates.add(OPENCOMPASS_DIR / "opencompass" / "configs")
    candidates.add(OPENCOMPASS_DIR / "configs")

    try:
        import site
        site_paths = [site.getusersitepackages()]
        try:
            site_paths.extend(site.getsitepackages())
        except Exception:
            pass
        for base in site_paths:
            if base:
                candidates.add(Path(base) / "opencompass" / "configs")
    except Exception:
        pass

    try:
        import importlib.util
        spec = importlib.util.find_spec("opencompass")
        if spec and spec.origin:
            candidates.add(Path(spec.origin).resolve().parent / "configs")
    except Exception:
        pass

    errors: List[str] = []
    for dst in sorted(candidates, key=lambda p: str(p).lower()):
        try:
            if src is not None and src.exists():
                try:
                    same = dst.resolve() == src.resolve()
                except Exception:
                    same = False
                if not same:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(src, dst, dirs_exist_ok=True)
            _write_mbpp_config_fallback(dst)
        except Exception as exc:
            errors.append(f"{dst}: {exc}")

    ready = [dst for dst in candidates if (dst / probe_rel).exists()]
    if not ready:
        raise RuntimeError(
            "Failed to expose/create OpenCompass MBPP config. "
            f"OPENCOMPASS_DIR={OPENCOMPASS_DIR}; src={src}; errors={' ; '.join(errors)}"
        )

def localize_opencompass_config_text(text: str) -> str:
    """Keep run_test-style OpenCompass config imports unchanged.

    The generated configs use ``with read_base(): from opencompass.configs...``.
    That is the format MMEngine expects.  We make the package-data path visible
    with ensure_opencompass_configs_available() instead of rewriting imports.
    """
    return text


def localize_opencompass_config_file(config_path: Path, work_dir: Path) -> Path:
    ensure_opencompass_configs_available()
    return config_path

def resolve_from_manifest(record: Dict[str, Any], manifest_path: Path, key: str) -> Path:
    """Resolve an absolute path, falling back to *_rel under manifest directory.

    This keeps the copied model_outputs tree portable after moving it from GPU
    to a local/CPU machine.
    """
    raw = record.get(key)
    if raw:
        path = Path(raw)
        if path.exists():
            return path
    rel_key = f"{key}_rel"
    rel = record.get(rel_key)
    if rel:
        candidate = manifest_path.parent / rel
        if candidate.exists() or key in {"work_dir", "config", "outputs_jsonl", "summary_jsonl", "trace_jsonl"}:
            return candidate
    if raw:
        return Path(raw)
    raise KeyError(f"Cannot resolve `{key}` from manifest record: {record.get('run_name')}")


def resolve_artifact_path(record: Dict[str, Any], manifest_path: Path, key: str, default: Path) -> Path:
    raw = record.get(key)
    if raw:
        path = Path(raw)
        if path.exists():
            return path
    rel = record.get(f"{key}_rel")
    if rel:
        return manifest_path.parent / rel
    if raw:
        return Path(raw)
    return default


def replace_leading_root(path_like: Any, output_root: Path) -> Path:
    path = Path(str(path_like))
    if path.is_absolute():
        try:
            rel = path.relative_to(ROOT / "outputs")
            return output_root / rel
        except ValueError:
            return path
    parts = path.parts
    if parts and parts[0] == "outputs":
        return output_root / Path(*parts[1:])
    return ROOT / path


def resolve_final_dir(record: Dict[str, Any], output_root: Path) -> Path:
    raw = record.get("final_work_dir")
    if raw:
        return replace_leading_root(raw, output_root)
    task = safe_name(str(record.get("task") or "runs"))
    compact = safe_name(str(record.get("compact_experiment") or record.get("experiment") or "experiment"))
    run_label = safe_name(str(record.get("run_label") or record.get("run_name") or "run"))
    return output_root / task / compact / run_label


def timestamp_exists(work_dir: Path, timestamp: str) -> bool:
    return (work_dir / timestamp).exists()


def latest_timestamp(work_dir: Path) -> Optional[str]:
    if not work_dir.exists():
        return None
    candidates = []
    for child in work_dir.iterdir():
        if child.is_dir() and (child / "configs").exists():
            candidates.append(child.name)
    return sorted(candidates)[-1] if candidates else None


def read_jsonl_count(path: Path) -> int:
    if not path.exists() or path.is_dir():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def mode_sequence(mode: str) -> List[str]:
    if mode == "both":
        return ["eval", "viz"]
    return [mode]


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists() and src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def copy_tree_if_exists(src: Path, dst: Path) -> None:
    if src.exists() and src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)


# ---------------------------------------------------------------------------
# OpenCompass result/detail extraction
# ---------------------------------------------------------------------------

DETAIL_KEYS = {
    "details",
    "detail",
    "samples",
    "sample_details",
    "predictions",
    "prediction",
    "preds",
    "outputs",
    "results",
    "data",
}
SAMPLE_HINT_KEYS = {
    "prediction",
    "pred",
    "output",
    "response",
    "generated",
    "answer",
    "gold",
    "target",
    "reference",
    "correct",
    "is_correct",
    "score",
    "accuracy",
    "passed",
}
METRIC_SKIP_KEYS = {
    "details",
    "detail",
    "samples",
    "sample_details",
    "predictions",
    "preds",
    "outputs",
    "data",
}


def load_structured_file(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl_list(path)
    if suffix == ".json":
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            return json.loads(path.read_text())
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as f:
            return pickle.load(f)
    return None


def is_sample_record(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    if not (set(obj.keys()) & SAMPLE_HINT_KEYS):
        return False
    # Avoid treating pure metric dictionaries as sample records.
    if len(obj) <= 5 and any(k in obj for k in ("accuracy", "score")) and not any(
        k in obj for k in ("prediction", "pred", "output", "answer", "gold", "target", "correct", "is_correct")
    ):
        return False
    return True


def extract_sample_records(obj: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen_lists: Set[int] = set()

    def visit(node: Any) -> None:
        if isinstance(node, list):
            node_id = id(node)
            if node_id in seen_lists:
                return
            seen_lists.add(node_id)
            if node and all(isinstance(x, dict) for x in node) and any(is_sample_record(x) for x in node):
                rows.extend([x for x in node if isinstance(x, dict)])
                return
            for item in node:
                visit(item)
            return
        if isinstance(node, dict):
            if is_sample_record(node):
                rows.append(node)
                return
            for key, value in node.items():
                if key in DETAIL_KEYS or isinstance(value, (list, dict)):
                    visit(value)

    visit(obj)
    return rows


def collect_numeric_metrics(obj: Any, prefix: str = "") -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in METRIC_SKIP_KEYS:
                continue
            new_key = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                metrics[new_key] = value
            elif isinstance(value, dict):
                metrics.update(collect_numeric_metrics(value, new_key))
    return metrics


def collect_opencompass_details(timestamp_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[str]]:
    rows: List[Dict[str, Any]] = []
    metrics: Dict[str, Any] = {}
    sources: List[str] = []
    search_dirs = [timestamp_dir / "results", timestamp_dir / "predictions"]
    for base in search_dirs:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix.lower() not in {".json", ".jsonl", ".pkl", ".pickle"}:
                continue
            try:
                obj = load_structured_file(path)
            except Exception as exc:
                print(f"[warn] cannot read OpenCompass artifact {path}: {exc}")
                continue
            file_rows = extract_sample_records(obj)
            if file_rows:
                for row in file_rows:
                    item = dict(row)
                    item.setdefault("_opencompass_source", str(path))
                    rows.append(item)
                sources.append(str(path))
            if path.is_relative_to(timestamp_dir / "results") if hasattr(path, "is_relative_to") else str(path).startswith(str(timestamp_dir / "results")):
                metrics.update(collect_numeric_metrics(obj, path.stem))
    return dedupe_rows(rows), metrics, sources


def dedupe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for row in rows:
        key_obj = {
            k: row.get(k)
            for k in ("sample_idx", "sample_id", "idx", "index", "prediction", "pred", "output", "answer", "gold", "correct", "score")
            if k in row
        }
        key = json.dumps(key_obj, sort_keys=True, ensure_ascii=False, default=json_default)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def first_present(row: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def to_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "correct", "yes", "pass", "passed", "1"}:
            return True
        if lowered in {"false", "incorrect", "wrong", "no", "fail", "failed", "0"}:
            return False
    return None


def normalize_eval_record(row: Dict[str, Any]) -> Dict[str, Any]:
    score = first_present(row, ["score", "acc", "accuracy", "em", "exact_match", "pass", "passed"])
    correct = first_present(row, ["correct", "is_correct", "passed", "pass", "accuracy", "acc"])
    correct_bool = to_bool(correct)
    if correct_bool is None and isinstance(score, (int, float)) and score in (0, 1):
        correct_bool = bool(score)
    normalized: Dict[str, Any] = {}
    if correct_bool is not None:
        normalized["correct"] = correct_bool
    if isinstance(score, bool):
        normalized["score"] = 1.0 if score else 0.0
    elif isinstance(score, (int, float)):
        normalized["score"] = float(score)
    elif correct_bool is not None:
        normalized["score"] = 1.0 if correct_bool else 0.0

    prediction = first_present(row, ["prediction", "pred", "output", "response", "generated", "infer_output"])
    gold = first_present(row, ["answer", "gold", "target", "reference", "label", "gt"])
    if prediction is not None:
        normalized["prediction"] = prediction
    if gold is not None:
        normalized["gold"] = gold

    # Preserve useful task-specific detail fields without making outputs too large.
    for key in [
        "sample_idx",
        "sample_id",
        "idx",
        "index",
        "metric",
        "dataset",
        "abbr",
        "_opencompass_source",
    ]:
        if key in row and key not in normalized:
            normalized[key] = row[key]
    return normalized


def sample_id_candidates(row: Dict[str, Any], fallback_index: int) -> List[Any]:
    values: List[Any] = []
    for key in ("sample_idx", "sample_id", "idx", "index", "id", "origin_idx"):
        if key in row and row[key] is not None:
            values.append(row[key])
            try:
                values.append(int(row[key]))
            except Exception:
                pass
    values.append(fallback_index)
    return values


def merge_eval_rows(raw_rows: List[Dict[str, Any]], eval_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_id: Dict[Any, Dict[str, Any]] = {}
    for idx, row in enumerate(eval_rows):
        for value in sample_id_candidates(row, idx):
            by_id.setdefault(value, row)

    scored_outputs: List[Dict[str, Any]] = []
    scores: List[Dict[str, Any]] = []
    for idx, raw in enumerate(raw_rows):
        raw = deepcopy(raw)
        match = None
        for value in sample_id_candidates(raw, idx):
            if value in by_id:
                match = by_id[value]
                break
        if match is None and idx < len(eval_rows):
            match = eval_rows[idx]
        eval_info = normalize_eval_record(match) if match else {}
        if eval_info:
            raw["eval"] = eval_info
            for key in ("correct", "score", "gold"):
                if key in eval_info:
                    raw[key] = eval_info[key]
            if "prediction" in eval_info and not raw.get("prediction"):
                raw["prediction"] = eval_info["prediction"]
        score_row = {
            "sample_idx": raw.get("sample_idx", raw.get("sample_id", idx)),
            "correct": raw.get("correct"),
            "score": raw.get("score"),
            "gold": raw.get("gold"),
            "prediction": raw.get("prediction"),
        }
        if eval_info.get("_opencompass_source"):
            score_row["opencompass_source"] = eval_info["_opencompass_source"]
        scores.append(score_row)
        scored_outputs.append(raw)
    return scored_outputs, scores


def aggregate_scores(scores: List[Dict[str, Any]], official_metrics: Dict[str, Any]) -> Dict[str, Any]:
    numeric_scores: List[float] = []
    correct_values: List[float] = []
    for row in scores:
        if isinstance(row.get("score"), (int, float)):
            numeric_scores.append(float(row["score"]))
        if isinstance(row.get("correct"), bool):
            correct_values.append(1.0 if row["correct"] else 0.0)
    metrics: Dict[str, Any] = {
        "n": len(scores),
        "scored_n": len(numeric_scores) or len(correct_values),
    }
    if numeric_scores:
        mean_score = sum(numeric_scores) / len(numeric_scores)
        metrics["score"] = round(mean_score * 100 if mean_score <= 1.0 else mean_score, 4)
    if correct_values:
        metrics["accuracy"] = round(sum(correct_values) / len(correct_values) * 100, 4)
    if official_metrics:
        metrics["opencompass_metrics"] = official_metrics
    return metrics


def materialize_evaluated_outputs(
    record: Dict[str, Any],
    manifest_path: Path,
    output_root: Path,
    reuse_timestamp: str,
    force: bool = True,
) -> Dict[str, Any]:
    work_dir = resolve_from_manifest(record, manifest_path, "work_dir")
    outputs_jsonl = resolve_artifact_path(record, manifest_path, "outputs_jsonl", work_dir / "outputs.jsonl")
    summary_jsonl = resolve_artifact_path(record, manifest_path, "summary_jsonl", work_dir / "summary.jsonl")
    trace_jsonl = resolve_artifact_path(record, manifest_path, "trace_jsonl", work_dir / "trace.jsonl")
    final_dir = resolve_final_dir(record, output_root)
    timestamp_dir = work_dir / reuse_timestamp

    if not outputs_jsonl.exists():
        raise FileNotFoundError(f"Missing raw outputs_jsonl: {outputs_jsonl}")
    raw_rows = read_jsonl_list(outputs_jsonl)
    eval_rows, official_metrics, detail_sources = collect_opencompass_details(timestamp_dir)
    if not eval_rows and not official_metrics:
        raise FileNotFoundError(
            f"No OpenCompass eval artifacts found under {timestamp_dir / 'results'} "
            f"or {timestamp_dir / 'predictions'}"
        )
    scored_outputs, scores = merge_eval_rows(raw_rows, eval_rows)
    metrics = aggregate_scores(scores, official_metrics)

    final_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for name in ["outputs.jsonl", "scores.jsonl", "scores.csv", "metrics.json", "eval_manifest.json"]:
            path = final_dir / name
            if path.exists():
                path.unlink()

    copy_if_exists(outputs_jsonl, final_dir / "raw_outputs.jsonl")
    copy_if_exists(summary_jsonl, final_dir / "summary.jsonl")
    copy_if_exists(trace_jsonl, final_dir / "trace.jsonl")
    for name in ["gpu.csv", "run.json", "oc_config.py", "visual_command.txt"]:
        copy_if_exists(work_dir / name, final_dir / name)
    copy_tree_if_exists(work_dir / "sample_traces", final_dir / "sample_traces")
    copy_tree_if_exists(timestamp_dir / "results", final_dir / "opencompass_results")
    copy_tree_if_exists(timestamp_dir / "predictions", final_dir / "opencompass_predictions")

    write_jsonl(final_dir / "outputs.jsonl", scored_outputs)
    write_jsonl(final_dir / "scores.jsonl", scores)
    write_csv(final_dir / "scores.csv", scores)
    write_json(final_dir / "metrics.json", metrics)

    if trace_jsonl.exists():
        sample_idx = 0
        if raw_rows:
            try:
                sample_idx = int(raw_rows[0].get("sample_idx", raw_rows[0].get("sample_id", 0)) or 0)
            except Exception:
                sample_idx = 0
        (final_dir / "visual_command.txt").write_text(
            f'python visual_arness_trace.py "{final_dir}" --sample-idx {sample_idx}\n',
            encoding="utf-8",
        )

    materialized = {
        "created_at": utc_now(),
        "model_name": record.get("model_name"),
        "task": record.get("task"),
        "experiment": record.get("experiment"),
        "benchmark": record.get("benchmark"),
        "run_label": record.get("run_label"),
        "run_name": record.get("run_name"),
        "reuse_timestamp": reuse_timestamp,
        "model_output_dir": str(work_dir),
        "raw_outputs_jsonl": str(outputs_jsonl),
        "final_output_dir": str(final_dir),
        "outputs_jsonl": str(final_dir / "outputs.jsonl"),
        "metrics_json": str(final_dir / "metrics.json"),
        "raw_n": len(raw_rows),
        "eval_detail_n": len(eval_rows),
        "detail_sources": detail_sources,
        "metrics": metrics,
    }
    write_json(final_dir / "eval_manifest.json", materialized)
    append_jsonl(output_root / "eval_manifest.jsonl", materialized)
    append_csv(
        output_root / "eval_manifest.csv",
        {
            "created_at": materialized["created_at"],
            "model_name": materialized.get("model_name") or "",
            "task": materialized.get("task") or "",
            "experiment": materialized.get("experiment") or "",
            "benchmark": materialized.get("benchmark") or "",
            "run_name": materialized.get("run_name") or "",
            "reuse_timestamp": reuse_timestamp,
            "raw_n": len(raw_rows),
            "eval_detail_n": len(eval_rows),
            "score": metrics.get("score", ""),
            "accuracy": metrics.get("accuracy", ""),
            "final_output_dir": str(final_dir),
        },
    )
    return materialized


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate saved model outputs with OpenCompass reuse mode and write final scored outputs/."
    )
    parser.add_argument(
        "--config",
        default="test_config.yaml",
        help=(
            "Optional YAML experiment config. When available and --root is omitted, "
            "run_outputs scans the matching model_outputs/<model>/<task>/<compact_experiment> "
            "folders from the config, e.g. model_outputs/iLLaDA/arness/mbpp_s6."
        ),
    )
    parser.add_argument(
        "--root",
        default=None,
        help=(
            "model_outputs root, an experiment folder, or an infer_manifest.jsonl file. "
            "If omitted, infer from --config; if the config is missing, falls back to model_outputs."
        ),
    )
    parser.add_argument(
        "--model-alias",
        default=None,
        help="Folder name under model_outputs used when deriving roots from --config, e.g. iLLaDA.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Final scored output root. Default: config execution.output_dir or outputs.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional filter by run_name, experiment name, task name, benchmark, or model_name.",
    )
    parser.add_argument(
        "--mode",
        choices=["eval", "viz", "both"],
        default="eval",
        help="OpenCompass mode. `eval` writes results; `both` runs eval then viz.",
    )
    parser.add_argument(
        "--reuse",
        default=None,
        help="Override OpenCompass reuse timestamp. Default: timestamp recorded by run_model.py.",
    )
    parser.add_argument(
        "--extra-opencompass-args",
        nargs="*",
        default=None,
        help="Extra args passed to OpenCompass after removing mode/work-dir/reuse controls.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even if a previous eval_manifest.jsonl entry exists for the same run/mode/timestamp.",
    )
    parser.add_argument(
        "--no-materialize",
        action="store_true",
        help="Only run OpenCompass eval/viz; do not copy scored artifacts to outputs/.",
    )
    args = parser.parse_args()

    selected = set(args.only or [])
    config: Optional[Dict[str, Any]] = None
    config_path: Optional[Path] = None

    candidate_config = resolve_under_root(args.config) if args.config else None
    if candidate_config is not None and candidate_config.exists():
        config_path = candidate_config
        config = load_yaml(config_path)

    if args.root:
        search_roots = [resolve_under_root(args.root)]
    elif config_path is not None:
        search_roots, config = manifest_search_roots_from_config(
            config_path,
            selected,
            args.model_alias,
            root_override=None,
        )
    else:
        search_roots = [resolve_under_root("model_outputs")]

    output_root = configured_output_root(config, args.output_root)

    manifests: List[Path] = []
    for root in search_roots:
        found = find_manifests(root)
        if found:
            print(f"[scan] {root} -> {len(found)} manifest(s)")
        else:
            print(f"[scan] {root} -> 0 manifest(s)")
        manifests.extend(found)

    # If the config-derived exact roots are stale/missing, fall back to the broader
    # model_outputs tree so copied runs can still be evaluated without hand editing.
    if not manifests and not args.root:
        fallback_root = configured_model_output_root(config or {}, None) if config else resolve_under_root("model_outputs")
        if config:
            model_name = model_alias(config.get("model", {}) or {}, args.model_alias)
            fallback_root = append_model_alias_once(fallback_root, model_name)
        manifests = find_manifests(fallback_root)
        if manifests:
            print(f"[scan:fallback] {fallback_root} -> {len(manifests)} manifest(s)")

    # Stable de-duplication.
    deduped_manifests: List[Path] = []
    seen_manifests: Set[str] = set()
    for manifest in manifests:
        key = str(manifest.resolve()) if manifest.exists() else str(manifest)
        if key not in seen_manifests:
            seen_manifests.add(key)
            deduped_manifests.append(manifest)
    manifests = deduped_manifests

    if not manifests:
        searched = ", ".join(str(x) for x in search_roots)
        raise SystemExit(f"No infer_manifest.jsonl found under: {searched}")

    ensure_opencompass_configs_available()
    env = build_env()
    extra_args = strip_opencompass_control_args(args.extra_opencompass_args or [])
    total = 0
    materialized_total = 0

    for manifest_path in manifests:
        exp_root = manifest_path.parent
        eval_manifest = exp_root / "eval_manifest.jsonl"
        eval_csv = exp_root / "eval_runs.csv"
        done_keys: Set[str] = set()
        if eval_manifest.exists() and not args.force:
            for item in read_jsonl(eval_manifest):
                # Only a successful OpenCompass subprocess should be treated as done.
                # Older versions recorded failed attempts in eval_manifest.jsonl before
                # returning non-zero, which made later runs print ``already evaluated``
                # and materialize stale/unscored folders without rerunning eval.
                rc = item.get("returncode")
                try:
                    rc_ok = int(rc) == 0
                except Exception:
                    rc_ok = False
                if not rc_ok:
                    continue
                done_keys.add(
                    f"{item.get('run_name')}|{item.get('mode')}|{item.get('reuse_timestamp')}"
                )

        for record in read_jsonl(manifest_path):
            run_name = record.get("run_name")
            experiment = record.get("experiment")
            task = record.get("task")
            benchmark = record.get("benchmark")
            model_name = record.get("model_name")
            if selected and not ({run_name, experiment, task, benchmark, model_name} & selected):
                continue
            if record.get("returncode") not in (0, None):
                print(
                    f"[skip:{run_name}] inference returncode={record.get('returncode')}",
                    flush=True,
                )
                continue

            config_path = resolve_from_manifest(record, manifest_path, "config")
            work_dir = resolve_from_manifest(record, manifest_path, "work_dir")
            config_path = localize_opencompass_config_file(config_path, work_dir)
            outputs_jsonl = resolve_artifact_path(record, manifest_path, "outputs_jsonl", work_dir / "outputs.jsonl")
            summary_jsonl = resolve_artifact_path(record, manifest_path, "summary_jsonl", work_dir / "summary.jsonl")
            trace_jsonl = resolve_artifact_path(record, manifest_path, "trace_jsonl", work_dir / "trace.jsonl")

            reuse_timestamp = args.reuse or record.get("opencompass_reuse_timestamp")
            if not reuse_timestamp:
                reuse_timestamp = latest_timestamp(work_dir)
            if not reuse_timestamp:
                raise SystemExit(f"No reusable OpenCompass timestamp found for {run_name}.")
            if not timestamp_exists(work_dir, reuse_timestamp):
                raise SystemExit(
                    f"Reuse timestamp does not exist for {run_name}: "
                    f"{work_dir / reuse_timestamp}"
                )

            last_returncode: Optional[int] = None
            last_elapsed: Optional[float] = None
            ran_eval_like_mode = False
            final_output_dir: Optional[Path] = None

            for oc_mode in mode_sequence(args.mode):
                done_key = f"{run_name}|{oc_mode}|{reuse_timestamp}"
                if done_key in done_keys:
                    print(f"[skip:{run_name}] already evaluated mode={oc_mode} reuse={reuse_timestamp}")
                    if oc_mode == "eval":
                        ran_eval_like_mode = True
                    continue

                total += 1
                command = [
                    sys.executable,
                    str(OPENCOMPASS_DIR / "run.py"),
                    str(config_path),
                    "-w",
                    str(work_dir),
                    "-m",
                    oc_mode,
                    "-r",
                    str(reuse_timestamp),
                ]
                command.extend(extra_args)

                print(f"\n[{oc_mode}:{run_name}] model: {model_name or 'unknown'}")
                print(f"[{oc_mode}:{run_name}] work_dir: {work_dir}")
                print(f"[{oc_mode}:{run_name}] reuse: {reuse_timestamp}")
                if args.dry_run:
                    print("[dry-run] " + " ".join(command))
                    returncode = None
                    elapsed = None
                else:
                    started = time.perf_counter()
                    returncode = run_command(command, OPENCOMPASS_DIR, env)
                    elapsed = round(time.perf_counter() - started, 3)
                    copy_aliases_for_visualizer(outputs_jsonl, summary_jsonl)

                last_returncode = returncode
                last_elapsed = elapsed
                if oc_mode == "eval":
                    ran_eval_like_mode = True

                eval_record = {
                    "created_at": utc_now(),
                    "mode": oc_mode,
                    "dry_run": args.dry_run,
                    "model_name": model_name,
                    "task": task,
                    "experiment": experiment,
                    "benchmark": benchmark,
                    "run_label": record.get("run_label"),
                    "run_name": run_name,
                    "config": str(config_path),
                    "work_dir": str(work_dir),
                    "outputs_jsonl": str(outputs_jsonl),
                    "summary_jsonl": str(summary_jsonl),
                    "trace_jsonl": str(trace_jsonl),
                    "visual_command": record.get("visual_command"),
                    "reuse_timestamp": reuse_timestamp,
                    "returncode": returncode,
                    "elapsed_seconds": elapsed,
                    "num_output_records": read_jsonl_count(outputs_jsonl),
                    "num_summary_records": read_jsonl_count(summary_jsonl),
                    "trace_exists": trace_jsonl.exists(),
                    "command": command,
                }
                append_jsonl(eval_manifest, eval_record)
                append_csv(
                    eval_csv,
                    {
                        "created_at": eval_record["created_at"],
                        "mode": oc_mode,
                        "model_name": model_name or "",
                        "run_name": run_name,
                        "benchmark": benchmark,
                        "reuse_timestamp": reuse_timestamp,
                        "returncode": "" if returncode is None else returncode,
                        "elapsed_seconds": "" if elapsed is None else elapsed,
                        "work_dir": str(work_dir),
                    },
                )
                if returncode not in (0, None):
                    return returncode

            if (
                not args.dry_run
                and not args.no_materialize
                and (ran_eval_like_mode or args.mode in {"eval", "both", "viz"})
                and last_returncode in (0, None)
            ):
                try:
                    materialized = materialize_evaluated_outputs(
                        record,
                        manifest_path,
                        output_root,
                        str(reuse_timestamp),
                        force=True,
                    )
                    materialized_total += 1
                    final_output_dir = Path(materialized["final_output_dir"])
                    print(f"[write:{run_name}] final scored outputs: {final_output_dir}")
                except Exception as exc:
                    print(f"[warn:{run_name}] failed to materialize scored outputs: {exc}")
                    if args.force:
                        raise

    print(f"\nEvaluated {total} OpenCompass run(s).")
    if not args.no_materialize:
        print(f"Materialized {materialized_total} final output folder(s) under: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
