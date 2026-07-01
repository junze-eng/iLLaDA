import argparse
import csv
import importlib
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from run_test import (
    ROOT,
    BENCHMARKS,
    as_list,
    collect_experiments,
    deep_merge,
    expand_matrix,
    load_yaml,
    ruler_prepared_condition_id,
    safe_name,
)

OPENCOMPASS_DIR = ROOT / "opencompass"
DEPTH_BY_POSITION = {"front": [0], "middle": [50], "back": [100], "end": [100]}
DATASET_META_KEYS = {"type", "abbr", "reader_cfg", "infer_cfg", "eval_cfg"}


def log(msg: str) -> None:
    print(msg, flush=True)


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["status"]
        rows = [{"status": "empty"}]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def dataset_size(dataset) -> Dict[str, Any]:
    if hasattr(dataset, "keys"):
        sizes: Dict[str, Any] = {}
        total = 0
        for split in dataset.keys():
            try:
                size = len(dataset[split])
            except Exception:
                size = None
            sizes[f"{split}_size"] = size
            if isinstance(size, int):
                total += size
        sizes["total_size"] = total if total else None
        return sizes
    try:
        return {"total_size": len(dataset)}
    except Exception:
        return {"total_size": None}


def benchmark_conditions(config: Dict[str, Any], selected: Set[str]) -> List[Dict[str, Any]]:
    defaults = config.get("defaults", {}) or {}
    rows = []
    for experiment in collect_experiments(config, selected):
        for benchmark in as_list(experiment.get("benchmark")):
            if not benchmark:
                continue
            for params in expand_matrix(experiment):
                rows.append({
                    "experiment": experiment.get("name"),
                    "task": experiment.get("task"),
                    "benchmark": benchmark,
                    "params": deep_merge(defaults, params),
                })
    return rows


def preflight_opencompass_benchmark(benchmark: str) -> Dict[str, Any]:
    if benchmark not in BENCHMARKS:
        return {"benchmark": benchmark, "status": "failed", "error": f"Unknown benchmark `{benchmark}`"}
    if str(OPENCOMPASS_DIR) not in sys.path:
        sys.path.insert(0, str(OPENCOMPASS_DIR))
    bench = BENCHMARKS[benchmark]
    try:
        module = importlib.import_module(bench["module"])
        dataset_cfgs = getattr(module, bench["var"])
        loaded = []
        for index, dataset_cfg in enumerate(dataset_cfgs):
            cfg = dict(dataset_cfg)
            dataset_type = cfg.pop("type")
            abbr = cfg.get("abbr", f"{benchmark}_{index}")
            load_kwargs = {key: value for key, value in cfg.items() if key not in DATASET_META_KEYS}
            dataset = dataset_type.load(**load_kwargs)
            loaded.append({
                "abbr": abbr,
                "dataset_type": getattr(dataset_type, "__name__", str(dataset_type)),
                **dataset_size(dataset),
            })
        return {"benchmark": benchmark, "status": "ok", "datasets": json.dumps(loaded, ensure_ascii=False, default=json_default)}
    except Exception as exc:
        return {"benchmark": benchmark, "status": "failed", "error": repr(exc)}


def iter_context_conditions(config: Dict[str, Any], selected: Set[str]) -> Iterable[Dict[str, Any]]:
    defaults = config.get("defaults", {}) or {}
    for experiment in collect_experiments(config, selected):
        if experiment.get("benchmark") != "ruler_niah_single_1":
            continue
        for idx, params in enumerate(expand_matrix(experiment)):
            merged = deep_merge(defaults, params)
            context_length = merged.get("context_length")
            needle_position = merged.get("needle_position")
            if context_length is None or needle_position is None:
                continue
            if str(needle_position) not in DEPTH_BY_POSITION:
                raise SystemExit(f"Unsupported needle_position `{needle_position}`.")
            run_name = safe_name(
                f"{experiment.get('name')}_{idx}_ctx{context_length}_{needle_position}_steps{merged.get('gen_steps')}"
            )
            yield {
                "run_name": run_name,
                "condition_id": ruler_prepared_condition_id(merged),
                "experiment": experiment.get("name"),
                "task": experiment.get("task"),
                "benchmark": "ruler_niah_single_1",
                "params": merged,
            }


def condition_output_path(output_dir: Path, condition: Dict[str, Any]) -> Path:
    return output_dir / "ruler_niah_single_1" / f"{condition['condition_id']}.jsonl"


def summarize_existing_condition(condition: Dict[str, Any], path: Path, model_max_seq_len: int) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    params = condition["params"]
    count = 0
    min_prompt = None
    max_prompt = None
    max_total = None
    truncated = 0
    underfilled = 0
    manifest_rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            count += 1
            apt = row.get("actual_prompt_tokens")
            att = row.get("actual_total_tokens")
            if isinstance(apt, int):
                min_prompt = apt if min_prompt is None else min(min_prompt, apt)
                max_prompt = apt if max_prompt is None else max(max_prompt, apt)
            if isinstance(att, int):
                max_total = att if max_total is None else max(max_total, att)
            truncated += int(bool(row.get("truncated", False)))
            underfilled += int(bool(row.get("underfilled", False)))
            manifest_rows.append({k: v for k, v in row.items() if k not in {"prompt", "answer"}})
    if count == 0:
        return None
    return {
        "condition": condition["condition_id"],
        "run_name": condition["run_name"],
        "experiment": condition["experiment"],
        "experiments_using": ",".join(condition.get("experiments_using", [])),
        "benchmark": condition["benchmark"],
        "prepared_jsonl": str(path),
        "requested_context_length": params["context_length"],
        "needle_position": params["needle_position"],
        "gen_length": int(params.get("gen_length", 128)),
        "gen_steps": params.get("gen_steps"),
        "gen_blocksize": params.get("gen_blocksize"),
        "num_samples": count,
        "model_max_seq_len": model_max_seq_len,
        "min_actual_prompt_tokens": min_prompt,
        "max_actual_prompt_tokens": max_prompt,
        "max_actual_total_tokens": max_total,
        "truncated_samples": truncated,
        "underfilled_samples": underfilled,
        "status": "skipped_existing",
        "sample_manifest_rows": manifest_rows,
    }


def load_tokenizers(model_path: str, need_model_tokenizer: bool = True):
    try:
        import tiktoken
    except ImportError as exc:
        raise SystemExit(f"Missing data preparation dependency: {exc}") from exc
    ruler_tokenizer = tiktoken.encoding_for_model("gpt-4")
    model_tokenizer = None
    if need_model_tokenizer:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise SystemExit(f"Missing data preparation dependency: {exc}") from exc
        model_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return ruler_tokenizer, model_tokenizer


def prompt_to_model_tokens(tokenizer, prompt: str) -> int:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        rendered = prompt
    encoded = tokenizer(rendered, add_special_tokens=False)
    return len(encoded["input_ids"])


def generate_ruler_dataset(params: Dict[str, Any], haystack_path: Path):
    if str(OPENCOMPASS_DIR) not in sys.path:
        sys.path.insert(0, str(OPENCOMPASS_DIR))
    from opencompass.datasets.ruler.ruler_niah import RulerNiahDataset
    return RulerNiahDataset.load(
        base_path=str(haystack_path.parent),
        file_path=haystack_path.name,
        tokens_to_generate=int(params.get("gen_length", 128)),
        max_seq_length=int(params["context_length"]),
        tokenizer_model="gpt-4",
        num_samples=int(params.get("num_samples", 20)),
        random_seed=int(params.get("seed", 42) or 42),
        num_needle_k=1,
        num_needle_v=1,
        num_needle_q=1,
        type_haystack="essay",
        depth_percents=DEPTH_BY_POSITION[str(params["needle_position"])],
    )


def dataset_rows(dataset) -> List[Dict[str, Any]]:
    return [dict(item) for item in dataset]


def prepare_condition(
    condition: Dict[str, Any],
    haystack_path: Path,
    output_dir: Path,
    ruler_tokenizer,
    model_tokenizer,
    model_max_seq_len: int,
    skip_length_check: bool,
) -> Dict[str, Any]:
    params = condition["params"]
    gen_length = int(params.get("gen_length", 128))
    dataset = generate_ruler_dataset(params, haystack_path)
    samples = dataset_rows(dataset)
    prepared_rows = []
    manifest_rows = []
    for sample_index, sample in enumerate(samples):
        prompt = sample["prompt"]
        nominal_prompt_tokens = len(ruler_tokenizer.encode(prompt))
        if skip_length_check:
            actual_prompt_tokens = None
            actual_total_tokens = None
            truncated = False
            fill_ratio = None
            underfilled = False
        else:
            actual_prompt_tokens = prompt_to_model_tokens(model_tokenizer, prompt)
            actual_total_tokens = actual_prompt_tokens + gen_length
            truncated = actual_total_tokens > model_max_seq_len
            target_budget = max(model_max_seq_len - gen_length, 1)
            fill_ratio = actual_prompt_tokens / target_budget
            underfilled = int(params["context_length"]) >= model_max_seq_len and fill_ratio < 0.85
        nominal_total_tokens = nominal_prompt_tokens + gen_length
        prepared = {
            **sample,
            "sample_index": sample_index,
            "condition": condition["run_name"],
            "condition_id": condition["condition_id"],
            "experiment": condition["experiment"],
            "benchmark": condition["benchmark"],
            "requested_context_length": int(params["context_length"]),
            "needle_position": params["needle_position"],
            "gen_length": gen_length,
            "gen_steps": int(params.get("gen_steps", gen_length)),
            "gen_blocksize": int(params.get("gen_blocksize", gen_length)),
            "model_max_seq_len": model_max_seq_len,
            "nominal_prompt_tokens": nominal_prompt_tokens,
            "nominal_total_tokens": nominal_total_tokens,
            "actual_prompt_tokens": actual_prompt_tokens,
            "actual_total_tokens": actual_total_tokens,
            "truncated": truncated,
            "underfilled": underfilled,
            "fill_ratio": round(fill_ratio, 6) if fill_ratio is not None else None,
        }
        prepared_rows.append(prepared)
        manifest_rows.append({k: v for k, v in prepared.items() if k not in {"prompt", "answer"}})
    condition_path = condition_output_path(output_dir, condition)
    write_jsonl(condition_path, prepared_rows)
    actual_prompts = [row["actual_prompt_tokens"] for row in prepared_rows if isinstance(row.get("actual_prompt_tokens"), int)]
    actual_totals = [row["actual_total_tokens"] for row in prepared_rows if isinstance(row.get("actual_total_tokens"), int)]
    return {
        "condition": condition["condition_id"],
        "run_name": condition["run_name"],
        "experiment": condition["experiment"],
        "experiments_using": ",".join(condition.get("experiments_using", [])),
        "benchmark": condition["benchmark"],
        "prepared_jsonl": str(condition_path),
        "requested_context_length": params["context_length"],
        "needle_position": params["needle_position"],
        "gen_length": gen_length,
        "gen_steps": params.get("gen_steps"),
        "gen_blocksize": params.get("gen_blocksize"),
        "num_samples": len(samples),
        "model_max_seq_len": model_max_seq_len,
        "min_actual_prompt_tokens": min(actual_prompts) if actual_prompts else None,
        "max_actual_prompt_tokens": max(actual_prompts) if actual_prompts else None,
        "max_actual_total_tokens": max(actual_totals) if actual_totals else None,
        "truncated_samples": sum(1 for row in prepared_rows if row["truncated"]),
        "underfilled_samples": sum(1 for row in prepared_rows if row["underfilled"]),
        "status": "prepared",
        "sample_manifest_rows": manifest_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and validate local RULER NIAH data before GPU runs.")
    parser.add_argument("--config", default="test_config.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--only", action="append", default=[], help="Limit to task or experiment name.")
    parser.add_argument("--dry-run", action="store_true", help="Only expand conditions and check paths.")
    parser.add_argument("--force", action="store_true", help="Regenerate prepared files even if they already exist.")
    parser.add_argument("--check-opencompass", action="store_true", help="Opt-in check/download for non-RULER OpenCompass datasets.")
    parser.add_argument("--skip-opencompass-check", action="store_true", help="Deprecated/no-op. Non-RULER checks are skipped by default.")
    parser.add_argument("--skip-length-check", action="store_true", help="Do not load iLLaDA tokenizer; prepare samples without actual model-token length validation.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_yaml(config_path)
    data_cfg = config.get("data", {}) or {}
    haystack_path = Path(data_cfg.get("ruler_haystack_path", "data/ruler/paul_graham_essay.jsonl"))
    if not haystack_path.is_absolute():
        haystack_path = ROOT / haystack_path
    output_dir = Path(args.output_dir or data_cfg.get("prepared_dir", "data/prepared"))
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    selected = set(args.only or [])
    benchmark_rows = benchmark_conditions(config, selected)
    unique_benchmarks = sorted({row["benchmark"] for row in benchmark_rows})
    log(f"Benchmark plan: {len(benchmark_rows)} run condition(s), {len(unique_benchmarks)} benchmark(s): {', '.join(unique_benchmarks)}")

    raw_conditions = list(iter_context_conditions(config, selected))
    conditions_by_id: Dict[str, Dict[str, Any]] = {}
    for condition in raw_conditions:
        existing = conditions_by_id.get(condition["condition_id"])
        if existing is None:
            conditions_by_id[condition["condition_id"]] = condition
            condition["experiments_using"] = [condition["experiment"]]
        else:
            existing.setdefault("experiments_using", []).append(condition["experiment"])
    conditions = list(conditions_by_id.values())
    log(f"Prepared-data plan: {len(conditions)} RULER condition(s)")
    for condition in conditions:
        params = condition["params"]
        exists = condition_output_path(output_dir, condition).exists()
        log(
            f"- {condition['condition_id']}: context={params['context_length']} "
            f"position={params['needle_position']} gen_steps={params.get('gen_steps')} "
            f"samples={params.get('num_samples')} used_by={len(condition.get('experiments_using', []))} "
            f"status={'exists' if exists else 'missing'}"
        )

    if args.dry_run:
        if not conditions:
            log("RULER haystack not needed for the selected experiments.")
        elif not haystack_path.exists():
            print(f"Missing RULER haystack file: {haystack_path}", file=sys.stderr, flush=True)
        else:
            log(f"Haystack OK: {haystack_path}")
        log(f"Prepared output dir: {output_dir}")
        return 2 if conditions and not haystack_path.exists() else 0

    preflight_rows = []
    if args.check_opencompass:
        for benchmark in unique_benchmarks:
            if benchmark == "ruler_niah_single_1":
                continue
            result = preflight_opencompass_benchmark(benchmark)
            preflight_rows.append({"kind": "opencompass_dataset", **result})
            if result.get("status") == "ok":
                log(f"OpenCompass dataset OK: {benchmark}")
            else:
                print(f"OpenCompass dataset FAILED: {benchmark} | {result.get('error')}", file=sys.stderr, flush=True)

    if conditions and not haystack_path.exists():
        preflight_rows.append({
            "kind": "ruler_haystack",
            "benchmark": "ruler_niah_single_1",
            "status": "failed",
            "path": str(haystack_path),
            "error": "missing local RULER haystack file",
        })
        write_csv(output_dir / "data_preflight_report.csv", preflight_rows)
        print(f"Missing RULER haystack file: {haystack_path}", file=sys.stderr, flush=True)
        print(f"Data preflight report: {output_dir / 'data_preflight_report.csv'}", flush=True)
        return 2

    if not conditions:
        write_csv(output_dir / "data_preflight_report.csv", preflight_rows)
        failed = [row for row in preflight_rows if row.get("status") != "ok"]
        log(f"Data preflight report: {output_dir / 'data_preflight_report.csv'}")
        return 4 if failed else 0

    model_cfg = deepcopy(config.get("model", {}) or {})
    model_path = str(model_cfg.get("path", "GSAI-ML/iLLaDA-8B-Instruct"))
    model_max_seq_len = int(model_cfg.get("max_seq_len", 8192))

    condition_summaries = []
    sample_manifest = []
    to_generate = []
    for condition in conditions:
        path = condition_output_path(output_dir, condition)
        if path.exists() and not args.force:
            summary = summarize_existing_condition(condition, path, model_max_seq_len)
            if summary is not None:
                log(f"[SKIP] existing prepared data: {path}")
                sample_manifest.extend(summary.pop("sample_manifest_rows"))
                condition_summaries.append(summary)
                continue
        to_generate.append(condition)

    ruler_tokenizer = None
    model_tokenizer = None
    if to_generate:
        log(f"Preparing {len(to_generate)} missing/stale RULER condition(s). Loading tokenizer(s)...")
        ruler_tokenizer, model_tokenizer = load_tokenizers(model_path, need_model_tokenizer=not args.skip_length_check)
    else:
        log("All RULER prepared files already exist. No tokenizer/model-token check needed.")

    for condition in to_generate:
        log(f"[PREPARE] {condition['condition_id']}")
        summary = prepare_condition(
            condition=condition,
            haystack_path=haystack_path,
            output_dir=output_dir,
            ruler_tokenizer=ruler_tokenizer,
            model_tokenizer=model_tokenizer,
            model_max_seq_len=model_max_seq_len,
            skip_length_check=args.skip_length_check,
        )
        sample_manifest.extend(summary.pop("sample_manifest_rows"))
        condition_summaries.append(summary)

    write_jsonl(output_dir / "prepared_manifest.jsonl", sample_manifest)
    write_csv(output_dir / "condition_summary.csv", condition_summaries)

    for summary in condition_summaries:
        preflight_rows.append({
            "kind": "ruler_prepared_condition",
            "benchmark": summary["benchmark"],
            "status": summary.get("status", "ok"),
            "condition": summary["condition"],
            "prepared_jsonl": summary["prepared_jsonl"],
            "num_samples": summary["num_samples"],
            "truncated_samples": summary["truncated_samples"],
            "underfilled_samples": summary["underfilled_samples"],
            "max_actual_total_tokens": summary["max_actual_total_tokens"],
        })
    write_csv(output_dir / "data_preflight_report.csv", preflight_rows)

    bad = [row for row in condition_summaries if row["truncated_samples"] or row["underfilled_samples"]]
    log(f"Wrote prepared data to: {output_dir}")
    log(f"Condition summary: {output_dir / 'condition_summary.csv'}")
    log(f"Data preflight report: {output_dir / 'data_preflight_report.csv'}")
    if bad:
        print("Length validation found risky conditions:", file=sys.stderr, flush=True)
        for row in bad:
            print(
                f"- {row['condition']}: truncated={row['truncated_samples']} "
                f"underfilled={row['underfilled_samples']} max_total={row['max_actual_total_tokens']}",
                file=sys.stderr,
                flush=True,
            )
        return 3
    failed = [row for row in preflight_rows if row.get("status") not in {"ok", "prepared", "skipped_existing"}]
    return 4 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
