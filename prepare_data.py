import argparse
import csv
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from run_test import (
    ROOT,
    collect_experiments,
    deep_merge,
    expand_matrix,
    load_yaml,
    ruler_prepared_condition_id,
    safe_name,
)


OPENCOMPASS_DIR = ROOT / "opencompass"
DEPTH_BY_POSITION = {"front": [0], "middle": [50], "back": [100], "end": [100]}


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
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def iter_context_conditions(config: Dict[str, Any], selected: set) -> Iterable[Dict[str, Any]]:
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


def load_tokenizers(model_path: str):
    try:
        import tiktoken
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(f"Missing data preparation dependency: {exc}") from exc

    ruler_tokenizer = tiktoken.encoding_for_model("gpt-4")
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
    sys.path.insert(0, str(OPENCOMPASS_DIR))
    from opencompass.datasets.ruler.ruler_niah import RulerNiahDataset

    base_path = str(haystack_path.parent)
    file_path = haystack_path.name
    return RulerNiahDataset.load(
        base_path=base_path,
        file_path=file_path,
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
    rows = []
    for item in dataset:
        rows.append(dict(item))
    return rows


def prepare_condition(
    condition: Dict[str, Any],
    haystack_path: Path,
    output_dir: Path,
    ruler_tokenizer,
    model_tokenizer,
    model_max_seq_len: int,
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
        actual_prompt_tokens = prompt_to_model_tokens(model_tokenizer, prompt)
        actual_total_tokens = actual_prompt_tokens + gen_length
        nominal_total_tokens = nominal_prompt_tokens + gen_length
        truncated = actual_total_tokens > model_max_seq_len
        target_budget = max(model_max_seq_len - gen_length, 1)
        fill_ratio = actual_prompt_tokens / target_budget
        underfilled = int(params["context_length"]) >= model_max_seq_len and fill_ratio < 0.85

        prepared = {
            **sample,
            "sample_index": sample_index,
            "condition": condition["run_name"],
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
            "fill_ratio": round(fill_ratio, 6),
        }
        prepared_rows.append(prepared)
        manifest_rows.append({k: v for k, v in prepared.items() if k not in {"prompt", "answer"}})

    condition_path = output_dir / "ruler_niah_single_1" / f"{condition['condition_id']}.jsonl"
    write_jsonl(condition_path, prepared_rows)

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
        "min_actual_prompt_tokens": min(row["actual_prompt_tokens"] for row in prepared_rows) if prepared_rows else None,
        "max_actual_prompt_tokens": max(row["actual_prompt_tokens"] for row in prepared_rows) if prepared_rows else None,
        "max_actual_total_tokens": max(row["actual_total_tokens"] for row in prepared_rows) if prepared_rows else None,
        "truncated_samples": sum(1 for row in prepared_rows if row["truncated"]),
        "underfilled_samples": sum(1 for row in prepared_rows if row["underfilled"]),
        "sample_manifest_rows": manifest_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and validate local RULER NIAH data before GPU runs.")
    parser.add_argument("--config", default="test_config.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--only", action="append", default=[], help="Limit to task or experiment name.")
    parser.add_argument("--dry-run", action="store_true", help="Only expand conditions and check paths.")
    parser.add_argument("--skip-tokenizer-check", action="store_true")
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
    print(f"Prepared-data plan: {len(conditions)} RULER condition(s)")
    for condition in conditions:
        params = condition["params"]
        print(
            f"- {condition['condition_id']}: context={params['context_length']} "
            f"position={params['needle_position']} gen_steps={params.get('gen_steps')} "
            f"samples={params.get('num_samples')} used_by={len(condition.get('experiments_using', []))}"
        )

    if not haystack_path.exists():
        print(f"Missing RULER haystack file: {haystack_path}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(f"Haystack OK: {haystack_path}")
        print(f"Prepared output dir: {output_dir}")
        return 0

    model_cfg = deepcopy(config.get("model", {}) or {})
    model_path = str(model_cfg.get("path", "GSAI-ML/iLLaDA-8B-Instruct"))
    model_max_seq_len = int(model_cfg.get("max_seq_len", 8192))

    if args.skip_tokenizer_check:
        ruler_tokenizer, model_tokenizer = load_tokenizers(model_path)
    else:
        ruler_tokenizer, model_tokenizer = load_tokenizers(model_path)

    condition_summaries = []
    sample_manifest = []
    for condition in conditions:
        summary = prepare_condition(
            condition=condition,
            haystack_path=haystack_path,
            output_dir=output_dir,
            ruler_tokenizer=ruler_tokenizer,
            model_tokenizer=model_tokenizer,
            model_max_seq_len=model_max_seq_len,
        )
        sample_manifest.extend(summary.pop("sample_manifest_rows"))
        condition_summaries.append(summary)

    write_jsonl(output_dir / "prepared_manifest.jsonl", sample_manifest)
    write_csv(output_dir / "condition_summary.csv", condition_summaries)

    bad = [row for row in condition_summaries if row["truncated_samples"] or row["underfilled_samples"]]
    print(f"Wrote prepared data to: {output_dir}")
    print(f"Condition summary: {output_dir / 'condition_summary.csv'}")
    if bad:
        print("Length validation found risky conditions:", file=sys.stderr)
        for row in bad:
            print(
                f"- {row['condition']}: truncated={row['truncated_samples']} "
                f"underfilled={row['underfilled_samples']} max_total={row['max_actual_total_tokens']}",
                file=sys.stderr,
            )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
