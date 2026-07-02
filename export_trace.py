#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_trace.py v6

ARness output manager + trace exporter.

What changed from the previous version
--------------------------------------
1. Output is organized by TASK/EXPERIMENT name, not run order.
2. Each task/experiment owns its output_path:
     outputs/arness/arness_trace_gsm8k_sample7/
     outputs/arness/arness_trace_mbpp_sample6/
3. Each condition folder is named by the actual config:
     gsm8k_sample7_len256_block64_steps128_thr0p6
4. Old numbered folders can be repaired into the new layout.
5. Trace export is integrated, so repaired/new folders immediately get:
     sample_traces/
       trace_metrics.jsonl
       trace_metrics.csv
       sample_0007/
         generation_chain.csv
         metrics.json
         problem_groundtruth_prediction.txt

Recommended usage
-----------------
Repair old outputs and export traces:
  python export_trace.py \
    --runs-root outputs/arness_all \
    --task-output-root outputs/arness \
    --canonicalize \
    --mode copy \
    --overwrite \
    --write-task-index

Export traces for already canonical outputs:
  python export_trace.py \
    --runs-root outputs/arness \
    --overwrite \
    --write-task-index

After this, visual scripts should read:
  outputs/arness/
not the old numbered folders.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# -----------------------------
# Basic IO helpers
# -----------------------------

def to_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        s = str(x).strip()
        if s == "" or s.lower() in {"none", "null", "nan"}:
            return default
        return int(float(s))
    except Exception:
        return default


def to_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        s = str(x).strip()
        if s == "" or s.lower() in {"none", "null", "nan"}:
            return default
        v = float(s)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def mean(values: Sequence[Any]) -> Optional[float]:
    vals = []
    for v in values:
        f = to_float(v)
        if f is not None and math.isfinite(f):
            vals.append(f)
    return sum(vals) / len(vals) if vals else None


def safe_part(x: Any) -> str:
    if x is None:
        return "none"
    s = str(x).strip()
    if s == "" or s.lower() in {"none", "null", "nan"}:
        return "none"
    s = s.replace(".", "p")
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "none"


def threshold_label(x: Any) -> str:
    v = to_float(x, None)
    if v is None:
        return "none"
    return f"{v:g}"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def get_field(record: Dict[str, Any], *names: str, default: Any = None) -> Any:
    for n in names:
        if n in record and record[n] not in (None, ""):
            return record[n]
    return default


# -----------------------------
# Metadata / canonical naming
# -----------------------------

def infer_gen_length(record: Dict[str, Any], traces: List[Dict[str, Any]]) -> int:
    gl = get_field(record, "gen_length", "generated_tokens", "max_new_tokens", "param_gen_length", default=None)
    if gl is not None:
        return to_int(gl, 0)
    max_mask = max([to_int(t.get("mask_count_before"), 0) for t in traces] or [0])
    if max_mask > 0:
        return max_mask
    max_pos = -1
    for t in traces:
        for p in t.get("selected_positions") or []:
            max_pos = max(max_pos, to_int(p, -1))
    return max_pos + 1 if max_pos >= 0 else 0


def infer_block_length(record: Dict[str, Any], traces: List[Dict[str, Any]], gen_length: int) -> int:
    bl = get_field(
        record,
        "block_length",
        "gen_blocksize",
        "param_gen_blocksize",
        "block_size",
        default=None,
    )
    if bl is not None:
        return max(1, to_int(bl, gen_length or 1))

    block_ids = sorted({to_int(t.get("block_idx"), -1) for t in traces})
    block_ids = [b for b in block_ids if b >= 0]
    if len(block_ids) > 1 and gen_length:
        return max(1, math.ceil(gen_length / len(block_ids)))
    return gen_length or 1


def fallback_summaries_from_trace(trace_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in trace_rows:
        sid = to_int(r.get("sample_idx", r.get("sample_id")), -1)
        if sid >= 0:
            grouped[sid].append(r)

    records = []
    for sid, rows in sorted(grouped.items()):
        first = rows[0]
        final = rows[-1]
        gen_length = infer_gen_length(first, rows)
        block_length = infer_block_length(first, rows, gen_length)
        records.append({
            "sample_idx": sid,
            "benchmark": first.get("benchmark", "unknown"),
            "gen_length": gen_length,
            "block_length": block_length,
            "gen_steps": max(to_int(r.get("step_idx"), 0) for r in rows) + 1,
            "token_selection_confidence_threshold": first.get("token_selection_confidence_threshold"),
            "completion_rate": final.get("current_completion_rate"),
            "final_mask_count": final.get("mask_count_after"),
        })
    return records


def dedup_summary_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for rec in rows:
        sid = rec.get("sample_idx", rec.get("sample_id"))
        if sid is None:
            sid = len(order)
        key = str(sid)
        if key not in by_id:
            order.append(key)
        by_id[key] = rec
    return [by_id[k] for k in order]


def dedup_trace_rows(trace_rows: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, Dict[Tuple[int, int, int], Dict[str, Any]]] = defaultdict(dict)
    for r in trace_rows:
        sid = to_int(r.get("sample_idx", r.get("sample_id")), -1)
        if sid < 0:
            continue
        key = (
            to_int(r.get("batch_item_idx"), 0),
            to_int(r.get("block_idx"), -1),
            to_int(r.get("step_idx"), -1),
        )
        grouped[sid][key] = r

    out = {}
    for sid, items in grouped.items():
        rows = list(items.values())
        rows.sort(key=lambda r: (
            to_int(r.get("step_idx"), 0),
            to_int(r.get("block_idx"), 0),
            to_int(r.get("batch_item_idx"), 0),
        ))
        out[sid] = rows
    return out


def metric_value(record: Dict[str, Any]) -> Optional[float]:
    for k in ["accuracy", "score", "pass", "primary_metric_value", "exact_match"]:
        v = to_float(record.get(k), None)
        if v is not None:
            return v
    return None


def infer_benchmark_from_name(path: Path) -> str:
    name = path.name.lower()
    if "gsm8k" in name or "gsm_8k" in name:
        return "gsm8k"
    if "mbpp" in name:
        return "mbpp"
    if "ruler" in name:
        return "ruler"
    return "unknown"


def infer_sample_from_name(path: Path) -> int:
    m = re.search(r"sample[_-]?(\d+)", path.name.lower())
    return int(m.group(1)) if m else 0


def condition_metadata(run_dir: Path) -> Dict[str, Any]:
    summary_rows = dedup_summary_rows(read_jsonl(run_dir / "summary.jsonl"))
    trace_rows = read_jsonl(run_dir / "trace.jsonl")
    trace_by_sample = dedup_trace_rows(trace_rows)

    if summary_rows:
        rec = summary_rows[-1]
    elif trace_rows:
        rec = fallback_summaries_from_trace(trace_rows)[-1]
    else:
        rec = {}

    sid = to_int(get_field(rec, "sample_idx", "sample_id", default=infer_sample_from_name(run_dir)), 0)
    traces = trace_by_sample.get(sid, trace_rows)

    benchmark = get_field(rec, "benchmark", default=None) or infer_benchmark_from_name(run_dir)
    gen_length = infer_gen_length(rec, traces)
    block_length = infer_block_length(rec, traces, gen_length)
    gen_steps = to_int(get_field(rec, "gen_steps", "steps", "param_gen_steps", default=None), 0)
    if not gen_steps and traces:
        gen_steps = max(to_int(r.get("step_idx"), -1) for r in traces) + 1

    th = get_field(
        rec,
        "token_selection_confidence_threshold",
        "param_token_selection_confidence_threshold",
        "threshold",
        default=None,
    )

    return {
        "benchmark": str(benchmark),
        "sample_idx": sid,
        "gen_length": gen_length,
        "block_length": block_length,
        "gen_steps": gen_steps,
        "threshold": th,
        "threshold_label": threshold_label(th),
        "raw_run_dir": str(run_dir),
    }


def experiment_name(meta: Dict[str, Any]) -> str:
    return f"arness_trace_{safe_part(meta['benchmark']).lower()}_sample{to_int(meta['sample_idx'], 0)}"


def condition_key(meta: Dict[str, Any]) -> str:
    bench = safe_part(meta["benchmark"]).lower()
    sid = to_int(meta["sample_idx"], 0)
    return (
        f"{bench}_sample{sid}"
        f"_len{safe_part(meta['gen_length'])}"
        f"_block{safe_part(meta['block_length'])}"
        f"_steps{safe_part(meta['gen_steps'])}"
        f"_thr{safe_part(meta['threshold_label'])}"
    )


def is_condition_dir(path: Path) -> bool:
    return bool(re.search(r"_sample\d+_len\d+_block\d+_steps\d+_thr", path.name))


def infer_run_dirs(root: Path, sample_traces_name: str) -> List[Path]:
    if (root / "trace.jsonl").exists():
        return [root]

    run_dirs = []
    for p in sorted(root.rglob("trace.jsonl")):
        if sample_traces_name in p.parts:
            continue
        run_dirs.append(p.parent)

    seen = set()
    out = []
    for rd in run_dirs:
        key = str(rd.resolve())
        if key not in seen:
            out.append(rd)
            seen.add(key)
    return out


def canonicalize_run_dirs(
    run_dirs: Sequence[Path],
    task_output_root: Path,
    mode: str,
    overwrite: bool,
) -> Tuple[List[Path], List[Dict[str, Any]], List[Dict[str, Any]]]:
    task_output_root.mkdir(parents=True, exist_ok=True)
    canonical_dirs: List[Path] = []
    index_rows: List[Dict[str, Any]] = []
    duplicate_rows: List[Dict[str, Any]] = []

    seen: Dict[str, int] = defaultdict(int)

    for rd in run_dirs:
        meta = condition_metadata(rd)
        exp = experiment_name(meta)
        key = condition_key(meta)
        canonical = task_output_root / exp / key

        already_canonical = is_condition_dir(rd) and rd.parent.name == exp
        if already_canonical:
            canonical = rd

        dup_id = f"{exp}/{key}"
        seen[dup_id] += 1
        duplicate_index = seen[dup_id]
        if duplicate_index > 1 and not overwrite and not already_canonical:
            canonical = task_output_root / exp / f"{key}_dup{duplicate_index}"
            duplicate_rows.append({
                **meta,
                "experiment_name": exp,
                "condition_key": key,
                "duplicate_index": duplicate_index,
                "raw_run_dir": str(rd),
                "canonical_run_dir": str(canonical),
            })

        row = {
            **meta,
            "experiment_name": exp,
            "condition_key": key,
            "duplicate_index": duplicate_index,
            "raw_run_dir": str(rd),
            "canonical_run_dir": str(canonical),
            "already_canonical": already_canonical,
        }
        index_rows.append(row)

        if already_canonical:
            canonical_dirs.append(canonical)
            continue

        if canonical.exists():
            if overwrite:
                shutil.rmtree(canonical)
            else:
                print(f"[SKIP] exists: {canonical}")
                canonical_dirs.append(canonical)
                continue

        canonical.parent.mkdir(parents=True, exist_ok=True)
        if mode == "copy":
            shutil.copytree(rd, canonical)
        elif mode == "move":
            shutil.move(str(rd), str(canonical))
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        canonical_dirs.append(canonical)
        print(f"[OK] {rd} -> {canonical}")

    return canonical_dirs, index_rows, duplicate_rows


# -----------------------------
# Trace export
# -----------------------------

def render_slots(slots: Sequence[Optional[str]], max_chars: int, compress_masks: bool) -> str:
    if compress_masks:
        parts: List[str] = []
        run = 0
        for tok in slots:
            if tok is None:
                run += 1
            else:
                if run:
                    parts.append(f"□x{run}")
                    run = 0
                parts.append(str(tok))
        if run:
            parts.append(f"□x{run}")
        s = "".join(parts)
    else:
        s = "".join("□" if tok is None else str(tok) for tok in slots)

    s = s.replace("\r", "\\r").replace("\n", "\\n")
    if max_chars > 0 and len(s) > max_chars:
        return s[:max_chars] + "...<truncated>"
    return s


def selected_to_local(pos: int, block_idx: int, block_length: int) -> Optional[int]:
    start = block_idx * block_length
    if start <= pos < start + block_length:
        return pos - start
    if 0 <= pos < block_length:
        return pos
    return None


def make_problem_text(record: Dict[str, Any]) -> str:
    question = get_field(record, "question", "prompt", "input", "problem", "query", default="")
    groundtruth = get_field(record, "answer", "groundtruth", "reference", "target", "gt", default="")
    prediction = get_field(record, "prediction", "decoded_prediction", "prediction_preview", "output", default="")
    score = metric_value(record)
    return (
        "QUESTION / PROMPT\n"
        "=================\n"
        f"{question}\n\n"
        "GROUNDTRUTH / REFERENCE\n"
        "=======================\n"
        f"{groundtruth}\n\n"
        "MODEL GENERATION\n"
        "================\n"
        f"{prediction}\n\n"
        "SCORE\n"
        "=====\n"
        f"{score}\n"
    )


def add_block_columns(
    out: Dict[str, Any],
    block_slots: Dict[int, List[Optional[str]]],
    num_blocks: int,
    max_cell_chars: int,
    compress_masks: bool,
) -> None:
    for b in range(num_blocks):
        visible = sum(x is not None for x in block_slots[b])
        total = len(block_slots[b])
        out[f"block_{b:02d}"] = render_slots(block_slots[b], max_cell_chars, compress_masks)
        out[f"block_{b:02d}_complete"] = f"{visible}/{total}"


def build_generation_chain(
    record: Dict[str, Any],
    traces: List[Dict[str, Any]],
    max_cell_chars: int,
    compress_masks: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    gen_length = infer_gen_length(record, traces)
    block_length = infer_block_length(record, traces, gen_length)
    num_blocks = max(1, math.ceil(gen_length / block_length)) if gen_length else 1

    for row in traces:
        if row.get("block_idx") is None or str(row.get("block_idx")).strip() == "":
            positions = row.get("selected_positions") or []
            first_pos = to_int(positions[0], 0) if positions else 0
            row["block_idx"] = min(num_blocks - 1, max(0, first_pos // block_length))

    block_slots: Dict[int, List[Optional[str]]] = {b: [None] * block_length for b in range(num_blocks)}
    local_round_counter: Dict[int, int] = defaultdict(int)
    block_selected_counts: Dict[int, List[int]] = defaultdict(list)
    block_confidences: Dict[int, List[float]] = defaultdict(list)
    block_reasons: Dict[int, List[str]] = defaultdict(list)

    chain_rows: List[Dict[str, Any]] = []
    initial: Dict[str, Any] = {
        "generation_step": -1,
        "active_block": "",
        "block_local_round": -1,
        "selected_count": 0,
        "transfer_reason": "initial_all_mask",
    }
    add_block_columns(initial, block_slots, num_blocks, max_cell_chars, compress_masks)
    chain_rows.append(initial)

    for row in traces:
        block_idx = min(num_blocks - 1, max(0, to_int(row.get("block_idx"), 0)))
        local_round = local_round_counter[block_idx]
        local_round_counter[block_idx] += 1

        positions = row.get("selected_positions") or []
        tokens = row.get("selected_decoded_tokens") or []
        confidences = row.get("selected_confidences") or []
        reason = str(row.get("transfer_reason", ""))

        for p, tok in zip(positions, tokens):
            lp = selected_to_local(to_int(p, -1), block_idx, block_length)
            if lp is not None and 0 <= lp < block_length:
                block_slots[block_idx][lp] = str(tok)

        block_selected_counts[block_idx].append(len(positions))
        for c in confidences:
            f = to_float(c)
            if f is not None and math.isfinite(f):
                block_confidences[block_idx].append(f)
        if reason:
            block_reasons[block_idx].append(reason)

        out: Dict[str, Any] = {
            "generation_step": to_int(row.get("step_idx"), len(chain_rows) - 1),
            "active_block": block_idx,
            "block_local_round": local_round,
            "selected_count": len(positions),
            "transfer_reason": reason,
        }
        add_block_columns(out, block_slots, num_blocks, max_cell_chars, compress_masks)
        chain_rows.append(out)

    final_visible_by_block = {
        f"block_{b:02d}_final_complete": f"{sum(x is not None for x in block_slots[b])}/{len(block_slots[b])}"
        for b in range(num_blocks)
    }

    block_stats = []
    for b in range(num_blocks):
        visible = sum(x is not None for x in block_slots[b])
        total = len(block_slots[b])
        block_stats.append({
            "block_idx": b,
            "complete": f"{visible}/{total}",
            "is_complete": visible == total,
            "local_rounds": local_round_counter[b],
            "selected_tokens_total": sum(block_selected_counts[b]),
            "mean_selected_count": mean(block_selected_counts[b]),
            "mean_confidence": mean(block_confidences[b]),
            "fallback_steps": sum(1 for x in block_reasons[b] if "fallback" in x),
            "threshold_pass_steps": sum(1 for x in block_reasons[b] if x == "threshold_pass"),
        })

    th = get_field(record, "token_selection_confidence_threshold", "param_token_selection_confidence_threshold")
    metrics: Dict[str, Any] = {
        "sample_idx": record.get("sample_idx", record.get("sample_id")),
        "benchmark": record.get("benchmark"),
        "gen_length": gen_length,
        "block_length": block_length,
        "num_blocks": num_blocks,
        "gen_steps": get_field(record, "gen_steps", "steps", default=len(traces)),
        "threshold": th,
        "threshold_label": threshold_label(th),
        "min_transfer_tokens": get_field(record, "min_transfer_tokens", "param_min_transfer_tokens"),
        "metric_value": metric_value(record),
        "completion_rate": get_field(record, "completion_rate", default=None),
        "actual_parallelism": get_field(record, "actual_parallelism", default=None),
        "actual_arness": get_field(record, "actual_arness", default=None),
        "threshold_pass_rate": get_field(record, "threshold_pass_rate", default=None),
        "fallback_rate": get_field(record, "fallback_rate", default=None),
        "elapsed_seconds": get_field(record, "elapsed_seconds", "latency_s", default=None),
        "tokens_per_second": get_field(record, "tokens_per_second", "tps", default=None),
        "visible_tps": get_field(record, "visible_tps", default=None),
        "actual_commit_tps": get_field(record, "actual_commit_tps", default=None),
        "final_mask_count": get_field(record, "final_mask_count", default=None),
        "failure_type": get_field(record, "failure_type", default=None),
        "trace_step_rows": len(traces),
        "generation_chain_rows": len(chain_rows),
        "block_stats": block_stats,
    }
    metrics.update(final_visible_by_block)

    gs = to_float(metrics.get("gen_steps"), None)
    if gs and gs != 0:
        metrics["planned_parallelism"] = gen_length / gs
        metrics["nominal_arness"] = gs / gen_length if gen_length else None
    else:
        metrics["planned_parallelism"] = None
        metrics["nominal_arness"] = None

    return chain_rows, metrics


def export_run_dir(
    run_dir: Path,
    sample_traces_name: str,
    overwrite: bool,
    max_cell_chars: int,
    compress_masks: bool,
) -> List[Dict[str, Any]]:
    summary_rows_raw = read_jsonl(run_dir / "summary.jsonl")
    trace_rows_raw = read_jsonl(run_dir / "trace.jsonl")
    if not trace_rows_raw:
        return []

    if not summary_rows_raw:
        summary_rows_raw = fallback_summaries_from_trace(trace_rows_raw)

    summary_rows = dedup_summary_rows(summary_rows_raw)
    traces_by_sample = dedup_trace_rows(trace_rows_raw)

    out_root = run_dir / sample_traces_name
    if out_root.exists() and overwrite:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    meta = condition_metadata(run_dir)
    exp = experiment_name(meta)
    key = condition_key(meta)

    all_metrics: List[Dict[str, Any]] = []

    for rec in summary_rows:
        sid = to_int(rec.get("sample_idx", rec.get("sample_id")), -1)
        if sid < 0:
            continue
        traces = traces_by_sample.get(sid, [])
        if not traces:
            continue

        sample_dir = out_root / f"sample_{sid:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        chain_rows, metrics = build_generation_chain(
            rec,
            traces,
            max_cell_chars=max_cell_chars,
            compress_masks=compress_masks,
        )

        metrics.update({
            "experiment_name": exp,
            "condition_key": key,
            "run_dir": str(run_dir),
            "sample_dir": str(sample_dir),
            "raw_summary_records_in_run": len(summary_rows_raw),
            "deduped_summary_records_in_run": len(summary_rows),
            "raw_trace_records_in_run": len(trace_rows_raw),
            "deduped_trace_records_for_sample": len(traces),
        })

        write_csv(sample_dir / "generation_chain.csv", chain_rows)
        write_json(sample_dir / "metrics.json", metrics)
        (sample_dir / "problem_groundtruth_prediction.txt").write_text(make_problem_text(rec), encoding="utf-8")

        flat = dict(metrics)
        flat.pop("block_stats", None)
        all_metrics.append(flat)

    write_jsonl(out_root / "trace_metrics.jsonl", all_metrics)
    write_csv(out_root / "trace_metrics.csv", all_metrics)
    return all_metrics


def write_task_indexes(root: Path, metric_rows: Sequence[Dict[str, Any]], index_rows: Sequence[Dict[str, Any]], duplicate_rows: Sequence[Dict[str, Any]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    write_jsonl(root / "trace_metrics_all.jsonl", metric_rows)
    write_csv(root / "trace_metrics_all.csv", metric_rows)
    write_jsonl(root / "trace_index.jsonl", index_rows)
    write_csv(root / "trace_index.csv", index_rows)
    write_csv(root / "duplicate_conditions.csv", duplicate_rows)

    by_exp: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in metric_rows:
        by_exp[str(r.get("experiment_name", "unknown"))].append(r)

    for exp, rows in by_exp.items():
        exp_dir = root / exp
        write_jsonl(exp_dir / "trace_metrics_all.jsonl", rows)
        write_csv(exp_dir / "trace_metrics_all.csv", rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="outputs/arness", help="Old or canonical ARness output root.")
    ap.add_argument("--task-output-root", default=None, help="Canonical task output root. Default: --runs-root.")
    ap.add_argument("--canonicalize", action="store_true", help="Copy/move old numbered folders into task/condition layout before exporting traces.")
    ap.add_argument("--mode", choices=["copy", "move"], default="copy", help="Canonicalization mode.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite sample_traces and canonical duplicate destinations.")
    ap.add_argument("--sample-traces-name", default="sample_traces")
    ap.add_argument("--max-cell-chars", type=int, default=0)
    ap.add_argument("--compress-masks", action="store_true")
    ap.add_argument("--write-task-index", action="store_true", help="Write root/task-level trace_index and trace_metrics_all files.")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    task_output_root = Path(args.task_output_root) if args.task_output_root else runs_root

    discovered = infer_run_dirs(runs_root, args.sample_traces_name)
    if not discovered:
        raise SystemExit(f"[ERROR] no trace.jsonl found under {runs_root}")

    index_rows: List[Dict[str, Any]] = []
    duplicate_rows: List[Dict[str, Any]] = []

    if args.canonicalize:
        run_dirs, index_rows, duplicate_rows = canonicalize_run_dirs(
            discovered,
            task_output_root=task_output_root,
            mode=args.mode,
            overwrite=args.overwrite,
        )
    else:
        run_dirs = discovered
        for rd in run_dirs:
            meta = condition_metadata(rd)
            row = {
                **meta,
                "experiment_name": experiment_name(meta),
                "condition_key": condition_key(meta),
                "raw_run_dir": str(rd),
                "canonical_run_dir": str(rd),
                "already_canonical": is_condition_dir(rd),
            }
            index_rows.append(row)

    all_metric_rows: List[Dict[str, Any]] = []
    for rd in run_dirs:
        rows = export_run_dir(
            rd,
            sample_traces_name=args.sample_traces_name,
            overwrite=args.overwrite,
            max_cell_chars=args.max_cell_chars,
            compress_masks=args.compress_masks,
        )
        all_metric_rows.extend(rows)
        print(f"[OK] exported {len(rows)} sample trace(s): {rd / args.sample_traces_name}")

    if args.write_task_index or args.canonicalize:
        write_task_indexes(task_output_root, all_metric_rows, index_rows, duplicate_rows)
        print(f"[OK] wrote task indexes under {task_output_root}")

    print(f"[DONE] total sample traces: {len(all_metric_rows)}")


if __name__ == "__main__":
    main()
