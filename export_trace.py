#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_arness_trace_csv_v5.py

Post-process iLLaDA ARness trace runs.

This version uses GLOBAL generation time in generation_chain.csv:
- Row -1 is the initial all-mask state.
- Then one row per original trace step, e.g. 0..255 for a 256-step trace.
- It does NOT output a separate global_timeline.csv.
- sample_traces is written under EACH original task/run folder by default.
- Each sample folder contains only:
    generation_chain.csv
    metrics.json
    problem_groundtruth_prediction.txt
- Each run folder also gets:
    sample_traces/trace_metrics.jsonl
    sample_traces/trace_metrics.csv

generation_chain.csv columns:
    generation_step
    active_block
    block_local_round
    selected_count
    transfer_reason
    block_00, block_00_complete
    block_01, block_01_complete
    ...

Where block_XX_complete is a compact visible/total string such as 16/64.
64/64 means that block is complete.

Usage:
  python export_arness_trace_csv.py --runs-root outputs/arness_all --overwrite

Default mask rendering is full mask characters: □□□...
Use --compress-masks to render □x64 instead.
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
from typing import Any, Dict, List, Optional, Sequence, Tuple


def safe_name(x: Any, max_len: int = 100) -> str:
    s = "none" if x is None else str(x)
    s = re.sub(r"[^A-Za-z0-9_.=-]+", "_", s).strip("_")
    return (s or "none")[:max_len]


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
        return float(s)
    except Exception:
        return default


def mean(values: Sequence[Any]) -> Optional[float]:
    vals = []
    for v in values:
        f = to_float(v)
        if f is not None and math.isfinite(f):
            vals.append(f)
    if not vals:
        return None
    return sum(vals) / len(vals)


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


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


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


def get_field(record: Dict[str, Any], *names: str, default: Any = None) -> Any:
    for n in names:
        if n in record and record[n] not in (None, ""):
            return record[n]
    return default


def infer_gen_length(record: Dict[str, Any], traces: List[Dict[str, Any]]) -> int:
    gl = get_field(record, "gen_length", "generated_tokens", "max_new_tokens", default=None)
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
    bl = get_field(record, "block_length", "gen_blocksize", "param_gen_blocksize", "block_size", default=None)
    if bl is not None:
        return max(1, to_int(bl, gen_length or 1))

    block_ids = sorted({to_int(t.get("block_idx"), -1) for t in traces})
    block_ids = [b for b in block_ids if b >= 0]
    if len(block_ids) > 1 and gen_length:
        return max(1, math.ceil(gen_length / len(block_ids)))
    return gen_length or 1


def selected_to_local(pos: int, block_idx: int, block_length: int) -> Optional[int]:
    start = block_idx * block_length
    if start <= pos < start + block_length:
        return pos - start
    if 0 <= pos < block_length:
        return pos
    return None


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
            "decoding_config_name": first.get("decoding_config_name"),
            "gen_length": gen_length,
            "block_length": block_length,
            "steps": max(to_int(r.get("step_idx"), 0) for r in rows) + 1,
            "token_selection_confidence_threshold": first.get("token_selection_confidence_threshold"),
            "completion_rate": final.get("current_completion_rate"),
            "final_mask_count": final.get("mask_count_after"),
        })
    return records


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


def build_generation_chain(record: Dict[str, Any], traces: List[Dict[str, Any]], max_cell_chars: int, compress_masks: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
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

    metrics: Dict[str, Any] = {
        "sample_idx": record.get("sample_idx", record.get("sample_id")),
        "benchmark": record.get("benchmark"),
        "run_name": record.get("run_name"),
        "decoding_config_name": record.get("decoding_config_name"),
        "gen_length": gen_length,
        "block_length": block_length,
        "num_blocks": num_blocks,
        "gen_steps": get_field(record, "gen_steps", "steps", default=len(traces)),
        "threshold": get_field(record, "token_selection_confidence_threshold", "param_token_selection_confidence_threshold"),
        "threshold_label": threshold_label(get_field(record, "token_selection_confidence_threshold", "param_token_selection_confidence_threshold")),
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


def export_run_dir(run_dir: Path, sample_traces_name: str, overwrite: bool, max_cell_chars: int, compress_masks: bool) -> List[Dict[str, Any]]:
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

        chain_rows, metrics = build_generation_chain(rec, traces, max_cell_chars=max_cell_chars, compress_masks=compress_masks)

        metrics["run_dir"] = str(run_dir)
        metrics["sample_dir"] = str(sample_dir)
        metrics["raw_summary_records_in_run"] = len(summary_rows_raw)
        metrics["deduped_summary_records_in_run"] = len(summary_rows)
        metrics["raw_trace_records_in_run"] = len(trace_rows_raw)
        metrics["deduped_trace_records_for_sample"] = len(traces)

        write_csv(sample_dir / "generation_chain.csv", chain_rows)
        write_json(sample_dir / "metrics.json", metrics)
        (sample_dir / "problem_groundtruth_prediction.txt").write_text(make_problem_text(rec), encoding="utf-8")

        flat = dict(metrics)
        flat.pop("block_stats", None)
        all_metrics.append(flat)

    write_jsonl(out_root / "trace_metrics.jsonl", all_metrics)
    write_csv(out_root / "trace_metrics.csv", all_metrics)
    return all_metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="outputs/arness_trace", help="Folder containing task/run condition folders.")
    ap.add_argument("--sample-traces-name", default="sample_traces", help="Subfolder name written under each original run folder.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing sample_traces under each run folder.")
    ap.add_argument("--max-cell-chars", type=int, default=0, help="Max cell chars; 0 means no truncation.")
    ap.add_argument("--compress-masks", action="store_true", help="Render masks as □xN instead of full □□□...")
    args = ap.parse_args()

    root = Path(args.runs_root)
    run_dirs = infer_run_dirs(root, args.sample_traces_name)
    if not run_dirs:
        raise SystemExit(f"[ERROR] no trace.jsonl found under {root}")

    total = 0
    for rd in run_dirs:
        rows = export_run_dir(
            rd,
            sample_traces_name=args.sample_traces_name,
            overwrite=args.overwrite,
            max_cell_chars=args.max_cell_chars,
            compress_masks=args.compress_masks,
        )
        total += len(rows)
        print(f"[OK] {rd}: wrote {len(rows)} sample trace folder(s) under {rd / args.sample_traces_name}")

    print(f"[DONE] total sample traces: {total}")


if __name__ == "__main__":
    main()
