"""Write per-sample ARness trace artifacts.

Output format:
  block_timeline.csv   block columns over generation steps
  step_events.csv      committed token positions/confidence/reason per step
  block_metrics.csv    per-block completion/fallback/pass summary
  sample_metrics.json  compact sample metadata
  plot_rows.csv        one flat row for plotting
  final_prediction.txt final decoded prediction
"""
from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


MASK_MARK = "[MASK]"


def _safe_name(value: Any, max_len: int = 120) -> str:
    text = "none" if value is None else str(value)
    text = text.replace(".", "p")
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "_", text).strip("_")
    return (text or "none")[:max_len]


def _num(value: Any, default: int = 0) -> int:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return int(float(str(value)))
    except Exception:
        return default


def _float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        text = str(value).strip().lower()
        if text in {"", "none", "null", "nan"}:
            return None
        return float(value)
    except Exception:
        return None


def _mean(values: Sequence[Any]) -> Optional[float]:
    vals = [_float_or_none(v) for v in values]
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else None


def _json_list(values: Sequence[Any], max_items: int = 64) -> str:
    vals = list(values or [])
    if len(vals) > max_items:
        vals = vals[:max_items] + [f"...(+{len(vals) - max_items})"]
    return json.dumps(vals, ensure_ascii=False)


def _compress_slots(slots: Sequence[Optional[str]], max_chars: int = 1600) -> str:
    parts: List[str] = []
    masks = 0
    for token in slots:
        if token is None:
            masks += 1
            continue
        if masks:
            parts.append(f"{MASK_MARK}x{masks}")
            masks = 0
        parts.append(str(token))
    if masks:
        parts.append(f"{MASK_MARK}x{masks}")
    text = "".join(parts).replace("\r", "\\r").replace("\n", "\\n")
    return text[:max_chars] + "...<truncated>" if max_chars > 0 and len(text) > max_chars else text


def _selected_to_local(position: Any, block_idx: int, block_length: int) -> Optional[int]:
    pos = _num(position, -1)
    start = block_idx * block_length
    if start <= pos < start + block_length:
        return pos - start
    if 0 <= pos < block_length:
        return pos
    return None


def _decode_token(tokenizer: Any, token_id: Any) -> str:
    if tokenizer is None:
        return str(token_id)
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return str(token_id)


def _step_records_for_sample(trace: Dict[str, Any], batch_item_idx: int, tokenizer: Any = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw in trace.get("step_stats") or []:
        if _num(raw.get("batch_item_idx"), 0) != batch_item_idx:
            continue
        row = dict(raw)
        if not row.get("selected_decoded_tokens"):
            row["selected_decoded_tokens"] = [
                _decode_token(tokenizer, token_id)
                for token_id in row.get("selected_token_ids") or []
            ]
        rows.append(row)
    rows.sort(key=lambda r: (_num(r.get("block_idx"), 0), _num(r.get("step_idx_in_block"), _num(r.get("step_idx"), 0))))
    return rows


def build_block_artifacts(
    record: Dict[str, Any],
    trace: Dict[str, Any],
    batch_item_idx: int = 0,
    tokenizer: Any = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    gen_length = _num(record.get("gen_length") or trace.get("gen_length"), 0)
    block_length = _num(record.get("block_length") or record.get("gen_blocksize") or trace.get("block_length"), gen_length or 1)
    block_length = max(block_length, 1)
    num_blocks = max(1, math.ceil(gen_length / block_length)) if gen_length else 1

    steps = _step_records_for_sample(trace, batch_item_idx, tokenizer=tokenizer)
    block_slots: Dict[int, List[Optional[str]]] = {idx: [None] * block_length for idx in range(num_blocks)}
    block_rounds: Dict[int, int] = defaultdict(int)
    step_events: List[Dict[str, Any]] = []
    timeline_rows: List[Dict[str, Any]] = []

    initial = {
        "global_step_idx": -1,
        "active_block_idx": "",
        "block_local_round": -1,
        "selected_count": 0,
        "transfer_reason": "initial_all_mask",
    }
    initial.update({f"block_{idx:02d}": _compress_slots(block_slots[idx]) for idx in range(num_blocks)})
    timeline_rows.append(initial)

    for step in steps:
        block_idx = min(num_blocks - 1, max(0, _num(step.get("block_idx"), 0)))
        local_round = _num(step.get("step_idx_in_block"), block_rounds[block_idx])
        block_rounds[block_idx] = max(block_rounds[block_idx], local_round + 1)

        positions = step.get("selected_positions") or []
        token_ids = step.get("selected_token_ids") or []
        decoded = step.get("selected_decoded_tokens") or [_decode_token(tokenizer, token_id) for token_id in token_ids]
        confidences = step.get("selected_confidences") or []

        visible_before = sum(slot is not None for slot in block_slots[block_idx])
        for position, token in zip(positions, decoded):
            local_pos = _selected_to_local(position, block_idx, block_length)
            if local_pos is not None and 0 <= local_pos < block_length:
                block_slots[block_idx][local_pos] = str(token)
        visible_after = sum(slot is not None for slot in block_slots[block_idx])
        mean_conf = _mean(confidences)

        step_events.append({
            "global_step_idx": _num(step.get("step_idx"), len(step_events)),
            "block_idx": block_idx,
            "block_local_round": local_round,
            "selected_count": len(positions),
            "mean_selected_confidence": round(mean_conf, 6) if mean_conf is not None else None,
            "min_selected_confidence": min(confidences) if confidences else None,
            "max_selected_confidence": max(confidences) if confidences else None,
            "transfer_reason": step.get("transfer_reason"),
            "selected_positions": _json_list(positions),
            "selected_token_ids": _json_list(token_ids),
            "selected_decoded_tokens": _json_list(decoded),
            "block_completion_rate": round(visible_after / block_length, 6),
            "block_visible_before": visible_before,
            "block_visible_after": visible_after,
            "mask_count_before": step.get("mask_count_before"),
            "mask_count_after": step.get("mask_count_after"),
            "scheduled_transfer_count": step.get("scheduled_transfer_count"),
            "threshold_passed_count": step.get("threshold_passed_count"),
            "fallback_forced_count": step.get("fallback_forced_count"),
            "actual_transfer_count": step.get("actual_transfer_count"),
            "current_completion_rate": step.get("current_completion_rate"),
            "cumulative_transferred_tokens": step.get("cumulative_transferred_tokens"),
            "block_state": _compress_slots(block_slots[block_idx]),
        })

        row = {
            "global_step_idx": _num(step.get("step_idx"), len(timeline_rows) - 1),
            "active_block_idx": block_idx,
            "block_local_round": local_round,
            "selected_count": len(positions),
            "transfer_reason": step.get("transfer_reason"),
        }
        row.update({f"block_{idx:02d}": _compress_slots(block_slots[idx]) for idx in range(num_blocks)})
        timeline_rows.append(row)

    block_metrics: List[Dict[str, Any]] = []
    for idx in range(num_blocks):
        events = [event for event in step_events if event["block_idx"] == idx]
        selected_total = sum(_num(event.get("selected_count"), 0) for event in events)
        final_visible = sum(slot is not None for slot in block_slots[idx])
        block_metrics.append({
            "block_idx": idx,
            "block_length": block_length,
            "local_rounds": len(events),
            "selected_tokens_total": selected_total,
            "final_completion_rate": round(final_visible / block_length, 6),
            "final_visible_tokens": final_visible,
            "final_mask_tokens": block_length - final_visible,
            "threshold_pass_steps": sum(event.get("transfer_reason") == "threshold_pass" for event in events),
            "fallback_steps": sum("fallback" in str(event.get("transfer_reason")) for event in events),
            "mean_selected_count_per_round": round(selected_total / len(events), 6) if events else None,
            "final_block_state": _compress_slots(block_slots[idx], max_chars=2400),
        })
    return step_events, block_metrics, timeline_rows


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def write_live_sample_artifacts(
    output_root: Any,
    record: Dict[str, Any],
    trace: Dict[str, Any],
    batch_item_idx: int = 0,
    response: Optional[str] = None,
    tokenizer: Any = None,
) -> Path:
    root = Path(output_root).expanduser().resolve()
    benchmark = _safe_name(record.get("benchmark"))
    sample_idx = _num(record.get("sample_idx"), 0)
    condition = "__".join([
        _safe_name(record.get("decoding_config_name") or "condition"),
        f"steps{_num(record.get('steps'), _num(record.get('gen_steps'), 0))}",
        f"thr{_safe_name(record.get('token_selection_confidence_threshold'))}",
        f"block{_num(record.get('block_length'), _num(record.get('gen_blocksize'), 0))}",
    ])
    if (root.parent / "config.json").exists() or (root.parent / "run.json").exists():
        sample_dir = root / f"sample_{sample_idx:04d}"
    else:
        sample_dir = root / benchmark / f"sample_{sample_idx:04d}" / condition
    sample_dir.mkdir(parents=True, exist_ok=True)

    step_events, block_metrics, timeline_rows = build_block_artifacts(record, trace, batch_item_idx=batch_item_idx, tokenizer=tokenizer)
    metric_keys = [
        "task_id", "benchmark", "sample_idx", "decoding_config_name", "failure_type",
        "elapsed_seconds", "tokens_per_second", "gen_length", "steps", "gen_steps",
        "block_length", "gen_blocksize", "token_selection_confidence_threshold",
        "min_transfer_tokens", "effective_parallelism", "arness", "completion_rate",
        "actual_parallelism", "actual_arness", "threshold_pass_rate", "fallback_rate",
        "final_mask_count", "cuda_max_memory_allocated_mb", "cuda_max_memory_reserved_mb",
    ]
    metrics = {key: record.get(key) for key in metric_keys if key in record}
    metrics["artifact_dir"] = str(sample_dir)
    metrics["num_blocks"] = len(block_metrics)
    metrics["trace_rows"] = len(step_events)

    (sample_dir / "sample_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (sample_dir / "final_prediction.txt").write_text(str(response if response is not None else record.get("prediction", "")), encoding="utf-8")
    _write_csv(sample_dir / "step_events.csv", step_events)
    _write_csv(sample_dir / "block_metrics.csv", block_metrics)
    _write_csv(sample_dir / "block_timeline.csv", timeline_rows)

    plot_row = dict(metrics)
    plot_row.update({
        "mean_block_completion": round(_mean([row.get("final_completion_rate") for row in block_metrics]) or 0.0, 6),
        "total_fallback_steps": sum(_num(row.get("fallback_steps"), 0) for row in block_metrics),
        "total_threshold_pass_steps": sum(_num(row.get("threshold_pass_steps"), 0) for row in block_metrics),
    })
    _write_csv(sample_dir / "plot_rows.csv", [plot_row])
    return sample_dir


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def export_rows(summary_rows: Sequence[Dict[str, Any]], trace_rows: Sequence[Dict[str, Any]], output_root: Any) -> List[Path]:
    by_sample: Dict[int, Dict[str, Any]] = defaultdict(lambda: {"step_stats": []})
    for row in trace_rows:
        idx = _num(row.get("sample_idx"), -1)
        if idx >= 0:
            by_sample[idx]["step_stats"].append(row)
    written: List[Path] = []
    for record in summary_rows:
        idx = _num(record.get("sample_idx"), -1)
        if idx < 0:
            continue
        trace = by_sample.get(idx, {"step_stats": []})
        if trace.get("step_stats"):
            written.append(write_live_sample_artifacts(output_root, record, trace, response=record.get("prediction")))
    return written


def export_run_dir(run_dir: Any, output_root: Any) -> List[Path]:
    run_dir = Path(run_dir)
    return export_rows(read_jsonl(run_dir / "summary.jsonl"), read_jsonl(run_dir / "trace.jsonl"), output_root)
