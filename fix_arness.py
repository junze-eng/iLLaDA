#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_arness_split_runs.py

Repair ARness Task3 outputs that were run separately and therefore ended up as
condition directories with the experiment name duplicated, for example:

  outputs/arness/arness_trace_gsm8k_sample7_gsm8k_sample7_len256_block64_steps64_thr0p6
  outputs/arness/arness_trace_gsm8k_sample7/arness_trace_gsm8k_sample7_gsm8k_sample7_len256_block64_steps64_thr0p6

into canonical layout:

  outputs/arness/arness_trace_gsm8k_sample7/
    gsm8k_sample7_len256_block64_steps64_thr0p6/

It then rebuilds experiment-level and global tables:

  outputs/arness/<experiment>/aggregate.csv
  outputs/arness/<experiment>/summary_all.csv
  outputs/arness/<experiment>/run_manifest.jsonl
  outputs/arness/<experiment>/repair_index.csv

  outputs/arness/aggregate_all.csv
  outputs/arness/summary_all.csv
  outputs/arness/repair_index_all.csv

The script does not rerun model inference. It only copies/moves files and rebuilds
CSV/JSONL indexes from per-condition outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

COND_RE = re.compile(
    r"^(?P<benchmark>gsm8k|mbpp)_sample(?P<sample>\d+)"
    r"_len(?P<gen_length>\d+)_block(?P<block>\d+)"
    r"_steps(?P<steps>\d+)_thr(?P<thr>[A-Za-z0-9p.-]+)$",
    re.IGNORECASE,
)
PREFIX_RE = re.compile(
    r"^(?P<experiment>arness_trace_(?P<benchmark>gsm8k|mbpp)_sample(?P<sample>\d+))_"
    r"(?P<condition>(?P=benchmark)_sample(?P=sample)_len\d+_block\d+_steps\d+_thr[A-Za-z0-9p.-]+)$",
    re.IGNORECASE,
)

RUN_MARKERS = ("aggregate.csv", "run_summary.csv", "summary.jsonl", "trace.jsonl", "config.json")
SKIP_PARTS = {"sample_traces", "visual_trace", "visual_arness_overview", "generated_configs", "__pycache__"}


def is_condition_payload_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if any(part in SKIP_PARTS for part in path.parts):
        return False
    return any((path / marker).exists() for marker in RUN_MARKERS)


def parse_condition_name(name: str) -> Optional[Dict[str, Any]]:
    m = PREFIX_RE.match(name)
    if m:
        cond = m.group("condition")
        cm = COND_RE.match(cond)
        if not cm:
            return None
        d = cm.groupdict()
        d["experiment"] = m.group("experiment")
        d["condition"] = cond
        d["old_prefixed"] = name
        return normalize_parsed(d)

    m = COND_RE.match(name)
    if m:
        d = m.groupdict()
        d["condition"] = name
        d["experiment"] = f"arness_trace_{d['benchmark'].lower()}_sample{d['sample']}"
        d["old_prefixed"] = None
        return normalize_parsed(d)

    return None


def normalize_parsed(d: Dict[str, Any]) -> Dict[str, Any]:
    benchmark = str(d["benchmark"]).lower()
    sample = int(d["sample"])
    gen_length = int(d["gen_length"])
    block = int(d["block"])
    steps = int(d["steps"])
    thr = str(d["thr"]).lower().replace(".", "p")
    if thr in {"nan", "null", "none", ""}:
        thr = "none"
    condition = f"{benchmark}_sample{sample}_len{gen_length}_block{block}_steps{steps}_thr{thr}"
    experiment = f"arness_trace_{benchmark}_sample{sample}"
    threshold: Optional[float]
    if thr == "none":
        threshold = None
    else:
        try:
            threshold = float(thr.replace("p", "."))
        except Exception:
            threshold = None
    return {
        "benchmark": benchmark,
        "sample_idx": sample,
        "gen_length": gen_length,
        "gen_blocksize": block,
        "block_length": block,
        "gen_steps": steps,
        "steps": steps,
        "threshold_label": thr,
        "token_selection_confidence_threshold": threshold,
        "experiment": experiment,
        "condition": condition,
        "old_prefixed": d.get("old_prefixed"),
    }


def scan_candidate_dirs(roots: Sequence[Path]) -> List[Tuple[Path, Dict[str, Any]]]:
    found: List[Tuple[Path, Dict[str, Any]]] = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if not is_condition_payload_dir(p):
                continue
            parsed = parse_condition_name(p.name)
            if not parsed:
                continue
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            found.append((p, parsed))
    return found


def same_path(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except Exception:
        return str(a) == str(b)


def copy_or_move(src: Path, dst: Path, mode: str, overwrite: bool, dry_run: bool) -> str:
    if same_path(src, dst):
        print(f"[KEEP] {src}")
        return "already_canonical"

    print(f"[PLAN] {src} -> {dst}")
    if dry_run:
        return "dry_run"

    if dst.exists():
        if overwrite:
            shutil.rmtree(dst)
        else:
            print(f"[SKIP] destination exists: {dst}")
            return "skipped_exists"
    dst.parent.mkdir(parents=True, exist_ok=True)

    if mode == "copy":
        shutil.copytree(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    else:
        raise ValueError(mode)
    return "ok"


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


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


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] failed reading {path}: {exc}")
        return pd.DataFrame()


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def clean_nan(v: Any) -> Any:
    try:
        if isinstance(v, float) and math.isnan(v):
            return None
    except Exception:
        pass
    return v


def df_to_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    rows = []
    for r in df.to_dict(orient="records"):
        rows.append({k: clean_nan(v) for k, v in r.items()})
    return rows


def condition_dirs_for_experiment(exp_dir: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    out: List[Tuple[Path, Dict[str, Any]]] = []
    if not exp_dir.exists():
        return out
    for p in sorted(exp_dir.iterdir()):
        if not is_condition_payload_dir(p):
            continue
        parsed = parse_condition_name(p.name)
        if parsed and parsed["condition"] == p.name:
            out.append((p, parsed))
    return out


def standardize_row(row: Dict[str, Any], cond_dir: Path, parsed: Dict[str, Any], source_file: str) -> Dict[str, Any]:
    row = dict(row)
    old_run = row.get("run_name") or row.get("decoding_config_name")
    cond = parsed["condition"]
    row["source_run_name"] = old_run
    row["run_name"] = cond
    row["decoding_config_name"] = cond
    row["experiment"] = parsed["experiment"]
    row["benchmark"] = parsed["benchmark"]
    row["condition_key"] = cond
    row["condition_dir"] = str(cond_dir)
    row["source_table"] = source_file

    # Fill/normalize config params used by plotting/report tables.
    row.setdefault("primary_metric_name", row.get("metric_name", "accuracy"))
    row["gen_length"] = row.get("gen_length", row.get("param_gen_length", parsed["gen_length"])) or parsed["gen_length"]
    row["gen_steps"] = row.get("gen_steps", row.get("param_gen_steps", parsed["gen_steps"])) or parsed["gen_steps"]
    row["gen_blocksize"] = row.get("gen_blocksize", row.get("param_gen_blocksize", parsed["gen_blocksize"])) or parsed["gen_blocksize"]
    row["block_length"] = row.get("block_length", parsed["block_length"])
    row["sample_idx"] = row.get("sample_idx", parsed["sample_idx"])
    row["threshold_label"] = parsed["threshold_label"]
    row["token_selection_confidence_threshold"] = row.get(
        "token_selection_confidence_threshold",
        row.get("param_token_selection_confidence_threshold", parsed["token_selection_confidence_threshold"]),
    )
    if row["token_selection_confidence_threshold"] in ("", "nan", "None"):
        row["token_selection_confidence_threshold"] = parsed["token_selection_confidence_threshold"]
    return row


def fallback_aggregate_from_sample_metrics(cond_dir: Path, parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for metrics_path in sorted(cond_dir.glob("sample_traces/sample_*/sample_metrics.json")):
        m = read_json(metrics_path)
        if not m:
            continue
        row = {
            "run_name": parsed["condition"],
            "experiment": parsed["experiment"],
            "benchmark": parsed["benchmark"],
            "decoding_config_name": parsed["condition"],
            "primary_metric_name": "accuracy" if parsed["benchmark"] == "gsm8k" else "score",
            "primary_metric_value": m.get("metric_value"),
            "latency_mean": m.get("elapsed_seconds"),
            "tokens_per_second_mean": m.get("tokens_per_second"),
            "peak_vram": m.get("cuda_max_memory_reserved_mb"),
            "actual_parallelism": m.get("actual_parallelism"),
            "completion_rate": m.get("completion_rate"),
            "gen_length": parsed["gen_length"],
            "gen_steps": parsed["gen_steps"],
            "gen_blocksize": parsed["gen_blocksize"],
            "token_selection_confidence_threshold": parsed["token_selection_confidence_threshold"],
            "num_samples": 1,
            "returncode": 0,
            "source_table": str(metrics_path),
        }
        rows.append(standardize_row(row, cond_dir, parsed, str(metrics_path)))
    return rows


def rebuild_tables(target_root: Path, only_experiments: Optional[Sequence[str]] = None) -> None:
    exp_dirs = sorted([p for p in target_root.iterdir() if p.is_dir() and p.name.startswith("arness_trace_")]) if target_root.exists() else []
    global_agg: List[Dict[str, Any]] = []
    global_sum: List[Dict[str, Any]] = []
    global_manifest: List[Dict[str, Any]] = []

    for exp_dir in exp_dirs:
        if only_experiments and exp_dir.name not in only_experiments:
            continue
        aggregate_rows: List[Dict[str, Any]] = []
        summary_rows: List[Dict[str, Any]] = []
        manifest_rows: List[Dict[str, Any]] = []

        for cond_dir, parsed in condition_dirs_for_experiment(exp_dir):
            agg_df = read_csv(cond_dir / "aggregate.csv")
            cond_agg_rows = [standardize_row(r, cond_dir, parsed, str(cond_dir / "aggregate.csv")) for r in df_to_rows(agg_df)]
            if not cond_agg_rows:
                cond_agg_rows = fallback_aggregate_from_sample_metrics(cond_dir, parsed)
            aggregate_rows.extend(cond_agg_rows)

            run_df = read_csv(cond_dir / "run_summary.csv")
            run_rows = [standardize_row(r, cond_dir, parsed, str(cond_dir / "run_summary.csv")) for r in df_to_rows(run_df)]
            if not run_rows:
                # If run_summary is missing, use plot_rows/sample_metrics as a summary-like row.
                plot_rows = []
                for plot_path in sorted(cond_dir.glob("sample_traces/sample_*/plot_rows.csv")):
                    plot_rows.extend(df_to_rows(read_csv(plot_path)))
                run_rows = [standardize_row(r, cond_dir, parsed, "sample_traces/*/plot_rows.csv") for r in plot_rows]
            summary_rows.extend(run_rows)

            cfg = read_json(cond_dir / "config.json")
            old_run = cfg.get("run_name")
            if cfg:
                cfg = dict(cfg)
                cfg["source_run_name"] = old_run
                cfg["run_name"] = parsed["condition"]
                cfg["condition_key"] = parsed["condition"]
                cfg["condition_dir"] = str(cond_dir)
                cfg["experiment"] = parsed["experiment"]
                cfg["benchmark"] = parsed["benchmark"]
                params = dict(cfg.get("params", {}))
                params.update({
                    "sample_indices": params.get("sample_indices", [parsed["sample_idx"]]),
                    "gen_length": parsed["gen_length"],
                    "gen_steps": parsed["gen_steps"],
                    "gen_blocksize": parsed["gen_blocksize"],
                    "token_selection_confidence_threshold": parsed["token_selection_confidence_threshold"],
                })
                cfg["params"] = params
                manifest_rows.append(cfg)
            else:
                manifest_rows.append({
                    "run_name": parsed["condition"],
                    "condition_key": parsed["condition"],
                    "condition_dir": str(cond_dir),
                    "task": "arness",
                    "experiment": parsed["experiment"],
                    "benchmark": parsed["benchmark"],
                    "params": {
                        "sample_indices": [parsed["sample_idx"]],
                        "gen_length": parsed["gen_length"],
                        "gen_steps": parsed["gen_steps"],
                        "gen_blocksize": parsed["gen_blocksize"],
                        "token_selection_confidence_threshold": parsed["token_selection_confidence_threshold"],
                    },
                })

        # Stable sort: benchmark, sample, len, block, steps descending-ish? Keep steps numeric ascending for plotting.
        aggregate_rows.sort(key=lambda r: (str(r.get("benchmark")), int(float(r.get("gen_steps") or 0)), str(r.get("threshold_label"))))
        summary_rows.sort(key=lambda r: (str(r.get("benchmark")), int(float(r.get("param_gen_steps") or r.get("gen_steps") or 0)), str(r.get("threshold_label"))))
        write_csv(exp_dir / "aggregate.csv", aggregate_rows)
        write_csv(exp_dir / "summary_all.csv", summary_rows)
        write_jsonl(exp_dir / "run_manifest.jsonl", manifest_rows)
        print(f"[OK] rebuilt {exp_dir / 'aggregate.csv'} ({len(aggregate_rows)} rows)")
        print(f"[OK] rebuilt {exp_dir / 'summary_all.csv'} ({len(summary_rows)} rows)")
        print(f"[OK] rebuilt {exp_dir / 'run_manifest.jsonl'} ({len(manifest_rows)} rows)")

        global_agg.extend(aggregate_rows)
        global_sum.extend(summary_rows)
        global_manifest.extend(manifest_rows)

    write_csv(target_root / "aggregate_all.csv", global_agg)
    write_csv(target_root / "summary_all.csv", global_sum)
    write_jsonl(target_root / "run_manifest_all.jsonl", global_manifest)
    print(f"[OK] rebuilt global aggregate: {target_root / 'aggregate_all.csv'} ({len(global_agg)} rows)")
    print(f"[OK] rebuilt global summary:   {target_root / 'summary_all.csv'} ({len(global_sum)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Repair split ARness condition dirs and rebuild total tables.")
    ap.add_argument("--roots", nargs="+", default=["outputs/arness"], help="Roots to scan for old/canonical condition dirs.")
    ap.add_argument("--target-root", default="outputs/arness", help="Canonical arness root.")
    ap.add_argument("--mode", choices=["copy", "move"], default="move")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-rebuild", action="store_true", help="Only rename/copy/move dirs, do not rebuild aggregate tables.")
    ap.add_argument("--only", nargs="*", default=None, help="Only process these experiment names.")
    args = ap.parse_args()

    roots = [Path(x) for x in args.roots]
    target_root = Path(args.target_root)
    candidates = scan_candidate_dirs(roots)

    repair_rows: List[Dict[str, Any]] = []
    for src, parsed in candidates:
        if args.only and parsed["experiment"] not in args.only:
            continue
        dst = target_root / parsed["experiment"] / parsed["condition"]
        status = copy_or_move(src, dst, args.mode, args.overwrite, args.dry_run)
        repair_rows.append({
            "experiment": parsed["experiment"],
            "benchmark": parsed["benchmark"],
            "sample_idx": parsed["sample_idx"],
            "condition_key": parsed["condition"],
            "source": str(src),
            "destination": str(dst),
            "status": status,
            "gen_length": parsed["gen_length"],
            "gen_steps": parsed["gen_steps"],
            "gen_blocksize": parsed["gen_blocksize"],
            "threshold_label": parsed["threshold_label"],
            "token_selection_confidence_threshold": parsed["token_selection_confidence_threshold"],
        })

    if not args.dry_run:
        write_csv(target_root / "repair_index_all.csv", repair_rows)
        # Per-experiment repair index.
        by_exp: Dict[str, List[Dict[str, Any]]] = {}
        for r in repair_rows:
            by_exp.setdefault(str(r["experiment"]), []).append(r)
        for exp, rows in by_exp.items():
            write_csv(target_root / exp / "repair_index.csv", rows)

    print(f"[SUMMARY] mapped condition dirs: {len(repair_rows)}")
    if not args.no_rebuild and not args.dry_run:
        rebuild_tables(target_root, only_experiments=args.only)
    elif args.dry_run:
        print("[DRY-RUN] no files changed; tables not rebuilt.")


if __name__ == "__main__":
    main()
