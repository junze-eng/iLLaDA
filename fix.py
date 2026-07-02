#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix.py

统一修复 iLLaDA outputs 里因为“顺序编号命名”导致混乱的 Task2 context 和 Task3 arness trace 输出。

只做两类事：
1. 检测 outputs 里的旧 arness trace 目录：
     outputs/.../arness_trace_gsm8k_sample7_gsm8k_13
     outputs/.../arness_trace_mbpp_sample6_mbpp_4
   按 test_config.yaml 里的 arness 配置展开顺序，移动/复制到：
     outputs/arness/arness_trace_gsm8k_sample7/
       gsm8k_sample7_len256_block64_steps64_thr0p8/

2. 检测 outputs 里的旧 context all 目录：
     outputs/.../task2_ruler_niah_1_2_4_8k_ruler_niah_single_1_1
   按 summary.jsonl 里的真实 context_length / needle_position 拆分，移动/复制到：
     outputs/context/task2_ruler_niah_1_2_4_8k/
       ruler_niah_single_1_ctx1024_posfront_len128_steps128_block128/

不会重新跑模型。
默认 copy，不破坏原始 outputs。
建议先 dry-run。

用法：
  cd /workspace/iLLaDA

  # 先看计划，不改文件
  python fix.py --config test_config.yaml --outputs-root outputs --dry-run

  # 执行修复，复制到新规范目录
  python fix.py --config test_config.yaml --outputs-root outputs --mode copy --overwrite

  # 如果确认要移动旧目录
  python fix.py --config test_config.yaml --outputs-root outputs --mode move --overwrite

可选：
  --tasks arness
  --tasks context
  --tasks arness context

输出：
  outputs/arness/fix_arness_index.csv
  outputs/arness/fix_arness_missing.csv
  outputs/context/fix_context_index.csv
  outputs/context/fix_context_missing.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# -------------------------
# common helpers
# -------------------------

def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as e:
        raise SystemExit("缺少 PyYAML：pip install pyyaml") from e
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
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


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


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


def to_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "" or s.lower() in {"none", "null", "nan"}:
            return default
        return int(float(s))
    except Exception:
        return default


def norm_pos(x: Any) -> str:
    return str(x).strip().lower()


def threshold_name(x: Any) -> str:
    if x is None:
        return "none"
    s = str(x).strip()
    if s == "" or s.lower() in {"none", "null", "nan"}:
        return "none"
    try:
        return f"{float(s):g}".replace(".", "p")
    except Exception:
        return safe_part(s)


def expand_experiment(exp: Dict[str, Any]) -> List[Dict[str, Any]]:
    base = dict(exp.get("params", {}))
    sweep = exp.get("sweep") or {}
    if not sweep:
        return [base]
    keys = list(sweep.keys())
    vals = [sweep[k] for k in keys]
    out = []
    for combo in itertools.product(*vals):
        row = dict(base)
        for k, v in zip(keys, combo):
            row[k] = v
        out.append(row)
    return out


def first_sample_idx(params: Dict[str, Any]) -> int:
    if params.get("sample_indices"):
        return int(params["sample_indices"][0])
    if "sample_idx" in params:
        return int(params["sample_idx"])
    return 0


def remove_dir_if_needed(dst: Path, overwrite: bool) -> bool:
    if not dst.exists():
        return True
    if overwrite:
        shutil.rmtree(dst)
        return True
    print(f"[SKIP] exists: {dst}")
    return False


def copy_or_move_dir(src: Path, dst: Path, mode: str, overwrite: bool, dry_run: bool) -> None:
    print(f"[PLAN] {src} -> {dst}")
    if dry_run:
        return
    if not remove_dir_if_needed(dst, overwrite):
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copytree(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    else:
        raise ValueError(mode)


def get_block(cfg: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    if name in cfg:
        return cfg[name]
    tasks = cfg.get("tasks")
    if isinstance(tasks, dict) and name in tasks:
        return tasks[name]
    return None


def scan_run_dirs(outputs_root: Path) -> List[Path]:
    """
    扫 outputs 下所有含 summary/trace/aggregate 的一级 run 目录。
    会跳过 sample_traces、visual，以及已经规范化的 condition 子目录。
    """
    run_dirs = []
    for p in outputs_root.rglob("*"):
        if not p.is_dir():
            continue
        parts = set(p.parts)
        if "sample_traces" in parts or "visual" in parts:
            continue
        if (p / "summary.jsonl").exists() or (p / "trace.jsonl").exists() or (p / "aggregate.csv").exists():
            run_dirs.append(p)
    # 去掉父子重复：如果父目录已经是 run dir，就不再收子目录
    run_dirs = sorted(set(run_dirs), key=lambda x: (len(x.parts), str(x)))
    selected = []
    for p in run_dirs:
        if any(str(p).startswith(str(q) + "/") for q in selected):
            continue
        selected.append(p)
    return selected


# -------------------------
# arness task3 fix
# -------------------------

def arness_condition_key(exp: Dict[str, Any], cond: Dict[str, Any]) -> str:
    benchmark = exp["benchmark"]
    params = exp.get("params", {})
    sample_idx = first_sample_idx(params)
    gen_length = cond.get("gen_length", params.get("gen_length"))
    block = cond.get("gen_blocksize", params.get("gen_blocksize", params.get("block_length")))
    steps = cond.get("gen_steps", params.get("gen_steps"))
    threshold = cond.get("token_selection_confidence_threshold", params.get("token_selection_confidence_threshold"))
    return (
        f"{safe_part(benchmark).lower()}_sample{sample_idx}"
        f"_len{safe_part(gen_length)}"
        f"_block{safe_part(block)}"
        f"_steps{safe_part(steps)}"
        f"_thr{threshold_name(threshold)}"
    )


def arness_output_path(arness_cfg: Dict[str, Any], exp: Dict[str, Any], root_override: Optional[str]) -> Path:
    if root_override:
        return Path(root_override) / exp["name"]
    if exp.get("output_path"):
        return Path(exp["output_path"])
    if arness_cfg.get("output_path"):
        return Path(arness_cfg["output_path"]) / exp["name"]
    return Path("outputs/arness") / exp["name"]


def trailing_index(path: Path, exp_name: str, benchmark: str) -> Optional[int]:
    name = path.name
    patterns = [
        rf"^{re.escape(exp_name)}_{re.escape(benchmark)}_(\d+)$",
        rf"^{re.escape(exp_name)}_(\d+)$",
    ]
    for pat in patterns:
        m = re.match(pat, name)
        if m:
            return int(m.group(1))
    return None


def fix_arness(cfg: Dict[str, Any], outputs_root: Path, root_override: Optional[str], mode: str, overwrite: bool, dry_run: bool) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    arness = get_block(cfg, "arness")
    if not arness:
        print("[SKIP] no arness block")
        return [], [], []

    all_dirs = scan_run_dirs(outputs_root)
    index_rows: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, Any]] = []
    extra_rows: List[Dict[str, Any]] = []

    for exp in arness.get("experiments", []):
        exp_name = exp.get("name")
        benchmark = exp.get("benchmark")
        if not exp_name or not benchmark:
            continue
        if not str(exp_name).startswith("arness_trace_"):
            continue

        conditions = expand_experiment(exp)
        expected = {i: cond for i, cond in enumerate(conditions, start=1)}
        task_out = arness_output_path(arness, exp, root_override)

        old = []
        for d in all_dirs:
            idx = trailing_index(d, exp_name, benchmark)
            if idx is not None:
                old.append((idx, d))
        old.sort(key=lambda x: (x[0], str(x[1])))

        print(f"\n[ARNESS] {exp_name}")
        print(f"  output_path: {task_out}")
        print(f"  expected: {len(conditions)}")
        print(f"  found old: {len(old)}")

        seen_idx = set()
        duplicate_counter: Dict[str, int] = defaultdict(int)

        for idx, src in old:
            if idx not in expected:
                extra_rows.append({
                    "task": "arness",
                    "experiment": exp_name,
                    "condition_index": idx,
                    "source": str(src),
                    "reason": "index_out_of_config_range",
                })
                continue

            cond = expected[idx]
            key = arness_condition_key(exp, cond)
            duplicate_counter[key] += 1
            dst_name = key if duplicate_counter[key] == 1 else f"{key}_dup{duplicate_counter[key]}"
            dst = task_out / dst_name
            copy_or_move_dir(src, dst, mode=mode, overwrite=overwrite, dry_run=dry_run)
            seen_idx.add(idx)

            index_rows.append({
                "task": "arness",
                "experiment": exp_name,
                "condition_index": idx,
                "condition_key": key,
                "source": str(src),
                "destination": str(dst),
                "benchmark": benchmark,
                "gen_length": cond.get("gen_length", exp.get("params", {}).get("gen_length")),
                "gen_blocksize": cond.get("gen_blocksize", exp.get("params", {}).get("gen_blocksize")),
                "gen_steps": cond.get("gen_steps"),
                "threshold": cond.get("token_selection_confidence_threshold"),
            })

        for idx, cond in expected.items():
            if idx not in seen_idx:
                missing_rows.append({
                    "task": "arness",
                    "experiment": exp_name,
                    "condition_index": idx,
                    "condition_key": arness_condition_key(exp, cond),
                    "expected_destination": str(task_out / arness_condition_key(exp, cond)),
                })

    out_root = Path(root_override) if root_override else Path(arness.get("output_path", "outputs/arness"))
    if not dry_run:
        write_csv(out_root / "fix_arness_index.csv", index_rows)
        write_jsonl(out_root / "fix_arness_index.jsonl", index_rows)
        write_csv(out_root / "fix_arness_missing.csv", missing_rows)
        write_csv(out_root / "fix_arness_extra.csv", extra_rows)

    return index_rows, missing_rows, extra_rows


# -------------------------
# context task2 fix
# -------------------------

def context_output_path(context_cfg: Dict[str, Any], exp: Dict[str, Any], root_override: Optional[str]) -> Path:
    if root_override:
        return Path(root_override) / exp["name"]
    if exp.get("output_path"):
        return Path(exp["output_path"])
    if context_cfg.get("output_path"):
        return Path(context_cfg["output_path"]) / exp["name"]
    return Path("outputs/context") / exp["name"]


def context_condition_key(benchmark: str, params: Dict[str, Any]) -> str:
    return (
        f"{safe_part(benchmark).lower()}"
        f"_ctx{safe_part(params.get('context_length'))}"
        f"_pos{safe_part(params.get('needle_position'))}"
        f"_len{safe_part(params.get('gen_length'))}"
        f"_steps{safe_part(params.get('gen_steps'))}"
        f"_block{safe_part(params.get('gen_blocksize'))}"
    )


def group_summary_by_context(summary_rows: List[Dict[str, Any]]) -> List[Tuple[Tuple[int, str], List[Dict[str, Any]]]]:
    groups: List[Tuple[Tuple[int, str], List[Dict[str, Any]]]] = []
    cur_key = None
    cur_rows: List[Dict[str, Any]] = []
    for r in summary_rows:
        ctx = to_int(r.get("context_length", r.get("param_context_length")), -1)
        pos = norm_pos(r.get("needle_position", r.get("param_needle_position", "")))
        key = (ctx, pos)
        if cur_key is None:
            cur_key = key
            cur_rows = [r]
        elif key == cur_key:
            cur_rows.append(r)
        else:
            groups.append((cur_key, cur_rows))
            cur_key = key
            cur_rows = [r]
    if cur_key is not None:
        groups.append((cur_key, cur_rows))
    return groups


def split_trace_by_groups(trace_rows: List[Dict[str, Any]], groups: List[Tuple[Tuple[int, str], List[Dict[str, Any]]]], gen_steps: int) -> List[List[Dict[str, Any]]]:
    chunks = []
    offset = 0
    for _, rows in groups:
        sample_ids = []
        for i, r in enumerate(rows):
            sample_ids.append(to_int(r.get("sample_idx"), i))
        n_samples = len(set(sample_ids)) if sample_ids else 0
        n = n_samples * gen_steps
        chunks.append(trace_rows[offset: offset + n])
        offset += n
    if offset < len(trace_rows) and chunks:
        chunks[-1].extend(trace_rows[offset:])
    return chunks


def matching_csv_rows(rows: List[Dict[str, Any]], ctx: int, pos: str) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        c = to_int(r.get("context_length", r.get("param_context_length")), -999)
        p = norm_pos(r.get("needle_position", r.get("param_needle_position", "")))
        if c == ctx and p == pos:
            out.append(r)
    return out


def fix_context(cfg: Dict[str, Any], outputs_root: Path, root_override: Optional[str], overwrite: bool, dry_run: bool) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    context = get_block(cfg, "context")
    if not context:
        print("[SKIP] no context block")
        return [], []

    all_dirs = scan_run_dirs(outputs_root)
    index_rows: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, Any]] = []

    for exp in context.get("experiments", []):
        exp_name = exp.get("name")
        benchmark = exp.get("benchmark")
        if not exp_name or not benchmark:
            continue
        # 只处理 task2 ruler context
        if not str(exp_name).startswith("task2_ruler"):
            continue

        conditions = expand_experiment(exp)
        expected_by_key: Dict[Tuple[int, str], Dict[str, Any]] = {}
        for cond in conditions:
            expected_by_key[(to_int(cond.get("context_length"), -1), norm_pos(cond.get("needle_position")))] = cond

        task_out = context_output_path(context, exp, root_override)

        old_dirs = []
        for d in all_dirs:
            if d.name.startswith(exp_name):
                old_dirs.append(d)
        old_dirs.sort(key=lambda p: str(p))

        print(f"\n[CONTEXT] {exp_name}")
        print(f"  output_path: {task_out}")
        print(f"  expected: {len(conditions)}")
        print(f"  found old: {len(old_dirs)}")

        seen = set()

        for src in old_dirs:
            summary = read_jsonl(src / "summary.jsonl")
            trace = read_jsonl(src / "trace.jsonl")
            agg = read_csv_rows(src / "aggregate.csv")
            run_sum = read_csv_rows(src / "run_summary.csv")

            if not summary:
                # 没 summary 无法安全拆分，跳过并记录
                index_rows.append({
                    "task": "context",
                    "experiment": exp_name,
                    "source": str(src),
                    "status": "skipped_no_summary",
                })
                continue

            groups = group_summary_by_context(summary)
            first_cond = None
            if groups and groups[0][0] in expected_by_key:
                first_cond = expected_by_key[groups[0][0]]
            gen_steps = to_int((first_cond or exp.get("params", {})).get("gen_steps"), 128)
            trace_chunks = split_trace_by_groups(trace, groups, gen_steps)

            for (gkey, rows), trace_chunk in zip(groups, trace_chunks):
                ctx, pos = gkey
                if gkey not in expected_by_key:
                    index_rows.append({
                        "task": "context",
                        "experiment": exp_name,
                        "source": str(src),
                        "context_length": ctx,
                        "needle_position": pos,
                        "status": "skipped_not_in_config",
                    })
                    continue

                cond = expected_by_key[gkey]
                key = context_condition_key(benchmark, cond)
                dst = task_out / key
                seen.add(gkey)

                official_agg = matching_csv_rows(agg, ctx, pos)
                official_run = matching_csv_rows(run_sum, ctx, pos)
                status = "complete" if official_agg and any(to_int(r.get("returncode"), 1) == 0 for r in official_agg) else "partial"

                print(f"[PLAN] {src} [{ctx}/{pos}, samples={len(rows)}, trace_rows={len(trace_chunk)}, {status}] -> {dst}")

                if not dry_run:
                    if remove_dir_if_needed(dst, overwrite):
                        dst.mkdir(parents=True, exist_ok=True)
                        write_jsonl(dst / "summary.jsonl", rows)
                        write_jsonl(dst / "trace.jsonl", trace_chunk)
                        write_csv(dst / "aggregate.csv", official_agg)
                        write_csv(dst / "run_summary.csv", official_run)
                        write_json(dst / "config.json", {
                            "task": "context",
                            "experiment": exp_name,
                            "benchmark": benchmark,
                            "run_name": key,
                            "params": cond,
                            "source_mixed_run_dir": str(src),
                            "repair_status": status,
                        })
                        if (src / "gpu_telemetry.csv").exists():
                            shutil.copy2(src / "gpu_telemetry.csv", dst / "gpu_telemetry.raw_from_mixed_run.csv")

                index_rows.append({
                    "task": "context",
                    "experiment": exp_name,
                    "condition_key": key,
                    "source": str(src),
                    "destination": str(dst),
                    "benchmark": benchmark,
                    "context_length": ctx,
                    "needle_position": pos,
                    "samples": len(rows),
                    "trace_rows": len(trace_chunk),
                    "status": status,
                    "has_official_aggregate": bool(official_agg),
                })

        for cond in conditions:
            key_tuple = (to_int(cond.get("context_length"), -1), norm_pos(cond.get("needle_position")))
            if key_tuple not in seen:
                key = context_condition_key(benchmark, cond)
                missing_rows.append({
                    "task": "context",
                    "experiment": exp_name,
                    "condition_key": key,
                    "context_length": key_tuple[0],
                    "needle_position": key_tuple[1],
                    "expected_destination": str(task_out / key),
                })

    out_root = Path(root_override) if root_override else Path(context.get("output_path", "outputs/context"))
    if not dry_run:
        write_csv(out_root / "fix_context_index.csv", index_rows)
        write_jsonl(out_root / "fix_context_index.jsonl", index_rows)
        write_csv(out_root / "fix_context_missing.csv", missing_rows)

    return index_rows, missing_rows


# -------------------------
# main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="test_config.yaml")
    ap.add_argument("--outputs-root", default="outputs", help="Scan old outputs under this root.")
    ap.add_argument("--arness-root", default=None, help="Override new arness root, default from config or outputs/arness.")
    ap.add_argument("--context-root", default=None, help="Override new context root, default from config or outputs/context.")
    ap.add_argument("--tasks", nargs="+", choices=["arness", "context"], default=["arness", "context"])
    ap.add_argument("--mode", choices=["copy", "move"], default="copy", help="Move/copy only applies to arness. Context mixed dirs are always split by copy.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    outputs_root = Path(args.outputs_root)

    if "arness" in args.tasks:
        ar_idx, ar_missing, ar_extra = fix_arness(
            cfg=cfg,
            outputs_root=outputs_root,
            root_override=args.arness_root,
            mode=args.mode,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        print("\n[ARNESS SUMMARY]")
        print(f"  fixed: {len(ar_idx)}")
        print(f"  missing: {len(ar_missing)}")
        print(f"  extra: {len(ar_extra)}")

    if "context" in args.tasks:
        ctx_idx, ctx_missing = fix_context(
            cfg=cfg,
            outputs_root=outputs_root,
            root_override=args.context_root,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        print("\n[CONTEXT SUMMARY]")
        print(f"  recovered/split: {len(ctx_idx)}")
        print(f"  missing: {len(ctx_missing)}")


if __name__ == "__main__":
    main()
