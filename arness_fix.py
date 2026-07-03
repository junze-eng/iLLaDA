#!/usr/bin/env python3
"""
Safe fixer for old GSM8K arness_trace outputs.

Default source:
  outputs/arness_trace/arness_trace_gsm8k_sample7_gsm8k_<idx>
Default target:
  outputs/arness/arness_trace_gsm8k_sample7/<config-named-run>

This script intentionally does NOT recursively rename internal OpenCompass log/eval folders.
It only copies/moves the run directory to the new top-level name, patches top-level config.json,
and copies/patches the generated config .py when found.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any

OLD_PREFIX = "arness_trace_gsm8k_sample7_gsm8k"
NEW_PREFIX = "arness_trace_gsm8k_sample7_gsm8k_sample7"
GEN_LENGTH = 256
GEN_BLOCKSIZE = 64
GEN_STEPS_SWEEP = [256, 128, 64, 32, 16]
THR_SWEEP = [None, 0.6, 0.7, 0.8, 0.9]


def thr_slug(thr: float | None) -> str:
    if thr is None:
        return "none"
    return str(thr).replace(".", "p")


def old_index_to_params(idx: int) -> tuple[int, float | None]:
    if idx < 1 or idx > len(GEN_STEPS_SWEEP) * len(THR_SWEEP):
        raise ValueError(f"old suffix _{idx} is outside supported range 1..{len(GEN_STEPS_SWEEP) * len(THR_SWEEP)}")
    zero = idx - 1
    steps = GEN_STEPS_SWEEP[zero // len(THR_SWEEP)]
    thr = THR_SWEEP[zero % len(THR_SWEEP)]
    return steps, thr


def new_run_name(steps: int, thr: float | None) -> str:
    return f"{NEW_PREFIX}_len{GEN_LENGTH}_block{GEN_BLOCKSIZE}_steps{steps}_thr{thr_slug(thr)}"


def find_generated_config(src_root: Path, old_name: str) -> Path | None:
    candidates = [
        src_root / "generated_configs" / f"{old_name}.py",
        src_root / f"{old_name}.py",
    ]
    for p in candidates:
        if p.exists():
            return p
    # bounded fallback: only look under source root generated configs if present
    gen_dir = src_root / "generated_configs"
    if gen_dir.exists():
        hits = list(gen_dir.glob(f"*{old_name}*.py"))
        if hits:
            return hits[0]
    return None


def patch_text_file(path: Path, old_name: str, new_name: str, src_root: Path, dst_root: Path) -> None:
    if not path.exists() or not path.is_file():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="ignore")
    text = text.replace(old_name, new_name)
    # Conservative path patching for generated OpenCompass configs.
    text = text.replace(str(src_root), str(dst_root))
    text = text.replace(str(src_root).replace("\\", "/"), str(dst_root).replace("\\", "/"))
    path.write_text(text, encoding="utf-8")


def patch_top_config_json(dest_dir: Path, old_name: str, new_name: str, steps: int, thr: float | None) -> None:
    cfg = dest_dir / "config.json"
    if not cfg.exists():
        return
    try:
        data: Any = json.loads(cfg.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] could not parse {cfg}: {e}")
        return

    def patch_obj(obj: Any) -> Any:
        if isinstance(obj, str):
            return obj.replace(old_name, new_name)
        if isinstance(obj, list):
            return [patch_obj(x) for x in obj]
        if isinstance(obj, dict):
            return {k: patch_obj(v) for k, v in obj.items()}
        return obj

    data = patch_obj(data)
    if isinstance(data, dict):
        data["run_name"] = new_name
        data["gen_steps"] = steps
        data["gen_length"] = GEN_LENGTH
        data["gen_blocksize"] = GEN_BLOCKSIZE
        data["token_selection_confidence_threshold"] = thr
        data["sample_indices"] = [7]
        params = data.get("params")
        if isinstance(params, dict):
            params["gen_steps"] = steps
            params["gen_length"] = GEN_LENGTH
            params["gen_blocksize"] = GEN_BLOCKSIZE
            params["token_selection_confidence_threshold"] = thr
            params["sample_indices"] = [7]
    cfg.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def copy_or_move_dir(src: Path, dest: Path, copy: bool, force: bool) -> str:
    if dest.exists():
        if not force:
            return "SKIP_EXISTS"
        if dest.is_dir():
            shutil.rmtree(dest)
        else:
            dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copytree(src, dest)
        return "COPIED"
    shutil.move(str(src), str(dest))
    return "MOVED"


def main() -> int:
    ap = argparse.ArgumentParser(description="Safely copy old GSM8K arness_trace runs into the new config-named arness layout.")
    ap.add_argument("--repo-root", default=".", help="Repository root. Default: current directory.")
    ap.add_argument("--src", default="outputs/arness_trace", help="Source root containing old arness_trace runs.")
    ap.add_argument("--dst", default="outputs/arness/arness_trace_gsm8k_sample7", help="Target root for new GSM8K arness runs.")
    ap.add_argument("--start", type=int, default=1, help="First old suffix index, inclusive.")
    ap.add_argument("--end", type=int, default=25, help="Last old suffix index, inclusive.")
    ap.add_argument("--apply", action="store_true", help="Actually copy/move files. Without this, dry-run only.")
    ap.add_argument("--copy", action="store_true", help="Copy instead of move. Recommended.")
    ap.add_argument("--force", action="store_true", help="Overwrite target directories/configs if they already exist.")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    src_root = (repo / args.src).resolve() if not Path(args.src).is_absolute() else Path(args.src).resolve()
    dst_root = (repo / args.dst).resolve() if not Path(args.dst).is_absolute() else Path(args.dst).resolve()
    gen_dst = dst_root / "generated_configs"

    print(f"[INFO] repo_root: {repo}")
    print(f"[INFO] source:    {src_root}")
    print(f"[INFO] target:    {dst_root}")
    print(f"[INFO] checking old-name range: _{args.start} .. _{args.end}")
    print("[INFO] safe mode: no recursive internal directory renames")

    rows: list[dict[str, Any]] = []
    found = missing = changed = skipped = 0

    for idx in range(args.start, args.end + 1):
        old_name = f"{OLD_PREFIX}_{idx}"
        src_dir = src_root / old_name
        steps, thr = old_index_to_params(idx)
        new_name = new_run_name(steps, thr)
        dest_dir = dst_root / new_name
        old_cfg = find_generated_config(src_root, old_name)
        new_cfg = gen_dst / f"{new_name}.py"

        row = {
            "idx": idx,
            "old_name": old_name,
            "new_name": new_name,
            "gen_steps": steps,
            "threshold": "none" if thr is None else thr,
            "src_dir": str(src_dir),
            "dest_dir": str(dest_dir),
            "old_generated_config": str(old_cfg) if old_cfg else "",
            "new_generated_config": str(new_cfg),
            "status": "",
        }

        if not src_dir.exists():
            missing += 1
            row["status"] = "MISS"
            rows.append(row)
            print(f"[MISS] {old_name} -> {new_name}")
            continue

        found += 1
        print(f"[PLAN] {old_name} -> {new_name}  steps={steps} thr={thr}")
        if not args.apply:
            row["status"] = "PLAN"
            rows.append(row)
            continue

        status = copy_or_move_dir(src_dir, dest_dir, args.copy, args.force)
        if status == "SKIP_EXISTS":
            skipped += 1
            row["status"] = status
            rows.append(row)
            print(f"[SKIP] target exists, use --force to overwrite: {dest_dir}")
            continue

        # Only patch top-level config.json and generated config copy. Do not rename internal log dirs.
        patch_top_config_json(dest_dir, old_name, new_name, steps, thr)
        if old_cfg:
            gen_dst.mkdir(parents=True, exist_ok=True)
            if new_cfg.exists() and args.force:
                new_cfg.unlink()
            if not new_cfg.exists():
                shutil.copy2(old_cfg, new_cfg)
            patch_text_file(new_cfg, old_name, new_name, src_root, dst_root)
        else:
            print(f"[WARN] generated config not found for {old_name}")

        changed += 1
        row["status"] = status
        rows.append(row)
        print(f"[{status}] {dest_dir}")

    if args.apply:
        dst_root.mkdir(parents=True, exist_ok=True)
    mapping_path = dst_root / "arness_trace_gsm8k_fix_mapping.csv"
    if args.apply or dst_root.exists():
        dst_root.mkdir(parents=True, exist_ok=True)
        with mapping_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["idx", "status"])
            writer.writeheader()
            writer.writerows(rows)

    print(f"[SUMMARY] found={found}, missing={missing}, changed={changed}, skipped={skipped}, checked={args.end - args.start + 1}")
    if not args.apply:
        print("[DRY-RUN] no files changed. Re-run with --apply --copy after checking the plan.")
    else:
        print(f"[DONE] wrote mapping: {mapping_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
