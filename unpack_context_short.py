#!/usr/bin/env python3
"""Extract context.zip with short directory names.

Example:
  python unpack_context_short.py --zip context.zip --out outputs/context_short --force

Long run directory:
  task2_..._ctx8192_posfront_len128_block32_steps128_thrnone

becomes:
  ctx8192_front

This avoids Windows path-length issues and makes visual inspection easier.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Tuple

try:
    import pandas as pd
except Exception:
    pd = None

SCRIPT_VERSION = "unpack_context_short_v1"

RUN_RE = re.compile(
    r"(?:^|/)("
    r".*?ctx(?P<ctx>\d+)_pos(?P<pos>front|middle|back)"
    r".*?"
    r")(?:/|$)"
)
RUN_SEG_RE = re.compile(r".*?ctx(?P<ctx>\d+)_pos(?P<pos>front|middle|back).*")
RUN_FILE_RE = re.compile(r".*?ctx(?P<ctx>\d+)_pos(?P<pos>front|middle|back).*?(\.[^.]+)$")


def short_run_name(name: str) -> str | None:
    m = RUN_SEG_RE.fullmatch(name)
    if not m:
        return None
    return f"ctx{m.group('ctx')}_{m.group('pos')}"


def short_file_name(name: str) -> str:
    m = RUN_FILE_RE.fullmatch(name)
    if m:
        return f"ctx{m.group('ctx')}_{m.group('pos')}{m.group(3)}"
    return name


def shorten_member_path(member: str) -> str:
    # Drop zip root folder "context/" if present.
    parts = [p for p in member.replace("\\", "/").split("/") if p and p != "."]
    if parts and parts[0] == "context":
        parts = parts[1:]

    new_parts = []
    for i, part in enumerate(parts):
        # Shorten generated config filenames too.
        if i > 0 and parts[i - 1] == "generated_configs":
            new_parts.append(short_file_name(part))
            continue

        sr = short_run_name(part)
        new_parts.append(sr if sr else part)

    return "/".join(new_parts)


def safe_join(root: Path, rel: str) -> Path:
    out = (root / rel).resolve()
    root_resolved = root.resolve()
    if not str(out).startswith(str(root_resolved)):
        raise RuntimeError(f"Unsafe path in zip: {rel}")
    return out


def extract_short(zip_path: Path, out_dir: Path, force: bool = False) -> Dict[str, str]:
    if out_dir.exists() and force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping: Dict[str, str] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            old = info.filename
            if old.endswith("/"):
                continue
            new_rel = shorten_member_path(old)
            if not new_rel:
                continue
            target = safe_join(out_dir, new_rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            mapping[old] = new_rel

    write_mapping(out_dir, mapping)
    patch_summary_tables(out_dir, mapping)
    return mapping


def write_mapping(out_dir: Path, mapping: Dict[str, str]) -> None:
    csv_path = out_dir / "short_name_map.csv"
    json_path = out_dir / "short_name_map.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["original_member", "short_path"])
        for old, new in sorted(mapping.items()):
            w.writerow([old, new])

    json_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")


def build_run_mapping(mapping: Dict[str, str]) -> Dict[str, str]:
    run_map = {}
    for old, new in mapping.items():
        old_parts = old.replace("\\", "/").split("/")
        new_parts = new.replace("\\", "/").split("/")
        for op in old_parts:
            sp = short_run_name(op)
            if sp:
                run_map[op] = sp
        # Also derive from full path regex if needed.
        m = RUN_RE.search(old)
        if m:
            run_map[m.group(1)] = f"ctx{m.group('ctx')}_{m.group('pos')}"
    return run_map


def replace_run_names_in_text(text: str, run_map: Dict[str, str]) -> str:
    for old, new in sorted(run_map.items(), key=lambda kv: len(kv[0]), reverse=True):
        text = text.replace(old, new)
    return text


def patch_summary_tables(out_dir: Path, mapping: Dict[str, str]) -> None:
    """Add original_run_name and replace run_name with short names in copied summary CSVs.

    This is intentionally conservative: if pandas is unavailable or a file cannot be
    parsed, leave the original copied file untouched.
    """
    run_map = build_run_mapping(mapping)
    if not run_map:
        return

    # Patch JSONL manifest textually.
    for name in ["run_manifest.jsonl", "fix_context_index.jsonl"]:
        p = out_dir / name
        if p.exists():
            txt = p.read_text(encoding="utf-8", errors="ignore")
            p.write_text(replace_run_names_in_text(txt, run_map), encoding="utf-8")

    if pd is None:
        return

    for name in ["aggregate.csv", "summary_all.csv", "fix_context_index.csv", "fix_context_missing.csv"]:
        p = out_dir / name
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue

        for col in list(df.columns):
            if col in {"run_name", "work_dir", "opencompass_summary_csv"} or df[col].dtype == object:
                # Only patch string-looking cells.
                try:
                    s = df[col].astype("string")
                    if col == "run_name" and "original_run_name" not in df.columns:
                        df.insert(df.columns.get_loc(col) + 1, "original_run_name", df[col])
                    df[col] = s.map(lambda x: replace_run_names_in_text(x, run_map) if x is not pd.NA else x)
                except Exception:
                    pass
        df.to_csv(p, index=False)


def print_tree_summary(out_dir: Path) -> None:
    print(f"\nExtracted short context tree to: {out_dir}")
    print("\nTop-level files/directories:")
    for p in sorted(out_dir.iterdir()):
        print(" -", p.name)

    runs = sorted([p.name for p in out_dir.iterdir() if p.is_dir() and re.match(r"ctx\d+_(front|middle|back)$", p.name)])
    if runs:
        print("\nShort run dirs:")
        for r in runs:
            print(" -", r)


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--zip", dest="zip_path", default="context.zip", help="Input context zip")
    ap.add_argument("--out", default="outputs/context_short", help="Output directory")
    ap.add_argument("--force", action="store_true", help="Delete output dir before extracting")
    args = ap.parse_args()

    zip_path = Path(args.zip_path)
    out_dir = Path(args.out)

    if not zip_path.exists():
        raise SystemExit(f"Zip file not found: {zip_path}")

    print(f"{SCRIPT_VERSION}")
    print(f"Input : {zip_path}")
    print(f"Output: {out_dir}")
    mapping = extract_short(zip_path, out_dir, force=args.force)
    print(f"Extracted {len(mapping)} files.")
    print_tree_summary(out_dir)


if __name__ == "__main__":
    main()
