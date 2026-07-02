#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
repair_arness_names.py

Compatibility wrapper for old numbered ARness folders.

It calls export_trace.py with:
  --canonicalize --mode copy --overwrite --write-task-index

Usage:
  python repair_arness_names.py \
    --old-roots outputs/arness_all outputs/arness_remain \
    --new-root outputs/arness
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-roots", nargs="+", default=["outputs/arness_all"], help="Old numbered roots.")
    ap.add_argument("--new-root", default="outputs/arness", help="New canonical root.")
    ap.add_argument("--export-script", default="export_trace.py")
    ap.add_argument("--mode", choices=["copy", "move"], default="copy")
    args = ap.parse_args()

    for old_root in args.old_roots:
        if not Path(old_root).exists():
            print(f"[SKIP] missing old root: {old_root}")
            continue
        cmd = [
            sys.executable,
            args.export_script,
            "--runs-root", old_root,
            "--task-output-root", args.new_root,
            "--canonicalize",
            "--mode", args.mode,
            "--overwrite",
            "--write-task-index",
        ]
        print("[RUN]", " ".join(cmd))
        subprocess.run(cmd, check=True)

    print(f"[DONE] canonical ARness output is under {args.new_root}")


if __name__ == "__main__":
    main()
