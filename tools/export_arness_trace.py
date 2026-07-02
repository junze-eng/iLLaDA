#!/usr/bin/env python3
"""Export existing summary.jsonl/trace.jsonl runs into per-sample trace folders."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


sys.path.insert(0, str(Path.cwd()))
from tools.arness_trace_writer import export_run_dir  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="outputs/arness_trace")
    parser.add_argument("--out", default="outputs/arness_trace/sample_traces")
    args = parser.parse_args()

    root = Path(args.runs_root)
    total = 0
    for run_dir in sorted(root.iterdir()) if root.exists() else []:
        if not run_dir.is_dir():
            continue
        if not (run_dir / "summary.jsonl").exists() or not (run_dir / "trace.jsonl").exists():
            continue
        written = export_run_dir(run_dir, args.out)
        print(f"[OK] {run_dir.name}: {len(written)} sample folders")
        total += len(written)
    print(f"[DONE] wrote {total} sample folders under {args.out}")


if __name__ == "__main__":
    main()
