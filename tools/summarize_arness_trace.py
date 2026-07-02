#!/usr/bin/env python3
"""Create one compact CSV from sample_traces/*/*/*/plot_rows.csv."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/arness_trace/sample_traces")
    parser.add_argument("--out", default="outputs/arness_trace/arness_trace_plot.csv")
    args = parser.parse_args()

    rows = []
    fields = []
    for path in Path(args.root).rglob("plot_rows.csv"):
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(row)
                for key in row:
                    if key not in fields:
                        fields.append(key)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] wrote {out}, rows={len(rows)}")


if __name__ == "__main__":
    main()
