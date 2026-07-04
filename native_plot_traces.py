#!/usr/bin/env python3
"""Plot native diffusion trace artifacts.

This script is intentionally lightweight and independent of OpenCompass.  It
works with the native output/eval directory layout and reads condition folders
containing trace.jsonl.  It is especially useful for W1-4B traces, where the
main questions are:
  - how many positions changed per diffusion step;
  - how quickly masks disappear;
  - whether updates are spread across the generated span or concentrated left-to-right.

Usage:
  python native_plot_traces.py --root native_outputs/w1_4b/native_w1_trace_focus --all
  python native_plot_traces.py --run native_outputs/w1_4b/native_w1_trace_focus/gsm8k/s7_l256_w1samgidd_w1st32
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def selected_count(row: Dict[str, Any]) -> int:
    if row.get("w1_selected_count") is not None:
        return safe_int(row.get("w1_selected_count"))
    positions = row.get("selected_positions") or []
    if isinstance(positions, list):
        return len(positions)
    return safe_int(row.get("selected_count"), 0)


def flatten_positions(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, int]]:
    points: List[Dict[str, int]] = []
    for row in rows:
        step = safe_int(row.get("step_idx"))
        positions = row.get("selected_positions") or []
        if isinstance(positions, list):
            for p in positions:
                try:
                    points.append({"step": step, "position": int(p)})
                except Exception:
                    pass
    return points


def write_trace_summary(run_dir: Path, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    steps = len(rows)
    counts = [selected_count(r) for r in rows]
    remain_after = [safe_int(r.get("mask_count_after"), -1) for r in rows if r.get("mask_count_after") is not None]
    points = flatten_positions(rows)
    summary = {
        "trace_path": str(run_dir / "trace.jsonl"),
        "steps_recorded": steps,
        "total_changed_positions": sum(counts),
        "avg_changed_positions_per_step": (sum(counts) / steps) if steps else None,
        "max_changed_positions_per_step": max(counts) if counts else None,
        "final_mask_count_after": remain_after[-1] if remain_after else None,
        "position_points": len(points),
    }
    (run_dir / "native_trace_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with (run_dir / "native_trace_steps.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["step_idx", "selected_count", "mask_count_before", "mask_count_after", "t", "sampler"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            meta = r.get("w1_metadata") or {}
            writer.writerow({
                "step_idx": r.get("step_idx"),
                "selected_count": selected_count(r),
                "mask_count_before": r.get("mask_count_before"),
                "mask_count_after": r.get("mask_count_after"),
                "t": r.get("t"),
                "sampler": meta.get("sampler") or r.get("sampler") or "",
            })
    return summary


def plot_run(run_dir: Path) -> Optional[Dict[str, Any]]:
    rows = read_jsonl(run_dir / "trace.jsonl")
    if not rows:
        return None

    # Import lazily so the evaluation path does not require matplotlib.
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: safe_int(r.get("step_idx")))
    steps = [safe_int(r.get("step_idx")) for r in rows]
    counts = [selected_count(r) for r in rows]
    before = [safe_int(r.get("mask_count_before"), 0) for r in rows]
    after = [safe_int(r.get("mask_count_after"), 0) for r in rows]

    summary = write_trace_summary(run_dir, rows)

    fig = plt.figure(figsize=(8, 4.5))
    plt.plot(steps, counts, marker="o", linewidth=1)
    plt.xlabel("diffusion step")
    plt.ylabel("changed / selected positions")
    plt.title("Native trace: positions changed per step")
    plt.tight_layout()
    fig.savefig(run_dir / "trace_selected_per_step.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 4.5))
    if any(v >= 0 for v in before):
        plt.plot(steps, before, marker="o", linewidth=1, label="before")
    if any(v >= 0 for v in after):
        plt.plot(steps, after, marker="o", linewidth=1, label="after")
    plt.xlabel("diffusion step")
    plt.ylabel("remaining mask count")
    plt.title("Native trace: mask count over time")
    plt.legend()
    plt.tight_layout()
    fig.savefig(run_dir / "trace_remaining_masks.png", dpi=180)
    plt.close(fig)

    points = flatten_positions(rows)
    if points:
        fig = plt.figure(figsize=(8, 4.5))
        plt.scatter([p["step"] for p in points], [p["position"] for p in points], s=8)
        plt.xlabel("diffusion step")
        plt.ylabel("generated-token position")
        plt.title("Native trace: changed positions")
        plt.tight_layout()
        fig.savefig(run_dir / "trace_position_scatter.png", dpi=180)
        plt.close(fig)

    return summary


def find_trace_dirs(root: Path) -> List[Path]:
    if root.is_file() and root.name == "trace.jsonl":
        return [root.parent]
    if (root / "trace.jsonl").exists():
        return [root]
    return sorted({p.parent for p in root.rglob("trace.jsonl")})


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot native W1/iLLaDA trace artifacts.")
    parser.add_argument("--root", default="native_outputs", help="Root directory to search when --all is set.")
    parser.add_argument("--run", help="Specific evaluated/model-output run directory containing trace.jsonl.")
    parser.add_argument("--all", action="store_true", help="Plot every trace.jsonl under --root.")
    args = parser.parse_args()

    if args.run:
        dirs = find_trace_dirs(Path(args.run))
    else:
        dirs = find_trace_dirs(Path(args.root)) if args.all else find_trace_dirs(Path(args.root))

    if not dirs:
        print("No trace.jsonl found.")
        return 1

    ok = 0
    for d in dirs:
        try:
            summary = plot_run(d)
            if summary is None:
                print(f"[skip] {d}: empty trace")
                continue
            ok += 1
            print(f"[plot] {d} | steps={summary.get('steps_recorded')} avg_changed={summary.get('avg_changed_positions_per_step')}")
        except Exception as exc:
            print(f"[error] {d}: {exc}")
    print(f"Plotted {ok}/{len(dirs)} trace run(s).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
