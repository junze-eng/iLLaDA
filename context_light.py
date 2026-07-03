#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
context_light_clean.py

A clean-output wrapper for context_light.py.

Why this exists
---------------
Some versions of context_light.py update the progress bar using carriage return
("\\r"). In some terminals / nohup logs, the previous line is not cleared, so
the log looks like:
  ... total_elapsed=4.1s | e/ [----...]

This wrapper runs the original context_light.py unchanged, but filters its
stdout/stderr stream so the progress output is readable.

Usage is the same as context_light.py:
  python context_light_clean.py \
    --output-dir outputs/context_light \
    --context-lengths 1024 2048 4096 8192 \
    --needle-positions front middle back \
    --num-samples-per-condition 20 \
    --sample-selection first \
    --gen-length 128 \
    --gen-steps 128 \
    --gen-blocksize 32

Background:
  nohup python context_light_clean.py ... > context_light.log 2>&1 &

It forwards all arguments to context_light.py.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


PROGRESS_MARKERS = (
    "sample_elapsed=",
    "total_elapsed=",
    "running ctx=",
    "eta ",
    "match=",
    "[----------------",
    "[################",
)


def is_progress_line(s: str) -> bool:
    return any(m in s for m in PROGRESS_MARKERS)


def clean_line(s: str) -> str:
    # Keep only the last carriage-return segment and remove common terminal clear codes.
    s = s.split("\r")[-1]
    s = s.replace("\x1b[K", "").replace("\x1b[2K", "")
    return s.rstrip("\n")


def emit(s: str, final: bool = False) -> None:
    if not s:
        return
    if sys.stdout.isatty() and is_progress_line(s):
        sys.stdout.write("\r\033[K" + s)
        if final:
            sys.stdout.write("\n")
        sys.stdout.flush()
    else:
        sys.stdout.write(s + "\n")
        sys.stdout.flush()


def main() -> int:
    here = Path(__file__).resolve().parent
    target = here / "context_light.py"
    if not target.exists():
        print(f"[ERROR] cannot find original context_light.py next to {Path(__file__).name}: {target}", file=sys.stderr)
        return 2

    cmd = [sys.executable, str(target)] + sys.argv[1:]
    print("[RUN]", " ".join(cmd), flush=True)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    assert proc.stdout is not None

    buf = ""
    last_progress = ""
    last_emit_ts = 0.0
    progress_interval = 10.0 if not sys.stdout.isatty() else 0.0

    while True:
        ch = proc.stdout.read(1)
        if ch == "":
            break

        if ch == "\r":
            line = clean_line(buf)
            buf = ""
            if line:
                if is_progress_line(line):
                    last_progress = line
                    now = time.time()
                    if sys.stdout.isatty() or (now - last_emit_ts >= progress_interval):
                        emit(line, final=False)
                        last_emit_ts = now
                else:
                    emit(line)
            continue

        if ch == "\n":
            line = clean_line(buf)
            buf = ""
            if line:
                # Print final progress/completion lines; also move to next line after TTY progress.
                if is_progress_line(line):
                    emit(line, final=True)
                else:
                    if sys.stdout.isatty() and last_progress:
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                        last_progress = ""
                    emit(line)
            continue

        buf += ch

    if buf.strip():
        line = clean_line(buf)
        if line:
            emit(line, final=True)

    rc = proc.wait()
    if sys.stdout.isatty() and last_progress:
        sys.stdout.write("\n")
        sys.stdout.flush()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
