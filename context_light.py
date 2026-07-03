#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
context_light.py clean launcher

用法：
  mv context_light.py context_light_raw.py
  cp /mnt/data/context_light.py .
  python context_light.py <原参数不变>

作用：
  调用 context_light_raw.py，并清理原脚本的 '\r' 进度条输出。
  这样命令入口仍然叫 context_light.py，不需要改你的跑法。
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


def _is_progress(s: str) -> bool:
    return any(m in s for m in PROGRESS_MARKERS)


def _clean(s: str) -> str:
    s = s.split("\r")[-1]
    s = s.replace("\x1b[K", "").replace("\x1b[2K", "")
    return s.rstrip("\n")


def _emit(s: str, final: bool = False) -> None:
    if not s:
        return
    if sys.stdout.isatty() and _is_progress(s):
        sys.stdout.write("\r\033[K" + s)
        if final:
            sys.stdout.write("\n")
    else:
        sys.stdout.write(s + "\n")
    sys.stdout.flush()


def main() -> int:
    here = Path(__file__).resolve().parent

    candidates = [
        here / "context_light_raw.py",
        here / "context_light.py.bak_progress",
        here / "context_light_original.py",
    ]
    target = next((p for p in candidates if p.exists()), None)

    if target is None:
        print(
            "[ERROR] 找不到原始脚本。请先执行：\n"
            "  mv context_light.py context_light_raw.py\n"
            "  cp /mnt/data/context_light.py .",
            file=sys.stderr,
        )
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
        bufsize=0,
        env=env,
    )
    assert proc.stdout is not None

    buf = ""
    last_log_ts = 0.0
    log_interval = 10.0 if not sys.stdout.isatty() else 0.0
    last_was_progress = False

    while True:
        ch = proc.stdout.read(1)
        if ch == "":
            break

        if ch == "\r":
            line = _clean(buf)
            buf = ""
            if line:
                if _is_progress(line):
                    now = time.time()
                    if sys.stdout.isatty() or now - last_log_ts >= log_interval:
                        _emit(line, final=False)
                        last_log_ts = now
                    last_was_progress = True
                else:
                    if sys.stdout.isatty() and last_was_progress:
                        sys.stdout.write("\n")
                    _emit(line)
                    last_was_progress = False
            continue

        if ch == "\n":
            line = _clean(buf)
            buf = ""
            if line:
                if _is_progress(line):
                    _emit(line, final=True)
                    last_was_progress = False
                else:
                    if sys.stdout.isatty() and last_was_progress:
                        sys.stdout.write("\n")
                    _emit(line)
                    last_was_progress = False
            continue

        buf += ch

    if buf.strip():
        _emit(_clean(buf), final=True)

    rc = proc.wait()
    if sys.stdout.isatty() and last_was_progress:
        sys.stdout.write("\n")
        sys.stdout.flush()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
