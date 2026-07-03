#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_context_light_progress_inplace.py

In-place patch for context_light.py v4.

It replaces only the LiveSpinner class with a clean version:
- TTY: uses "\r\033[K" to clear the line before each update.
- Non-TTY / nohup log: does not use "\r"; prints at most once every 10 seconds.
- On exit: clears the spinner line cleanly.

Usage:
  cd /workspace/iLLaDA
  cp /mnt/data/fix_context_light_progress_inplace.py .
  python fix_context_light_progress_inplace.py context_light.py

Backup:
  context_light.py.bak_spinner
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

NEW_CLASS = """class LiveSpinner:
    \"""Tiny dependency-free spinner used while one long model generation is running.

    Cleaned version:
    - In an interactive terminal, refresh one line and clear leftovers with ANSI K.
    - In redirected logs / nohup, avoid carriage-return spam and print at most
      one running-status line every 10 seconds.
    \"""

    def __init__(self, message_fn, interval: float = 0.5, log_interval: float = 10.0):
        self.message_fn = message_fn
        self.interval = interval
        self.log_interval = log_interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._is_tty = sys.stdout.isatty()
        self._last_log_time = 0.0

    def __enter__(self):
        def _run() -> None:
            frames = "|/-\\\\"
            i = 0
            while not self._stop.is_set():
                msg = self.message_fn()
                now = time.perf_counter()

                if self._is_tty:
                    # Clear whole line first. This prevents stale characters like "e"
                    # from the previous longer progress line.
                    sys.stdout.write("\\r\\033[K" + frames[i % len(frames)] + " " + msg)
                    sys.stdout.flush()
                else:
                    # In logs, carriage returns are unreadable. Print a throttled line.
                    if now - self._last_log_time >= self.log_interval:
                        sys.stdout.write(frames[i % len(frames)] + " " + msg + "\\n")
                        sys.stdout.flush()
                        self._last_log_time = now

                i += 1
                self._stop.wait(self.interval)

            if self._is_tty:
                sys.stdout.write("\\r\\033[K")
                sys.stdout.flush()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._is_tty:
            sys.stdout.write("\\r\\033[K")
            sys.stdout.flush()
"""


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("context_light.py")
    if not path.exists():
        print(f"[ERROR] not found: {path}", file=sys.stderr)
        return 2

    text = path.read_text(encoding="utf-8")

    if "class LiveSpinner:" not in text:
        print("[ERROR] class LiveSpinner not found.", file=sys.stderr)
        return 2

    pattern = re.compile(
        r'class LiveSpinner:\n'
        r'(?:    .*\n)*?'
        r'\n'
        r'def make_running_message\(',
        re.DOTALL,
    )

    replacement = NEW_CLASS + "\n\ndef make_running_message("
    new_text, n = pattern.subn(replacement, text, count=1)

    if n != 1:
        print("[ERROR] failed to replace LiveSpinner block safely.", file=sys.stderr)
        return 2

    bak = path.with_suffix(path.suffix + ".bak_spinner")
    if not bak.exists():
        bak.write_text(text, encoding="utf-8")

    path.write_text(new_text, encoding="utf-8")
    print(f"[OK] patched LiveSpinner in {path}")
    print(f"[OK] backup saved to {bak}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
