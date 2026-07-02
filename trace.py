#!/usr/bin/env python3
"""Compatibility entrypoint for exporting ARness trace artifacts.

This wraps export_trace.py so existing runs can be repaired/exported with:

  python trace.py --runs-root outputs/arness
"""
from __future__ import annotations

from export_trace import main


if __name__ == "__main__":
    main()
