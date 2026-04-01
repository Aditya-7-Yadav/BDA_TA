#!/usr/bin/env python3
"""
Job 1 — Play Count Mapper
=========================
Input  : listening_history.csv lines (tab-separated after header strip)
Output : track_id \t 1

Each mapper reads a chunk of the listening history file and emits
one (track_id, 1) pair per non-skipped play event.
"""
import sys

for line in sys.stdin:
    line = line.strip()
    if not line or line.startswith("user_id"):   # skip header
        continue

    parts = line.split(",")
    if len(parts) < 6:
        continue

    # CSV columns: user_id,track_id,timestamp,play_duration_ms,skipped,...
    track_id = parts[1]
    skipped  = parts[4]

    # Count only completed (non-skipped) plays
    if skipped == "0":
        print(f"{track_id}\t1")
