#!/usr/bin/env python3
"""
Job 1 — Play Count Reducer
===========================
Input  : sorted (track_id \t 1) lines from all mappers
Output : track_id \t total_play_count

Classic word-count reducer — sums counts per track_id.
"""
import sys

current_track = None
current_count = 0

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    parts = line.split("\t")
    if len(parts) != 2:
        continue

    track_id, count_str = parts
    try:
        count = int(count_str)
    except ValueError:
        continue

    if current_track == track_id:
        current_count += count
    else:
        if current_track is not None:
            print(f"{current_track}\t{current_count}")
        current_track = track_id
        current_count = count

# Flush last key
if current_track is not None:
    print(f"{current_track}\t{current_count}")
