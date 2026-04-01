#!/usr/bin/env python3
"""
Job 3 — Track Popularity Score Reducer
========================================
Input  : sorted (track_id \t C|S) lines
Output : track_id \t completed \t skipped \t score

score = completed * 2 + skipped * 1
"""
import sys

current_track = None
completed = 0
skipped   = 0

WEIGHT_COMPLETE = 2
WEIGHT_SKIP     = 1

def emit(track_id, completed, skipped):
    score = completed * WEIGHT_COMPLETE + skipped * WEIGHT_SKIP
    print(f"{track_id}\t{completed}\t{skipped}\t{score}")

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    parts = line.split("\t")
    if len(parts) != 2:
        continue

    track_id, event_type = parts

    if current_track == track_id:
        if event_type == "C":
            completed += 1
        else:
            skipped += 1
    else:
        if current_track is not None:
            emit(current_track, completed, skipped)
        current_track = track_id
        completed = 1 if event_type == "C" else 0
        skipped   = 0 if event_type == "C" else 1

if current_track is not None:
    emit(current_track, completed, skipped)
