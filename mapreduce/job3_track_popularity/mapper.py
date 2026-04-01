#!/usr/bin/env python3
"""
Job 3 — Track Popularity Score Mapper
=======================================
Computes a weighted popularity score per track:
  score = completed_plays * 2  +  skipped_plays * 1

This rewards tracks that people actually finish while still counting skips
as weak engagement signals.

Input  : listening_history.csv (stdin)
Output : track_id \t C (completed, weight 2) or S (skipped, weight 1)
"""
import sys

for line in sys.stdin:
    line = line.strip()
    if not line or line.startswith("user_id"):
        continue

    parts = line.split(",")
    if len(parts) < 5:
        continue

    track_id = parts[1]
    skipped  = parts[4]

    if skipped == "0":
        print(f"{track_id}\tC")   # completed play
    else:
        print(f"{track_id}\tS")   # skipped play
