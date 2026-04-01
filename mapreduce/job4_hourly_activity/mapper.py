#!/usr/bin/env python3
"""
Job 4 — Hourly Listening Activity Mapper
==========================================
Analyses when users listen to music across hours of day and days of week.

Input  : listening_history.csv (stdin)
Output : hour_of_day|day_of_week \t 1

day_of_week: 0=Monday … 6=Sunday
"""
import sys

for line in sys.stdin:
    line = line.strip()
    if not line or line.startswith("user_id"):
        continue

    parts = line.split(",")
    if len(parts) < 7:
        continue

    # CSV: user_id,track_id,timestamp,play_duration_ms,skipped,hour_of_day,day_of_week
    hour_of_day  = parts[5]
    day_of_week  = parts[6]

    print(f"{hour_of_day}|{day_of_week}\t1")
