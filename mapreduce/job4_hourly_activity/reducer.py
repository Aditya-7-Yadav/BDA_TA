#!/usr/bin/env python3
"""
Job 4 — Hourly Listening Activity Reducer
===========================================
Input  : sorted (hour|day \t 1) lines
Output : hour_of_day \t day_of_week \t event_count \t day_name

Useful for plotting a heatmap: X=hour, Y=day, colour=play_count.
"""
import sys

DAY_NAMES = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

current_key   = None
current_count = 0

def emit(key: str, count: int):
    hour, day = key.split("|")
    day_name = DAY_NAMES[int(day)] if int(day) < 7 else "Unknown"
    print(f"{hour}\t{day}\t{count}\t{day_name}")

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    parts = line.split("\t")
    if len(parts) != 2:
        continue

    key, count_str = parts
    try:
        count = int(count_str)
    except ValueError:
        continue

    if current_key == key:
        current_count += count
    else:
        if current_key is not None:
            emit(current_key, current_count)
        current_key   = key
        current_count = count

if current_key is not None:
    emit(current_key, current_count)
