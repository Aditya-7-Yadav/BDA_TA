#!/usr/bin/env python3
"""
Job 2 — User × Genre Affinity Mapper
======================================
This is a two-input join job using Hadoop Streaming's side-data approach.
We load tracks.csv into memory (passed via -files or a distributed cache)
and join on track_id to emit (user_id|genre, 1) pairs.

Input  : listening_history.csv (stdin)
Side   : /data/tracks.csv loaded into memory via TRACKS_FILE env var
Output : user_id|genre \t 1
"""
import sys
import os
import csv

# ── Load track → genre lookup from side-file ─────────────────────────────────
tracks_file = os.environ.get("TRACKS_FILE", "/data/tracks.csv")
track_genre: dict[str, str] = {}

try:
    with open(tracks_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            track_genre[row["track_id"]] = row["genre"]
except FileNotFoundError:
    sys.stderr.write(f"WARN: tracks file not found at {tracks_file}\n")

# ── Process listening events ──────────────────────────────────────────────────
for line in sys.stdin:
    line = line.strip()
    if not line or line.startswith("user_id"):
        continue

    parts = line.split(",")
    if len(parts) < 5:
        continue

    user_id  = parts[0]
    track_id = parts[1]
    skipped  = parts[4]

    if skipped == "1":   # only count genuine listens
        continue

    genre = track_genre.get(track_id)
    if genre:
        print(f"{user_id}|{genre}\t1")
