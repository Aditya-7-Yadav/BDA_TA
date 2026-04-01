#!/usr/bin/env python3
"""
Job 2 — User × Genre Affinity Reducer
=======================================
Input  : sorted (user_id|genre \t count) lines
Output : user_id \t genre \t total_plays

Also computes per-user genre rank (output sorted by plays descending per user).
Since standard streaming reducers can't easily sort within a group after
aggregation, we buffer all genres per user and emit sorted.
"""
import sys
from collections import defaultdict

# Buffer: user_id → {genre: count}
user_genre_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

current_key   = None
current_count = 0

def flush(key: str, count: int):
    """Emit buffered user-genre totals sorted by plays desc."""
    if key is None:
        return
    user_id, genre = key.split("|", 1)
    user_genre_counts[user_id][genre] += count

# First pass: aggregate
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
        flush(current_key, current_count)
        current_key   = key
        current_count = count

flush(current_key, current_count)

# Second pass: emit sorted output
for user_id, genre_map in user_genre_counts.items():
    for genre, plays in sorted(genre_map.items(), key=lambda x: -x[1]):
        print(f"{user_id}\t{genre}\t{plays}")
