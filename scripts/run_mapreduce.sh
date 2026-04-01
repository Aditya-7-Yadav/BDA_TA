#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_mapreduce.sh — Run all 4 Hadoop Streaming MapReduce jobs
#
# Run inside the namenode container:
#   docker exec -it namenode bash /scripts/run_mapreduce.sh
#
# Each job reads from /spotify/*.csv in HDFS and writes results to
# /spotify/mapreduce/<job_name>/
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

HADOOP_STREAMING_JAR=$(find $HADOOP_HOME -name "hadoop-streaming*.jar" | head -1)
if [ -z "$HADOOP_STREAMING_JAR" ]; then
    echo "ERROR: Could not find hadoop-streaming jar in $HADOOP_HOME"
    exit 1
fi
echo "Using streaming jar: $HADOOP_STREAMING_JAR"

# Python binary inside the container
PYTHON_BIN=$(which python3 2>/dev/null || which python 2>/dev/null)
if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: Python not found in container."
    echo "       Make sure you are using the hadoop-python custom image."
    exit 1
fi
echo "Using Python: $PYTHON_BIN"

HDFS_INPUT="/spotify/listening_history.csv"
HDFS_TRACKS="/spotify/tracks.csv"

# ─── Helper ──────────────────────────────────────────────────────────────────
run_job() {
    local JOB_NAME="$1"
    local MAPPER="$2"
    local REDUCER="$3"
    local INPUT="$4"
    local OUTPUT="/spotify/mapreduce/${JOB_NAME}"
    local EXTRA_ARGS="${5:-}"

    echo ""
    echo "============================================"
    echo "  Running Job: ${JOB_NAME}"
    echo "============================================"

    # Clean output path
    hdfs dfs -rm -r -f "$OUTPUT" 2>/dev/null || true

    hadoop jar "$HADOOP_STREAMING_JAR" \
        -files "$MAPPER","$REDUCER" \
        -mapper  "$(basename "$MAPPER")" \
        -reducer "$(basename "$REDUCER")" \
        -input   "$INPUT" \
        -output  "$OUTPUT" \
        -numReduceTasks 2 \
        $EXTRA_ARGS \
        && echo "  SUCCESS → ${OUTPUT}" \
        || { echo "  FAILED: ${JOB_NAME}"; exit 1; }
}

# ─── Job 1: Play Counts ──────────────────────────────────────────────────────
run_job "play_counts" \
    "/mapreduce/job1_play_counts/mapper.py" \
    "/mapreduce/job1_play_counts/reducer.py" \
    "$HDFS_INPUT"

# ─── Job 2: User × Genre Affinity ────────────────────────────────────────────
# tracks.csv is distributed as a side file so the mapper can load it
echo ""
echo "============================================"
echo "  Running Job: user_genre_affinity"
echo "============================================"
hdfs dfs -rm -r -f /spotify/mapreduce/user_genre 2>/dev/null || true

hadoop jar "$HADOOP_STREAMING_JAR" \
    -files "/mapreduce/job2_user_genre_affinity/mapper.py",\
"/mapreduce/job2_user_genre_affinity/reducer.py" \
    -mapper  "mapper.py" \
    -reducer "reducer.py" \
    -input   "$HDFS_INPUT" \
    -output  "/spotify/mapreduce/user_genre" \
    -numReduceTasks 2 \
    -cmdenv  "TRACKS_FILE=/data/tracks.csv" \
    && echo "  SUCCESS → /spotify/mapreduce/user_genre" \
    || echo "  FAILED (user_genre) — continuing"

# ─── Job 3: Track Popularity Score ───────────────────────────────────────────
run_job "track_popularity" \
    "/mapreduce/job3_track_popularity/mapper.py" \
    "/mapreduce/job3_track_popularity/reducer.py" \
    "$HDFS_INPUT"

# ─── Job 4: Hourly Activity Heatmap ──────────────────────────────────────────
run_job "hourly_activity" \
    "/mapreduce/job4_hourly_activity/mapper.py" \
    "/mapreduce/job4_hourly_activity/reducer.py" \
    "$HDFS_INPUT"

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  All MapReduce jobs complete!"
echo "  Results in HDFS /spotify/mapreduce/"
echo "============================================"
hdfs dfs -ls /spotify/mapreduce/

echo ""
echo "  Sample — Play Counts (top 10):"
hdfs dfs -cat /spotify/mapreduce/play_counts/part-* 2>/dev/null \
    | sort -t$'\t' -k2 -rn \
    | head -10

echo ""
echo "  Sample — Hourly Activity (top 10 slots):"
hdfs dfs -cat /spotify/mapreduce/hourly_activity/part-* 2>/dev/null \
    | sort -t$'\t' -k3 -rn \
    | head -10
