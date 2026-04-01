#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# init_hdfs.sh — Run inside the namenode container
#   1. Wait for HDFS to be out of safe-mode
#   2. Create /spotify directory tree
#   3. Upload CSV data files
#   4. Set permissions
#
# Usage:
#   docker exec -it namenode bash /scripts/init_hdfs.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

echo "========================================"
echo "  Spotify HDFS Initialisation"
echo "========================================"

# ── 1. Wait for NameNode to leave safe-mode ───────────────────────────────────
echo ""
echo "[1/4] Waiting for HDFS safe-mode to exit..."
MAX_WAIT=120
WAITED=0
until hdfs dfsadmin -safemode get 2>/dev/null | grep -q "Safe mode is OFF"; do
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: HDFS did not leave safe-mode after ${MAX_WAIT}s"
        exit 1
    fi
    echo "  Still in safe-mode, waiting 5s..."
    sleep 5
    WAITED=$((WAITED + 5))
done
echo "  HDFS is ready."

# ── 2. Create directory tree ──────────────────────────────────────────────────
echo ""
echo "[2/4] Creating HDFS directories..."

DIRS=(
    /spotify
    /spotify/input
    /spotify/output
    /spotify/mapreduce
    /spotify/mapreduce/play_counts
    /spotify/mapreduce/user_genre
    /spotify/mapreduce/track_popularity
    /spotify/mapreduce/hourly_activity
)

for DIR in "${DIRS[@]}"; do
    if hdfs dfs -test -d "$DIR" 2>/dev/null; then
        echo "  Already exists: $DIR"
    else
        hdfs dfs -mkdir -p "$DIR"
        echo "  Created: $DIR"
    fi
done

# Set open permissions so Jupyter can write without auth issues
hdfs dfs -chmod -R 777 /spotify

# ── 3. Upload data files ──────────────────────────────────────────────────────
echo ""
echo "[3/4] Uploading data files to HDFS..."

for CSV in /data/tracks.csv /data/users.csv /data/listening_history.csv; do
    FILENAME=$(basename "$CSV")
    HDFS_PATH="/spotify/${FILENAME}"
    if [ -f "$CSV" ]; then
        # Remove old copy first (overwrite)
        hdfs dfs -rm -f "$HDFS_PATH" 2>/dev/null || true
        hdfs dfs -put "$CSV" "$HDFS_PATH"
        SIZE=$(hdfs dfs -du -h "$HDFS_PATH" | awk '{print $1, $2}')
        echo "  Uploaded: $HDFS_PATH  ($SIZE)"
    else
        echo "  WARN: $CSV not found, skipping (generate data first)"
    fi
done

# ── 4. Verify ─────────────────────────────────────────────────────────────────
echo ""
echo "[4/4] HDFS /spotify directory listing:"
hdfs dfs -ls -h /spotify/

echo ""
echo "========================================"
echo "  HDFS initialisation complete!"
echo "  NameNode UI: http://localhost:9870"
echo "========================================"
