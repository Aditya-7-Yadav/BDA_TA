#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_spark.sh — Submit all Spark jobs from the Jupyter container
#
# Usage (from host, after `docker compose up -d`):
#   docker exec -it jupyter bash /home/jovyan/work/../scripts/run_spark.sh
#
# Or open a terminal in JupyterLab and run:
#   bash ~/work/../scripts/run_spark.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SPARK_HOME=${SPARK_HOME:-/usr/local/spark}
SPARK_SUBMIT="$SPARK_HOME/bin/spark-submit"
WORK_DIR="/home/jovyan/work"

echo "========================================"
echo "  Spotify Spark Pipeline"
echo "========================================"

# ─── 1. Analytics ─────────────────────────────────────────────────────────────
echo ""
echo "[1/3] Running analytics.py..."
"$SPARK_SUBMIT" \
    --master "local[*]" \
    --driver-memory 2g \
    "$WORK_DIR/spark/analytics.py" \
    && echo "  analytics.py — DONE" \
    || { echo "  analytics.py — FAILED"; exit 1; }

# ─── 2. ALS Collaborative Filtering ───────────────────────────────────────────
echo ""
echo "[2/3] Running als_recommendations.py..."
"$SPARK_SUBMIT" \
    --master "local[*]" \
    --driver-memory 3g \
    --packages "org.apache.spark:spark-mllib_2.12:3.3.0" \
    "$WORK_DIR/spark/als_recommendations.py" \
    && echo "  als_recommendations.py — DONE" \
    || { echo "  als_recommendations.py — FAILED"; exit 1; }

# ─── 3. Content-Based Filtering ───────────────────────────────────────────────
echo ""
echo "[3/3] Running content_based.py..."
"$SPARK_SUBMIT" \
    --master "local[*]" \
    --driver-memory 2g \
    "$WORK_DIR/spark/content_based.py" \
    && echo "  content_based.py — DONE" \
    || { echo "  content_based.py — FAILED"; exit 1; }

echo ""
echo "========================================"
echo "  All Spark jobs complete!"
echo "========================================"
