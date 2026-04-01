"""
analytics.py
============
Spark SQL + DataFrame analytics for the Spotify case study.

Computes:
  1. Top 20 most-played tracks globally
  2. Genre popularity ranking (play share %)
  3. Artist leaderboard (total plays)
  4. User cohort analysis: free vs premium listening behaviour
  5. Audio-feature K-Means clustering of tracks
  6. Temporal trends: monthly plays over 2023

Input  : HDFS /spotify/{tracks,users,listening_history}.csv
Output : HDFS /spotify/output/analytics_{1..6}/
"""

from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F
from pyspark.sql.functions import col, count, avg, sum as _sum, when, lit, month
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

spark = (
    SparkSession.builder
    .appName("SpotifyAnalytics")
    .config("spark.sql.shuffle.partitions", "50")
    .config("spark.driver.memory", "2g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

HDFS_BASE = "hdfs://namenode:9000/spotify"


def save(df, name, show_n=10):
    out = f"{HDFS_BASE}/output/{name}"
    df.write.csv(out, header=True, mode="overwrite")
    print(f"  [saved] {out}")
    df.show(show_n, truncate=False)


# ─── Load ─────────────────────────────────────────────────────────────────────

print("\n=== Loading data ===")
history = spark.read.csv(f"{HDFS_BASE}/listening_history.csv", header=True, inferSchema=True)
tracks  = spark.read.csv(f"{HDFS_BASE}/tracks.csv",            header=True, inferSchema=True)
users   = spark.read.csv(f"{HDFS_BASE}/users.csv",             header=True, inferSchema=True)
history.createOrReplaceTempView("history")
tracks.createOrReplaceTempView("tracks")
users.createOrReplaceTempView("users")

# ─── 1. Top 20 Most-Played Tracks ─────────────────────────────────────────────

print("\n=== [1] Top 20 most-played tracks ===")

top_tracks = spark.sql("""
    SELECT
        h.track_id,
        t.name          AS track_name,
        t.artist_name,
        t.genre,
        t.popularity    AS spotify_popularity,
        COUNT(*)        AS total_plays,
        SUM(CASE WHEN h.skipped=0 THEN 1 ELSE 0 END)   AS completed_plays,
        SUM(CASE WHEN h.skipped=1 THEN 1 ELSE 0 END)   AS skipped_plays,
        ROUND(SUM(CASE WHEN h.skipped=0 THEN 1 ELSE 0 END)*100.0/COUNT(*), 1)
                        AS completion_rate_pct
    FROM history h
    JOIN tracks  t ON h.track_id = t.track_id
    GROUP BY h.track_id, t.name, t.artist_name, t.genre, t.popularity
    ORDER BY completed_plays DESC
    LIMIT 20
""")
save(top_tracks, "analytics_top_tracks")

# ─── 2. Genre Popularity ──────────────────────────────────────────────────────

print("\n=== [2] Genre popularity ===")

genre_stats = spark.sql("""
    SELECT
        t.genre,
        COUNT(*)        AS total_plays,
        COUNT(DISTINCT h.user_id)   AS unique_listeners,
        COUNT(DISTINCT h.track_id)  AS unique_tracks,
        ROUND(AVG(CASE WHEN h.skipped=0 THEN 1.0 ELSE 0.0 END)*100, 1)
                        AS avg_completion_pct
    FROM history h
    JOIN tracks  t ON h.track_id = t.track_id
    GROUP BY t.genre
    ORDER BY total_plays DESC
""")

total_plays = history.count()
genre_stats = genre_stats.withColumn(
    "play_share_pct",
    F.round(col("total_plays") / lit(total_plays) * 100, 2)
)
save(genre_stats, "analytics_genre_popularity")

# ─── 3. Artist Leaderboard ────────────────────────────────────────────────────

print("\n=== [3] Artist leaderboard ===")

artist_leaderboard = spark.sql("""
    SELECT
        t.artist_id,
        t.artist_name,
        t.genre,
        COUNT(*)                AS total_plays,
        COUNT(DISTINCT h.user_id) AS unique_listeners,
        COUNT(DISTINCT t.track_id) AS track_count
    FROM history h
    JOIN tracks  t ON h.track_id = t.track_id
    WHERE h.skipped = 0
    GROUP BY t.artist_id, t.artist_name, t.genre
    ORDER BY total_plays DESC
    LIMIT 30
""")
save(artist_leaderboard, "analytics_artist_leaderboard")

# ─── 4. Free vs Premium Cohort Analysis ───────────────────────────────────────

print("\n=== [4] Free vs premium cohort analysis ===")

cohort = spark.sql("""
    SELECT
        u.subscription_type,
        COUNT(*)                AS total_plays,
        COUNT(DISTINCT h.user_id)                           AS users,
        ROUND(COUNT(*) / COUNT(DISTINCT h.user_id), 1)      AS avg_plays_per_user,
        ROUND(AVG(h.play_duration_ms)/1000, 1)              AS avg_listen_duration_sec,
        ROUND(SUM(CASE WHEN h.skipped=1 THEN 1.0 ELSE 0.0 END)/COUNT(*)*100, 1)
                                AS skip_rate_pct
    FROM history h
    JOIN users   u ON h.user_id = u.user_id
    GROUP BY u.subscription_type
""")
save(cohort, "analytics_cohort_free_premium")

# ─── 5. K-Means Audio Feature Clustering ──────────────────────────────────────
#
# Groups tracks into K=6 audio-feature clusters (e.g. "energetic dance",
# "mellow acoustic", "dark heavy", etc.) using their audio fingerprint.

print("\n=== [5] K-Means audio feature clustering ===")

FEATURE_COLS = [
    "energy", "danceability", "acousticness",
    "valence", "instrumentalness", "speechiness",
    "tempo", "loudness"
]

assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features_raw", handleInvalid="skip")
scaler    = StandardScaler(inputCol="features_raw", outputCol="features",
                           withStd=True, withMean=True)
kmeans    = KMeans(k=6, seed=42, featuresCol="features", predictionCol="cluster")

tracks_feat = assembler.transform(tracks.dropna(subset=FEATURE_COLS))
scaler_m    = scaler.fit(tracks_feat)
tracks_sc   = scaler_m.transform(tracks_feat)
km_model    = kmeans.fit(tracks_sc)

clustered = km_model.transform(tracks_sc)

cluster_profile = (
    clustered
    .groupBy("cluster")
    .agg(
        count("*").alias("track_count"),
        F.round(avg("energy"), 3).alias("avg_energy"),
        F.round(avg("danceability"), 3).alias("avg_danceability"),
        F.round(avg("acousticness"), 3).alias("avg_acousticness"),
        F.round(avg("valence"), 3).alias("avg_valence"),
        F.round(avg("tempo"), 1).alias("avg_tempo"),
    )
    .orderBy("cluster")
)

silhouette = ClusteringEvaluator().evaluate(clustered)
print(f"  Silhouette score (k=6): {silhouette:.4f}")

save(cluster_profile, "analytics_kmeans_clusters")

# Save track-cluster mapping
(
    clustered
    .select("track_id", "name", "artist_name", "genre", "cluster")
    .write.csv(f"{HDFS_BASE}/output/analytics_track_clusters", header=True, mode="overwrite")
)
print(f"  Track-cluster mapping → {HDFS_BASE}/output/analytics_track_clusters")

# ─── 6. Monthly Listening Trends ──────────────────────────────────────────────

print("\n=== [6] Monthly listening trends ===")

monthly = spark.sql("""
    SELECT
        SUBSTRING(timestamp, 1, 7)   AS year_month,
        COUNT(*)                     AS total_plays,
        COUNT(DISTINCT user_id)      AS active_users,
        ROUND(SUM(play_duration_ms)/3600000.0, 1) AS total_listen_hours
    FROM history
    GROUP BY SUBSTRING(timestamp, 1, 7)
    ORDER BY year_month
""")
save(monthly, "analytics_monthly_trends")

print("\n=== All analytics complete ===\n")
spark.stop()
