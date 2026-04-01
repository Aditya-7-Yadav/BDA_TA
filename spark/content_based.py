"""
content_based.py
================
Content-based filtering using track audio features.

Approach
--------
Each track is represented as a feature vector of its audio attributes:
  [energy, danceability, acousticness, valence, instrumentalness,
   speechiness, tempo_norm, loudness_norm]

For a given user, we compute their "taste profile" as the weighted
mean of feature vectors of tracks they have listened to (weight =
play weight: 2 for complete, 1 for skip).

Recommendations are the K tracks (not yet heard) with highest
cosine similarity to that taste profile.

Input
-----
  /spotify/listening_history.csv  (HDFS)
  /spotify/tracks.csv             (HDFS)

Output
------
  /spotify/output/content_based_recs/  — top-10 content-based recs per user

Usage:
  spark-submit /home/jovyan/work/spark/content_based.py
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, udf
from pyspark.sql.types import ArrayType, FloatType, DoubleType
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.linalg import Vectors, DenseVector
import pyspark.sql.functions as F
import math

spark = (
    SparkSession.builder
    .appName("SpotifyContentBased")
    .config("spark.sql.shuffle.partitions", "50")
    .config("spark.driver.memory", "2g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

HDFS_BASE  = "hdfs://namenode:9000/spotify"
TOP_K      = 10
FEATURE_COLS = [
    "energy", "danceability", "acousticness",
    "valence", "instrumentalness", "speechiness",
]

# ─── 1. Load ──────────────────────────────────────────────────────────────────

print("\n=== [1] Loading data ===")

history = spark.read.csv(f"{HDFS_BASE}/listening_history.csv", header=True, inferSchema=True)
tracks  = spark.read.csv(f"{HDFS_BASE}/tracks.csv",            header=True, inferSchema=True)

# ─── 2. Normalise Audio Features ─────────────────────────────────────────────

print("\n=== [2] Normalising track audio features ===")

# Normalise tempo [60, 200] and loudness [-40, 0] to [0, 1]
tracks = (
    tracks
    .withColumn("tempo_norm",    (col("tempo") - 60.0) / 140.0)
    .withColumn("loudness_norm", (col("loudness") + 40.0) / 40.0)
)

all_feature_cols = FEATURE_COLS + ["tempo_norm", "loudness_norm"]

assembler = VectorAssembler(inputCols=all_feature_cols, outputCol="features_raw")
tracks_vec = assembler.transform(tracks)

scaler = MinMaxScaler(inputCol="features_raw", outputCol="features")
scaler_model = scaler.fit(tracks_vec)
tracks_scaled = scaler_model.transform(tracks_vec)

tracks_scaled.cache()
print(f"  Track feature vectors computed: {tracks_scaled.count():,}")

# ─── 3. Build User Taste Profiles ─────────────────────────────────────────────
#
# taste_profile[u] = Σ(weight_i * features_i) / Σ(weight_i)
# where the sum is over all tracks i that user u has listened to.
#
# We do this via a Pandas-on-Spark approach: collect per-user play weights
# then broadcast track features.

print("\n=== [3] Building user taste profiles ===")

play_weights = (
    history
    .withColumn("weight", when(col("skipped") == 0, lit(2.0)).otherwise(lit(1.0)))
    .select("user_id", "track_id", "weight")
)

# Join to get track features alongside play weights
plays_with_features = (
    play_weights
    .join(
        tracks_scaled.select("track_id", "features"),
        "track_id"
    )
)

# Convert features vector to array for aggregation
def vec_to_list(v):
    return v.toArray().tolist() if v is not None else None

vec_to_list_udf = udf(vec_to_list, ArrayType(DoubleType()))

plays_with_features = plays_with_features.withColumn(
    "feat_arr", vec_to_list_udf(col("features"))
)

# Compute weighted sum per user in Python (collect per user)
# For large datasets this would be done in Spark natively;
# here we use a UDF approach for clarity.
n_features = len(all_feature_cols)

from pyspark.sql.types import StructType, StructField, StringType, ArrayType, DoubleType as DT

@F.pandas_udf(ArrayType(DoubleType()))
def weighted_mean_profile(weights_series, feat_series):
    import numpy as np
    import pandas as pd

    results = []
    # These are grouped per user — pandas_udf receives the whole group
    total_w = weights_series.sum()
    if total_w == 0:
        results = [0.0] * n_features
    else:
        feature_matrix = np.array(feat_series.tolist())
        w = weights_series.values[:, np.newaxis]
        results = (feature_matrix * w).sum(axis=0) / total_w
        results = results.tolist()
    return pd.Series([results])


taste_profiles = (
    plays_with_features
    .groupBy("user_id")
    .agg(
        weighted_mean_profile(col("weight"), col("feat_arr")).alias("taste_profile")
    )
)

taste_profiles.cache()
print(f"  User taste profiles: {taste_profiles.count():,}")

# ─── 4. Cosine Similarity & Recommendation ───────────────────────────────────

print("\n=== [4] Computing cosine similarity (Spark cross-join on samples) ===")

# For a full production system, approximate nearest-neighbour (ANN) would be used.
# Here we demonstrate the approach on a sample of 1000 users for performance.

SAMPLE_USERS = 1000
taste_sample = taste_profiles.limit(SAMPLE_USERS)

# Collect track feature vectors as broadcast variable
tracks_broadcast_data = (
    tracks_scaled
    .select("track_id", "features")
    .rdd
    .map(lambda r: (r["track_id"], r["features"].toArray().tolist()))
    .collectAsMap()
)
bc_tracks = spark.sparkContext.broadcast(tracks_broadcast_data)

# Collect which tracks each user has already heard
heard_tracks = (
    history
    .groupBy("user_id")
    .agg(F.collect_set("track_id").alias("heard"))
)

def cosine_sim(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na  = math.sqrt(sum(x*x for x in a))
    nb  = math.sqrt(sum(x*x for x in b))
    return dot / (na * nb + 1e-9)

@udf(ArrayType(ArrayType(FloatType())))
def top_k_recs(taste_profile, heard_list):
    """Return list of [track_id, score] for top-K unseen tracks."""
    if taste_profile is None:
        return []
    heard_set = set(heard_list or [])
    scores = []
    for tid, feat in bc_tracks.value.items():
        if tid in heard_set:
            continue
        score = cosine_sim(taste_profile, feat)
        scores.append((tid, score))
    scores.sort(key=lambda x: -x[1])
    return [[t, round(s, 6)] for t, s in scores[:TOP_K]]


content_recs = (
    taste_sample
    .join(heard_tracks, "user_id", "left")
    .withColumn("recommendations", top_k_recs(col("taste_profile"), col("heard")))
    .select("user_id", F.explode("recommendations").alias("rec"))
    .select("user_id",
            col("rec")[0].alias("track_id"),
            col("rec")[1].cast(DoubleType()).alias("similarity_score"))
    .join(
        tracks.select("track_id", "name", "artist_name", "genre"),
        "track_id"
    )
    .orderBy("user_id", F.desc("similarity_score"))
)

content_recs.cache()
print(f"  Content-based recommendation pairs: {content_recs.count():,}")
content_recs.show(20, truncate=False)

# ─── 5. Save ──────────────────────────────────────────────────────────────────

print("\n=== [5] Saving content-based recommendations ===")

content_recs.write.csv(
    f"{HDFS_BASE}/output/content_based_recs",
    header=True, mode="overwrite"
)
print(f"  Saved → {HDFS_BASE}/output/content_based_recs")

print("\n=== Content-based pipeline complete ===\n")
spark.stop()
