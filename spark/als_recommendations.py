"""
als_recommendations.py
======================
Spark MLlib ALS (Alternating Least Squares) collaborative filtering
for Spotify track recommendations.

Algorithm
---------
ALS factorises the user–item interaction matrix R (shape U×I) into
two low-rank matrices: U (users × k) and V (items × k) such that
R ≈ U · V^T.  It alternates between fixing U and solving for V, then
fixing V and solving for U, using regularised least squares each step.

Input
-----
  /spotify/listening_history.csv  (HDFS)
  /spotify/tracks.csv             (HDFS)

Output
------
  /spotify/output/als_recommendations/  — top-10 recommendations per user
  /spotify/output/als_model/            — saved ALS model artefacts

Usage (from Jupyter terminal or docker exec):
  spark-submit /home/jovyan/work/spark/als_recommendations.py
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum as _sum, when, lit
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
import pyspark.sql.functions as F

# ─── Spark Session ────────────────────────────────────────────────────────────

spark = (
    SparkSession.builder
    .appName("SpotifyALS")
    .config("spark.sql.shuffle.partitions", "50")
    .config("spark.driver.memory", "2g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

HDFS_BASE = "hdfs://namenode:9000/spotify"

# ─── 1. Load Data ─────────────────────────────────────────────────────────────

print("\n=== [1] Loading data from HDFS ===")

history = spark.read.csv(
    f"{HDFS_BASE}/listening_history.csv",
    header=True, inferSchema=True
)
tracks = spark.read.csv(
    f"{HDFS_BASE}/tracks.csv",
    header=True, inferSchema=True
)

print(f"  Listening events : {history.count():,}")
print(f"  Tracks           : {tracks.count():,}")

# ─── 2. Build Implicit Ratings Matrix ─────────────────────────────────────────
#
# ALS in Spark can work in implicit mode (no explicit ratings).
# We compute a "confidence score" per (user, track):
#   - completed play  : weight 2
#   - skipped play    : weight 1
# Then aggregate into a single implicit rating per (user, track).

print("\n=== [2] Computing implicit ratings ===")

ratings_raw = (
    history
    .withColumn("weight", when(col("skipped") == 0, lit(2)).otherwise(lit(1)))
    .groupBy("user_id", "track_id")
    .agg(_sum("weight").alias("implicit_rating"))
)

# ALS requires integer user/item IDs — encode string IDs to integers
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

user_indexer  = StringIndexer(inputCol="user_id",  outputCol="user_idx")
track_indexer = StringIndexer(inputCol="track_id", outputCol="track_idx")

pipeline = Pipeline(stages=[user_indexer, track_indexer])
model    = pipeline.fit(ratings_raw)
ratings  = model.transform(ratings_raw)

ratings = ratings.withColumn("user_idx",  col("user_idx").cast("int"))
ratings = ratings.withColumn("track_idx", col("track_idx").cast("int"))

print(f"  Rating pairs: {ratings.count():,}")
ratings.show(5)

# ─── 3. Train / Test Split ────────────────────────────────────────────────────

print("\n=== [3] Train / test split (80 / 20) ===")
train, test = ratings.randomSplit([0.8, 0.2], seed=42)
print(f"  Train : {train.count():,}   Test : {test.count():,}")

# ─── 4. ALS Model ─────────────────────────────────────────────────────────────
#
# Key hyperparameters:
#   rank          — latent factor dimension  (k)
#   maxIter       — number of ALS iterations
#   regParam      — L2 regularisation  (λ)
#   implicitPrefs — True: treats ratings as confidence, not direct ratings
#   alpha         — confidence scaling for implicit feedback  c_{ui} = 1 + α·r_{ui}
#   coldStartStrategy — 'drop' removes NaN predictions for unseen users/items

print("\n=== [4] Training ALS model ===")

als = ALS(
    rank=50,
    maxIter=15,
    regParam=0.1,
    alpha=40.0,
    implicitPrefs=True,
    userCol="user_idx",
    itemCol="track_idx",
    ratingCol="implicit_rating",
    coldStartStrategy="drop",
    seed=42,
)

als_model = als.fit(train)

# ─── 5. Evaluate ──────────────────────────────────────────────────────────────

print("\n=== [5] Evaluating on test set ===")

predictions = als_model.transform(test)
predictions.cache()

# RMSE (lower is better)
evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="implicit_rating",
    predictionCol="prediction",
)
rmse = evaluator.evaluate(predictions)
print(f"  RMSE  = {rmse:.4f}")

# MAE
evaluator_mae = RegressionEvaluator(
    metricName="mae",
    labelCol="implicit_rating",
    predictionCol="prediction",
)
mae = evaluator_mae.evaluate(predictions)
print(f"  MAE   = {mae:.4f}")

# ─── 6. Generate Top-10 Recommendations Per User ──────────────────────────────

print("\n=== [6] Generating top-10 recommendations per user ===")

recs_raw = als_model.recommendForAllUsers(10)

# Decode integer indices back to string IDs using the index models
user_id_df  = model.stages[0].labels   # user StringIndexerModel
track_id_df = model.stages[1].labels

# Build reverse-lookup DataFrames
from pyspark.sql.types import StringType, IntegerType, StructType, StructField

user_map = spark.createDataFrame(
    [(i, lbl) for i, lbl in enumerate(user_id_df)],
    schema=StructType([
        StructField("user_idx",  IntegerType(), False),
        StructField("user_id_decoded", StringType(), False),
    ])
)
track_map = spark.createDataFrame(
    [(i, lbl) for i, lbl in enumerate(track_id_df)],
    schema=StructType([
        StructField("track_idx", IntegerType(), False),
        StructField("track_id_decoded", StringType(), False),
    ])
)

# Explode recommendations array
recs_exploded = (
    recs_raw
    .select("user_idx", F.explode("recommendations").alias("rec"))
    .select("user_idx",
            col("rec.track_idx").alias("track_idx"),
            col("rec.rating").alias("predicted_score"))
)

# Join decoded IDs
recs_decoded = (
    recs_exploded
    .join(user_map,  "user_idx")
    .join(track_map, "track_idx")
    .join(tracks.select("track_id", "name", "artist_name", "genre"),
          tracks["track_id"] == col("track_id_decoded"))
    .select(
        col("user_id_decoded").alias("user_id"),
        col("track_id_decoded").alias("track_id"),
        "name", "artist_name", "genre",
        "predicted_score",
    )
    .orderBy("user_id", F.desc("predicted_score"))
)

recs_decoded.cache()
print(f"  Total recommendation pairs: {recs_decoded.count():,}")
recs_decoded.show(20, truncate=False)

# ─── 7. Save Outputs ──────────────────────────────────────────────────────────

print("\n=== [7] Saving outputs ===")

# Recommendations to HDFS
recs_decoded.write.csv(
    f"{HDFS_BASE}/output/als_recommendations",
    header=True, mode="overwrite"
)
print(f"  Recommendations → {HDFS_BASE}/output/als_recommendations")

# Save model
als_model.save(f"{HDFS_BASE}/output/als_model")
print(f"  Model           → {HDFS_BASE}/output/als_model")

# Metrics summary to HDFS
metrics_df = spark.createDataFrame(
    [("RMSE", rmse), ("MAE", mae)],
    ["metric", "value"]
)
metrics_df.write.csv(
    f"{HDFS_BASE}/output/als_metrics",
    header=True, mode="overwrite"
)

print("\n=== ALS pipeline complete ===\n")
spark.stop()
