# Spotify Music Recommendation System — Big Data Case Study

A full big-data pipeline that replicates core components of Spotify's
recommendation architecture using **Hadoop**, **Apache Spark**, and **Python**,
all running in Docker.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          Docker Network                          │
│                                                                  │
│  ┌────────────┐   ┌────────────┐   ┌──────────────────────────┐ │
│  │  namenode  │   │  datanode  │   │   resourcemanager        │ │
│  │  HDFS NN   │◄──│  HDFS DN   │   │   YARN RM                │ │
│  │  :9870 UI  │   │  :9864 UI  │   │   :8088 UI               │ │
│  └─────┬──────┘   └────────────┘   └────────────┬─────────────┘ │
│        │                                         │               │
│        │  HDFS RPC (:9000)         YARN (:8032)  │               │
│        │                                         │               │
│  ┌─────▼──────────────────────────────────────── ▼─────────────┐ │
│  │                       jupyter                                │ │
│  │   JupyterLab + PySpark + Hadoop Client                       │ │
│  │   :8888 (Lab UI)   :4040 (Spark UI)                         │ │
│  │                                                              │ │
│  │   notebooks/spotify_case_study.ipynb  ← main entry point    │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─────────────┐   ┌─────────────┐                              │
│  │ nodemanager │   │historyserver│                              │
│  │  YARN NM    │   │  MR :8188   │                              │
│  └─────────────┘   └─────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- **Docker Desktop** (with Docker Compose v2) — already confirmed installed
- 6 GB free RAM (8 GB recommended)
- 5 GB free disk space

---

## Quick Start

### Step 1 — Start the cluster

```bash
cd C:\Users\adity\repos\bdata
docker compose up -d
```

First run will pull ~5 images and take 3–5 minutes. Watch progress:

```bash
docker compose logs -f
```

### Step 2 — Verify services are up

| Service | URL |
|---|---|
| JupyterLab | http://localhost:8888 |
| HDFS NameNode UI | http://localhost:9870 |
| YARN ResourceManager | http://localhost:8088 |
| MapReduce History | http://localhost:8188 |
| Spark UI (during jobs) | http://localhost:4040 |

Get the Jupyter token:
```bash
docker logs jupyter 2>&1 | grep token
```

### Step 3 — Generate data & initialise HDFS

```bash
# Initialise Hadoop Distributed File System (HDFS) directory structure and upload CSVs
docker exec -it namenode bash /scripts/init_hdfs.sh
```

This script:
1. Waits for HDFS to leave safe-mode
2. Creates `/spotify/` directory tree
3. Uploads `tracks.csv`, `users.csv`, `listening_history.csv`

> **Note**: Data generation (~500k rows) happens automatically inside
> the Jupyter notebook on first run.

### Step 4 — Open the notebook

1. Go to http://localhost:8888
2. Navigate to `notebooks/` → open `spotify_case_study.ipynb`
3. Run all cells (Kernel → Restart & Run All)

The notebook is self-contained and will:
- Install missing Python packages
- Generate synthetic data
- Upload to HDFS
- Run all analytics locally
- Train the ALS model
- Produce visualisations

### Step 5 — Run MapReduce jobs (optional, needs YARN)

```bash
docker exec -it namenode bash /scripts/run_mapreduce.sh
```

Four jobs will be submitted to YARN:
| Job | Input | Output |
|---|---|---|
| 1 — Play Counts | listening_history.csv | /spotify/mapreduce/play_counts/ |
| 2 — User Genre Affinity | listening_history.csv | /spotify/mapreduce/user_genre/ |
| 3 — Track Popularity Score | listening_history.csv | /spotify/mapreduce/track_popularity/ |
| 4 — Hourly Activity | listening_history.csv | /spotify/mapreduce/hourly_activity/ |

Track job progress at http://localhost:8088

---

## Project Structure

```
bdata/
├── docker-compose.yml          ← All 6 services
├── hadoop.env                  ← Shared Hadoop configuration env vars
│
├── hadoop-config/              ← XML configs mounted into Jupyter for HDFS access
│   ├── core-site.xml
│   ├── hdfs-site.xml
│   ├── mapred-site.xml
│   └── yarn-site.xml
│
├── data/
│   └── generate_data.py        ← Synthetic dataset generator
│       (10k users, 5k tracks, 500k listening events)
│
├── mapreduce/                  ← Python Streaming MapReduce jobs
│   ├── job1_play_counts/       mapper.py + reducer.py
│   ├── job2_user_genre_affinity/
│   ├── job3_track_popularity/
│   └── job4_hourly_activity/
│
├── spark/                      ← PySpark jobs
│   ├── als_recommendations.py  ← ALS collaborative filtering
│   ├── content_based.py        ← Cosine similarity content filtering
│   └── analytics.py            ← Spark SQL analytics (top tracks, genres, K-Means)
│
├── scripts/
│   ├── init_hdfs.sh            ← Initialise HDFS, upload data
│   ├── run_mapreduce.sh        ← Submit all 4 MapReduce jobs
│   └── run_spark.sh            ← Submit all Spark jobs
│
└── notebooks/
    └── spotify_case_study.ipynb  ← MAIN ENTRY POINT (interactive case study)
```

---

## Dataset

| Table | Rows | Key Fields |
|---|---|---|
| `tracks.csv` | 5,000 | track_id, name, artist, genre, tempo, energy, danceability, acousticness, valence, instrumentalness, speechiness, loudness |
| `users.csv` | 10,000 | user_id, age, gender, country, subscription_type, preferred_genres |
| `listening_history.csv` | ~500,000 | user_id, track_id, timestamp, play_duration_ms, skipped, hour_of_day, day_of_week |

---

## Case Study Sections (notebook)

| # | Section | Technology |
|---|---|---|
| 1 | Environment & data generation | Python (stdlib only) |
| 2 | Upload to HDFS | hdfs CLI via subprocess |
| 3 | Exploratory Data Analysis | Pandas, Matplotlib, Seaborn |
| 4 | MapReduce results analysis | Pandas (local simulation) |
| 5 | ALS Collaborative Filtering | PySpark MLlib ALS |
| 6 | Content-Based Filtering | Sklearn cosine similarity |
| 7 | K-Means audio clustering | Sklearn KMeans + PCA |
| 8 | Hybrid recommendations | Weighted score combination |
| 9 | Conclusions | — |

---

## Stopping the Cluster

```bash
docker compose down          # stop containers (keeps HDFS data volumes)
docker compose down -v       # stop + delete all volumes (clean slate)
```

---

## Troubleshooting

**Jupyter can't reach HDFS:**
- Check `HADOOP_CONF_DIR` is set: `docker exec jupyter env | grep HADOOP`
- Verify namenode is up: `docker logs namenode | tail -20`

**MapReduce job fails:**
- Check YARN logs at http://localhost:8088
- Ensure data was uploaded: `docker exec namenode hdfs dfs -ls /spotify/`

**Out of memory:**
- Reduce `YARN_CONF_yarn_nodemanager_resource_memory___mb` in `hadoop.env`
- Reduce `--driver-memory` in `spark/` scripts

**"Safe mode" errors:**
- Wait 60s after `docker compose up` before running init_hdfs.sh
