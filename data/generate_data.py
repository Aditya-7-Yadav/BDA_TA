"""
generate_data.py — Synthetic Spotify-like dataset generator
============================================================
Produces three CSV files in /data (or ./data from project root):
  - users.csv          : 10 000 user profiles
  - tracks.csv         : 5 000 track metadata + audio features
  - listening_history.csv : ~500 000 play events

Usage (inside Jupyter notebook or standalone):
    python generate_data.py [--out /path/to/output]

All random state is seeded so results are reproducible.
"""

import os
import sys
import random
import argparse
import csv
from datetime import datetime, timedelta

SEED = 42
random.seed(SEED)

# ─── Domain Constants ────────────────────────────────────────────────────────

GENRES = [
    "Pop", "Hip-Hop", "Rock", "Electronic", "R&B",
    "Latin", "Country", "Jazz", "Classical", "Indie",
    "Metal", "Reggae", "Blues", "Soul", "Folk",
]

COUNTRIES = [
    "US", "GB", "DE", "BR", "IN", "MX", "AU", "CA", "FR", "SE",
    "NG", "KR", "JP", "AR", "ID", "IT", "ES", "NL", "PL", "ZA",
]

SUBSCRIPTION_TYPES = ["free", "premium"]

FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Jamie",
    "Avery", "Quinn", "Skyler", "Drew", "Blake", "Cameron", "Dana",
    "Emery", "Finley", "Harper", "Hayden", "Jesse", "Kendall",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
    "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson",
]

ARTIST_FIRST = [
    "DJ", "MC", "Young", "Big", "Lil", "The", "Dark", "Electric",
    "Cosmic", "Neon", "Crystal", "Golden", "Silver", "Midnight", "Solar",
]
ARTIST_LAST = [
    "Wave", "Storm", "Echo", "Pulse", "Vibe", "Groove", "Beat",
    "Rhythm", "Flow", "Sound", "Tone", "Melody", "Harmony", "Bass",
]

ADJECTIVES = ["Blue", "Red", "Dark", "Bright", "Lost", "Found", "New", "Old",
              "Last", "First", "Wild", "Quiet", "Loud", "Deep", "High"]
NOUNS = ["Night", "Day", "Road", "Dream", "Heart", "Soul", "Mind", "Fire",
         "Rain", "Sun", "Moon", "Star", "Time", "Love", "Life"]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def randu(lo, hi):
    return random.uniform(lo, hi)

def randi(lo, hi):
    return random.randint(lo, hi)

def randc(seq):
    return random.choice(seq)

def rand_audio_feature():
    return round(randu(0.0, 1.0), 4)


# ─── Generator Functions ─────────────────────────────────────────────────────

def generate_artists(n=500):
    artists = {}
    for i in range(n):
        aid = f"A{i:05d}"
        name = f"{randc(ARTIST_FIRST)} {randc(ARTIST_LAST)}"
        genre = randc(GENRES)
        artists[aid] = {"artist_id": aid, "name": name, "genre": genre}
    return artists


def generate_tracks(artists, n=5000):
    artist_ids = list(artists.keys())
    tracks = []
    for i in range(n):
        tid = f"T{i:06d}"
        artist = artists[randc(artist_ids)]
        title = f"{randc(ADJECTIVES)} {randc(NOUNS)}"
        release_year = randi(2000, 2024)
        duration_ms = randi(120_000, 360_000)   # 2–6 minutes
        popularity = randi(0, 100)
        tempo = round(randu(60, 200), 1)
        tracks.append({
            "track_id": tid,
            "name": title,
            "artist_id": artist["artist_id"],
            "artist_name": artist["name"],
            "genre": artist["genre"],
            "duration_ms": duration_ms,
            "popularity": popularity,
            "release_year": release_year,
            "tempo": tempo,
            "energy": rand_audio_feature(),
            "danceability": rand_audio_feature(),
            "acousticness": rand_audio_feature(),
            "valence": rand_audio_feature(),       # musical positiveness
            "instrumentalness": rand_audio_feature(),
            "loudness": round(randu(-40, 0), 2),   # dB
            "speechiness": rand_audio_feature(),
        })
    return tracks


def generate_users(n=10_000):
    users = []
    for i in range(n):
        uid = f"U{i:07d}"
        age = randi(13, 70)
        # Assign 2–4 preferred genres that will bias listening patterns
        num_pref = randi(2, 4)
        pref_genres = random.sample(GENRES, num_pref)
        users.append({
            "user_id": uid,
            "name": f"{randc(FIRST_NAMES)} {randc(LAST_NAMES)}",
            "age": age,
            "gender": randc(["M", "F", "NB"]),
            "country": randc(COUNTRIES),
            "subscription_type": randc(SUBSCRIPTION_TYPES),
            "preferred_genres": "|".join(pref_genres),
            "account_created_year": randi(2010, 2024),
        })
    return users


def generate_listening_history(users, tracks, n_events=500_000):
    """
    Simulate listening events with realistic biases:
    - Users prefer tracks that match their genre preferences (70 % of plays)
    - Popular tracks have higher play probability
    - Timestamps cluster on evenings and weekends
    - ~20 % of plays are skips (play_duration < 30 s)
    """
    # Pre-build genre → track index for fast lookup
    genre_tracks: dict[str, list] = {}
    for t in tracks:
        genre_tracks.setdefault(t["genre"], []).append(t)

    # Popularity-weighted track pool (global)
    pop_weights = [t["popularity"] + 1 for t in tracks]  # +1 avoids zero weight
    total_weight = sum(pop_weights)
    norm_weights = [w / total_weight for w in pop_weights]

    # Base time: start of 2023
    base_ts = datetime(2023, 1, 1)

    history = []
    for _ in range(n_events):
        user = randc(users)
        pref_genres = user["preferred_genres"].split("|")

        # 70 % genre-affinity pick, 30 % global popularity pick
        if random.random() < 0.70:
            preferred = randc(pref_genres)
            candidate_pool = genre_tracks.get(preferred, tracks)
            track = randc(candidate_pool)
        else:
            # Weighted random sample (approximate via cumulative)
            r = random.random()
            cumulative = 0.0
            track = tracks[-1]
            for t, w in zip(tracks, norm_weights):
                cumulative += w
                if r <= cumulative:
                    track = t
                    break

        # Timestamp: random day in 2023, bias toward evenings
        day_offset = randi(0, 364)
        hour = random.choices(
            range(24),
            weights=[1,1,1,1,1,1,2,3,4,4,4,4,4,4,4,5,6,8,10,10,9,8,6,3],
            k=1
        )[0]
        ts = base_ts + timedelta(days=day_offset, hours=hour,
                                 minutes=randi(0, 59), seconds=randi(0, 59))

        # Play duration: skip (~20 %) or full listen
        if random.random() < 0.20:
            play_duration_ms = randi(5_000, 29_000)   # skipped
            skipped = True
        else:
            # Listened 60–100 % of track
            fraction = randu(0.60, 1.0)
            play_duration_ms = int(track["duration_ms"] * fraction)
            skipped = False

        history.append({
            "user_id": user["user_id"],
            "track_id": track["track_id"],
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "play_duration_ms": play_duration_ms,
            "skipped": int(skipped),
            "hour_of_day": hour,
            "day_of_week": ts.weekday(),   # 0=Mon … 6=Sun
        })

    return history


# ─── CSV Writers ─────────────────────────────────────────────────────────────

def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written {len(rows):,} rows → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(out_dir: str):
    print("=" * 60)
    print("  Spotify Synthetic Dataset Generator")
    print("=" * 60)

    print("\n[1/4] Generating artists …")
    artists = generate_artists(500)

    print("[2/4] Generating tracks …")
    tracks = generate_tracks(artists, 5_000)

    print("[3/4] Generating users …")
    users = generate_users(10_000)

    print("[4/4] Generating listening history (~500 000 events) …")
    history = generate_listening_history(users, tracks, 500_000)

    print("\nWriting CSV files …")
    write_csv(
        os.path.join(out_dir, "tracks.csv"),
        tracks,
        ["track_id","name","artist_id","artist_name","genre","duration_ms",
         "popularity","release_year","tempo","energy","danceability",
         "acousticness","valence","instrumentalness","loudness","speechiness"],
    )
    write_csv(
        os.path.join(out_dir, "users.csv"),
        list(users),
        ["user_id","name","age","gender","country","subscription_type",
         "preferred_genres","account_created_year"],
    )
    write_csv(
        os.path.join(out_dir, "listening_history.csv"),
        history,
        ["user_id","track_id","timestamp","play_duration_ms","skipped",
         "hour_of_day","day_of_week"],
    )

    print("\nDone!  Files written to:", out_dir)
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic Spotify dataset")
    parser.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)),
                        help="Output directory (default: same dir as this script)")
    args = parser.parse_args()
    main(args.out)
