import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ════════════════════════════════════════════════════════
print("📦 Loading dataset...")
df = pd.read_csv("dataset.csv")
print(f"Shape: {df.shape}")
print(df.head())

# ════════════════════════════════════════════════════════
# STEP 2 — CLEAN DATA
# ════════════════════════════════════════════════════════
print("\n🧹 Cleaning data...")
df.drop_duplicates(subset=["track_name", "artists"], inplace=True)
df.dropna(subset=["track_name", "artists", "energy", "danceability"], inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"Cleaned shape: {df.shape}")

# ════════════════════════════════════════════════════════
# STEP 3 — EDA VISUALIZATIONS
# ════════════════════════════════════════════════════════
print("\n🔍 Running EDA...")
sns.set_theme(style="whitegrid")

# Top 10 most popular artists
plt.figure(figsize=(12, 5))
top_artists = df.groupby("artists")["popularity"].mean().sort_values(ascending=False).head(10)
sns.barplot(x=top_artists.values, y=top_artists.index, palette="viridis")
plt.title("Top 10 Artists by Popularity")
plt.xlabel("Average Popularity")
plt.tight_layout()
plt.savefig("plot_01_top_artists.png", dpi=150)
plt.show()

# Energy vs Danceability
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df.sample(2000), x="energy", y="danceability",
                hue="valence", palette="RdYlGn", alpha=0.6)
plt.title("Energy vs Danceability (coloured by Mood/Valence)")
plt.tight_layout()
plt.savefig("plot_02_energy_vs_dance.png", dpi=150)
plt.show()

# Feature distributions
features = ["energy", "danceability", "valence", "acousticness", "tempo"]
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for i, feat in enumerate(features):
    axes[i].hist(df[feat], bins=40, color="steelblue", edgecolor="none", alpha=0.8)
    axes[i].set_title(feat)
plt.suptitle("Audio Feature Distributions")
plt.tight_layout()
plt.savefig("plot_03_feature_distributions.png", dpi=150)
plt.show()

# Popularity distribution
plt.figure(figsize=(8, 4))
sns.histplot(df["popularity"], bins=50, kde=True, color="coral")
plt.title("Song Popularity Distribution")
plt.tight_layout()
plt.savefig("plot_04_popularity.png", dpi=150)
plt.show()

# ════════════════════════════════════════════════════════
# STEP 4 — FEATURE ENGINEERING + SCALING
# ════════════════════════════════════════════════════════
print("\n⚙️  Engineering features...")
FEATURES = ["energy", "danceability", "valence", "acousticness",
            "instrumentalness", "tempo", "loudness", "speechiness"]

X = df[FEATURES].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ════════════════════════════════════════════════════════
# STEP 5 — MOOD CLUSTERING (KMeans)
# ════════════════════════════════════════════════════════
print("\n🎭 Clustering songs by mood...")

# Find optimal K using elbow method
inertias = []
K_range = range(2, 10)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(7, 4))
plt.plot(K_range, inertias, "bo-")
plt.title("Elbow Method — Optimal Number of Clusters")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.tight_layout()
plt.savefig("plot_05_elbow.png", dpi=150)
plt.show()

# Train with K=5 (5 mood clusters)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["mood_cluster"] = kmeans.fit_predict(X_scaled)

# Label the clusters
MOOD_LABELS = {
    0: "Chill & Acoustic",
    1: "High Energy",
    2: "Happy & Danceable",
    3: "Dark & Intense",
    4: "Calm & Instrumental"
}
df["mood"] = df["mood_cluster"].map(MOOD_LABELS)

print("\nMood cluster distribution:")
print(df["mood"].value_counts())

# Visualize clusters
plt.figure(figsize=(9, 5))
sns.scatterplot(data=df.sample(3000), x="energy", y="valence",
                hue="mood", palette="Set2", alpha=0.6)
plt.title("Song Mood Clusters")
plt.tight_layout()
plt.savefig("plot_06_mood_clusters.png", dpi=150)
plt.show()

# ════════════════════════════════════════════════════════
# STEP 6 — SONG RECOMMENDER (Cosine Similarity)
# ════════════════════════════════════════════════════════
print("\n🎵 Building recommender...")

# Sample 10,000 songs to avoid memory issues
# Always keep popular songs + random sample
popular = df[df["popularity"] >= 70]
rest = df[df["popularity"] < 70].sample(n=10000-len(popular), random_state=42)
df = pd.concat([popular, rest]).reset_index(drop=True)
X = df[FEATURES].copy()
X_scaled = scaler.fit_transform(X)
df["mood_cluster"] = kmeans.predict(X_scaled)
df["mood"] = df["mood_cluster"].map(MOOD_LABELS)
similarity_matrix = cosine_similarity(X_scaled)

def recommend_songs(song_name, n=5):
    matches = df[df["track_name"].str.lower() == song_name.lower()]
    if matches.empty:
        # Try partial match
        matches = df[df["track_name"].str.lower().str.contains(song_name.lower())]
    if matches.empty:
        print(f"Song '{song_name}' not found in dataset.")
        return None

    idx = matches.index[0]
    found_song = df.loc[idx, "track_name"]
    found_artist = df.loc[idx, "artists"]
    found_mood = df.loc[idx, "mood"]

    print(f"\nFound: '{found_song}' by {found_artist}")
    print(f"Mood: {found_mood}")
    print(f"\nTop {n} recommendations:")

    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:n]

    results = []
    for i, (song_idx, score) in enumerate(sim_scores):
        row = df.loc[song_idx]
        print(f"{i+1}. {row['track_name']} — {row['artists']} "
              f"(Mood: {row['mood']}, Score: {score:.3f})")
        results.append(row)
    return pd.DataFrame(results)

# ════════════════════════════════════════════════════════
# STEP 7 — TEST THE RECOMMENDER
# ════════════════════════════════════════════════════════
print("\n" + "="*50)
print("TESTING RECOMMENDER")
print("="*50)

test_songs = ["Blinding Lights", "Shape of You", "Bohemian Rhapsody"]
for song in test_songs:
    recommend_songs(song, n=5)
    print()

# Save processed data for Streamlit app
df.to_csv("processed_songs.csv", index=False)
np.save("similarity_matrix.npy", similarity_matrix)
print("\n✅ Data saved! Ready for Streamlit app.")