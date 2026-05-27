import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="🎵 Spotify Recommender", layout="centered")
st.title("🎵 Spotify Song Recommender")
st.markdown("Type a song name and get recommendations based on audio features!")

FEATURES = ["energy", "danceability", "valence", "acousticness",
            "instrumentalness", "tempo", "loudness", "speechiness"]

MOOD_LABELS = {
    0: "Chill & Acoustic",
    1: "High Energy",
    2: "Happy & Danceable",
    3: "Dark & Intense",
    4: "Calm & Instrumental"
}

MOOD_COLORS = {
    "Chill & Acoustic": "🌿",
    "High Energy": "⚡",
    "Happy & Danceable": "💃",
    "Dark & Intense": "🌑",
    "Calm & Instrumental": "🎹"
}

@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    df.drop_duplicates(subset=["track_name", "artists"], inplace=True)
    df.dropna(subset=FEATURES + ["track_name", "artists"], inplace=True)
    popular = df[df["popularity"] >= 70]
    rest = df[df["popularity"] < 70].sample(
        n=min(10000 - len(popular), len(df[df["popularity"] < 70])),
        random_state=42
    )
    df = pd.concat([popular, rest]).reset_index(drop=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df["mood_cluster"] = kmeans.fit_predict(X_scaled)
    df["mood"] = df["mood_cluster"].map(MOOD_LABELS)
    sim_matrix = cosine_similarity(X_scaled)
    return df, sim_matrix

with st.spinner("Loading music data... this takes ~30 seconds on first load ⏳"):
    df, similarity_matrix = load_data()

st.success(f"✅ {len(df):,} songs loaded!")

song_input = st.text_input("🔍 Enter a song name:", placeholder="e.g. Blinding Lights")
n_recs = st.slider("Number of recommendations:", 3, 10, 5)

if st.button("Get Recommendations 🎶") and song_input:
    matches = df[df["track_name"].str.lower().str.contains(song_input.lower())]
    if matches.empty:
        st.error(f"Song '{song_input}' not found. Try another name!")
    else:
        idx = matches.index[0]
        song = df.loc[idx]
        mood_icon = MOOD_COLORS.get(song["mood"], "🎵")
        st.success(f"Found: **{song['track_name']}** by {song['artists']}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mood", f"{mood_icon} {song['mood']}")
        col2.metric("Energy", f"{song['energy']:.2f}")
        col3.metric("Danceability", f"{song['danceability']:.2f}")
        col4.metric("Popularity", f"{int(song['popularity'])}")
        st.markdown("---")
        st.subheader(f"🎯 Top {n_recs} Recommendations")
        sim_scores = list(enumerate(similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [s for s in sim_scores if s[0] != idx][:n_recs]
        for i, (song_idx, score) in enumerate(sim_scores):
            row = df.loc[song_idx]
            mood_icon = MOOD_COLORS.get(row["mood"], "🎵")
            with st.expander(f"{i+1}. {row['track_name']} — {row['artists']}"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Mood", f"{mood_icon} {row['mood']}")
                c2.metric("Energy", f"{row['energy']:.2f}")
                c3.metric("Danceability", f"{row['danceability']:.2f}")
                c4.metric("Match Score", f"{score:.3f}")

st.markdown("---")
st.caption("Built with ❤️ using Spotify data, KMeans clustering & cosine similarity")