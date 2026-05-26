import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ──────────────────────────────────────
st.set_page_config(page_title="🎵 Spotify Recommender", layout="centered")

st.title("🎵 Spotify Song Recommender")
st.markdown("Type a song name and get personalized recommendations based on audio features!")

# ── Load data ─────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("processed_songs.csv")
    FEATURES = ["energy", "danceability", "valence", "acousticness",
                "instrumentalness", "tempo", "loudness", "speechiness"]
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])
    sim_matrix = cosine_similarity(X_scaled)
    return df, sim_matrix

df, similarity_matrix = load_data()

# ── Mood colors ───────────────────────────────────────
MOOD_COLORS = {
    "Chill & Acoustic": "🌿",
    "High Energy": "⚡",
    "Happy & Danceable": "💃",
    "Dark & Intense": "🌑",
    "Calm & Instrumental": "🎹"
}

# ── Search ────────────────────────────────────────────
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