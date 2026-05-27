# 🎵 Spotify Song Recommender

Analyzes 100k+ Spotify songs and recommends similar tracks based on audio features like energy, danceability, valence, and tempo. Songs are clustered into 5 mood groups using KMeans, and recommendations are made using cosine similarity.

## Tech Stack
Python, Pandas, Scikit-learn, Matplotlib, Streamlit

## Setup
```bash
git clone https://github.com/anjanadc06/spotify-recommender.git
cd spotify-recommender
pip install -r requirements.txt
python recommender.py    # run analysis + build recommender
streamlit run app.py     # launch web app
```

## Mood Clusters
🌿 Chill & Acoustic | ⚡ High Energy | 💃 Happy & Danceable | 🌑 Dark & Intense | 🎹 Calm & Instrumental