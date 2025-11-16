# app.py

import streamlit as st
import pandas as pd
from recommender import content_based_recommend, sentiment_based_recommend, hybrid_recommend

# =========================
# Load Data
# =========================
movies_df = pd.read_csv('movies.csv')  # must include columns: 'title', 'overview', 'genres', 'avg_sentiment'

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender System")

# =========================
# User Input
# =========================
movie_choice = st.selectbox("Select a movie:", movies_df['title'].sort_values())

if st.button("Get Recommendations"):
    # Content-Based
    st.subheader("✨ Content-Based Recommendations")
    content_movies = content_based_recommend(movie_choice, movies_df)
    if not content_movies.empty:
        for idx, row in content_movies.iterrows():
            st.markdown(f"**{row['title']}** ({row['genres']})\n\n{row['overview']}")
            st.markdown("---")
    else:
        st.write("No recommendations found.")
    
    # Sentiment-Based
    st.subheader("😊 Sentiment-Based Recommendations")
    sentiment_movies = sentiment_based_recommend(movie_choice, movies_df)
    if not sentiment_movies.empty:
        for idx, row in sentiment_movies.iterrows():
            st.markdown(f"**{row['title']}** ({row['genres']})\n\n{row['overview']}")
            st.markdown("---")
    else:
        st.write("No recommendations found.")
    
    # Hybrid
    st.subheader("🔀 Hybrid Recommendations")
    hybrid_movies = hybrid_recommend(movie_choice, movies_df)
    if not hybrid_movies.empty:
        for idx, row in hybrid_movies.iterrows():
            st.markdown(f"**{row['title']}** ({row['genres']})\n\n{row['overview']}")
            st.markdown("---")
    else:
        st.write("No recommendations found.")
