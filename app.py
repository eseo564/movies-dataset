import streamlit as st
import pandas as pd
from data_utils import preprocess_movies, compute_sentiment_scores
from recommender import recommend_movies

# Load dataset (or pre-fetch via API)
@st.cache_data
def load_movies():
    df = pd.read_csv("movies_metadata.csv")  # Or fetch from API
    df = compute_sentiment_scores(df)
    tfidf_matrix = preprocess_movies(df)
    return df, tfidf_matrix

st.set_page_config(page_title="Movie Recommender", page_icon=":clapper:", layout="wide")
st.image("assets/logo.png", width=150)
st.title("🎬 Hybrid Movie Recommender")

movie_df, tfidf_matrix = load_movies()

movie_choice = st.selectbox("Choose a movie:", movie_df['title'].tolist())
num_recs = st.slider("Number of recommendations:", 5, 20, 10)

if st.button("Recommend"):
    recommendations = recommend_movies(movie_choice, movie_df, tfidf_matrix, top_n=num_recs)
    
    if recommendations.empty:
        st.warning("No recommendations found!")
    else:
        for idx, row in recommendations.iterrows():
            st.markdown(f"### {row['title']} ({row.get('year','N/A')})")
            st.image(row['poster'], width=150) if row['poster'] else None
            st.write(row['overview'])
            st.write(f"**Genres:** {row.get('genres','N/A')}")
            st.write("---")
