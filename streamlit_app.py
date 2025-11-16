# streamlit_app.py

import streamlit as st
from data_utils import get_movie_details, get_popular_movies
from recommender import content_based_recommendations, sentiment_based_recommendations, hybrid_recommendations

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender")

# --- Movie selection ---
movie_name = st.text_input("Enter a movie name:")

if movie_name:
    # Fetch movie details from API
    movie_details = get_movie_details(movie_name)
    
    if movie_details:
        st.subheader("Movie Details")
        st.write(f"**Title:** {movie_details.get('title')}")
        st.write(f"**Genres:** {', '.join([g['name'] for g in movie_details.get('genres', [])])}")
        st.write(f"**Director:** {', '.join([d['name'] for d in movie_details.get('credits', {}).get('crew', []) if d['job']=='Director'])}")
        st.write(f"**Cast:** {', '.join([c['name'] for c in movie_details.get('credits', {}).get('cast', [])[:5]])}")
        st.write(f"**Runtime:** {movie_details.get('runtime')} min")
        st.write(f"**Overview:** {movie_details.get('overview')}")

        # Fetch a list of popular movies for recommendation input
        all_movies = get_popular_movies()  # This should return a list of dicts with 'title' and 'overview'

        # --- Recommendations ---
        st.subheader("Recommendations")

        # Content-based
        cb_movies = content_based_recommendations(movie_name, all_movies)
        if cb_movies:
            st.write("**Content-Based:**")
            for m in cb_movies:
                st.write(f"- {m}")
        else:
            st.write("No content-based recommendations found.")

        # Sentiment-based
        sb_movies = sentiment_based_recommendations(movie_name, all_movies)
        if sb_movies:
            st.write("**Sentiment-Based:**")
            for m in sb_movies:
                st.write(f"- {m}")
        else:
            st.write("No sentiment-based recommendations found.")

        # Hybrid
        hybrid_movies = hybrid_recommendations(movie_name, all_movies)
        if hybrid_movies:
            st.write("**Hybrid:**")
            for m in hybrid_movies:
                st.write(f"- {m}")
        else:
            st.write("No hybrid recommendations found.")
    else:
        st.error("Movie not found. Please try another title.")
else:
    st.info("Please enter a movie name to get details and recommendations.")
