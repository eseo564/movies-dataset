import streamlit as st
from data_utils import get_movie_details, get_filtered_movies, get_person_id
from recommender import content_based_recommendations, sentiment_based_recommendations, hybrid_recommendations

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender")

# --- Movie input ---
movie_name = st.text_input("Enter a movie name:")

# --- Filters ---
st.subheader("Filters")
col1, col2 = st.columns(2)

with col1:
    start_year = st.number_input("Start Year", min_value=1900, max_value=2030, value=2000)
    end_year = st.number_input("End Year", min_value=1900, max_value=2030, value=2025)
    min_rating = st.slider("Minimum Rating", 0.0, 10.0, 7.0)
    min_votes = st.number_input("Minimum Votes", min_value=0, value=100)
    
with col2:
    min_runtime = st.number_input("Min Runtime (min)", min_value=0, value=60)
    max_runtime = st.number_input("Max Runtime (min)", min_value=0, value=180)
    language = st.text_input("Language (ISO code, e.g., 'en')", value="en")
    director_name = st.text_input("Director (optional)")

genre_ids = st.text_input("Genre IDs (comma-separated, optional, e.g., 28,12 for Action & Adventure)")
genre_operator = st.selectbox("Genre Operator", ["AND", "OR"])

# Convert inputs
person_id = get_person_id(director_name) if director_name else None
genre_ids_list = [int(g.strip()) for g in genre_ids.split(",")] if genre_ids else None

if movie_name:
    movie_details = get_movie_details(movie_name)
    
    if movie_details:
        st.subheader("Movie Details")
        st.write(f"**Title:** {movie_details.get('title')}")
        st.write(f"**Genres:** {', '.join([g['name'] for g in movie_details.get('genres', [])])}")
        st.write(f"**Director:** {', '.join([d['name'] for d in movie_details.get('credits', {}).get('crew', []) if d['job']=='Director'])}")
        st.write(f"**Cast:** {', '.join([c['name'] for c in movie_details.get('credits', {}).get('cast', [])[:5]])}")
        st.write(f"**Runtime:** {movie_details.get('runtime')} min")
        st.write(f"**Overview:** {movie_details.get('overview')}")

        # --- Fetch filtered movies ---
        all_movies = get_filtered_movies(
            start_year=start_year,
            end_year=end_year,
            min_rating=min_rating,
            min_votes=min_votes,
            min_runtime=min_runtime,
            max_runtime=max_runtime,
            language=language,
            person_id=person_id,
            genre_ids=genre_ids_list,
            genre_operator=genre_operator
        )

        st.subheader("Recommendations")
        
        cb_movies = content_based_recommendations(movie_name, all_movies)
        st.write("**Content-Based:**", cb_movies if cb_movies else "No recommendations found.")

        sb_movies = sentiment_based_recommendations(movie_name, all_movies)
        st.write("**Sentiment-Based:**", sb_movies if sb_movies else "No recommendations found.")

        hybrid_movies = hybrid_recommendations(movie_name, all_movies)
        st.write("**Hybrid:**", hybrid_movies if hybrid_movies else "No recommendations found.")
        
    else:
        st.error("Movie not found. Please try another title.")
else:
    st.info("Please enter a movie name to get details and recommendations.")
