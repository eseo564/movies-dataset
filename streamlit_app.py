import streamlit as st
from tmdb_utils import get_movie_details, get_popular_movies
from recommender import content_based_recommendations

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender")

# Input movie
movie_input = st.text_input("Enter a movie name:")

# Fetch popular movies for recommendation pool
all_movies_raw = get_popular_movies()
all_movies = [{'title': m['title'], 'overview': m.get('overview', '')} for m in all_movies_raw]

if movie_input:
    movie = get_movie_details(movie_input)
    
    if movie:
        # Display movie details
        st.subheader(movie['title'])
        st.write("Genres:", [g['name'] for g in movie.get('genres', [])])
        st.write("Director:", next((c['name'] for c in movie['credits']['crew'] if c['job'] == 'Director'), 'N/A'))
        st.write("Cast:", [c['name'] for c in movie['credits']['cast'][:5]])
        st.write("Runtime:", movie.get('runtime', 'N/A'), "minutes")
        st.write("Overview:", movie.get('overview', 'N/A'))
        
        # Recommendations
        recommendations = content_based_recommendations(movie['title'], all_movies)
        if recommendations:
            st.subheader("You may also like:")
            for rec in recommendations:
                st.write(f"• {rec}")
        else:
            st.info("No recommendations found.")
    else:
        st.error("Movie not found. Try another title.")
