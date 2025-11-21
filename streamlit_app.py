import streamlit as st
import pandas as pd
import altair as alt
from data_utils import get_movie_details, get_filtered_movies, get_person_id
from recommender import (
    content_based_recommendations,
    sentiment_based_recommendations,
    hybrid_recommendations,
    explain_similarity
)

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender")

# =========================
# Movie input
# =========================
movie_name = st.text_input("Enter a movie name:")

# =========================
# Filters
# =========================
st.subheader("Filters")
col1, col2 = st.columns(2)

with col1:
    start_year = st.number_input("Start Year", min_value=1900, max_value=2030, value=2000)
    end_year = st.number_input("End Year", min_value=1900, max_value=2030, value=2025)
    min_rating = st.slider("Minimum Rating", 0.0, 10.0, 7.0)
    min_votes = st.number_input("Minimum Votes", min_value=0, value=100)
    decade = st.selectbox("Decade", ["Any"] + [f"{i}s" for i in range(1950, 2030, 10)])
    if decade != "Any":
        start_year = int(decade[:4])
        end_year = start_year + 9

with col2:
    min_runtime = st.number_input("Min Runtime (min)", min_value=0, value=60)
    max_runtime = st.number_input("Max Runtime (min)", min_value=0, value=180)
    language = st.text_input("Language (ISO code, e.g., 'en')", value="en")
    director_name = st.text_input("Director (optional)")
    certification = st.text_input("Certification (optional, e.g., 'PG-13')")

genre_ids = st.text_input("Genre IDs (comma-separated, optional, e.g., 28,12)")
genre_operator = st.selectbox("Genre Operator", ["AND", "OR"])

# Convert inputs
person_id = get_person_id(director_name) if director_name else None
genre_ids_list = [int(g.strip()) for g in genre_ids.split(",")] if genre_ids else None

# =========================
# User Watchlist
# =========================
if 'watchlist' not in st.session_state:
    st.session_state['watchlist'] = []

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
        if movie_details.get('poster_path'):
            st.image(f"https://image.tmdb.org/t/p/w200{movie_details['poster_path']}")

        if st.button("Add to Watchlist"):
            st.session_state['watchlist'].append(movie_name)
            st.success(f"'{movie_name}' added to watchlist!")

        # =========================
        # Fetch filtered movies
        # =========================
        all_movies = get_filtered_movies(
            start_year=start_year,
            end_year=end_year,
            min_rating=min_rating,
            min_votes=min_votes,
            min_runtime=min_runtime,
            max_runtime=max_runtime,
            language=language,
            certification=certification,
            person_id=person_id,
            genre_ids=genre_ids_list,
            genre_operator=genre_operator
        )

        # Ensure input movie is in the list
        input_movie_entry = {
            'title': movie_details['title'],
            'overview': movie_details.get('overview', ''),
            'id': movie_details['id'],
            'genres': [g['id'] for g in movie_details.get('genres', [])],
            'poster_path': movie_details.get('poster_path'),
            'release_date': movie_details.get('release_date', '')
        }
        if input_movie_entry['title'] not in [m['title'] for m in all_movies]:
            all_movies.append(input_movie_entry)

        # Remove movies with empty overview
        all_movies = [m for m in all_movies if m.get('overview')]

        # =========================
        # Recommendations
        # =========================
        st.subheader("Recommendations")

        # Content-Based
        cb_movies = content_based_recommendations(movie_name, all_movies)
        if cb_movies:
            st.write("**Content-Based Recommendations:**")
            for m in cb_movies:
                explanation = explain_similarity(movie_name, m, all_movies)
                st.markdown(f"- **{m}** → {explanation}")
        else:
            st.write("No recommendations found.")

        # Sentiment-Based
        sb_movies = sentiment_based_recommendations(movie_name, all_movies)
        if sb_movies:
            st.write("**Sentiment-Based Recommendations:**")
            for m in sb_movies:
                explanation = explain_similarity(movie_name, m, all_movies)
                st.markdown(f"- **{m}** → {explanation}")
        else:
            st.write("No recommendations found.")

        # Hybrid
        hybrid_movies = hybrid_recommendations(movie_name, all_movies)
        if hybrid_movies:
            st.write("**Hybrid Recommendations:**")
            for m in hybrid_movies:
                explanation = explain_similarity(movie_name, m, all_movies)
                st.markdown(f"- **{m}** → {explanation}")
        else:
            st.write("No recommendations found.")

        # =========================
        # Watchlist-Based Aggregated Recommendations
        # =========================
        if st.session_state['watchlist']:
            st.subheader("🎟️ Your Watchlist")
            for w in st.session_state['watchlist']:
                st.write(f"- {w}")

            st.subheader("🔁 Watchlist-Based Recommendations")
            watchlist_recs = []
            for w_movie in st.session_state['watchlist']:
                recs = content_based_recommendations(w_movie, all_movies)
                watchlist_recs.extend([r for r in recs if r not in watchlist_recs])
            if watchlist_recs:
                for r in watchlist_recs[:5]:
                    st.write(f"- {r}")
            else:
                st.write("No watchlist-based recommendations available.")

        # =========================
        # Data Visualization: Genre & Release Year
        # =========================
# Data Visualization: Genre & Release Year
# =========================
if all_movies:
    df = pd.DataFrame(all_movies)
    df['release_year'] = pd.to_datetime(df.get('release_date', pd.Series([None]*len(df))), errors='coerce').dt.year
    df['genres_str'] = df['genres'].apply(lambda g: ",".join(map(str, g)) if g else "None")

    st.subheader("📊 Filtered Movies Overview")

    # Genre Distribution
    genre_chart = alt.Chart(df).mark_bar().encode(
        x='genres_str:N',
        y='count()',
        tooltip=['genres_str', 'count()']
    ).properties(title="Genre Distribution of Filtered Movies")
    st.altair_chart(genre_chart, use_container_width=True)

    # Release Year Distribution
    year_chart = alt.Chart(df.dropna(subset=['release_year'])).mark_bar().encode(
        x=alt.X('release_year:O', title='Release Year'),
        y=alt.Y('count()', title='Number of Movies'),
        tooltip=[alt.Tooltip('release_year:O', title='Year'), alt.Tooltip('count()', title='Count')]
    ).properties(title="Release Year Distribution of Filtered Movies")
    st.altair_chart(year_chart, use_container_width=True)
