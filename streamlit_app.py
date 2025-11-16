# streamlit_app.py
"""
Streamlit Movie Recommendation App
- Uses a local CSV for box-office / genre visualizations
- Uses TMDB API to fetch live metadata and build a recommendation dataset
- Implements content-based, sentiment-based and hybrid recommenders
- Includes filters, watchlist, similarity explanations, and simple visualizations

Before running:
- Set environment variable TMDB_API_KEY in your hosting environment (or secrets)
  e.g. TMDB_API_KEY=your_api_key_here
- If running locally: export TMDB_API_KEY=...
"""

import os
import time
from functools import lru_cache

import altair as alt
import pandas as pd
import numpy as np
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Ensure required NLTK data is available
try:
    _ = SentimentIntensityAnalyzer()
except Exception:
    nltk.download("vader_lexicon")
    _ = SentimentIntensityAnalyzer()

# -----------------------
# Configuration & Helpers
# -----------------------
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
if not TMDB_API_KEY:
    st.warning(
        "TMDB API key not found. Set TMDB_API_KEY as an environment variable or "
        "in your Hugging Face / Streamlit secrets. The app will still show CSV visualizations "
        "but recommendations require a valid TMDB API key."
    )

BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p/w342"  # poster size

HEADERS = {"Authorization": f"Bearer {os.getenv('TMDB_BEARER')}" } if os.getenv("TMDB_BEARER") else None
# If you prefer v3 key usage, we will pass api_key in query params.

# -----------------------
# TMDB API functions
# -----------------------
@st.cache_data(show_spinner=False)
def tmdb_get(endpoint: str, params: dict = None):
    """
    Generic TMDB GET wrapper. Uses either v3 api_key or v4 bearer token.
    """
    if params is None:
        params = {}

    # Prefer bearer if set via TMDB_BEARER env var; otherwise use v3 API key
    if HEADERS:
        url = f"{BASE_URL}{endpoint}"
        res = requests.get(url, headers={**HEADERS, "accept": "application/json"}, params=params, timeout=10)
    else:
        params = {**params, "api_key": TMDB_API_KEY}
        url = f"{BASE_URL}{endpoint}"
        res = requests.get(url, params=params, timeout=10)

    res.raise_for_status()
    return res.json()


@st.cache_data(show_spinner=False)
def search_tmdb_movie(query: str, page: int = 1):
    return tmdb_get("/search/movie", {"query": query, "page": page})


@st.cache_data(show_spinner=False)
def get_movie_details(movie_id: int):
    # request credits and keywords in the same call
    return tmdb_get(f"/movie/{movie_id}", {"append_to_response": "credits,keywords,release_dates"})

@st.cache_data(show_spinner=False)
def get_popular_movies(page=1):
    return tmdb_get("/movie/popular", {"page": page})


@st.cache_data(show_spinner=False)
def get_trending(media_type="movie", time_window="week"):
    return tmdb_get(f"/trending/{media_type}/{time_window}")


# -----------------------
# Build TMDB-backed dataset
# -----------------------
@st.cache_data(show_spinner=True)
def build_tmdb_dataset(num_movies: int = 300):
    """
    Fetch a set of popular movies from TMDB and return a DataFrame enriched with:
      - id, title, overview, year, runtime, genres (list), rating, votes, cast (list), director, language, poster_path
    This is intentionally simple (polls / paginates / fetch details) and caches results.
    """
    movies = []
    page = 1
    # get popular movies across pages until we have enough
    while len(movies) < num_movies:
        try:
            resp = get_popular_movies(page=page)
        except Exception as e:
            st.error(f"Error fetching popular movies: {e}")
            break
        for item in resp.get("results", []):
            mid = item.get("id")
            try:
                details = get_movie_details(mid)
            except Exception:
                # Skip on any error for a single movie
                continue

            # extract director and top cast
            director = None
            cast_list = []
            for person in details.get("credits", {}).get("crew", []):
                if person.get("job") == "Director":
                    director = person.get("name")
                    break
            for c in details.get("credits", {}).get("cast", [])[:6]:
                cast_list.append(c.get("name"))

            # handle release year
            rel = details.get("release_date") or details.get("release_dates", {}).get("results", [])
            year = None
            if details.get("release_date"):
                year = details.get("release_date")[:4]
            else:
                year = None

            genres = [g["name"] for g in details.get("genres", [])] if details.get("genres") else []

            movies.append(
                {
                    "id": details.get("id"),
                    "title": details.get("title"),
                    "overview": details.get("overview") or "",
                    "year": year,
                    "runtime": details.get("runtime"),
                    "genres": genres,
                    "rating": details.get("vote_average") or 0.0,
                    "votes": details.get("vote_count") or 0,
                    "cast": cast_list,
                    "director": director,
                    "language": details.get("original_language"),
                    "poster_path": details.get("poster_path"),
                }
            )
            if len(movies) >= num_movies:
                break
        page += 1
        # safety
        if page > 20:
            break

    df = pd.DataFrame(movies)
    # keep deterministic types
    df["year"] = df["year"].fillna("").astype(str)
    df["genres"] = df["genres"].apply(lambda g: g if isinstance(g, list) else [])
    df["cast"] = df["cast"].apply(lambda c: c if isinstance(c, list) else [])
    return df


# -----------------------
# NLP + Feature extraction
# -----------------------
@st.cache_data(show_spinner=False)
def build_nlp_features(df_tmdb: pd.DataFrame):
    """
    Build TF-IDF matrix and sentiment scores.
    Returns: tfidf_vectorizer, tfidf_matrix, sentiment_scores (Series)
    """
    overviews = df_tmdb["overview"].fillna("").astype(str).tolist()
    tfidf = TfidfVectorizer(stop_words="english", max_features=20000)
    try:
        matrix = tfidf.fit_transform(overviews)
    except ValueError:
        # fallback if all overview strings are empty
        matrix = tfidf.fit_transform([""] * len(overviews))

    sia = SentimentIntensityAnalyzer()
    sentiments = df_tmdb["overview"].fillna("").apply(lambda x: sia.polarity_scores(str(x))["compound"])

    return tfidf, matrix, sentiments


# -----------------------
# Recommendation functions
# -----------------------
def recommend_content(df_tmdb, tfidf_matrix, movie_title, top_k=10):
    """Return top_k similar movies by TF-IDF cosine similarity on overview."""
    if movie_title not in df_tmdb["title"].values:
        return pd.DataFrame()  # not found
    idx = df_tmdb.index[df_tmdb["title"] == movie_title][0]
    sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]
    sim_idx = np.argsort(sims)[-top_k - 1:-1][::-1]  # exclude self
    result = df_tmdb.iloc[sim_idx].copy()
    result["score"] = sims[sim_idx]
    return result.reset_index(drop=True)


def recommend_sentiment(df_tmdb, sentiment_series, movie_title, top_k=10):
    """Return top_k movies with closest sentiment score to the movie_title's overview."""
    if movie_title not in df_tmdb["title"].values:
        return pd.DataFrame()
    target_sent = float(sentiment_series[df_tmdb["title"] == movie_title].iloc[0])
    temp = df_tmdb.copy()
    temp["sent_diff"] = (sentiment_series - target_sent).abs()
    return temp.sort_values("sent_diff").iloc[1 : top_k + 1].reset_index(drop=True)


def recommend_hybrid(df_tmdb, tfidf_matrix, sentiment_series, movie_title, top_k=10, w_content=0.7, w_sent=0.3):
    """
    Hybrid: combine normalized content similarity and inverse sentiment difference.
    """
    content_df = recommend_content(df_tmdb, tfidf_matrix, movie_title, top_k * 3)
    senti_df = recommend_sentiment(df_tmdb, sentiment_series, movie_title, top_k * 3)
    if content_df.empty or senti_df.empty:
        return pd.DataFrame()
    # Merge on id/title
    merged = content_df.merge(senti_df[["id", "sent_diff"]], on="id", how="inner")
    # normalize content score and sent_diff
    merged["content_norm"] = (merged["score"] - merged["score"].min()) / (
        merged["score"].max() - merged["score"].min() + 1e-9
    )
    merged["sent_inv"] = 1 - (merged["sent_diff"] - merged["sent_diff"].min()) / (
        merged["sent_diff"].max() - merged["sent_diff"].min() + 1e-9
    )
    merged["hybrid_score"] = w_content * merged["content_norm"] + w_sent * merged["sent_inv"]
    return merged.sort_values("hybrid_score", ascending=False).head(top_k).reset_index(drop=True)


# -----------------------
# Utilities for UI
# -----------------------
def get_poster_url(poster_path):
    if not poster_path:
        return None
    return IMAGE_BASE + poster_path


def format_movie_card(row):
    title = row.get("title")
    year = row.get("year")
    rating = row.get("rating")
    runtime = row.get("runtime")
    genres = ", ".join(row.get("genres") or [])
    director = row.get("director") or "N/A"
    cast = ", ".join(row.get("cast") or [])
    poster = get_poster_url(row.get("poster_path"))
    return title, year, rating, runtime, genres, director, cast, poster


def explain_similarity(tfidf_vectorizer, tfidf_matrix, base_idx, neighbor_idx, top_n=6):
    """
    Find top contributing terms to the similarity between base_idx and neighbor_idx.
    We'll look at the absolute weight of terms in the neighbor TF-IDF vector limited to top_n.
    """
    feat = tfidf_vectorizer.get_feature_names_out()
    base_vec = tfidf_matrix[base_idx].toarray().reshape(-1)
    neigh_vec = tfidf_matrix[neighbor_idx].toarray().reshape(-1)
    # elementwise product highlights shared terms
    shared = base_vec * neigh_vec
    top_indices = np.argsort(shared)[-top_n:][::-1]
    terms = [feat[i] for i in top_indices if shared[i] > 0]
    return terms


# -----------------------
# Load local CSV for visualization (your original file)
# -----------------------
@st.cache_data(show_spinner=False)
def load_local_csv(path="data/movies_genres_summary.csv"):
    if not os.path.exists(path):
        st.info("Local CSV not found: keep a file at data/movies_genres_summary.csv to enable gross/genre visualizations.")
        return pd.DataFrame()
    df_local = pd.read_csv(path)
    return df_local


# -----------------------
# App layout
# -----------------------
st.sidebar.title("Controls")
data_mode = st.sidebar.radio("Recommendation dataset source", ("TMDB popular (live)", "Use smaller TMDB sample"))
num_movies = st.sidebar.slider("Number of TMDB movies to index", 100, 800, 300, step=50)

# load local CSV for visualization & left column display
df_local = load_local_csv()

# Build/Load TMDB dataset
with st.spinner("Building TMDB dataset (cached)..."):
    if TMDB_API_KEY:
        df_tmdb = build_tmdb_dataset(num_movies)
        tfidf_vectorizer, tfidf_matrix, sentiment_series = build_nlp_features(df_tmdb)
    else:
        df_tmdb = pd.DataFrame()
        tfidf_vectorizer, tfidf_matrix, sentiment_series = None, None, None


# Top bar
st.title("🎬 Movie Recommender — TMDB + Local CSV")
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
        """
        ## Find movies, get recommendations, and explore trends
        Use the search box to find a movie (from the indexed TMDB set). Use filters to refine results.
        """
    )
with col2:
    st.markdown("**Watchlist**")
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = []

    if st.session_state.watchlist:
        for item in st.session_state.watchlist:
            st.markdown(f"- {item}")
    else:
        st.markdown("_Your watchlist is empty._")


# ---------- Search & filters ----------
st.sidebar.header("Search & Filters")
search_title = st.sidebar.text_input("Search movie title (from indexed TMDB set)")
query_nl = st.sidebar.text_input("Natural-language query (describe what you want)")

years = st.sidebar.slider("Release year (min, max)", 1900, 2026, (2000, 2024))
min_rating = st.sidebar.slider("Min rating", 0.0, 10.0, 5.5)
min_votes = st.sidebar.number_input("Min vote count", min_value=0, value=50, step=10)
runtime_min, runtime_max = st.sidebar.slider("Runtime (minutes)", 0, 400, (0, 240))
lang = st.sidebar.selectbox("Language (original)", options=["Any"] + sorted(df_tmdb["language"].dropna().unique().tolist()) if not df_tmdb.empty else ["Any"])
genre_filter = st.sidebar.multiselect("Genres", options=sorted({g for row in df_tmdb["genres"].fillna("").tolist() for g in (row or [])}) if not df_tmdb.empty else [])
actor_filter = st.sidebar.text_input("Actor (type exact name or part)")
director_filter = st.sidebar.text_input("Director (type exact name or part)")

# ---------- Search results & recommendations ----------
main_col, right_col = st.columns([3, 1])

with main_col:
    st.subheader("Search and Recommendations")
    # If user provided natural-language query, find top matches in overviews
    if query_nl and not df_tmdb.empty:
        st.markdown(f"**Natural-language query:** {query_nl}")
        q_vec = tfidf_vectorizer.transform([query_nl])
        sims = cosine_similarity(q_vec, tfidf_matrix)[0]
        top_idx = np.argsort(sims)[-10:][::-1]
        results = df_tmdb.iloc[top_idx].copy()
        results["score"] = sims[top_idx]
        st.write("Top matches for your description:")
        for _, row in results.iterrows():
            title, year, rating, runtime, genres, director, cast, poster = format_movie_card(row)
            cols = st.columns([1, 4])
            with cols[0]:
                if poster:
                    st.image(poster, width=100)
            with cols[1]:
                st.markdown(f"**{title}** ({year}) — rating: {rating} — {genres}")
                st.markdown(f"{row['overview'][:350]}...")
                st.button("Add to watchlist", key=f"nl_add_{row['id']}", on_click=lambda t=row['title']: st.session_state.watchlist.append(t))

    # If user provided a title search, show movie details and recommendations
    if search_title and not df_tmdb.empty:
        # try to find exact or close title in df_tmdb
        matches = df_tmdb[df_tmdb["title"].str.contains(search_title, case=False, na=False)]
        if matches.empty:
            st.info("No matching titles found in the indexed TMDB set. Try different spelling or increase indexed movie count.")
        else:
            # show first match detail
            movie_row = matches.iloc[0]
            st.markdown(f"### {movie_row['title']} ({movie_row['year']})")
            cols = st.columns([1, 3])
            with cols[0]:
                poster = get_poster_url(movie_row.get("poster_path"))
                if poster:
                    st.image(poster, width=180)
            with cols[1]:
                st.markdown(f"**Genres:** {', '.join(movie_row['genres'])}")
                st.markdown(f"**Director:** {movie_row['director']}")
                st.markdown(f"**Cast:** {', '.join(movie_row['cast'])}")
                st.markdown(f"**Runtime:** {movie_row['runtime']} minutes")
                st.markdown(f"**Rating:** {movie_row['rating']} ({movie_row['votes']} votes)")
                st.markdown(f"**Overview:** {movie_row['overview'][:800]}")

                if st.button("Add to watchlist", key=f"add_{movie_row['id']}"):
                    st.session_state.watchlist.append(movie_row["title"])
                    st.success("Added to watchlist")

            # Apply filters to dataset for candidate pool
            candidates = df_tmdb.copy()
            candidates = candidates[candidates["year"].apply(lambda y: int(y) if y.isdigit() else 0).between(years[0], years[1])]
            candidates = candidates[candidates["rating"] >= min_rating]
            candidates = candidates[candidates["votes"] >= min_votes]
            candidates = candidates[(candidates["runtime"].fillna(0) >= runtime_min) & (candidates["runtime"].fillna(0) <= runtime_max)]
            if lang != "Any":
                candidates = candidates[candidates["language"] == lang]
            if genre_filter:
                candidates = candidates[candidates["genres"].apply(lambda g: set(genre_filter).issubset(set(g)))]
            if actor_filter:
                candidates = candidates[candidates["cast"].apply(lambda c: any(actor_filter.lower() in (x or "").lower() for x in c))]
            if director_filter:
                candidates = candidates[candidates["director"].apply(lambda d: director_filter.lower() in (d or "").lower() if d else False)]

            # recommendations
            st.markdown("#### Content-based recommendations")
            content_recs = recommend_content(candidates.reset_index(drop=True), tfidf_matrix, movie_row["title"], top_k=8)
            if content_recs.empty:
                st.write("No content-based recommendations found (increase indexed movies or relax filters).")
            else:
                for idx, r in content_recs.iterrows():
                    cols = st.columns([1, 4])
                    with cols[0]:
                        poster = get_poster_url(r.get("poster_path"))
                        if poster:
                            st.image(poster, width=80)
                    with cols[1]:
                        st.markdown(f"**{r['title']}** ({r['year']}) — {r['genres']}")
                        st.markdown(f"Score: {r['score']:.3f} — Rating: {r['rating']} — Votes: {r['votes']}")
                        # explanation of similarity
                        try:
                            base_idx = df_tmdb.index[df_tmdb["title"] == movie_row["title"]][0]
                            neigh_idx = df_tmdb.index[df_tmdb["id"] == r["id"]][0]
                            terms = explain_similarity(tfidf_vectorizer, tfidf_matrix, base_idx, neigh_idx, top_n=6)
                            if terms:
                                st.markdown(f"**Shared terms:** {', '.join(terms)}")
                        except Exception:
                            pass
                        st.button("Add to watchlist", key=f"cont_add_{r['id']}", on_click=lambda t=r['title']: st.session_state.watchlist.append(t))

            st.markdown("#### Sentiment-based recommendations")
            senti_recs = recommend_sentiment(candidates.reset_index(drop=True), sentiment_series, movie_row["title"], top_k=8)
            if senti_recs.empty:
                st.write("No sentiment-based recommendations found.")
            else:
                for idx, r in senti_recs.iterrows():
                    cols = st.columns([1, 4])
                    with cols[0]:
                        poster = get_poster_url(r.get("poster_path"))
                        if poster:
                            st.image(poster, width=80)
                    with cols[1]:
                        st.markdown(f"**{r['title']}** — sentiment diff: {r['sent_diff']:.3f} — Rating: {r['rating']}")
                        st.button("Add to watchlist", key=f"senti_add_{r['id']}", on_click=lambda t=r['title']: st.session_state.watchlist.append(t))

            st.markdown("#### Hybrid recommendations")
            hybrid_recs = recommend_hybrid(candidates.reset_index(drop=True), tfidf_matrix, sentiment_series, movie_row["title"], top_k=8)
            if hybrid_recs.empty:
                st.write("No hybrid recommendations found.")
            else:
                for idx, r in hybrid_recs.iterrows():
                    cols = st.columns([1, 4])
                    with cols[0]:
                        poster = get_poster_url(r.get("poster_path"))
                        if poster:
                            st.image(poster, width=80)
                    with cols[1]:
                        st.markdown(f"**{r['title']}** — hybrid score: {r['hybrid_score']:.3f} — Rating: {r['rating']}")
                        st.button("Add to watchlist", key=f"hyb_add_{r['id']}", on_click=lambda t=r['title']: st.session_state.watchlist.append(t))


# ---------- Right column: trending and visualizations ----------
with right_col:
    st.subheader("Trending (TMDB)")
    if TMDB_API_KEY:
        try:
            trend = get_trending()
            for t in trend.get("results", [])[:6]:
                st.markdown(f"- **{t.get('title') or t.get('name')}** ({t.get('media_type')})")
        except Exception as e:
            st.write("Could not load trending:", e)
    else:
        st.write("Provide TMDB API key to show trending.")

    st.markdown("---")
    st.subheader("Top genres (from local CSV)")
    if not df_local.empty:
        top_genres = df_local.groupby("genre")["gross"].sum().sort_values(ascending=False).head(10)
        st.bar_chart(top_genres)
        st.markdown("You can keep your CSV to show box office visualizations while recommendations come from TMDB.")
    else:
        st.info("Local CSV not loaded. Upload your local CSV to data/movies_genres_summary.csv to enable local visualizations.")


# ---------- Footer: simple diagnostics & deployment info ----------
st.markdown("---")
st.write("## Diagnostics")
st.write(f"TMDB API key present: {'Yes' if TMDB_API_KEY else 'No'}")
st.write(f"Indexed TMDB movies: {len(df_tmdb)}")
st.write("Tip: Increase 'Number of TMDB movies to index' in the sidebar for broader recommendations (cost: more API calls at first build).")

st.markdown("## Deployment notes")
st.markdown(
    """
- Add your TMDB API key as an environment secret:
  - **Hugging Face Spaces**: Settings → Secrets → add `TMDB_API_KEY` = your_key
  - **Streamlit Cloud**: Advanced settings → Secrets
- Include `requirements.txt` (provided) in the repo
- If you rely on local CSV, include it under `data/movies_genres_summary.csv` in the repo
"""
)
