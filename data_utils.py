import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import os

nltk.download('vader_lexicon')

# TMDb API configuration
TMDB_API_KEY = os.getenv("TMDB_API_KEY")  # Store key in environment variable
TMDB_BASE_URL = "https://api.themoviedb.org/3"

def fetch_movie_data(movie_title):
    """
    Fetch movie details from TMDb API.
    Returns dict with title, overview, genres, poster URL, release year.
    """
    search_url = f"{TMDB_BASE_URL}/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": movie_title}
    response = requests.get(search_url, params=params)
    data = response.json()
    
    if data['results']:
        movie = data['results'][0]
        return {
            "title": movie.get("title"),
            "overview": movie.get("overview", ""),
            "genres": [g['name'] for g in movie.get("genre_ids", [])],
            "poster": f"https://image.tmdb.org/t/p/w500{movie.get('poster_path')}" if movie.get("poster_path") else None,
            "year": movie.get("release_date", "")[:4]
        }
    else:
        return None

def preprocess_movies(movie_df):
    """
    Fill missing overviews and compute TF-IDF vectors.
    """
    movie_df['overview'] = movie_df['overview'].fillna('')
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movie_df['overview'])
    return tfidf_matrix

def compute_sentiment_scores(movie_df):
    sia = SentimentIntensityAnalyzer()
    movie_df['sentiment'] = movie_df['overview'].apply(lambda x: sia.polarity_scores(x)['compound'])
    return movie_df
