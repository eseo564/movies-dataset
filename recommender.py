# recommender.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ------------------------
# Content-Based Recommendation
# ------------------------
def content_based_recommendations(movie_title, all_movies, top_n=3):
    """
    Recommend movies based on overview similarity
    """
    if not all_movies:
        return []

    # Build overview TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english')
    overviews = [m.get('overview', '') for m in all_movies]
    tfidf_matrix = tfidf.fit_transform(overviews)

    # Map title to index
    indices = {m['title']: i for i, m in enumerate(all_movies)}
    idx = indices.get(movie_title)
    if idx is None:
        return []

    # Compute similarity
    sim_scores = list(enumerate(cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Return top N similar movies excluding the movie itself
    top_indices = [i for i, score in sim_scores[1:top_n+1]]
    return [all_movies[i]['title'] for i in top_indices]


# ------------------------
# Sentiment-Based Recommendation (simplified)
# ------------------------
def sentiment_based_recommendations(movie_title, all_movies, top_n=3):
    """
    Recommend movies with similar "sentiment" in the overview.
    This is a simple mock using positive/negative word counts.
    """
    if not all_movies:
        return []

    positive_words = ['good', 'great', 'love', 'excellent', 'amazing', 'fun']
    negative_words = ['bad', 'boring', 'poor', 'worst', 'awful', 'hate']

    def sentiment_score(text):
        text = text.lower()
        pos = sum(text.count(w) for w in positive_words)
        neg = sum(text.count(w) for w in negative_words)
        return pos - neg

    scores = np.array([sentiment_score(m.get('overview', '')) for m in all_movies])
    
    # Map title to index
    indices = {m['title']: i for i, m in enumerate(all_movies)}
    idx = indices.get(movie_title)
    if idx is None:
        return []

    # Compute similarity in sentiment
    movie_score = scores[idx]
    sim_scores = [(i, 1 - abs(movie_score - scores[i])) for i in range(len(all_movies))]
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Return top N excluding the movie itself
    top_indices = [i for i, score in sim_scores[1:top_n+1]]
    return [all_movies[i]['title'] for i in top_indices]


# ------------------------
# Hybrid Recommendation
# ------------------------
def hybrid_recommendations(movie_title, all_movies, top_n=3):
    """
    Combine content similarity and sentiment similarity
    """
    if not all_movies:
        return []

    # Content similarity
    tfidf = TfidfVectorizer(stop_words='english')
    overviews = [m.get('overview', '') for m in all_movies]
    tfidf_matrix = tfidf.fit_transform(overviews)
    indices = {m['title']: i for i, m in enumerate(all_movies)}
    idx = indices.get(movie_title)
    if idx is None:
        return []

    content_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]

    # Sentiment similarity
    positive_words = ['good', 'great', 'love', 'excellent', 'amazing', 'fun']
    negative_words = ['bad', 'boring', 'poor', 'worst', 'awful', 'hate']

    def sentiment_score(text):
        text = text.lower()
        pos = sum(text.count(w) for w in positive_words)
        neg = sum(text.count(w) for w in negative_words)
        return pos - neg

    scores = np.array([sentiment_score(m.get('overview', '')) for m in all_movies])
    movie_score = scores[idx]
    sentiment_sim = 1 - np.abs(movie_score - scores)

    # Hybrid: simple average
    hybrid_sim = (content_sim + sentiment_sim) / 2
    sim_scores = list(enumerate(hybrid_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    top_indices = [i for i, score in sim_scores[1:top_n+1]]
    return [all_movies[i]['title'] for i in top_indices]
