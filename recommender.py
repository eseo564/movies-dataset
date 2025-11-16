from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def content_based_recommendations(movie_title, all_movies):
    """Return up to 3 recommended movies based on overview similarity"""
    if not all_movies:
        return []

    overviews = [m.get('overview', '') for m in all_movies]
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(overviews)

    indices = {m['title']: i for i, m in enumerate(all_movies)}
    idx = indices.get(movie_title)
    if idx is None:
        return []

    sim_scores = list(enumerate(cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    top_indices = [i for i, _ in sim_scores[1:4]]
    return [all_movies[i]['title'] for i in top_indices]

# Placeholder functions (implement sentiment & hybrid logic similarly)
def sentiment_based_recommendations(movie_title, all_movies):
    """Dummy sentiment-based recommendations"""
    return content_based_recommendations(movie_title, all_movies)

def hybrid_recommendations(movie_title, all_movies):
    """Dummy hybrid recommendations"""
    return content_based_recommendations(movie_title, all_movies)
