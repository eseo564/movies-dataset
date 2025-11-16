from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def content_based_recommendations(movie_title, all_movies):
    """
    all_movies: list of dicts with 'title' and 'overview'
    Returns up to 3 recommended movie titles
    """
    if not all_movies:
        return []

    # Build overview matrix
    tfidf = TfidfVectorizer(stop_words='english')
    overviews = [m.get('overview', '') for m in all_movies]
    tfidf_matrix = tfidf.fit_transform(overviews)

    # Map title to index
    indices = {m['title']: i for i, m in enumerate(all_movies)}
    
    idx = indices.get(movie_title)
    if idx is None:
        return []

    sim_scores = list(enumerate(cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top 3 recommendations (excluding the movie itself)
    top_indices = [i for i, score in sim_scores[1:4]]
    return [all_movies[i]['title'] for i in top_indices]
