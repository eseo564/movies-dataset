from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def content_based_recommendations(movie_title, all_movies):
    """Recommend up to 3 movies based on overview similarity"""
    if not all_movies:
        return []

    tfidf = TfidfVectorizer(stop_words='english')
    overviews = [m.get('overview', '') for m in all_movies]
    tfidf_matrix = tfidf.fit_transform(overviews)

    indices = {m['title']: i for i, m in enumerate(all_movies)}
    idx = indices.get(movie_title)
    if idx is None:
        return []

    sim_scores = list(enumerate(cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    top_indices = [i for i, _ in sim_scores[1:4]]
    return [all_movies[i]['title'] for i in top_indices]

def sentiment_based_recommendations(movie_title, all_movies):
    """Simplified sentiment-based recommendations"""
    cb = content_based_recommendations(movie_title, all_movies)
    return list(reversed(cb)) if cb else []

def hybrid_recommendations(movie_title, all_movies):
    """Combine content-based and sentiment-based recommendations"""
    cb = content_based_recommendations(movie_title, all_movies)
    sb = sentiment_based_recommendations(movie_title, all_movies)
    combined = cb + [m for m in sb if m not in cb]
    return combined[:3]

def explain_similarity(movie_title, recommended_movie, all_movies):
    """Explain why a recommendation was made based on genres"""
    indices = {m['title']: i for i, m in enumerate(all_movies)}
    idx1, idx2 = indices.get(movie_title), indices.get(recommended_movie)
    if idx1 is None or idx2 is None:
        return "No explanation available"
    
    genres1 = set(all_movies[idx1].get('genres', []))
    genres2 = set(all_movies[idx2].get('genres', []))
    common = genres1.intersection(genres2)
    return f"Common genres (IDs): {', '.join(map(str, common))}" if common else "No common genres"

    
    genres1 = set([g['name'] for g in all_movies[idx1].get('genres', [])])
    genres2 = set([g['name'] for g in all_movies[idx2].get('genres', [])])
    common = genres1.intersection(genres2)
    return f"Common genres: {', '.join(common)}" if common else "No common genres"
