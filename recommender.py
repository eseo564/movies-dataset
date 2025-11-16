import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def recommend_movies(title, movie_df, tfidf_matrix, top_n=10, sentiment_weight=0.3):
    """
    Hybrid recommendation:
    - Content similarity using TF-IDF
    - Sentiment adjustment
    """
    if title not in movie_df['title'].values:
        return []

    idx = movie_df[movie_df['title'] == title].index[0]
    content_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    
    # Hybrid score: weighted sum of content similarity and sentiment
    sentiment_scores = movie_df['sentiment'].values
    hybrid_score = (1 - sentiment_weight) * content_sim + sentiment_weight * sentiment_scores

    # Exclude the queried movie
    hybrid_score[idx] = -1

    top_indices = hybrid_score.argsort()[::-1][:top_n]
    return movie_df.iloc[top_indices]
