# recommender.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# =========================
# Content-Based Recommendation
# =========================
def content_based_recommend(movie_title, movies_df, top_n=3):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['overview'].fillna(''))
    
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    if movie_title not in movies_df['title'].values:
        return pd.DataFrame()  # empty if movie not found
    
    idx = movies_df[movies_df['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [i for i in sim_scores if i[0] != idx]  # exclude itself
    
    movie_indices = [i[0] for i in sim_scores[:top_n]]
    return movies_df.iloc[movie_indices]

# =========================
# Sentiment-Based Recommendation
# =========================
def sentiment_based_recommend(movie_title, movies_df, top_n=3):
    if movie_title not in movies_df['title'].values:
        return pd.DataFrame()
    
    selected_sentiment = movies_df.loc[movies_df['title'] == movie_title, 'avg_sentiment'].values[0]
    movies_df = movies_df.copy()
    movies_df['sentiment_diff'] = abs(movies_df['avg_sentiment'] - selected_sentiment)
    movies_df_filtered = movies_df[movies_df['title'] != movie_title]
    return movies_df_filtered.nsmallest(top_n, 'sentiment_diff')

# =========================
# Hybrid Recommendation
# =========================
def hybrid_recommend(movie_title, movies_df, top_n=3, content_weight=0.7, sentiment_weight=0.3):
    if movie_title not in movies_df['title'].values:
        return pd.DataFrame()
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['overview'].fillna(''))
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    idx = movies_df[movies_df['title'] == movie_title].index[0]
    content_scores = cosine_sim[idx]
    
    selected_sentiment = movies_df.loc[idx, 'avg_sentiment']
    sentiment_scores = 1 - abs(movies_df['avg_sentiment'] - selected_sentiment)
    
    combined_scores = content_weight * content_scores + sentiment_weight * sentiment_scores
    combined_scores[idx] = -1  # exclude itself
    top_indices = combined_scores.argsort()[::-1][:top_n]
    
    return movies_df.iloc[top_indices]
