# data_utils.py
import requests

API_KEY = "YOUR_TMDB_API_KEY"  # <-- replace with your actual TMDB API key
BASE_URL = "https://api.themoviedb.org/3"

def get_movie_details(title):
    """Fetch movie details from TMDB by title"""
    search_url = f"{BASE_URL}/search/movie?api_key={API_KEY}&query={title}"
    response = requests.get(search_url)
    
    # Check if request was successful
    if response.status_code != 200:
        return None

    data = response.json()
    
    # Safely check 'results'
    if data.get('results'):
        movie_id = data['results'][0]['id']
        details_url = f"{BASE_URL}/movie/{movie_id}?api_key={API_KEY}&append_to_response=credits"
        details_response = requests.get(details_url)
        if details_response.status_code != 200:
            return None
        details = details_response.json()
        return details
    return None


def get_popular_movies():
    """Fetch popular movies from TMDB"""
    url = f"{BASE_URL}/movie/popular?api_key={API_KEY}&language=en-US&page=1"
    response = requests.get(url)
    if response.status_code != 200:
        return []

    data = response.json()
    return data.get('results', [])
