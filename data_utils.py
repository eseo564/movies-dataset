import requests

API_KEY = "YOUR_TMDB_API_KEY"
BASE_URL = "https://api.themoviedb.org/3"

def get_movie_details(title):
    """Fetch movie details from TMDB by title"""
    search_url = f"{BASE_URL}/search/movie?api_key={API_KEY}&query={title}"
    response = requests.get(search_url).json()
    
    if response['results']:
        movie_id = response['results'][0]['id']
        details_url = f"{BASE_URL}/movie/{movie_id}?api_key={API_KEY}&append_to_response=credits"
        details = requests.get(details_url).json()
        return details
    return None


def get_popular_movies():
    """Fetch popular movies from TMDB"""
    url = f"{BASE_URL}/movie/popular?api_key={API_KEY}&language=en-US&page=1"
    response = requests.get(url).json()
    return response.get('results', [])
