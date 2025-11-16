import requests

API_KEY = "765f721b002191fdc6a324061701eed7"
BASE_URL = "https://api.themoviedb.org/3"

def get_movie_details(title):
    """Fetch movie details from TMDB by title"""
    search_url = f"{BASE_URL}/search/movie?api_key={API_KEY}&query={title}"
    response = requests.get(search_url).json()
    
    if response.get('results'):
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

def get_person_id(name):
    """Search for a person by name to get their TMDB ID"""
    url = f"{BASE_URL}/search/person?api_key={API_KEY}&query={name}"
    response = requests.get(url).json()
    results = response.get('results', [])
    if results:
        return results[0]['id']
    return None

def get_filtered_movies(
    start_year=None,
    end_year=None,
    min_rating=None,
    min_votes=None,
    min_runtime=None,
    max_runtime=None,
    language=None,
    certification=None,
    person_id=None,
    genre_ids=None,
    genre_operator="AND"
):
    """Fetch movies using multiple filters"""
    url = f"{BASE_URL}/discover/movie?api_key={API_KEY}&language=en-US&page=1"
    
    if start_year:
        url += f"&primary_release_date.gte={start_year}-01-01"
    if end_year:
        url += f"&primary_release_date.lte={end_year}-12-31"
    if min_rating:
        url += f"&vote_average.gte={min_rating}"
    if min_votes:
        url += f"&vote_count.gte={min_votes}"
    if min_runtime:
        url += f"&with_runtime.gte={min_runtime}"
    if max_runtime:
        url += f"&with_runtime.lte={max_runtime}"
    if language:
        url += f"&with_original_language={language}"
    if certification:
        url += f"&certification_country=US&certification.lte={certification}"
    if person_id:
        url += f"&with_cast={person_id}"
    if genre_ids:
        if genre_operator.upper() == "OR":
            genre_query = "|".join(map(str, genre_ids))
        else:
            genre_query = ",".join(map(str, genre_ids))
        url += f"&with_genres={genre_query}"
    
    response = requests.get(url).json()
    return response.get('results', [])
