import ast
import numpy as np
import pandas as pd

# -------------------------------
# Load datasets
# -------------------------------
movies = pd.read_csv(
    '/Users/bharadwaj/Desktop/ML-Projects/Movie_recommendor/tmdb_5000_movies.csv'
)
credits = pd.read_csv(
    '/Users/bharadwaj/Desktop/ML-Projects/Movie_recommendor/tmdb_5000_credits.csv'
)

# -------------------------------
# Merge datasets
# -------------------------------
movies = movies.merge(credits, on='title')

# -------------------------------
# Select relevant columns
# -------------------------------
movies = movies[
    ['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']
]

# -------------------------------
# Data cleaning
# -------------------------------
movies.dropna(inplace=True)
movies.drop_duplicates(inplace=True)

# -------------------------------
# Helper functions
# -------------------------------
def convert(text):
    """Extract 'name' values from JSON-like strings"""
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

def fetch_director(text):
    """Extract director name"""
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

def collapse(L):
    """Remove spaces from words"""
    return [i.replace(" ", "") for i in L]

# -------------------------------
# Feature engineering
# -------------------------------
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert).apply(lambda x: x[:3])
movies['crew'] = movies['crew'].apply(fetch_director)

movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)

movies['overview'] = movies['overview'].apply(lambda x: x.split())

# -------------------------------
# Create tags
# -------------------------------
movies['tags'] = (
    movies['overview']
    + movies['genres']
    + movies['keywords']
    + movies['cast']
    + movies['crew']
)

# -------------------------------
# Final dataframe
# -------------------------------
new = movies[['movie_id', 'title', 'tags']].copy()
new['tags'] = new['tags'].apply(lambda x: " ".join(x).lower())

# -------------------------------
# Vectorization
# -------------------------------
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new['tags']).toarray()

# -------------------------------
# Cosine similarity
# -------------------------------
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)

# -------------------------------
# Recommendation function
# -------------------------------
def recommend(movie):
    if movie not in new['title'].values:
        print(" Movie not found in database")
        return

    index = new[new['title'] == movie].index[0]

    distances = sorted(
        list(enumerate(similarity[index])),
        reverse=True,
        key=lambda x: x[1]
    )

    print(f"\n Movies similar to '{movie}':\n")
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)

# -------------------------------
# Example call
# -------------------------------
recommend('The Lego Movie')
