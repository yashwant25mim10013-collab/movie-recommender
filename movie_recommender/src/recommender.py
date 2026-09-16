"""
recommender.py
---------------
Core logic of the Content-Based Movie Recommendation System.

HOW IT WORKS (in short):
1. We take 5 text features of each movie: genres, keywords, tagline, cast, director
2. Combine them into one big string per movie
3. Convert all combined strings into TF-IDF vectors (numeric form of text)
4. Compute cosine similarity between every pair of movies
5. When a user gives a movie name, find the closest matching title,
   then return the movies with the highest similarity score to it.
"""

import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SELECTED_FEATURES = ["genres", "keywords", "tagline", "cast", "director"]


class MovieRecommender:
    def __init__(self, csv_path: str):
        """Load the dataset and build the similarity matrix once."""
        self.movies_data = pd.read_csv(csv_path)
        self._prepare_data()
        self._build_similarity_matrix()

    def _prepare_data(self):
        """Fill missing text values and combine the 5 features into one string."""
        for feature in SELECTED_FEATURES:
            self.movies_data[feature] = self.movies_data[feature].fillna("")

        self.movies_data["combined_features"] = (
            self.movies_data["genres"] + " " +
            self.movies_data["keywords"] + " " +
            self.movies_data["tagline"] + " " +
            self.movies_data["cast"] + " " +
            self.movies_data["director"]
        )

        # 'index' column is used later to map a movie title back to its row.
        # If the CSV doesn't already have one, create it from the row position.
        if "index" not in self.movies_data.columns:
            self.movies_data["index"] = self.movies_data.index

        self.list_of_all_titles = self.movies_data["title"].tolist()

    def _build_similarity_matrix(self):
        """Convert text to TF-IDF vectors and compute cosine similarity."""
        vectorizer = TfidfVectorizer()
        feature_vectors = vectorizer.fit_transform(self.movies_data["combined_features"])
        self.similarity = cosine_similarity(feature_vectors)

    def find_closest_title(self, movie_name: str):
        """Fuzzy-match the user's typed movie name to an actual title in the dataset."""
        matches = difflib.get_close_matches(movie_name, self.list_of_all_titles)
        if not matches:
            return None
        return matches[0]

    def recommend(self, movie_name: str, top_n: int = 10):
        """
        Return a list of up to `top_n` recommended movie titles similar to
        `movie_name`. Returns an empty list if no close match was found.
        """
        close_match = self.find_closest_title(movie_name)
        if close_match is None:
            return []

        movie_index = self.movies_data[self.movies_data.title == close_match]["index"].values[0]
        similarity_scores = list(enumerate(self.similarity[movie_index]))
        sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        recommendations = []
        for idx, score in sorted_scores:
            title = self.movies_data[self.movies_data.index == idx]["title"].values[0]
            if title == close_match:
                continue  # skip the movie itself
            recommendations.append(title)
            if len(recommendations) >= top_n:
                break

        return recommendations
