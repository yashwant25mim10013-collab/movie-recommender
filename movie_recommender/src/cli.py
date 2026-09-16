"""
cli.py
-------
Simple command-line version of the movie recommender.

Run with:  python src/cli.py
"""

import os
import sys

# Allow running this file directly (adds project root to path)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recommender import MovieRecommender

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "movies.csv")


def main():
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at: {DATA_PATH}")
        print("Please download 'movies.csv' (TMDB 5000 Movie Dataset) and place it in the data/ folder.")
        print("See README.md for the download link.")
        return

    print("Loading dataset and building similarity matrix... (this may take a few seconds)")
    recommender = MovieRecommender(DATA_PATH)
    print("Ready!\n")

    movie_name = input("Enter your favourite movie name: ").strip()
    matched_title = recommender.find_closest_title(movie_name)

    if matched_title is None:
        print(f"\nSorry, no close match found for '{movie_name}' in the dataset.")
        return

    print(f"\nClosest match found: {matched_title}")
    recommendations = recommender.recommend(movie_name, top_n=10)

    print("\nMovies suggested for you:\n")
    for i, title in enumerate(recommendations, start=1):
        print(f"{i}. {title}")


if __name__ == "__main__":
    main()
