# ____ __Movie Recommendation System______

A **content-based movie recommendation system** built with Python and
scikit-learn. Give it a movie you want to watch or  like, and it suggests similar movies based
on genre, cast, director, keywords, and tagline — using TF-IDF vectorization
and cosine similarity.

## -----> How It Works

1. Each movie's `genres`, `keywords`, `tagline`, `cast`, and `director` are
   combined into a single text string.
2. All combined strings are converted into TF-IDF vectors (numeric
   representations of text, weighting rare/important words higher).
3. **Cosine similarity** is computed between every pair of movies —
   this gives a similarity score (0 to 1) between any two movies.
4. When you type a movie name, the closest matching title is found
   (using fuzzy string matching, so typos are okay), and the movies
   with the highest similarity scores to it are returned.

## ----> Project Structure

```
movie_recommender/
├── data/
│   └── movies.csv            
├── src/
│   ├── __init__.py
│   ├── recommender.py       
│   └── cli.py                
├── requirements.txt
├── .gitignore
└── README.md
```

## -----> Dataset

This project uses the **TMDB 5000 Movie Dataset**. Download `movies.csv`
(sometimes named `tmdb_5000_movies.csv`) from Kaggle and place it inside
the `data/` folder as `data/movies.csv`:

 https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

The dataset must have at least these columns: `index`, `title`, `genres`,
`keywords`, `tagline`, `cast`, `director`. (If your CSV doesn't have an
`index` column, the code will auto-generate one.)

## ----> How to Run

1. **Clone the repo / open the project folder**

   ```bash
   git clone <your-repo-url>
   cd movie_recommender
   ```
2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
3. **Add the dataset**
   Place `movies.csv` inside the `data/` folder (see Dataset section above).
4. **Run it**

   ```bash
   python src/cli.py
   ```

   It will ask for a movie name and print recommendations.

## ---->   Using it in your own code

```python
from src.recommender import MovieRecommender

recommender = MovieRecommender("data/movies.csv")

# Get top 10 similar movies
recommendations = recommender.recommend("Iron Man", top_n=10)
print(recommendations)

# Just find the closest matching title (fuzzy match)
closest = recommender.find_closest_title("ironman")
print(closest)  # -> "Iron Man"
```

## ---->Core Functions (`src/recommender.py`)

| Function                            | What it does                                                                      |
| ----------------------------------- | --------------------------------------------------------------------------------- |
| `MovieRecommender(csv_path)`      | Loads the CSV, combines features, builds the TF-IDF similarity matrix (done once) |
| `find_closest_title(movie_name)`  | Fuzzy-matches a typed movie name to the closest real title in the dataset         |
| `recommend(movie_name, top_n=10)` | Returns a list of the`top_n` most similar movie titles                          |

## ---->  Tech Stack

- **Python** — core language
- **pandas** — data loading and manipulation
- **scikit-learn** — `TfidfVectorizer` and `cosine_similarity`
- **difflib** — fuzzy matching of movie titles (handles typos)

## ------> Example

```
Enter your favourite movie name: Iron Man

Movies suggested for you:

1. Iron Man 2
2. Iron Man 3
3. Avengers: Age of Ultron
4. The Avengers
5. Captain America: Civil War
6. Captain America: The Winter Soldier
7. Ant-Man
8. X-Men
9. X-Men: Apocalypse
10. X2
```
