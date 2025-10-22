# CineRec-AI-Movie-Recommender-System


CineRec is an interactive web application that recommends movies based on content similarity. This project demonstrates a complete data science pipeline, from raw data cleaning and feature engineering to model building and deployment as a user-friendly web app.

(Replace the URL above with a screenshot of your app. You can upload one to imgur.com)

Features

Content-Based Filtering: Recommends movies by analyzing their plot, genres, keywords, cast, and crew.

NLP Vectorization: Uses CountVectorizer to convert text data into a high-dimensional vector space.

Machine Learning: Implements Cosine Similarity to find the mathematical "closeness" between movies.

Interactive UI: Built with Streamlit for a clean, responsive user interface.

External API Integration: Fetches movie posters in real-time from the OMDb API.

How It Works: The Data Science Pipeline

The recommendation engine is built in two stages:

1. Offline Preprocessing (preprocessor.py)

This script runs once to build the recommendation model.

Data Loading: Loads the raw tmdb_5000_movies.csv and tmdb_5000_credits.csv datasets.

Data Cleaning: Merges the datasets, drops irrelevant columns, and handles missing values.

Feature Engineering:

Parses complex JSON columns (genres, keywords, cast, crew) to extract lists of names (e.g., ['Action', 'Adventure']).

Selects only the top 3 actors and the director to create focused features.

Removes spaces from names (e.g., "James Cameron" -> "JamesCameron") to treat them as unique tags.

Tag Creation: Combines all text features (overview, genres,...) into a single "bag of words" string for each movie, called tags`.

NLP & Modeling:

Stemming: Applies the Porter Stemmer (nltk) to all tags to normalize words (e.g., "loving" -> "love").

Vectorization: Uses CountVectorizer to convert the tags of all 4,800+ movies into a sparse matrix of 5,000 dimensions (the top 5,000 most frequent words).

Similarity Calculation: Computes the Cosine Similarity between every movie vector, resulting in a (4800 x 4800) matrix where each cell represents the similarity score between two movies.

Saving Artifacts: Saves the final processed DataFrame (movies.pkl) and the similarity matrix (similarity.pkl) to disk.

2. Live Web Application (app.py)

This is the Streamlit app that provides the user interface.

Load Models: Loads the movies.pkl and similarity.pkl files into memory.

User Input: Creates a simple dropdown (st.selectbox) for the user to pick a movie.

Get Recommendations: When a movie is selected:

It finds the movie's index in the similarity matrix.

Sorts that movie's row by similarity score (highest to lowest).

Selects the top 5 most similar movies.

Display Results:

Displays the names of the 5 recommended movies.

Calls the OMDb API using the movie titles to fetch their official poster URLs.

Renders the posters in a clean layout using st.columns.

How to Run This Project

Follow these steps to set up and run the project on your local machine.

1. Prerequisites

Python 3.7+

Kaggle Account (for the dataset)

OMDb API Key (free)

2. Step-by-Step Setup

Step 1: Clone the Repository

git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME


Step 2: Get the OMDb API Key

Go to http://www.omdbapi.com/apikey.aspx

Select the "FREE" plan and activate your key via email.

Open app.py and paste your key into the OMDB_API_KEY variable on line 8.

Step 3: Get the Dataset

Download the TMDB 5000 Movie Dataset from Kaggle:
https://www.kaggle.com/datasets/tmdb/tmdb-5000-movie-dataset

Unzip the file and place tmdb_5000_movies.csv and tmdb_5000_credits.csv directly into your project folder.

Step 4: Create a Virtual Environment
It's highly recommended to use a virtual environment.

# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate


Step 5: Install Required Libraries

pip install -r requirements.txt


Step 6: Run the Preprocessor (One-Time Only)
This script will build your model files (.pkl).

python preprocessor.py


This will take a minute. It will print "Preprocessing complete!" when finished.

Step 7: Run the Streamlit App!

streamlit run app.py


Your app will automatically open in your default web browser.

Technologies Used

Python

Streamlit: For the web app interface.

Pandas: For data manipulation and loading.

Scikit-learn: For CountVectorizer and cosine_similarity.

NLTK: For natural language processing (stemming).

Requests: For making API calls to OMDb.

Pickle: For saving and loading the ML model.
