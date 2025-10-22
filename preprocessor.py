import pandas as pd
import ast # For safely evaluating string representations of lists/dicts
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import sys

# --- Helper Functions ---

def convert_json_list(obj):
    """
    Converts a JSON-like string representation of a list of objects 
    into a list of names.
    Example: '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]' 
    -> ['Action', 'Adventure']
    """
    L = []
    try:
        for i in ast.literal_eval(obj):
            L.append(i['name'])
    except (ValueError, SyntaxError, TypeError):
        pass # Ignore malformed data
    return L

def convert_json_list_director(obj):
    """
    Converts a JSON-like string representation of a crew list 
    to find and return the Director's name.
    Example: '[{"job": "Director", "name": "James Cameron"}, ...]' 
    -> ['James Cameron']
    """
    L = []
    try:
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
                break # We only need the main director
    except (ValueError, SyntaxError, TypeError):
        pass
    return L

def convert_json_list_cast(obj, limit=3):
    """
    Converts a JSON-like string representation of a cast list 
    into a list of the top 'limit' actors' names.
    Example: '[{"name": "Chris Evans"}, {"name": "Robert Downey Jr."}, ...]' 
    -> ['Chris Evans', 'Robert Downey Jr.', 'Mark Ruffalo']
    """
    L = []
    counter = 0
    try:
        for i in ast.literal_eval(obj):
            if counter != limit:
                L.append(i['name'])
                counter += 1
            else:
                break
    except (ValueError, SyntaxError, TypeError):
        pass
    return L

def stem(text):
    """
    Stems a given string of text.
    Example: 'loving loved loves' -> 'love love love'
    """
    ps = PorterStemmer()
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

def preprocess_data(movies, credits):
    """
    Main preprocessing pipeline.
    """
    # Merge datasets
    movies = movies.merge(credits, on='title')
    
    # --- Feature Selection ---
    # Keep only relevant columns for content-based filtering
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    
    # --- Data Cleaning & Transformation ---
    
    # Drop rows with missing essential data
    movies.dropna(inplace=True)
    
    # Apply helper functions to convert JSON-like strings
    print("Converting JSON data...")
    movies['genres'] = movies['genres'].apply(convert_json_list)
    movies['keywords'] = movies['keywords'].apply(convert_json_list)
    movies['cast'] = movies['cast'].apply(convert_json_list_cast)
    movies['crew'] = movies['crew'].apply(convert_json_list_director)
    
    # Handle missing 'overview' (which we'll treat as an empty list for now)
    movies['overview'] = movies['overview'].apply(lambda x: [x] if isinstance(x, str) else [])

    # --- Feature Engineering ---
    
    # Remove spaces from names to create unique tags
    # 'James Cameron' -> 'JamesCameron'
    # This prevents the model from confusing 'James Cameron' with 'James Franco'
    print("Removing spaces from tags...")
    movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
    
    # Combine all features into a single 'tags' column (list of strings)
    print("Creating 'tags' column...")
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    
    # Convert the 'tags' list into a single space-separated string
    new_df = movies[['movie_id', 'title', 'tags']]
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
    
    # Convert all tags to lowercase
    new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
    
    # Apply stemming to the tags
    print("Applying stemming...")
    
    # Download 'punkt' tokenizer if not already present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK 'punkt' tokenizer...")
        nltk.download('punkt')
        
    new_df['tags'] = new_df['tags'].apply(stem)
    
    return new_df

def create_model(df):
    """
    Creates the CountVectorizer model and computes the cosine similarity matrix.
    """
    print("Creating CountVectorizer model...")
    # Initialize CountVectorizer
    # max_features=5000: Keep the 5000 most frequent words
    # stop_words='english': Remove common English words (like 'the', 'is', 'in')
    cv = CountVectorizer(max_features=5000, stop_words='english')
    
    # Transform the 'tags' into a sparse matrix of word counts
    vectors = cv.fit_transform(df['tags']).toarray()
    
    print("Calculating cosine similarity...")
    # Calculate the cosine similarity between all pairs of movies
    # This creates a (n_movies x n_movies) matrix
    similarity = cosine_similarity(vectors)
    
    return similarity

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Movie Recommender Preprocessing Script ---")
    
    # Define file paths
    movies_file = 'tmdb_5000_movies.csv'
    credits_file = 'tmdb_5000_credits.csv'
    
    # Check if data files exist
    if not os.path.exists(movies_file) or not os.path.exists(credits_file):
        print(f"Error: Could not find data files.")
        print(f"Please download '{movies_file}' and '{credits_file}'")
        print("from https://www.kaggle.com/datasets/tmdb/tmdb-5000-movie-dataset")
        print("and place them in the same folder as this script.")
        sys.exit(1) # Exit the script with an error code

    # Load data
    print(f"Loading '{movies_file}' and '{credits_file}'...")
    try:
        movies_df = pd.read_csv(movies_file)
        credits_df = pd.read_csv(credits_file)
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        sys.exit(1)

    # Run preprocessing
    processed_df = preprocess_data(movies_df, credits_df)
    
    # Create the model
    similarity_matrix = create_model(processed_df)
    
    # --- Save Artifacts ---
    # We save the processed_df (which has movie_id, title, tags)
    # and the similarity_matrix.
    
    print("Saving processed data to 'movies.pkl'...")
    with open('movies.pkl', 'wb') as f:
        pickle.dump(processed_df, f)
        
    print("Saving similarity matrix to 'similarity.pkl'...")
    with open('similarity.pkl', 'wb') as f:
        pickle.dump(similarity_matrix, f)
        
    print("\n--- Preprocessing complete! ---")
    print("You can now run the Streamlit app:")
    print("streamlit run app.py")

