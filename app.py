import streamlit as st
import pickle
import pandas as pd
import time
import os
import urllib.parse # Used to encode text for URLs
import requests # Added for API calls

# --- API Key ---
# IMPORTANT: Paste your OMDb API key here (e.g., "a1b2c3d4")
OMDB_API_KEY = "a826ab66"


# --- Helper Functions ---

def fetch_poster(movie_title):
    """Fetches the movie poster URL from the OMDb API using the movie title."""
    
    # Check if API key is set
    if not OMDB_API_KEY or OMDB_API_KEY == "YOUR_API_KEY_HERE":
        # Return a placeholder if the API key is missing
        encoded_title = urllib.parse.quote(movie_title)
        return f"https://placehold.co/500x750/333333/FFFFFF?text=API+Key+Missing"
    
    # URL-encode the title for the API call
    encoded_title = urllib.parse.quote(movie_title)
    url = f"http://www.omdbapi.com/?t={encoded_title}&apikey={OMDB_API_KEY}"
    
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for bad status codes
        data = response.json()
        
        # Check if OMDb found the movie and has a poster
        if data.get('Response') == 'True' and data.get('Poster') not in [None, 'N/A']:
            # Return the poster URL
            return data['Poster']
        else:
            # Return a placeholder if no poster is found
            encoded_title = urllib.parse.quote(movie_title)
            return f"https://placehold.co/500x750/333333/FFFFFF?text=Poster+Not+Found"
    
    except requests.exceptions.RequestException:
        # Return a placeholder if the API call fails
        encoded_title = urllib.parse.quote(movie_title)
        return f"https://placehold.co/500x750/333333/FFFFFF?text=API+Error"


@st.cache_data
def load_data():
    """Loads the pickled movie list and similarity matrix."""
    if not os.path.exists('movies.pkl') or not os.path.exists('similarity.pkl'):
        st.error("Model files not found! Please run `preprocessor.py` first.")
        st.stop()
        
    try:
        with open('movies.pkl', 'rb') as f:
            movies = pickle.load(f)
        
        with open('similarity.pkl', 'rb') as f:
            similarity = pickle.load(f)
            
        return movies, similarity
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        st.stop()

def recommend(movie, movies_df, similarity_matrix, num_recommendations=5):
    """
    Recommends movies based on the selected movie.
    """
    try:
        # Get the index of the selected movie
        movie_index = movies_df[movies_df['title'] == movie].index[0]
    except IndexError:
        st.error("Movie not found in the database. This shouldn't happen.")
        return [], []

    # Get the similarity scores for the selected movie
    distances = similarity_matrix[movie_index]
    
    # Sort the movies based on similarity (descending) and get top indices
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:num_recommendations+1]
    
    recommended_movies = []
    recommended_posters = []
    
    for i in movies_list:
        # Get the movie title from the dataframe
        movie_title = movies_df.iloc[i[0]].title
        
        recommended_movies.append(movie_title)
        # Fetch placeholder poster using the movie TITLE
        recommended_posters.append(fetch_poster(movie_title))
        
    return recommended_movies, recommended_posters

# --- Main App UI ---

# Set page config
st.set_page_config(page_title="CineRec Recommender", layout="wide")

# Check for API key at the start
if not OMDB_API_KEY or OMDB_API_KEY == "YOUR_API_KEY_HERE":
    st.warning("Please add your OMDb_API_KEY at the top of `app.py` to see movie posters. Get one for free from omdbapi.com.")

# --- Load Data ---
movies_df, similarity = load_data()


# --- UI Layout ---
st.title('🎬 CineRec: Movie Recommender')
st.markdown("Find your next favorite movie! Select a movie you love, and we'll suggest 5 similar ones.")

# Main user input: A select box for movies
selected_movie_name = st.selectbox(
    'Type or select a movie from the dropdown:',
    movies_df['title'].values
)

# Recommendation button
if st.button('Recommend', type="primary", use_container_width=True):
    if selected_movie_name:
        with st.spinner(f'Finding movies similar to "{selected_movie_name}"...'):
            time.sleep(1) # Small delay to make it feel like it's "thinking"
            names, posters = recommend(selected_movie_name, movies_df, similarity)
            
            st.subheader("Here are your recommendations:")
            
            # Display recommendations in 5 columns
            cols = st.columns(5)
            for i in range(5):
                with cols[i]:
                    st.image(posters[i], caption=names[i], use_container_width=True)
                    # Add a centered link to Google the movie
                    st.markdown(f"<h5 style='text-align: center; font-weight: normal;'><a href='https://www.google.com/search?q={names[i]} movie' target='_blank'>{names[i]}</a></h5>", unsafe_allow_html=True)

    else:
        st.error("Please select a movie.")

# --- Footer ---
st.markdown("---")
st.markdown("Built with Python, Streamlit, Pandas, and Scikit-learn.")
st.markdown("Data from [TMDB](https://www.themoviedb.org/).")


