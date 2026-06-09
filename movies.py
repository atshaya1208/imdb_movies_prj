import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os

def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    movie_path = os.path.join(BASE_DIR, 'movie_df.pkl')
    tfidf_path = os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')
    matrix_path = os.path.join(BASE_DIR, 'tfidf_matrix.pkl')
    
    with open(movie_path, 'rb') as f:
        df = pickle.load(f)
        
    with open(tfidf_path, 'rb') as f:
        tfidf = pickle.load(f)
        
    with open(matrix_path, 'rb') as f:
        X = pickle.load(f)
        
    return df, tfidf, X

df, tfidf, X = load_data()

st.title(" Storyline-Based Movie Recommender")
st.write("Type a movie plot idea below, and we'll find the 5 most similar movies!")

user_input = st.text_area("Enter Storyline:", placeholder="e.g., A group of astronauts travel through a wormhole in search of a new home for humanity...")

if st.button("Recommend"):
    if user_input.strip() == "":
        st.warning("Please enter a storyline first!")
    else:
        with st.spinner("Finding recommendations..."):
        
            user_vector = tfidf.transform([user_input]).toarray()
            
            similarity_scores = cosine_similarity(user_vector, X)[0]
            
            top_indices = similarity_scores.argsort()[-5:][::-1]
            
            st.success("Here are your top 5 recommendations:")
            for rank, idx in enumerate(top_indices, 1):
                movie_title = df.iloc[idx]['title'] 
                movie_storyline = df.iloc[idx].get('storyline', df.iloc[idx].get('overview', 'No description available.'))

                score = similarity_scores[idx]
                
                with st.expander(f"**{rank}. {movie_title}** — *Match Score: {score:.1%}*"):
                    st.markdown(f"**Plot Summary:**")
                    st.write(movie_storyline)