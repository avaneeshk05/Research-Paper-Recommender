from flask import Flask, request, jsonify, render_template

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import spacy
import re
import os

#new
from dotenv import load_dotenv
from flask_cors import CORS
load_dotenv() #new

app = Flask(__name__)


#Get environ vars
app.config['DEBUG']=os.environ.get('FLASK_DEBUG') #new

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to preprocess text using spaCy
def preprocess_text_spacy(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove numbers & punctuation
    doc = nlp(text)
    words = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(words)

# Load models and data
print("Loading saved models and data...")
knn = joblib.load('knn_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
pca = joblib.load('pca_model.pkl')
df = pd.read_csv('df_updated.csv')
hybrid_features = np.load("hybrid_features.npy", allow_pickle=True)

# Load SBERT model
sbert_model = SentenceTransformer('all-mpnet-base-v2')

@app.route('/')
def home():
    """Renders the homepage with the input form."""
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_papers():
    """
    API endpoint to recommend papers based on input summary.
    """
    data = request.json
    if 'summary' not in data:
        return jsonify({"error": "Missing 'summary' field"}), 400

    input_summary = data['summary']
    top_n = data.get('top_n', 5)  # Default to 5 recommendations

    # Preprocess input summary
    processed_summary = preprocess_text_spacy(input_summary)

    # Convert to TF-IDF vector
    tfidf_vector = tfidf_vectorizer.transform([processed_summary]).toarray()

    # Reduce dimensionality using PCA
    tfidf_reduced = pca.transform(tfidf_vector)

    # Get SBERT embedding
    sbert_embedding = sbert_model.encode([processed_summary])

    # Normalize features
    tfidf_reduced = normalize(tfidf_reduced)
    sbert_embedding = normalize(sbert_embedding)

    # Concatenate TF-IDF and SBERT features
    hybrid_feature = np.hstack((tfidf_reduced, sbert_embedding))

    # Find nearest neighbors
    distances, indices = knn.kneighbors(hybrid_feature.reshape(1, -1), n_neighbors=top_n + 1)

    # Retrieve recommended paper titles
    recommended_titles = df.iloc[indices.flatten()[1:top_n+1]]['title'].tolist()
    recommended_ids = df.iloc[indices.flatten()[1:top_n+1]]['id'].tolist()
    urls=[]
    for i in recommended_ids:
        v=i.split("-")
        if v[0]=="cs":
            urls.append(f"https://arxiv.org/pdf/cs/{v[1]}")
        else:
            urls.append(f"https://arxiv.org/pdf/{v[1]}")
    return jsonify({"recommended_papers": recommended_titles,"urls":urls})

if __name__ == '__main__':
    app.run()
