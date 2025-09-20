from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load dataset
DATA_PATH = os.path.join("data", "internships.csv")
df = pd.read_csv(DATA_PATH)

df['text'] = df[['title', 'description', 'skills']].fillna('').agg(' '.join, axis=1)

# Prepare TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['text'].tolist())

@app.route('/recommend', methods=['POST'])
def recommend():
    body = request.get_json()
    profile_text = body.get("profile", "")
    if not profile_text:
        return jsonify({"error": "Profile cannot be empty"}), 400
    
    profile_vec = tfidf.transform([profile_text])
    sims = cosine_similarity(profile_vec, tfidf_matrix).flatten()
    
    top_idx = np.argsort(sims)[::-1][:5]
    results = []
    for idx in top_idx:
        item = df.iloc[idx]
        results.append({
            "title": item['title'],
            "description": item['description'],
            "skills": item['skills'],
            "location": item['location'],
            "duration": item['duration'],
            "organization": item['organization'],
            "score": float(sims[idx])
        })
    return jsonify(results)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
