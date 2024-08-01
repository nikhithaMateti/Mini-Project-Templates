from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the dataset and prepare the recommendation model
df = pd.read_csv('HyderabadResturants.csv')
df.dropna(how='any', inplace=True)
df.drop_duplicates(inplace=True)
df = df.rename(columns={'price for one': 'price', 'ratings': 'rating', 'names': 'name'})

feature = df["cuisine"].tolist()
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(feature)
similarity = cosine_similarity(tfidf_matrix)
indices = pd.Series(df.index, index=df['name']).drop_duplicates()

def restaurant_recommendation(name, similarity=similarity):
    index = indices.get(name, None)
    if index is None:
        return []

    similarity_scores = list(enumerate(similarity[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:11]
    restaurant_indices = [i[0] for i in similarity_scores]
    recommendations = df.iloc[restaurant_indices][['name']].copy()
    return recommendations['name'].tolist()

@app.route('/recommend', methods=['GET'])
def recommend():
    name = request.args.get('name')
    if not name:
        return jsonify([])

    recommendations = restaurant_recommendation(name)
    return jsonify(recommendations)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
