
from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json
    # Add your prediction logic here
    return jsonify({"result": "Prediction successful!"})


with open('model_law.pkl', 'rb') as file:
    model_law = pickle.load(file)

with open('model_outcome.pkl', 'rb') as file:
    model_outcome = pickle.load(file)

with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# Function to check if input is valid (based on cosine similarity)
def is_input_valid(new_input, X_train):
    new_input_tfidf = tfidf_vectorizer.transform([new_input])
    similarity_scores = cosine_similarity(new_input_tfidf, tfidf_vectorizer.transform(X_train))
    max_similarity = np.max(similarity_scores)
    return max_similarity >= 0.5

# API endpoint to predict the relevant law and outcome
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    new_input = data['input']

    # Here, you should pass X_train from your model, this could be an array of case facts or some stored data
    X_train = ["dummy data"]  # Replace this with the actual training data

    if is_input_valid(new_input, X_train):
        new_input_tfidf = tfidf_vectorizer.transform([new_input])
        predicted_law = model_law.predict(new_input_tfidf)[0]
        predicted_outcome = model_outcome.predict(new_input_tfidf)[0]
        return jsonify({
            'predicted_law': predicted_law,
            'predicted_outcome': predicted_outcome
        })
    else:
        return jsonify({'error': 'Input is out of bounds or too different from training data'}), 400

if __name__ == '__main__':
    app.run(debug=True)

