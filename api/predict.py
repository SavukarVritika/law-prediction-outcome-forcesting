from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the trained model and vectorizer
with open('model_law.pkl', 'rb') as file:
    model_law = pickle.load(file)

with open('model_outcome.pkl', 'rb') as file:
    model_outcome = pickle.load(file)

with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# Dummy training data for similarity check (Replace with actual training data)
X_train = ["Case Facts"]  # Replace this with actual case facts data

# Function to check if the input is valid based on cosine similarity
def is_input_valid(new_input, X_train):
    new_input_tfidf = tfidf_vectorizer.transform([new_input])
    similarity_scores = cosine_similarity(new_input_tfidf, tfidf_vectorizer.transform(X_train))
    max_similarity = np.max(similarity_scores)
    return max_similarity >= 0.5

# API endpoint to predict the relevant law and outcome
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get the input data from the request
    new_input = data.get('input')  # Extract the 'input' field from the request data

    if new_input:  # Check if input is provided
        if is_input_valid(new_input, X_train):  # Check if the input is valid based on cosine similarity
            new_input_tfidf = tfidf_vectorizer.transform([new_input])  # Transform input to TF-IDF
            predicted_law = model_law.predict(new_input_tfidf)[0]  # Predict the relevant law
            predicted_outcome = model_outcome.predict(new_input_tfidf)[0]  # Predict the outcome

            return jsonify({
                'predicted_law': predicted_law,
                'predicted_outcome': predicted_outcome
            })
        else:
            return jsonify({'error': 'Input is out of bounds or too different from training data'}), 400
    else:
        return jsonify({'error': 'No input provided'}), 400

# Run the app (for local testing, not needed in Vercel)
if __name__ == '__main__':
    app.run(debug=True)
