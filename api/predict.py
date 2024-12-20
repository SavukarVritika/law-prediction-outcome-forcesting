from http.server import BaseHTTPRequestHandler
import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load models and vectorizer
model_law = joblib.load('./models/model_law.pkl')
model_outcome = joblib.load('./models/model_outcome.pkl')
tfidf_vectorizer = joblib.load('./models/tfidf_vectorizer.pkl')

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Read input data from the request body
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        input_data = json.loads(post_data)

        # Check if input is provided
        if 'case_facts' not in input_data:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'Error: case_facts is required')
            return

        # Transform and predict
        new_input = input_data['case_facts']
        new_input_tfidf = tfidf_vectorizer.transform([new_input])
        predicted_law = model_law.predict(new_input_tfidf)[0]
        predicted_outcome = model_outcome.predict(new_input_tfidf)[0]

        # Prepare response
        response = {
            "predicted_law": predicted_law,
            "predicted_outcome": predicted_outcome
        }

        # Send response
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
