from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data = request.get_json()
    feature1 = data['feature1']
    feature2 = data['feature2']
    
    # Preprocess and predict
    input_data = np.array([feature1, feature2]).reshape(1, -1)
    prediction = model.predict(input_data)
    
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
