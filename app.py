from flask import Flask, render_template, request, jsonify
from new_ml import LawPredictor
import os

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Initialize the model
predictor = None

def init_model():
    global predictor
    try:
        # First try to load the trained model
        predictor = LawPredictor.load_model()
    except:
        # If loading fails, create and train a new model
        predictor = LawPredictor()
        predictor.train("draft_mini_dataset.csv")

# Initialize the model when the app starts
init_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get case facts from JSON request
        data = request.get_json()
        case_facts = data.get('case_facts')
        
        # Make sure case facts is not empty
        if not case_facts or len(case_facts.strip()) == 0:
            return jsonify({
                'success': False,
                'error': 'Please enter case facts'
            })
        
        # Make prediction using the model
        prediction = predictor.predict(case_facts)
        
        # Return the prediction
        return jsonify({
            'law': prediction['law'],
            'outcome': prediction['outcome']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
