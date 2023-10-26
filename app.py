from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from heart_disease import train_and_predict

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return "Hello, World"


@app.route('/predict', methods=['POST'])
def predict():
    # Get the user input from the POST request as JSON
    input_data = request.get_json()

    #make sure input_data has all expected data 
    expected_keys = [
        'age', 'sex', 'ChestPainType', 'BP', 'Cholesterol', 'FBSOver120',
        'EKGResults', 'MaxHR', 'ExerciseAngina', 'STdepression', 'SlopeOfST',
        'NumberOfVesselsFluro', 'Thallium'
    ]
    if not all(key in input_data for key in expected_keys):
        error_message = 'Invalid input data'
        print("Error:", error_message)
        return jsonify({'error': error_message})

    # Convert input_data to integers and store them in a list
    input_data_list = [int(input_data[key]) for key in expected_keys]

    # Use the train_and_predict function to make a prediction
    result = train_and_predict(input_data_list)

    # Return the result as JSON
    return jsonify({'result': result})


if __name__ == "__main__":
    app.run(debug=True)
