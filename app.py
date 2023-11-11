from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import os
from heart_disease import predict_heart_disease
from lung_cancer import predict_lung_cancer
from diabetes import predict_diabetes
app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return "HealthAI Backend"

@app.route('/predict-heart-disease', methods=['POST'])
def predict_heart():
    input_data = request.get_json()

    expected_keys = [
        'age', 'sex', 'ChestPainType', 'BP', 'Cholesterol', 'FBSOver120',
        'EKGResults', 'MaxHR', 'ExerciseAngina', 'STdepression', 'SlopeOfST',
        'NumberOfVesselsFluro', 'Thallium'
    ]
    
    if not all(key in input_data for key in expected_keys):
        error_message = 'Invalid input data'
        print("Error:", error_message)
        return jsonify({'error': error_message})

    input_data_list = [int(input_data[key]) for key in expected_keys]

    result = predict_heart_disease(input_data_list)

    return jsonify({'result': result})

@app.route('/predict-lung-disease', methods=['POST'])
def predict_lung():
    input_data = request.get_json()

    expected_keys = [
        'GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
        'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING',
        'COUGHING', 'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN'
    ]
    
    if not all(key in input_data for key in expected_keys):
        error_message = 'Invalid input data'
        print("Error:", error_message)
        return jsonify({'error': error_message})
    
    input_data_list = [(input_data[key]) for key in expected_keys]

    result = predict_lung_cancer(input_data_list)

    return jsonify({'result': result})


@app.route('/predict-diabetes', methods=['POST'])
def predict_diab():
    input_data = request.get_json()

    expected_keys = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
        'DiabetesPedigreeFunction', 'Age'
    ]
    
    if not all(key in input_data for key in expected_keys):
        error_message = 'Invalid input data'
        print("Error:", error_message)
        return jsonify({'error': error_message})
    
    input_data_list = [(input_data[key]) for key in expected_keys]

    result = predict_diabetes(input_data_list)

    return jsonify({'result': result})



@app.route('/download-heart-data-csv', methods=['GET'])
def download_heart_data_csv():
    data_path = 'Data/Heart_Disease_Prediction.csv'
    
    # Check if the CSV file exists
    if os.path.exists(data_path):
        return send_file(data_path, as_attachment=True)
    else:
        return jsonify({'error': 'CSV file not found'})

@app.route('/download-heart-data-accuracy-txt', methods=['GET'])
def download_heart_data_accuracy_txt():
    data_path = 'Data/heart_disease_accuracy.txt'
    
    # Check if the text file exists
    if os.path.exists(data_path):
        return send_file(data_path, as_attachment=True)
    else:
        return jsonify({'error': 'Text file not found'})



@app.route('/download-lung-data-csv', methods=['GET'])
def download_lung_data_csv():
    data_path = 'Data/Lung_Cancer_Detection.csv'
    
    # Check if the CSV file exists
    if os.path.exists(data_path):
        return send_file(data_path, as_attachment=True)
    else:
        return jsonify({'error': 'CSV file not found'})
    
@app.route('/download-lung-data-accuracy-txt', methods=['GET'])
def download_lung_data_accuracy_txt():
    data_path = 'Data/lung_cancer_accuracy.txt'
    
    # Check if the text file exists
    if os.path.exists(data_path):
        return send_file(data_path, as_attachment=True)
    else:
        return jsonify({'error': 'Text file not found'})
    
@app.route('/download-diabetes-data-csv', methods=['GET'])
def download_diabetes_data_csv():
    data_path = 'Data/Diabetes_Detection.csv'
    
    # Check if the CSV file exists
    if os.path.exists(data_path):
        return send_file(data_path, as_attachment=True)
    else:
        return jsonify({'error': 'CSV file not found'})
    
@app.route('/download-diabetes-data-accuracy-txt', methods=['GET'])
def download_diabetes_data_accuracy_txt():
    data_path = 'Data/diabetes_accuracy.txt'
    
    # Check if the text file exists
    if os.path.exists(data_path):
        return send_file(data_path, as_attachment=True)
    else:
        return jsonify({'error': 'Text file not found'})


if __name__ == "__main__":
    app.run(debug=True)
