import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import accuracy_score

def train_heart_disease_model():
    # Load the heart disease dataset
    heart_data = pd.read_csv('Data/Heart_Disease_Prediction.csv')

    # Define features (X) and target (Y)
    X = heart_data.drop(columns='Heart Disease', axis=1)
    Y = heart_data['Heart Disease']

    # Split the data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(Y_test, Y_pred)
    # Save the trained model to a file
    joblib.dump(model, 'Models/heart_disease_model.joblib')

    # Save accuracy to a file
    with open('Data/heart_disease_accuracy.txt', 'w') as accuracy_file:
        accuracy_file.write(f'Accuracy: {accuracy:.2f}')

def predict_heart_disease(input_data):
    # Check if the model file exists
    try:
        model = joblib.load('Models/heart_disease_model.joblib')
    except FileNotFoundError:
        # If the file doesn't exist, train a new model
        train_heart_disease_model()
        model = joblib.load('Models/heart_disease_model.joblib')

    # Create a dictionary for input data
    input_data_dict = {
        'Age': [input_data[0]],
        'Sex': [input_data[1]],
        'Chest pain type': [input_data[2]],
        'BP': [input_data[3]],
        'Cholesterol': [input_data[4]],
        'FBS over 120': [input_data[5]],
        'EKG results': [input_data[6]],
        'Max HR': [input_data[7]],
        'Exercise angina': [input_data[8]],
        'ST depression': [input_data[9]],
        'Slope of ST': [input_data[10]],
        'Number of vessels fluro': [input_data[11]],
        'Thallium': [input_data[12]]
    }

    # Create a DataFrame with the same columns as during training
    input_data_df = pd.DataFrame(input_data_dict)

    # Make a prediction using the trained model
    prediction = model.predict(input_data_df)

    # Return 1 if the prediction is 1 (Presence), else return 0 (Absence)
    return prediction[0]

train_heart_disease_model()