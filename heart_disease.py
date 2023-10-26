import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_and_predict(input_data):
    # dataset
    heart_data = pd.read_csv('Data/Heart_Disease_Prediction.csv')

    # Define features (X) and target (Y)
    X = heart_data.drop(columns='Heart Disease', axis=1)
    Y = heart_data['Heart Disease']

    # Split the data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    # use Logistic Regression  to
    model = LogisticRegression()
    model.fit(X_train, Y_train)

    # Convert the input data to a pandas DataFrame with appropriate column names
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
    input_data_df = pd.DataFrame(input_data_dict)

    # Make a prediction using the trained model
    prediction = model.predict(input_data_df)

    # Return "Presence" if the prediction is "Presence," else return "Absence"
    return 1 if prediction[0] == "Presence" else 0

