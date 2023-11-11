import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_diabetes_model():
    # Load the dataset
    diabetes_data = pd.read_csv('Data/Diabetes_Detection.csv')

    # Define features (X) and target (Y)
    X = diabetes_data.drop(columns='Outcome', axis=1)
    Y = diabetes_data['Outcome']

     # Split the data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(Y_test, Y_pred)

    # Save the trained model and preprocessing steps to a file
    joblib.dump((model, X.columns), 'Models/diabetes_model.joblib')

    # Save accuracy to a file
    with open('Data/diabetes_accuracy.txt', 'w') as accuracy_file:
        accuracy_file.write(f'Accuracy: {accuracy:.2f}')

def predict_diabetes(input_data):
    # Check if the model file exists
    try:
        model, columns = joblib.load('Models/diabetes_model.joblib')
    except FileNotFoundError:
        # If the file doesn't exist, train a new model
        train_diabetes_model()
        model, columns = joblib.load('Models/diabetes_model.joblib')

    # Create a dictionary for input data
    input_data_dict = {
        'Pregnancies': [input_data[0]],
        'Glucose': [input_data[1]],
        'BloodPressure': [input_data[2]],
        'SkinThickness': [input_data[3]],
        'Insulin': [input_data[4]],
        'BMI': [input_data[5]],
        'DiabetesPedigreeFunction': [input_data[6]],
        'Age': [input_data[7]],
    }

    # Create a DataFrame with the same columns as during training
    input_data_df = pd.DataFrame(input_data_dict, columns=columns)

    # Make a prediction using the trained model
    prediction = model.predict(input_data_df)
    return 1 if prediction[0] == 1 else 0