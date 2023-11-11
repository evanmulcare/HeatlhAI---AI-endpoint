import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_lung_cancer_model():
    # Load the dataset
    lung_data = pd.read_csv('Data/Lung_Cancer_Detection.csv')

    # Define features (X) and target (Y)
    X = lung_data.drop(columns='LUNG_CANCER', axis=1)
    Y = lung_data['LUNG_CANCER']

    # Perform one-hot encoding for the 'GENDER' feature
    X = pd.get_dummies(X, columns=['GENDER'], drop_first=True)

    # Split the data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    # Create and train a Logistic Regression model
    model = LogisticRegression(random_state=2)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(Y_test, Y_pred)

    # Save the trained model and preprocessing steps to a file
    joblib.dump((model, X.columns), 'Models/lung_cancer_model.joblib')

    # Save accuracy to a file
    with open('Data/lung_cancer_accuracy.txt', 'w') as accuracy_file:
        accuracy_file.write(f'Accuracy: {accuracy:.2f}')

def predict_lung_cancer(input_data):
    # Check if the model file exists
    try:
        model, columns = joblib.load('Models/lung_cancer_model.joblib')
    except FileNotFoundError:
        # If the file doesn't exist, train a new model
        train_lung_cancer_model()
        model, columns = joblib.load('Models/lung_cancer_model.joblib')

    # Create a dictionary for input data
    input_data_dict = {
        'AGE': [input_data[1]],
        'SMOKING': [input_data[2]],
        'YELLOW_FINGERS': [input_data[3]],
        'ANXIETY': [input_data[4]],
        'PEER_PRESSURE': [input_data[5]],
        'CHRONIC DISEASE': [input_data[6]],
        'FATIGUE': [input_data[7]],
        'ALLERGY': [input_data[8]],
        'WHEEZING': [input_data[9]],
        'ALCOHOL CONSUMING': [input_data[10]],
        'COUGHING': [input_data[11]],
        'SHORTNESS OF BREATH': [input_data[12]],
        'SWALLOWING DIFFICULTY': [input_data[13]],
        'CHEST PAIN': [input_data[14]],
        'GENDER_M': [1 if input_data[0] == 'M' else 0]
    }

    # Create a DataFrame with the same columns as during training
    input_data_df = pd.DataFrame(input_data_dict, columns=columns)

    # Make a prediction using the trained model
    prediction = model.predict(input_data_df)
    return 1 if prediction[0] == "YES" else 0
