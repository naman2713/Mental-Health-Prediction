import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pickle

def train_model():
    df = pd.read_csv("dataset/abcd.csv")

    # Drop any unnamed columns that might be present
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Handle NaN values
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].fillna(df[column].mode()[0])
        elif df[column].dtype.name == 'category':
            df[column] = df[column].fillna(df[column].mode()[0])
        else:
            df[column] = df[column].fillna(df[column].median())

    # Separate the target variable
    target = 'MentalStressLevel'
    y = df[target]
    X = df.drop(columns=[target])

    # Encode the target variable
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Save the label encoder for the target variable
    joblib.dump(label_encoder, 'label_encoder.pkl')

    # One-hot encode categorical features
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))

    # Combine the encoded categorical columns with the rest of the data
    X = X.drop(columns=categorical_cols)
    X = pd.concat([X, X_encoded], axis=1)

    # Save the one-hot encoder and feature names for later use
    joblib.dump(encoder, 'onehot_encoder.pkl')
    joblib.dump(X.columns, 'feature_names.joblib')

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the best parameters for RandomForestClassifier
    best_params_rf = {
        'max_depth': 10,
        'max_features': 'sqrt',
        'min_samples_leaf': 1,
        'min_samples_split': 10,
        'n_estimators': 300
    }

    # Initialize the RandomForestClassifier with the best parameters
    rf_classifier = RandomForestClassifier(random_state=42, **best_params_rf)

    # Fit the model on the training data
    rf_classifier.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred_rf = rf_classifier.predict(X_test)

    # Evaluate the model on the test data
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print(f"RandomForest Accuracy on Test Data: {accuracy_rf:.2f}")

    print("\nRandomForest Classification Report on Test Data:")
    print(classification_report(y_test, y_pred_rf))

    feature_importances = rf_classifier.feature_importances_
    feature_names = X.columns
    importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importances_df = importances_df.sort_values(by='Importance', ascending=False)
    print("\nFeature Importances:")
    print(importances_df)

    # Save the model using joblib
    pickle.dump(rf_classifier, open('model.pkl', 'wb'))

def predict(data):
    # Load the model and encoders
    model = pickle.load(open('model.pkl', 'rb'))
    feature_names = joblib.load('feature_names.joblib')
    encoder = joblib.load('onehot_encoder.pkl')

    # Create a DataFrame from the input data
    df = pd.DataFrame([data])

    # One-hot encode the categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    encoded_df = pd.DataFrame(encoder.transform(df[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))

    # Combine the encoded columns with the original DataFrame
    df = df.drop(columns=categorical_cols)
    df = pd.concat([df, encoded_df], axis=1)

    # Ensure the input data has the same columns as the training data
    df = df.reindex(columns=feature_names, fill_value=0)

    # Make a prediction
    prediction = model.predict(df)
    return prediction[0]

# Train the model when this script is run
if __name__ == '__main__':
    train_model()
