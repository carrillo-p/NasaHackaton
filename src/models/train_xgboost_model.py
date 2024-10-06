import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.features.build_climate_features import build_features

def load_data(file_path):
    """
    Load data with features for training.
    
    Args:
    file_path (str): Path to the CSV file with processed data.
    
    Returns:
    tuple: X (features) and y (target variable)
    """
    print(f"Attempting to load data from: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: Input file does not exist: {file_path}")
        print("Attempting to build features...")
        
        input_path = os.path.join(project_root, "data", "processed", "preprocessed_climate_data.csv")
        build_features(input_path, file_path)
        
        if not os.path.exists(file_path):
            print("Error: Failed to create the feature file.")
            return None, None
    
    df = pd.read_csv(file_path)
    print("Data loaded successfully.")
    print(f"Columns in the dataset: {df.columns.tolist()}")
    
    if 'Favorable_Condition' not in df.columns:
        print("Error: 'Favorable_Condition' column not found in the data.")
        return None, None
    
    # Drop non-numeric columns
    df = df.select_dtypes(include=[np.number])
    
    y = df['Favorable_Condition']
    X = df.drop(['Favorable_Condition'], axis=1)
    
    print(f"Features used for training: {X.columns.tolist()}")
    return X, y

def train_xgboost_model(X, y):
    """
    Train an XGBoost model for classification.
    
    Args:
    X (pd.DataFrame): Features dataset.
    y (pd.Series): Target variable.
    
    Returns:
    xgb.XGBClassifier: Trained XGBoost model.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
    print("\nTop 10 Most Important Features:")
    print(feature_importance_df.head(10))
    
    return model

def save_model(model, file_path):
    """
    Save the trained model to a file.
    
    Args:
    model (xgb.XGBClassifier): Trained XGBoost model.
    file_path (str): Path to save the model.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")

def main():
    data_path = os.path.join(project_root, "data", "processed", "climate_data_with_features.csv")
    model_path = os.path.join(project_root, "models", "xgboost", "xgboost_climate_model.joblib")
    
    X, y = load_data(data_path)
    if X is None or y is None:
        print("Error: Failed to load data. Exiting.")
        return
    
    model = train_xgboost_model(X, y)
    save_model(model, model_path)

if __name__ == "__main__":
    main()