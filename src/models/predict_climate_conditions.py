import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.features.build_climate_features import build_features

def load_model(model_path):
    """
    Load the trained XGBoost model.
    
    Args:
    model_path (str): Path to the saved model file.
    
    Returns:
    xgb.XGBClassifier: Loaded XGBoost model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)

def prepare_input_data(input_data, feature_file_path):
    """
    Prepare input data for prediction.
    
    Args:
    input_data (dict): Dictionary with input data.
    feature_file_path (str): Path to the feature file to get the correct columns.
    
    Returns:
    pd.DataFrame: DataFrame ready for prediction.
    """
    if not os.path.exists(feature_file_path):
        print(f"Feature file not found: {feature_file_path}")
        print("Attempting to build features...")
        input_path = os.path.join(project_root, "data", "processed", "preprocessed_climate_data.csv")
        build_features(input_path, feature_file_path)
        
        if not os.path.exists(feature_file_path):
            raise FileNotFoundError(f"Failed to create feature file: {feature_file_path}")

    feature_df = pd.read_csv(feature_file_path)
    required_features = feature_df.select_dtypes(include=[np.number]).columns.drop('Favorable_Condition')
    
    df = pd.DataFrame([input_data])
    
    # Create dummy values for rolling averages, lag features, and interaction features
    for col in ['Temperature', 'Absolute_Humidity', 'UV_Index', 'Precipitation_Probability']:
        for window in [3, 7, 14]:
            df[f'{col}_rolling_{window}h'] = df[col]
        for lag in [1, 3, 7]:
            df[f'{col}_lag_{lag}h'] = df[col]
    
    df['temp_humidity_interaction'] = df['Temperature'] * df['Absolute_Humidity']
    df['uv_precip_interaction'] = df['UV_Index'] * df['Precipitation_Probability']
    
    # Ensure all required features are present
    for feature in required_features:
        if feature not in df.columns:
            df[feature] = 0  # Use a default value of 0 for missing features
    
    return df[required_features]

def predict_climate_condition(model, input_data, feature_file_path):
    """
    Make a prediction of favorable climate conditions.
    
    Args:
    model (xgb.XGBClassifier): Loaded XGBoost model.
    input_data (dict): Dictionary with input data.
    feature_file_path (str): Path to the feature file.
    
    Returns:
    int: 1 if conditions are favorable, 0 otherwise.
    """
    df = prepare_input_data(input_data, feature_file_path)
    prediction = model.predict(df)
    return prediction[0]

def main():
    model_path = os.path.join(project_root, "models", "xgboost", "xgboost_climate_model.joblib")
    feature_file_path = os.path.join(project_root, "data", "processed", "climate_data_with_features.csv")
    
    try:
        model = load_model(model_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure that you have trained the model before running predictions.")
        return

    # Example input data
    input_data = {
        'Temperature': 25,
        'Absolute_Humidity': 12,
        'Heat_Index': 27,
        'Precipitation_Probability': 20,
        'UV_Index': 5,
        'Evapotranspiration': 3,
        'Drought_Index': 1
    }
    
    try:
        prediction = predict_climate_condition(model, input_data, feature_file_path)
        print(f"Prediction: {'Favorable conditions' if prediction == 1 else 'Unfavorable conditions'}")
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()