import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from load_nasa_data import load_nasa_data, validate_data

current_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(current_dir, "..", "..", "data", "raw", "nasa_climate_data.csv")
output_path = os.path.join(current_dir, "..", "..", "data", "processed", "preprocessed_climate_data.csv")

def handle_missing_values(df):
    """
    Handle missing values in the DataFrame.
    
    Args:
    df (pd.DataFrame): DataFrame with possible missing values.
    
    Returns:
    pd.DataFrame: DataFrame with missing values handled.
    """
    # Separate numeric and non-numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
    
    # Fill numeric columns with their median
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    
    # Fill non-numeric columns with their mode
    for col in non_numeric_columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

def normalize_features(df):
    """
    Normalize numeric features of the DataFrame.
    
    Args:
    df (pd.DataFrame): DataFrame with features to normalize.
    
    Returns:
    pd.DataFrame: DataFrame with normalized features.
    """
    scaler = StandardScaler()
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df

def create_favorable_condition_label(df):
    """
    Create a label for favorable climate conditions.
    
    Args:
    df (pd.DataFrame): DataFrame with climate data.
    
    Returns:
    pd.DataFrame: DataFrame with new label column.
    """
    df['Favorable_Condition'] = (
        (df['t_2m:C'] > 15) & (df['t_2m:C'] < 30) &
        (df['absolute_humidity_2m:gm3'] > 5) & (df['absolute_humidity_2m:gm3'] < 15) &
        (df['uv:idx'] < 6) &
        (df['prob_precip_1h:p'] < 30)
    ).astype(int)
    return df

def preprocess_data(input_file_path, output_file_path):
    """
    Preprocess climate data and save the result.
    
    Args:
    input_file_path (str): Path to the input file.
    output_file_path (str): Path to save the preprocessed file.
    """
    df = load_nasa_data(input_file_path)
    if df is None or not validate_data(df):
        print("Error in loading or validating the data.")
        return

    # Convert date to datetime before handling missing values
    df['validdate'] = pd.to_datetime(df['validdate'])

    df = handle_missing_values(df)
    df = normalize_features(df)
    df = create_favorable_condition_label(df)

    # Select relevant columns and rename for clarity
    df = df[['validdate', 't_2m:C', 'absolute_humidity_2m:gm3', 'heat_index:C', 
             'prob_precip_1h:p', 'uv:idx', 'evapotranspiration_1h:mm', 
             'drought_index:idx', 'Favorable_Condition']]
    
    df.columns = ['Date', 'Temperature', 'Absolute_Humidity', 'Heat_Index',
                  'Precipitation_Probability', 'UV_Index', 'Evapotranspiration',
                  'Drought_Index', 'Favorable_Condition']

    df.to_csv(output_file_path, index=False)
    print(f"Preprocessed data saved to {output_file_path}")

if __name__ == "__main__":
    preprocess_data(input_path, output_path)