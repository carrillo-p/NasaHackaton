import pandas as pd
import numpy as np
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
input_path = os.path.join(project_root, "data", "processed", "preprocessed_climate_data.csv")
output_path = os.path.join(project_root, "data", "processed", "climate_data_with_features.csv")

def create_rolling_averages(df, columns, window_sizes=[3, 7, 14]):
    """
    Create rolling averages for specified columns.
    
    Args:
    df (pd.DataFrame): DataFrame with climate data.
    columns (list): List of columns to calculate rolling averages.
    window_sizes (list): List of window sizes for rolling averages.
    
    Returns:
    pd.DataFrame: DataFrame with new rolling average features.
    """
    df = df.sort_values('Date')
    for col in columns:
        for window in window_sizes:
            df[f'{col}_rolling_{window}h'] = df[col].rolling(window=window).mean()
    return df

def create_lag_features(df, columns, lag_periods=[1, 3, 7]):
    """
    Create lag features for specified columns.
    
    Args:
    df (pd.DataFrame): DataFrame with climate data.
    columns (list): List of columns to create lag features.
    lag_periods (list): List of lag periods in hours.
    
    Returns:
    pd.DataFrame: DataFrame with new lag features.
    """
    df = df.sort_values('Date')
    for col in columns:
        for lag in lag_periods:
            df[f'{col}_lag_{lag}h'] = df[col].shift(lag)
    return df

def create_interaction_features(df):
    """
    Create interaction features between relevant variables.
    
    Args:
    df (pd.DataFrame): DataFrame with climate data.
    
    Returns:
    pd.DataFrame: DataFrame with new interaction features.
    """
    df['temp_humidity_interaction'] = df['Temperature'] * df['Absolute_Humidity']
    df['uv_precip_interaction'] = df['UV_Index'] * df['Precipitation_Probability']
    return df

def build_features(input_file_path, output_file_path):
    """
    Build additional features from preprocessed data.
    
    Args:
    input_file_path (str): Path to the preprocessed data file.
    output_file_path (str): Path to save the file with new features.
    """
    print(f"Loading data from: {input_file_path}")
    df = pd.read_csv(input_file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    columns_for_rolling = ['Temperature', 'Absolute_Humidity', 'UV_Index', 'Precipitation_Probability']
    df = create_rolling_averages(df, columns_for_rolling)
    
    columns_for_lag = ['Temperature', 'Absolute_Humidity', 'UV_Index', 'Precipitation_Probability']
    df = create_lag_features(df, columns_for_lag)
    
    df = create_interaction_features(df)
    
    # Handle NaN values created by rolling averages and lag features
    df = df.fillna(df.mean())
    
    df.to_csv(output_file_path, index=False)
    print(f"Data with new features saved to {output_file_path}")

if __name__ == "__main__":
    build_features(input_path, output_path)