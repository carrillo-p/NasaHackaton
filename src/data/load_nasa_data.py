import pandas as pd
import os

def load_nasa_data(file_path):
    """
    Load NASA climate data from a CSV file.
    
    Args:
    file_path (str): Path to the NASA data CSV file.
    
    Returns:
    pd.DataFrame: DataFrame with loaded climate data.
    """
    print(f"Attempting to load file from: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return None
    
    try:
        # Use sep=';' to specify semicolon as the separator
        df = pd.read_csv(file_path, sep=';')
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print("First few rows of the data:")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def validate_data(df):
    """
    Validate that the DataFrame contains the expected columns.
    
    Args:
    df (pd.DataFrame): DataFrame to validate.
    
    Returns:
    bool: True if the DataFrame is valid, False otherwise.
    """
    expected_columns = [
        "lat", "lon", "validdate", "t_2m:C", "absolute_humidity_2m:gm3",
        "heat_index:C", "prob_precip_1h:p", "uv:idx", "evapotranspiration_1h:mm",
        "drought_index:idx", "frost_days:d"
    ]
    
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return False
    return True

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "..", "..", "data", "raw", "nasa_climate_data.csv")
    print(f"Attempting to load file from: {data_path}")
    df = load_nasa_data(data_path)
    if df is not None and validate_data(df):
        print("Data loaded and validated successfully.")
    else:
        print("Error in loading or validating the data.")