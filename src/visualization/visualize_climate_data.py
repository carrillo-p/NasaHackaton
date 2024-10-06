import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import numpy as np

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

def load_data(file_path):
    """
    Load processed climate data.
    
    Args:
    file_path (str): Path to the CSV file with processed climate data.
    
    Returns:
    pd.DataFrame: DataFrame with climate data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def plot_correlation_heatmap(df, output_path):
    """
    Create a correlation heatmap for climate variables.
    
    Args:
    df (pd.DataFrame): DataFrame with climate data.
    output_path (str): Path to save the heatmap.
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    plt.figure(figsize=(20, 16))
    sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap of Climate Variables')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Correlation heatmap saved to {output_path}")

def plot_feature_importance(df, output_path):
    """
    Create a bar plot of feature importance.
    
    Args:
    df (pd.DataFrame): DataFrame with climate data.
    output_path (str): Path to save the plot.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    feature_importance = numeric_df.drop('Favorable_Condition', axis=1).corrwith(numeric_df['Favorable_Condition']).abs().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    feature_importance[:20].plot(kind='bar')  # Plot top 20 features
    plt.title('Top 20 Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Absolute Correlation with Favorable_Condition')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Feature importance plot saved to {output_path}")

def plot_favorable_conditions_distribution(df, output_path):
    """
    Create a pie chart of the distribution of favorable/unfavorable conditions.
    
    Args:
    df (pd.DataFrame): DataFrame with climate data.
    output_path (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 10))
    df['Favorable_Condition'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
    plt.title('Distribution of Favorable/Unfavorable Climate Conditions')
    plt.ylabel('')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Favorable conditions distribution plot saved to {output_path}")

def plot_time_series(df, output_path):
    """
    Create a time series plot of key climate variables.
    
    Args:
    df (pd.DataFrame): DataFrame with climate data.
    output_path (str): Path to save the plot.
    """
    plt.figure(figsize=(16, 12))
    plt.plot(df['Date'], df['Temperature'], label='Temperature')
    plt.plot(df['Date'], df['Absolute_Humidity'], label='Absolute Humidity')
    plt.plot(df['Date'], df['UV_Index'], label='UV Index')
    plt.title('Time Series of Key Climate Variables')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Time series plot saved to {output_path}")

def main():
    data_path = os.path.join(project_root, "data", "processed", "climate_data_with_features.csv")
    output_dir = os.path.join(project_root, "visualizations")
    os.makedirs(output_dir, exist_ok=True)

    try:
        df = load_data(data_path)
        
        plot_correlation_heatmap(df, os.path.join(output_dir, "correlation_heatmap.png"))
        plot_feature_importance(df, os.path.join(output_dir, "feature_importance.png"))
        plot_favorable_conditions_distribution(df, os.path.join(output_dir, "favorable_conditions_distribution.png"))
        plot_time_series(df, os.path.join(output_dir, "time_series_plot.png"))
        
        print("All visualizations have been generated successfully.")
    except Exception as e:
        print(f"An error occurred during visualization: {e}")

if __name__ == "__main__":
    main()