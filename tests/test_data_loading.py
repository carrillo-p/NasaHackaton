import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.models.train_xgboost_model import train_xgboost_model
from src.models.predict_climate_conditions import predict_climate_condition, prepare_input_data

class TestXGBoostModel(unittest.TestCase):
    
    def setUp(self):
        # Create test data
        np.random.seed(42)
        self.X = pd.DataFrame({
            'Temperature': np.random.uniform(15, 35, 100),
            'Relative Humidity': np.random.uniform(30, 80, 100),
            'UV Index': np.random.uniform(0, 11, 100),
            'Precipitation Probability': np.random.uniform(0, 100, 100)
        })
        self.y = (
            (self.X['Temperature'] > 20) & 
            (self.X['Temperature'] < 30) & 
            (self.X['Relative Humidity'] > 40) & 
            (self.X['Relative Humidity'] < 70) & 
            (self.X['UV Index'] < 6) & 
            (self.X['Precipitation Probability'] < 30)
        ).astype(int)
        
        # Save test data to a CSV file
        self.test_file_path = os.path.join(project_root, 'data', 'processed', 'test_climate_data.csv')
        pd.concat([self.X, self.y.rename('Favorable_Condition')], axis=1).to_csv(self.test_file_path, index=False)
    
    def tearDown(self):
        # Remove the temporary CSV file after tests
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)
    
    def test_train_xgboost_model(self):
        model = train_xgboost_model(self.X, self.y)
        self.assertIsNotNone(model)
    
    def test_predict_climate_condition(self):
        model = train_xgboost_model(self.X, self.y)
        
        input_data = {
            'Temperature': 25,
            'Relative Humidity': 50,
            'UV Index': 5,
            'Precipitation Probability': 20
        }
        
        prediction = predict_climate_condition(model, input_data, self.test_file_path)
        self.assertIn(prediction, [0, 1])
    
    def test_prepare_input_data(self):
        input_data = {
            'Temperature': 25,
            'Relative Humidity': 50,
            'UV Index': 5,
            'Precipitation Probability': 20
        }
        
        prepared_data = prepare_input_data(input_data, self.test_file_path)
        self.assertIsInstance(prepared_data, pd.DataFrame)
        self.assertEqual(prepared_data.shape, (1, 4))

if __name__ == '__main__':
    unittest.main()