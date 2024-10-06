import unittest
import pandas as pd
import numpy as np
import os
import sys

# Añadir el directorio src al path para poder importar los módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models.train_xgboost_model import train_xgboost_model
from models.predict_climate_conditions import predict_climate_condition, prepare_input_data

class TestXGBoostModel(unittest.TestCase):
    
    def setUp(self):
        # Crear datos de prueba
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
    
    def test_train_xgboost_model(self):
        model = train_xgboost_model(self.X, self.y)
        self.assertIsNotNone(model)
    
    def test_predict_climate_condition(self):
        model = train_xgboost_model(self.X, self.y)
        
        input_data = {
            'Temperature': 25,
            'Heat Index': 27,
            'Relative Humidity': 50,
            'Absolute Humidity': 12,
            'CO2': 400,
            'Precipitation Probability': 20,
            'UV Index': 5,
            'Evapotranspiration': 3,
            'Drought Index': 1
        }
        
        prediction = predict_climate_condition(model, input_data)
        self.assertIn(prediction, [0, 1])
    
    def test_prepare_input_data(self):
        input_data = {
            'Temperature': 25,
            'Heat Index': 27,
            'Relative Humidity': 50,
            'Absolute Humidity': 12,
            'CO2': 400,
            'Precipitation Probability': 20,
            'UV Index': 5,
            'Evapotranspiration': 3,
            'Drought Index': 1
        }
        
        prepared_data = prepare_input_data(input_data)
        self.assertIsInstance(prepared_data, pd.DataFrame)
        self.assertEqual(prepared_data.shape, (1, 9))

if __name__ == '__main__':
    unittest.main()