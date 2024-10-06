# ONYVA: AI-Powered Climate Prediction for Sustainable Agriculture

## Project Overview

ONYVA is an innovative AI-powered platform designed to help farmers make informed decisions about their agricultural practices by leveraging NASA's Earth science data, Meteomatics API (as part of the NASA Space Apps Challenge), and local IoT sensor information. This project aims to enhance crop yield, optimize resource usage, and promote sustainable farming practices through advanced climate prediction and analysis.

## Features

- Integration of NASA Earth science data and Meteomatics API data with local IoT sensor data
- Advanced climate condition prediction using machine learning
- Real-time monitoring of soil moisture, temperature, and other critical parameters
- User-friendly mobile application for farmers to access insights and recommendations

## Technical Architecture

### Data Sources

1. NASA Earth Science Data:
   - Agriculture and Water Management Pathfinder
   - Floods Data Pathfinder
   - Extreme Heat Data Pathfinder

2. Meteomatics API (NASA Space Apps Challenge resource):
   - Provides high-resolution weather and climate data

3. Local IoT Sensor Data:
   - Soil moisture
   - Temperature
   - Salinity

### Data Processing

- AppEEARS (Application for Extracting and Exploring Analysis Ready Samples)
- NASA Earthdata Search
- Giovanni for data visualization and analysis
- Meteomatics API for retrieving weather and climate data

### AI Model

Our AI model is built using Python and leverages the following libraries:

- pandas for data manipulation
- scikit-learn for machine learning algorithms
- XGBoost for gradient boosting
- matplotlib and seaborn for data visualization

The core components of our AI model include:

1. Data Preprocessing (`preprocess_climate_data.py`):
   - Handles missing values
   - Normalizes features
   - Creates a 'Favorable_Condition' label based on climate parameters

2. Feature Engineering (`build_climate_features.py`):
   - Creates rolling averages for key climate variables
   - Generates lag features to capture temporal patterns
   - Constructs interaction features to model complex relationships

3. Model Training (`train_xgboost_model.py`):
   - Utilizes XGBoost for its high performance and ability to handle complex datasets
   - Performs feature importance analysis to understand key predictors

4. Prediction (`predict_climate_conditions.py`):
   - Makes predictions on new data using the trained model
   - Provides interpretable results for farmers

5. Visualization (`visualize_climate_data.py`):
   - Generates correlation heatmaps to understand relationships between variables
   - Creates feature importance plots to identify key predictors
   - Produces time series plots to visualize trends in climate data

## Data Parameters

The model is trained on the following parameters obtained from the Meteomatics API:

- lat: Latitude
- lon: Longitude
- validdate: Date and time of the data point
- t_2m:C: Temperature at 2 meters above ground in Celsius
- absolute_humidity_2m:gm3: Absolute humidity at 2 meters in g/m³
- heat_index:C: Heat index in Celsius
- prob_precip_1h:p: Probability of precipitation in the next hour (percentage)
- uv:idx: UV index
- evapotranspiration_1h:mm: Evapotranspiration in the last hour in millimeters
- drought_index:idx: Drought index
- frost_days:d: Number of frost days

These parameters provide a comprehensive view of the climate conditions relevant to agricultural decision-making.

## How It Works

1. Data Collection: The system continuously collects data from NASA's Earth science datasets, Meteomatics API, and local IoT sensors.

2. Data Preprocessing: Raw data is cleaned, normalized, and prepared for analysis.

3. Feature Engineering: The system creates advanced features to capture complex patterns in the data.

4. Model Training: An XGBoost model is trained on historical data to predict favorable farming conditions.

5. Prediction: The trained model makes predictions on current and forecasted climate conditions.

6. Visualization: Results are visualized through various plots and charts for easy interpretation.

7. User Interface: Farmers access insights and recommendations through a user-friendly mobile application.

## Future Developments

- Integration with more NASA datasets for enhanced prediction accuracy
- Implementation of deep learning models for complex pattern recognition
- Development of crop-specific prediction models
- Integration with automated farming systems for direct action based on predictions

## Contribution to Sustainable Agriculture

ONYVA contributes to sustainable agriculture by:
- Optimizing water usage through precise irrigation recommendations
- Reducing the use of fertilizers and pesticides by predicting optimal application times
- Improving crop yields through data-driven decision making
- Enhancing farmers' resilience to extreme weather events through early warnings

By combining NASA's global Earth science data, Meteomatics API data, and local sensor information, ONYVA provides a powerful tool for farmers to adapt to changing climate conditions and promote sustainable agricultural practices.

## Acknowledgments

This project is part of the NASA Space Apps Challenge 2024. We gratefully acknowledge NASA and Meteomatics for providing access to their data and APIs, which have been instrumental in developing this solution.
