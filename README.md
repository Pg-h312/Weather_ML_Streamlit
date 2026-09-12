Weather Prediction & Monitoring Dashboard

A machine learning-powered Weather Prediction & Monitoring Dashboard built with Python, Pandas, NumPy, Scikit-learn, Plotly, and Streamlit. The application provides interactive weather monitoring, analytics, forecasting, air-quality monitoring, alerts, and ML-based temperature prediction.

Note: The included dataset is generated using Python for project/demo purposes. It is not official observed weather data.

🚀 Features
🌤️ Current Weather Monitoring
🕐 Hourly Weather Forecast
📅 7-Day Weather Analysis
🤖 Machine Learning Temperature Prediction
📊 Weather Analytics
🌫️ Air Quality / AQI Monitoring
🗺️ Interactive Weather Map
🚨 Weather Alerts
📜 Historical Weather Data
📥 Download Weather Data as CSV
⚙️ Dashboard Settings
📍 Multiple Indian City Selection
🔄 Refresh functionality
📈 Interactive Plotly charts
📊 KPI cards for important weather metrics
🏙️ Supported Cities

The generated dataset contains weather information for:

Delhi
Mumbai
Bengaluru
Chennai
Kolkata
Hyderabad
Lucknow
Pune
Jaipur
Ahmedabad
🤖 Machine Learning

The project uses a Random Forest Regressor to predict the next-hour temperature.

ML Features

The model uses:

Temperature
Humidity
Pressure
Wind Speed
Cloud Cover
Precipitation
Hour
Day of Year
Temperature Lag 1
Temperature Lag 3
Temperature Lag 6
Model Evaluation

The application displays:

MAE — Mean Absolute Error
RMSE — Root Mean Squared Error
R² Score — Model performance
Actual vs Predicted Temperature
Feature Importance
🛠️ Technology Stack
Technology	Purpose
Python	Application & data generation
Pandas	Data processing
NumPy	Numerical calculations
Scikit-learn	Machine Learning
Random Forest	Temperature prediction
Plotly	Interactive visualizations
Streamlit	Dashboard UI
Databricks Apps	Cloud deployment

Spark is not required for this application.

📁 Project Structure
Weather_ML_Streamlit/
│
├── data/
│   └── weather_prediction.csv
│
├── app.py
├── generate_data.py
├── requirements.txt
├── app.yaml
├── .gitignore
└── README.md
⚙️ Installation
1. Clone the repository
git clone https://github.com/Pg-h312/Weather_ML_Streamlit.git
2. Open the project
cd Weather_ML_Streamlit
3. Create virtual environment
python -m venv venv
4. Activate virtual environment — Windows PowerShell
venv\Scripts\activate

If PowerShell blocks activation:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
venv\Scripts\activate
5. Install all libraries
pip install streamlit pandas numpy plotly scikit-learn requests

Or use:

pip install -r requirements.txt
📊 Generate Dataset

Run:

python generate_data.py

This creates:

data/weather_prediction.csv
▶️ Run the Application

Run:

python -m streamlit run app.py

The dashboard will open in your browser at:

http://localhost:8501
🖥️ Dashboard Modules
Dashboard

Provides an overview of:

Current temperature
Humidity
Wind speed
Cloud cover
Precipitation
Visibility
Pressure
UV index
Temperature trend
ML prediction trend
Current Weather

Displays detailed weather information for the selected city.

Hourly Forecast

Provides hourly:

Temperature
Humidity
Wind
Pressure
Rain probability
Cloud cover
7-Day Forecast

Provides daily:

Average temperature
Minimum temperature
Maximum temperature
Humidity
Precipitation
Wind speed
ML Prediction

Shows:

Random Forest
      ↓
Weather Features
      ↓
Temperature Prediction
      ↓
MAE / RMSE / R²
      ↓
Actual vs Predicted
Weather Analytics

Includes:

Temperature vs Humidity
Temperature vs Wind
Correlation Matrix
Interactive Plotly visualizations
Air Quality

Displays:

AQI
AQI status
Humidity
AQI trend
Weather Map

Shows weather conditions across the supported Indian cities using an interactive map.

Alerts

Generates alerts for:

High temperature
High humidity
Strong wind
High rain probability
Poor AQI
High UV index
☁️ Databricks Deployment

The project can also be deployed as a Databricks App using Streamlit.

Required files:

app.py
app.yaml
requirements.txt
data/
app.yaml
command:
  - streamlit
  - run
  - app.py
requirements.txt
streamlit>=1.45.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
scikit-learn>=1.3.0
requests>=2.31.0

The application reads the CSV directly:

data/weather_prediction.csv

No Spark cluster is required.

🔄 Data Flow
Python
  ↓
generate_data.py
  ↓
weather_prediction.csv
  ↓
Pandas
  ↓
Data Processing
  ↓
Random Forest ML Model
  ↓
Temperature Prediction
  ↓
Streamlit Dashboard
  ↓
Databricks Apps
🎯 Project Objective

The main objective of this project is to build an interactive weather intelligence platform that combines:

Data Generation + Data Processing + Machine Learning + Visualization + Web Dashboard + Cloud Deployment

It can be used as a portfolio project to demonstrate practical knowledge of Python, Machine Learning, Streamlit, data analytics, and Databricks Apps.

👩‍💻 Author

Priya Gupta

📄 License

This project is intended for educational, portfolio, and demonstration purposes.
