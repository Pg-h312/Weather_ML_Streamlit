import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Weather AI | Prediction & Monitoring",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROFESSIONAL CSS
# ============================================================

st.markdown(
    """
    <style>

    /* =========================
       MAIN APPLICATION
       ========================= */

    .stApp {
        background-color: #f4f7fb;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }


    /* =========================
       SIDEBAR
       ========================= */

    [data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid #1e293b;
    }

    [data-testid="stSidebar"] * {
        color: #f8fafc;
    }

    [data-testid="stSidebar"] .stRadio label {
        color: #e2e8f0 !important;
        font-size: 15px;
        font-weight: 500;
    }

    [data-testid="stSidebar"] .stSelectbox label {
        color: #cbd5e1 !important;
        font-weight: 600;
    }

    [data-testid="stSidebar"] hr {
        border-color: #334155;
    }


    /* =========================
       HEADINGS
       ========================= */

    h1 {
        color: #0f172a !important;
        font-weight: 800 !important;
    }

    h2 {
        color: #0f172a !important;
        font-weight: 750 !important;
    }

    h3 {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    p {
        color: #475569;
    }


    /* =========================
       METRIC CARDS
       ========================= */

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dbe4ef;
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
        min-height: 115px;
    }

    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-size: 14px !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-size: 27px !important;
        font-weight: 800 !important;
    }


    /* =========================
       BUTTONS
       ========================= */

    .stButton > button {
        background: #0f766e;
        color: white;
        border: none;
        border-radius: 10px;
        min-height: 42px;
        font-weight: 700;
    }

    .stButton > button:hover {
        background: #115e59;
        color: white;
        border: none;
    }


    /* =========================
       DOWNLOAD BUTTON
       ========================= */

    .stDownloadButton > button {
        background: #2563eb;
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 700;
    }

    .stDownloadButton > button:hover {
        background: #1d4ed8;
        color: white;
    }


    /* =========================
       SELECTBOX
       ========================= */

    [data-baseweb="select"] > div {
        border-radius: 10px;
    }


    /* =========================
       DATAFRAME
       ========================= */

    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }


    /* =========================
       ALERTS
       ========================= */

    [data-testid="stAlert"] {
        border-radius: 12px;
    }


    /* =========================
       FOOTER
       ========================= */

    .footer-text {
        text-align: center;
        color: #64748b;
        font-size: 13px;
        padding-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATA PATH
# ============================================================

DATA_PATH = os.path.join(
    "data",
    "weather_prediction.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()

    try:

        data = pd.read_csv(DATA_PATH)

        if "timestamp" not in data.columns:
            return pd.DataFrame()

        data["timestamp"] = pd.to_datetime(
            data["timestamp"],
            errors="coerce"
        )

        data = data.dropna(
            subset=["timestamp"]
        )

        if "city" not in data.columns:
            return pd.DataFrame()

        data = data.sort_values(
            ["city", "timestamp"]
        ).reset_index(drop=True)

        return data

    except Exception:
        return pd.DataFrame()


df = load_data()


# ============================================================
# DATA ERROR HANDLING
# ============================================================

if df.empty:

    st.error(
        "❌ Weather dataset could not be loaded."
    )

    st.info(
        "Please make sure this file exists:"
    )

    st.code(
        "data/weather_prediction.csv"
    )

    st.info(
        "Then run: python generate_data.py"
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "## 🌤️ Weather AI"
)

st.sidebar.caption(
    "Prediction & Monitoring"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Current Weather",
        "Hourly Forecast",
        "7-Day Forecast",
        "ML Prediction",
        "Weather Analytics",
        "Air Quality",
        "Weather Map",
        "Alerts",
        "Historical Data",
        "Download Data",
        "Settings"
    ]
)

st.sidebar.markdown("---")


# ============================================================
# CITY SELECTOR
# ============================================================

cities = sorted(
    df["city"].dropna().unique().tolist()
)

if not cities:
    st.error("No cities found in the dataset.")
    st.stop()

selected_city = st.sidebar.selectbox(
    "📍 Selected Location",
    cities
)


city_df = df[
    df["city"] == selected_city
].copy()


if city_df.empty:
    st.error("No weather data available for selected city.")
    st.stop()


latest = city_df.iloc[-1]


# ============================================================
# SIDEBAR REFRESH
# ============================================================

auto_refresh = st.sidebar.checkbox(
    "Auto Refresh",
    value=False
)

if auto_refresh:
    st.sidebar.success(
        "Auto Refresh Enabled"
    )

st.sidebar.markdown("---")

st.sidebar.caption(
    "Last Updated"
)

st.sidebar.caption(
    latest["timestamp"].strftime(
        "%d %b %Y • %H:%M"
    )
)


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns(
    [6, 1]
)

with header_left:

    st.title(
        "Weather Prediction & Monitoring Dashboard"
    )

    st.caption(
        "Weather Data  •  Machine Learning  •  Forecast  •  Analytics"
    )


with header_right:

    if st.button(
        "🔄 Refresh",
        use_container_width=True
    ):

        st.cache_data.clear()
        st.rerun()


# ============================================================
# HELPER FUNCTION
# ============================================================

def safe_value(row, column, default=0):

    if column in row.index:

        value = row[column]

        if pd.isna(value):
            return default

        return value

    return default


# ============================================================
# ML MODEL
# ============================================================

@st.cache_data
def train_model(city_data):

    features = [
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "cloud_cover",
        "precipitation",
        "hour",
        "day_of_year",
        "temperature_lag_1",
        "temperature_lag_3",
        "temperature_lag_6"
    ]

    target = "target_temperature"

    required = features + [target]

    missing = [
        col
        for col in required
        if col not in city_data.columns
    ]

    if missing:
        return None

    model_data = city_data.dropna(
        subset=required
    ).copy()

    if len(model_data) < 100:
        return None

    model_data = model_data.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    X = model_data[features]
    y = model_data[target]

    split_index = int(
        len(model_data) * 0.80
    )

    if split_index <= 0 or split_index >= len(model_data):
        return None

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results = pd.DataFrame(
        {
            "timestamp": model_data.iloc[
                split_index:
            ]["timestamp"].values,

            "actual": y_test.values,

            "predicted": predictions
        }
    )

    importance = pd.DataFrame(
        {
            "Feature": features,
            "Importance": model.feature_importances_
        }
    ).sort_values(
        "Importance",
        ascending=False
    )

    return {
        "model": model,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "results": results,
        "features": features,
        "importance": importance
    }


model_result = train_model(
    city_df
)


# ============================================================
# PAGE 1 - DASHBOARD
# ============================================================

if page == "Dashboard":

    st.subheader(
        f"🌤️ Current Weather — {selected_city}"
    )

    st.caption(
        "Latest monitored weather conditions"
    )

    st.write("")

    # Hero weather section
    hero1, hero2, hero3, hero4 = st.columns(4)

    with hero1:

        st.metric(
            "🌡️ Temperature",
            f"{safe_value(latest, 'temperature'):.1f} °C"
        )

    with hero2:

        st.metric(
            "💧 Humidity",
            f"{safe_value(latest, 'humidity'):.0f} %"
        )

    with hero3:

        st.metric(
            "💨 Wind",
            f"{safe_value(latest, 'wind_speed'):.1f} km/h"
        )

    with hero4:

        st.metric(
            "☁️ Condition",
            str(
                safe_value(
                    latest,
                    "condition",
                    "Unknown"
                )
            )
        )

    st.write("")

    # Second KPI row
    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "🌧️ Rain Probability",
            f"{safe_value(latest, 'precipitation_probability'):.0f} %"
        )

    with c2:

        st.metric(
            "👁️ Visibility",
            f"{safe_value(latest, 'visibility'):.1f} km"
        )

    with c3:

        st.metric(
            "🔵 Pressure",
            f"{safe_value(latest, 'pressure'):.0f} hPa"
        )

    with c4:

        st.metric(
            "☀️ UV Index",
            f"{safe_value(latest, 'uv_index'):.1f}"
        )

    st.divider()

    # Temperature chart
    st.subheader(
        "📈 Temperature Trend"
    )

    chart_df = city_df.tail(48).copy()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=chart_df["timestamp"],
            y=chart_df["temperature"],
            mode="lines+markers",
            name="Actual Temperature",
            line=dict(width=3)
        )
    )

    if (
        model_result is not None
        and not model_result["results"].empty
    ):

        prediction_df = model_result[
            "results"
        ].tail(48)

        fig.add_trace(
            go.Scatter(
                x=prediction_df["timestamp"],
                y=prediction_df["predicted"],
                mode="lines",
                name="ML Prediction",
                line=dict(
                    dash="dash",
                    width=3
                )
            )
        )

    fig.update_layout(
        height=430,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(
            l=20,
            r=20,
            t=30,
            b=20
        ),
        xaxis_title="Time",
        yaxis_title="Temperature °C"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Analytics
    left, right = st.columns(2)

    with left:

        humidity_fig = px.line(
            chart_df,
            x="timestamp",
            y="humidity",
            markers=True,
            title="Humidity Trend"
        )

        humidity_fig.update_layout(
            height=350,
            template="plotly_white"
        )

        st.plotly_chart(
            humidity_fig,
            use_container_width=True
        )

    with right:

        wind_fig = px.line(
            chart_df,
            x="timestamp",
            y="wind_speed",
            markers=True,
            title="Wind Speed Trend"
        )

        wind_fig.update_layout(
            height=350,
            template="plotly_white"
        )

        st.plotly_chart(
            wind_fig,
            use_container_width=True
        )


# ============================================================
# PAGE 2 - CURRENT WEATHER
# ============================================================

elif page == "Current Weather":

    st.subheader(
        f"🌤️ Current Weather — {selected_city}"
    )

    st.caption(
        "Detailed current weather conditions"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Temperature",
            f"{safe_value(latest, 'temperature'):.1f} °C"
        )

    with c2:

        temp = safe_value(
            latest,
            "temperature"
        )

        st.metric(
            "Feels Like",
            f"{temp + 1.5:.1f} °C"
        )

    with c3:

        st.metric(
            "Humidity",
            f"{safe_value(latest, 'humidity'):.0f} %"
        )

    with c4:

        st.metric(
            "Condition",
            str(
                safe_value(
                    latest,
                    "condition",
                    "Unknown"
                )
            )
        )

    st.write("")

    c5, c6, c7, c8 = st.columns(4)

    with c5:

        st.metric(
            "Wind",
            f"{safe_value(latest, 'wind_speed'):.1f} km/h"
        )

    with c6:

        st.metric(
            "Wind Direction",
            f"{int(safe_value(latest, 'wind_direction'))}°"
        )

    with c7:

        st.metric(
            "Pressure",
            f"{safe_value(latest, 'pressure'):.0f} hPa"
        )

    with c8:

        st.metric(
            "Visibility",
            f"{safe_value(latest, 'visibility'):.1f} km"
        )

    st.divider()

    st.subheader(
        "Weather Details"
    )

    details = pd.DataFrame(
        {
            "Parameter": [
                "Temperature",
                "Humidity",
                "Pressure",
                "Wind Speed",
                "Wind Direction",
                "Cloud Cover",
                "Precipitation",
                "Rain Probability",
                "Visibility",
                "UV Index",
                "AQI",
                "Condition"
            ],

            "Value": [
                f"{safe_value(latest, 'temperature'):.2f} °C",
                f"{safe_value(latest, 'humidity'):.1f} %",
                f"{safe_value(latest, 'pressure'):.1f} hPa",
                f"{safe_value(latest, 'wind_speed'):.1f} km/h",
                f"{int(safe_value(latest, 'wind_direction'))}°",
                f"{safe_value(latest, 'cloud_cover'):.1f} %",
                f"{safe_value(latest, 'precipitation'):.2f} mm",
                f"{safe_value(latest, 'precipitation_probability'):.1f} %",
                f"{safe_value(latest, 'visibility'):.1f} km",
                f"{safe_value(latest, 'uv_index'):.1f}",
                str(safe_value(latest, "aqi", 0)),
                str(safe_value(latest, "condition", "Unknown"))
            ]
        }
    )

    st.dataframe(
        details,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PAGE 3 - HOURLY FORECAST
# ============================================================

elif page == "Hourly Forecast":

    st.subheader(
        f"🕐 Hourly Forecast — {selected_city}"
    )

    hourly = city_df.tail(24).copy()

    fig = px.line(
        hourly,
        x="timestamp",
        y="temperature",
        markers=True,
        title="24 Hour Temperature Trend"
    )

    fig.update_layout(
        height=430,
        template="plotly_white",
        xaxis_title="Time",
        yaxis_title="Temperature °C"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    left, right = st.columns(2)

    with left:

        rain_fig = px.bar(
            hourly,
            x="timestamp",
            y="precipitation_probability",
            title="Rain Probability"
        )

        rain_fig.update_layout(
            template="plotly_white",
            height=350
        )

        st.plotly_chart(
            rain_fig,
            use_container_width=True
        )

    with right:

        pressure_fig = px.line(
            hourly,
            x="timestamp",
            y="pressure",
            title="Pressure Trend",
            markers=True
        )

        pressure_fig.update_layout(
            template="plotly_white",
            height=350
        )

        st.plotly_chart(
            pressure_fig,
            use_container_width=True
        )

    st.subheader(
        "Hourly Weather Details"
    )

    columns = [
        "timestamp",
        "temperature",
        "humidity",
        "wind_speed",
        "precipitation_probability",
        "pressure",
        "cloud_cover"
    ]

    available = [
        col
        for col in columns
        if col in hourly.columns
    ]

    st.dataframe(
        hourly[available],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PAGE 4 - 7 DAY FORECAST
# ============================================================

elif page == "7-Day Forecast":

    st.subheader(
        f"📅 7-Day Forecast — {selected_city}"
    )

    daily = (
        city_df
        .set_index("timestamp")
        .resample("D")
        .agg(
            {
                "temperature": [
                    "mean",
                    "min",
                    "max"
                ],
                "humidity": "mean",
                "precipitation": "sum",
                "wind_speed": "mean"
            }
        )
        .tail(7)
    )

    daily.columns = [
        "avg_temperature",
        "min_temperature",
        "max_temperature",
        "humidity",
        "precipitation",
        "wind_speed"
    ]

    daily = daily.reset_index()

    fig = px.bar(
        daily,
        x="timestamp",
        y="avg_temperature",
        title="7-Day Average Temperature"
    )

    fig.update_layout(
        template="plotly_white",
        height=430
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        daily.round(2),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PAGE 5 - ML PREDICTION
# ============================================================

elif page == "ML Prediction":

    st.subheader(
        "🤖 Machine Learning Prediction"
    )

    st.caption(
        f"Random Forest temperature prediction for {selected_city}"
    )

    if model_result is None:

        st.warning(
            "Not enough data or required ML columns are missing."
        )

    else:

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "MAE",
                f"{model_result['mae']:.2f} °C"
            )

        with c2:

            st.metric(
                "RMSE",
                f"{model_result['rmse']:.2f} °C"
            )

        with c3:

            st.metric(
                "R² Score",
                f"{model_result['r2']:.3f}"
            )

        st.success(
            "Random Forest model trained successfully."
        )

        results = model_result[
            "results"
        ].tail(48)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=results["timestamp"],
                y=results["actual"],
                mode="lines+markers",
                name="Actual"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=results["timestamp"],
                y=results["predicted"],
                mode="lines+markers",
                name="Predicted"
            )
        )

        fig.update_layout(
            title="Actual vs ML Predicted Temperature",
            height=450,
            template="plotly_white",
            hovermode="x unified",
            xaxis_title="Time",
            yaxis_title="Temperature °C"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader(
            "ML Feature Importance"
        )

        importance = model_result[
            "importance"
        ]

        importance_fig = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Random Forest Feature Importance"
        )

        importance_fig.update_layout(
            template="plotly_white",
            height=450
        )

        st.plotly_chart(
            importance_fig,
            use_container_width=True
        )

        st.subheader(
            "🔮 Recent ML Predictions"
        )

        prediction_table = results.tail(10).copy()

        prediction_table["Difference"] = (
            prediction_table["predicted"]
            - prediction_table["actual"]
        )

        st.dataframe(
            prediction_table.round(2),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# PAGE 6 - WEATHER ANALYTICS
# ============================================================

elif page == "Weather Analytics":

    st.subheader(
        f"📊 Weather Analytics — {selected_city}"
    )

    left, right = st.columns(2)

    with left:

        scatter1 = px.scatter(
            city_df.tail(1000),
            x="humidity",
            y="temperature",
            size="cloud_cover",
            title="Temperature vs Humidity"
        )

        scatter1.update_layout(
            template="plotly_white",
            height=400
        )

        st.plotly_chart(
            scatter1,
            use_container_width=True
        )

    with right:

        scatter2 = px.scatter(
            city_df.tail(1000),
            x="wind_speed",
            y="temperature",
            color="condition",
            title="Temperature vs Wind"
        )

        scatter2.update_layout(
            template="plotly_white",
            height=400
        )

        st.plotly_chart(
            scatter2,
            use_container_width=True
        )

    st.subheader(
        "Correlation Matrix"
    )

    numeric_columns = [
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "cloud_cover",
        "precipitation",
        "visibility",
        "uv_index",
        "aqi"
    ]

    available_numeric = [
        col
        for col in numeric_columns
        if col in city_df.columns
    ]

    if len(available_numeric) >= 2:

        corr = city_df[
            available_numeric
        ].corr()

        corr_fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            title="Weather Feature Correlation"
        )

        corr_fig.update_layout(
            template="plotly_white",
            height=550
        )

        st.plotly_chart(
            corr_fig,
            use_container_width=True
        )


# ============================================================
# PAGE 7 - AIR QUALITY
# ============================================================

elif page == "Air Quality":

    st.subheader(
        f"🌫️ Air Quality — {selected_city}"
    )

    aqi = int(
        safe_value(
            latest,
            "aqi",
            0
        )
    )

    if aqi <= 50:
        status = "Good"
    elif aqi <= 100:
        status = "Moderate"
    elif aqi <= 150:
        status = "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        status = "Unhealthy"
    elif aqi <= 300:
        status = "Very Unhealthy"
    else:
        status = "Hazardous"

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "AQI",
            aqi
        )

    with c2:

        st.metric(
            "Status",
            status
        )

    with c3:

        st.metric(
            "Humidity",
            f"{safe_value(latest, 'humidity'):.0f}%"
        )

    st.divider()

    st.subheader(
        "AQI Trend"
    )

    aqi_df = city_df.tail(48)

    fig = px.line(
        aqi_df,
        x="timestamp",
        y="aqi",
        title="Air Quality Index Trend",
        markers=True
    )

    fig.update_layout(
        template="plotly_white",
        height=430
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# PAGE 8 - WEATHER MAP
# ============================================================

elif page == "Weather Map":

    st.subheader(
        "🗺️ India Weather Map"
    )

    latest_all = (
        df.sort_values("timestamp")
        .groupby("city")
        .tail(1)
        .copy()
    )

    required_map = [
        "latitude",
        "longitude",
        "temperature",
        "city"
    ]

    if all(
        col in latest_all.columns
        for col in required_map
    ):

        fig = px.scatter_geo(
            latest_all,
            lat="latitude",
            lon="longitude",
            size="temperature",
            color="temperature",
            hover_name="city",
            hover_data=[
                "temperature",
                "humidity",
                "wind_speed",
                "aqi"
            ],
            scope="asia",
            projection="natural earth",
            title="Current Weather Across Indian Cities"
        )

        fig.update_geos(
            showcountries=True,
            showland=True,
            fitbounds="locations"
        )

        fig.update_layout(
            template="plotly_white",
            height=650,
            margin=dict(
                l=0,
                r=0,
                t=50,
                b=0
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "Latitude/Longitude information is not available."
        )


# ============================================================
# PAGE 9 - ALERTS
# ============================================================

elif page == "Alerts":

    st.subheader(
        f"🚨 Weather Alerts — {selected_city}"
    )

    alerts = []

    temperature = safe_value(
        latest,
        "temperature"
    )

    humidity = safe_value(
        latest,
        "humidity"
    )

    wind_speed = safe_value(
        latest,
        "wind_speed"
    )

    rain_probability = safe_value(
        latest,
        "precipitation_probability"
    )

    uv_index = safe_value(
        latest,
        "uv_index"
    )

    aqi = safe_value(
        latest,
        "aqi"
    )

    if temperature >= 40:
        alerts.append(
            "🔥 High temperature alert"
        )

    if humidity >= 85:
        alerts.append(
            "💧 Very high humidity"
        )

    if wind_speed >= 30:
        alerts.append(
            "💨 Strong wind alert"
        )

    if rain_probability >= 70:
        alerts.append(
            "🌧️ High rain probability"
        )

    if aqi >= 150:
        alerts.append(
            "🌫️ Poor air quality alert"
        )

    if uv_index >= 7:
        alerts.append(
            "☀️ High UV index"
        )

    if not alerts:

        st.success(
            "✅ No major weather alerts for the selected location."
        )

    else:

        for alert in alerts:

            st.warning(
                alert
            )

    st.divider()

    st.subheader(
        "Current Monitoring Values"
    )

    monitoring = pd.DataFrame(
        {
            "Parameter": [
                "Temperature",
                "Wind Speed",
                "Rain Probability",
                "AQI",
                "UV Index",
                "Humidity"
            ],

            "Value": [
                f"{temperature:.1f} °C",
                f"{wind_speed:.1f} km/h",
                f"{rain_probability:.1f} %",
                aqi,
                f"{uv_index:.1f}",
                f"{humidity:.1f} %"
            ]
        }
    )

    st.dataframe(
        monitoring,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PAGE 10 - HISTORICAL DATA
# ============================================================

elif page == "Historical Data":

    st.subheader(
        f"📜 Historical Weather Data — {selected_city}"
    )

    min_date = city_df[
        "timestamp"
    ].dt.date.min()

    max_date = city_df[
        "timestamp"
    ].dt.date.max()

    default_start = max(
        min_date,
        max_date - pd.Timedelta(days=7)
    )

    date_range = st.date_input(
        "Select Date Range",
        value=(
            default_start,
            max_date
        ),
        min_value=min_date,
        max_value=max_date
    )

    if isinstance(
        date_range,
        (tuple, list)
    ) and len(date_range) == 2:

        start_date = date_range[0]
        end_date = date_range[1]

        historical = city_df[
            (
                city_df["timestamp"].dt.date
                >= start_date
            )
            &
            (
                city_df["timestamp"].dt.date
                <= end_date
            )
        ].copy()

    else:

        historical = city_df.copy()

    st.metric(
        "Records",
        f"{len(historical):,}"
    )

    st.dataframe(
        historical,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PAGE 11 - DOWNLOAD DATA
# ============================================================

elif page == "Download Data":

    st.subheader(
        "⬇️ Download Weather Data"
    )

    st.caption(
        "Download weather data for analysis and reporting."
    )

    csv_data = city_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download CSV",
        data=csv_data,
        file_name=(
            selected_city.lower()
            + "_weather_data.csv"
        ),
        mime="text/csv",
        use_container_width=True
    )

    st.divider()

    st.subheader(
        "Dataset Summary"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Rows",
            f"{len(city_df):,}"
        )

    with c2:

        st.metric(
            "Columns",
            len(city_df.columns)
        )

    with c3:

        st.metric(
            "Cities",
            df["city"].nunique()
        )


# ============================================================
# PAGE 12 - SETTINGS
# ============================================================

elif page == "Settings":

    st.subheader(
        "⚙️ Dashboard Settings"
    )

    st.markdown(
        "### Dashboard Preferences"
    )

    refresh = st.slider(
        "Refresh Interval",
        min_value=10,
        max_value=300,
        value=60,
        step=10
    )

    show_charts = st.checkbox(
        "Show Interactive Charts",
        value=True
    )

    show_ml = st.checkbox(
        "Show Machine Learning Information",
        value=True
    )

    st.divider()

    st.markdown(
        "### Application Information"
    )

    st.info(
        """
        Weather Prediction & Monitoring Dashboard

        Technology:
        • Python
        • Pandas
        • NumPy
        • Scikit-learn
        • Random Forest
        • Plotly
        • Streamlit
        • Databricks Apps

        Spark is not required.
        """
    )

    st.success(
        f"Selected location: {selected_city}"
    )

    st.write(
        f"Refresh interval configured: {refresh} seconds"
    )

    if show_charts:

        st.write(
            "Interactive charts: Enabled"
        )

    else:

        st.write(
            "Interactive charts: Disabled"
        )

    if show_ml:

        st.write(
            "Machine Learning information: Enabled"
        )

    else:

        st.write(
            "Machine Learning information: Disabled"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer-text">
        Weather Prediction & Monitoring Dashboard
        <br>
        Built with Python • Machine Learning • Streamlit • Databricks
    </div>
    """,
    unsafe_allow_html=True
)