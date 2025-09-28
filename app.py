import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
from datetime import datetime
import pytz
import os
import json
from typing import List, Optional

# --- Helper Function for AQI Calculation ---
# This function determines the AQI category and associated health advice.
def get_aqi_info(aqi: float):
    """
    Determines the AQI category, color, and health advisory based on the AQI value.
    """
    aqi = round(aqi)
    if 0 <= aqi <= 50:
        return {
            "name": "Good",
            "color": "#00e400",
            "advisory": "Air quality is satisfactory, and air pollution poses little or no risk."
        }
    elif 51 <= aqi <= 100:
        return {
            "name": "Moderate",
            "color": "#ffff00",
            "advisory": "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution."
        }
    elif 101 <= aqi <= 150:
        return {
            "name": "Unhealthy for Sensitive Groups",
            "color": "#ff7e00",
            "advisory": "Members of sensitive groups may experience health effects. The general public is less likely to be affected."
        }
    elif 151 <= aqi <= 200:
        return {
            "name": "Unhealthy",
            "color": "#ff0000",
            "advisory": "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects."
        }
    elif 201 <= aqi <= 300:
        return {
            "name": "Very Unhealthy",
            "color": "#8f3f97",
            "advisory": "Health alert: The risk of health effects is increased for everyone."
        }
    elif aqi > 300:
        return {
            "name": "Hazardous",
            "color": "#7e0023",
            "advisory": "Health warning of emergency conditions: everyone is more likely to be affected."
        }
    return {
        "name": "Unknown",
        "color": "#808080",
        "advisory": "AQI value is out of the standard range."
    }

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="AURA | AQI Prediction Interface",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Custom CSS for Futuristic UI ---
def load_css():
    """Load and inject custom CSS for styling."""
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(180deg, #0f0f1a 0%, #141e30 100%); color: #e0e0e0; }
        [data-testid=\"stSidebar\"] { background: rgba(255,255,255,.05); backdrop-filter: blur(10px); border-right: 1px solid rgba(255,255,255,.1); }
        h1, h2 { color: #00f2ff; text-shadow: 0 0 5px #00f2ff, 0 0 10px #00f2ff; }
        [data-testid=\"stMetric\"] { background: rgba(0,242,255,.1); border: 1px solid #00f2ff; border-radius: 10px; padding: 20px; box-shadow: 0 0 15px rgba(0,242,255,.2); }
        .stButton>button { border: 2px solid #00f2ff; border-radius: 10px; color: #00f2ff; background: transparent; transition: all .3s ease; width: 100%; }
        .stButton>button:hover { background: #00f2ff; color: #0f0f1a; box-shadow: 0 0 20px #00f2ff; }
        [data-testid=\"stAlert\"] { background: rgba(255,255,255,.08); border-left: 5px solid #ff00c1; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# --- 3. Model Loading (Updated with Error Handling & Feature Validation) ---
@st.cache_resource
def load_model():
    """
    Load the pre-trained model from the .pkl file with robust error checking.

    IMPORTANT:
    - The saved artifact SHOULD be a scikit-learn Pipeline that includes any preprocessing
      (scaling/encoding/selection) and the final estimator.
    - Exporting only a bare regressor is not recommended and may cause incorrect predictions.

    The model should expose .predict(X). If available, .feature_names_in_ will be used
    to validate/align input features. Otherwise, we fall back to the UI feature order.

    HOW TO TRAIN & EXPORT A PIPELINE (example):
    -------------------------------------------------
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor
    import joblib

    num_features = ['PM10','PM2_5','NO2','SO2','O3','Temperature','Humidity','WindSpeed']
    pre = ColumnTransformer([
        ('scale', StandardScaler(), num_features),
    ], remainder='drop')
    pipe = Pipeline([
        ('pre', pre),
        ('model', RandomForestRegressor(random_state=42))
    ])
    pipe.fit(X_train[num_features], y_train)
    joblib.dump(pipe, 'model/Air_quality_index.pkl')
    -------------------------------------------------
    """
    model_path = os.path.join('model', 'Air_quality_index.pkl')

    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at '{model_path}'. Run the notebook to generate it.")
        return None

    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        if not hasattr(model, 'predict'):
            st.error(
                f"❌ Loaded object (type: {type(model).__name__}) is not a valid model (no '.predict()'). "
                "Re-train and export a scikit-learn Pipeline that includes preprocessing."
            )
            return None

        # Try to infer training feature names from the pipeline or estimator
        feature_names = getattr(model, 'feature_names_in_', None)
        if feature_names is None and hasattr(model, 'named_steps'):
            for _, step in model.named_steps.items():
                if hasattr(step, 'feature_names_in_'):
                    feature_names = getattr(step, 'feature_names_in_')
                    break
        try:
            setattr(model, '_training_feature_names', list(feature_names) if feature_names is not None else None)
        except Exception:
            pass
        return model

    except pickle.UnpicklingError:
        st.error(f"❌ Failed to unpickle '{model_path}'. File may be corrupted. Re-run the notebook.")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error loading model: {e}")
        return None

# --- 4. Plotly Gauge Chart ---
def create_gauge(aqi_value, aqi_info):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi_value,
        number={'suffix': " AQI", 'font': {'size': 50, 'color': aqi_info['color']}},
        title={'text': f"{aqi_info['name']}", 'font': {'size': 24, 'color': '#e0e0e0'}},
        gauge={
            'axis': {'range': [0, 500], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': aqi_info['color'], 'thickness': 0.3},
            'bgcolor': "rgba(255, 255, 255, 0.1)",
            'borderwidth': 2,
            'bordercolor': "#e0e0e0",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(0, 228, 0, 0.5)'},
                {'range': [51, 100], 'color': 'rgba(255, 255, 0, 0.5)'},
                {'range': [101, 150], 'color': 'rgba(255, 126, 0, 0.5)'},
                {'range': [151, 200], 'color': 'rgba(255, 0, 0, 0.5)'},
                {'range': [201, 300], 'color': 'rgba(143, 63, 151, 0.5)'},
                {'range': [301, 500], 'color': 'rgba(126, 0, 35, 0.5)'}
            ],
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "#e0e0e0", 'family': "Arial"}, height=350)
    return fig

# --- 5. Utilities for diagnostics ---
EXPECTED_FEATURE_ORDER: List[str] = [
    'PM10', 'PM2_5', 'NO2', 'SO2', 'O3', 'Temperature', 'Humidity', 'WindSpeed'
]

def validate_and_align_features(df: pd.DataFrame, model_obj) -> Optional[pd.DataFrame]:
    """Validate and align input DataFrame columns to match training features."""
    training_cols = getattr(model_obj, '_training_feature_names', None)
    if training_cols is not None:
        missing = [c for c in training_cols if c not in df.columns]
        extra = [c for c in df.columns if c not in training_cols]
        if missing:
            st.error(f"❌ Missing input features required by the model: {missing}")
            return None
        if extra:
            st.warning(f"ℹ️ Extra features provided will be ignored: {extra}")
        return df.reindex(columns=list(training_cols))
    # Fallback to UI order
    missing = [c for c in EXPECTED_FEATURE_ORDER if c not in df.columns]
    if missing:
        st.error(f"❌ Missing input features: {missing}")
        return None
    return df.reindex(columns=EXPECTED_FEATURE_ORDER)

# --- 6. Main Application Logic ---
load_css()
model = load_model()

# --- Header ---
st.title("AURA: Atmospheric Quality & Risk Analyzer")
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("An advanced interface for real-time air quality prediction using machine learning.")
with col2:
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    st.markdown(f"""
    <div style=\"text-align: right; color: #aaa;\">
        Location: Nagpur, India<br/>
        Timestamp: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- Sidebar ---
with st.sidebar:
    st.header("System Input")
    st.markdown("Provide environmental telemetry for analysis.")
    with st.form(key='prediction_form'):
        PM10 = st.slider('PM10 (µg/m³)', 0.0, 500.0, 148.7, 0.1, help="Particulate Matter 10 concentration.")
        PM2_5 = st.slider('PM2.5 (µg/m³)', 0.0, 300.0, 100.2, 0.1, help="Particulate Matter 2.5 concentration.")
        NO2 = st.slider('NO2 (µg/m³)', 0.0, 300.0, 102.3, 0.1, help="Nitrogen Dioxide concentration.")
        SO2 = st.slider('SO2 (µg/m³)', 0.0, 200.0, 50.5, 0.1, help="Sulfur Dioxide concentration.")
        O3 = st.slider('O3 (µg/m³)', 0.0, 300.0, 125.0, 0.1, help="Ozone concentration.")
        Temperature = st.slider('Temperature (°C)', -10.0, 50.0, 15.0, 0.1, help="Average temperature.")
        Humidity = st.slider('Humidity (%)', 0.0, 100.0, 50.0, 0.1, help="Relative humidity.")
        WindSpeed = st.slider('Wind Speed (km/h)', 0.0, 50.0, 8.0, 0.1, help="Average wind speed.")
        submit_button = st.form_submit_button(label='Initiate Analysis')

    st.markdown("---")
    st.header("System Status")
    model_path = os.path.join('model', 'Air_quality_index.pkl')
    if os.path.exists(model_path):
        st.success("✅ Model file found.")
        if model is not None:
            st.success("✅ Model loaded successfully.")
        else:
            st.error("❌ Model failed to load.")
            st.info("The .pkl file may be corrupted or not a Pipeline. Please re-train and export a scikit-learn Pipeline including preprocessing.")
    else:
        st.error("❌ Model file not found.")
        st.info("Please ensure 'Air_quality_index.pkl' is located inside the 'model' directory.")

# --- Main Dashboard Display ---
if model is None:
    st.error("SYSTEM OFFLINE: Prediction model is unavailable. Please check the System Status panel in the sidebar for details.")
else:
    if 'aqi_result' not in st.session_state:
        st.session_state.aqi_result = None

    if submit_button:
        with st.spinner("Analyzing atmospheric data... Engaging prediction model..."):
            input_features = {
                'PM10': PM10,
                'PM2_5': PM2_5,
                'NO2': NO2,
                'SO2': SO2,
                'O3': O3,
                'Temperature': Temperature,
                'Humidity': Humidity,
                'WindSpeed': WindSpeed
            }
            input_df = pd.DataFrame([input_features])

            # --- Diagnostic logging: show features and values, and alignment status ---
            st.subheader("Diagnostics: Input Snapshot")
            st.code(json.dumps(input_features, indent=2), language="json")
            st.write("Input DataFrame columns:", list(input_df.columns))

            aligned_df = validate_and_align_features(input_df, model)
            if aligned_df is None:
                st.stop()
            st.write("Aligned DataFrame columns:", list(aligned_df.columns))
            st.dataframe(aligned_df)

            try:
                prediction = model.predict(aligned_df)
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                st.info("Ensure the saved artifact is a scikit-learn Pipeline handling any required preprocessing.")
                st.stop()

            predicted_aqi = float(prediction[0])

            # --- Diagnostic: detect degenerate model outputs ---
            history = st.session_state.get('prediction_history', [])
            history.append({'inputs': input_features, 'prediction': predicted_aqi})
            st.session_state['prediction_history'] = history[-5:]  # keep last 5

            warn_degenerate = False
            if len(st.session_state['prediction_history']) >= 2:
                hist = st.session_state['prediction_history']
                unique_inputs = {json.dumps(h['inputs'], sort_keys=True) for h in hist}
                unique_preds = {h['prediction'] for h in hist}
                if len(unique_inputs) > 1 and len(unique_preds) == 1:
                    warn_degenerate = True
            if warn_degenerate:
                st.warning(
                    "⚠️ Diagnostic: Multiple diverse inputs produced an identical prediction. "
                    "This often indicates a model/export issue (e.g., missing preprocessing or wrong feature order). "
                    "Please retrain and export a scikit-learn Pipeline from the notebook, ensuring feature names and order match."
                )

