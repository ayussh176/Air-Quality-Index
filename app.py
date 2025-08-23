import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
from datetime import datetime
import pytz
import os

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
    st.markdown("""
    <style>
        /* Main background and theme */
        .stApp {
            background: linear-gradient(180deg, #0f0f1a 0%, #141e30 100%);
            color: #e0e0e0;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Custom title and headers */
        h1, h2 {
            color: #00f2ff; /* Neon Blue */
            text-shadow: 0 0 5px #00f2ff, 0 0 10px #00f2ff;
        }
        
        /* Metric styling */
        [data-testid="stMetric"] {
            background: rgba(0, 242, 255, 0.1);
            border: 1px solid #00f2ff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 0 15px rgba(0, 242, 255, 0.2);
        }
        
        /* Button styling */
        .stButton>button {
            border: 2px solid #00f2ff;
            border-radius: 10px;
            color: #00f2ff;
            background-color: transparent;
            transition: all 0.3s ease;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #00f2ff;
            color: #0f0f1a;
            box-shadow: 0 0 20px #00f2ff;
        }

        /* Info box styling */
        [data-testid="stAlert"] {
            background: rgba(255, 255, 255, 0.08);
            border-left: 5px solid #ff00c1; /* Neon Pink */
        }
    </style>
    """, unsafe_allow_html=True)

# --- 3. Model Loading (Updated with Error Handling) ---
@st.cache_resource
def load_model():
    """Load the pre-trained model from the .pkl file with robust error checking."""
    # Path based on your directory structure
    model_path = os.path.join('model', 'Air_quality_index.pkl')
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at '{model_path}'. Run the notebook to generate it.")
        return None

    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        
        if not hasattr(model, 'predict'):
            st.error(f"❌ Loaded object (type: {type(model).__name__}) is not a valid model (no '.predict()' method). Re-run the notebook.")
            return None

        return model

    except pickle.UnpicklingError:
        st.error(f"❌ Failed to unpickle '{model_path}'. File may be corrupted. Re-run the notebook.")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error loading model: {e}")
        return None

# --- 4. Plotly Gauge Chart ---
def create_gauge(aqi_value, aqi_info):
    """Create a futuristic gauge chart using Plotly."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi_value,
        number={'suffix': " AQI", 'font': {'size': 50, 'color': aqi_info['color']}},
        title={'text': f"<b>{aqi_info['name']}</b>", 'font': {'size': 24, 'color': '#e0e0e0'}},
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
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "#e0e0e0", 'family': "Arial"},
        height=350
    )
    return fig

# --- 5. Main Application Logic ---
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
    <div style='text-align: right; color: #aaa;'>
        <b>Location:</b> Nagpur, India<br>
        <b>Timestamp:</b> {now.strftime('%Y-%m-%d %H:%M:%S %Z')}
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
            st.info("The .pkl file may be corrupted. Please re-run your notebook to create a valid model file.")
    else:
        st.error("❌ Model file not found.")
        st.info(f"Please ensure 'Air_quality_index.pkl' is located inside the 'model' directory.")

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
            
            prediction = model.predict(input_df)
            predicted_aqi = prediction[0]
            
            st.session_state.aqi_result = {
                "value": predicted_aqi,
                "info": get_aqi_info(predicted_aqi),
                "inputs": input_features
            }

    if st.session_state.aqi_result:
        result = st.session_state.aqi_result
        st.header("Analysis Complete: Results")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### Health Protocol Advisory")
            st.info(result['info']["advisory"])
            st.markdown("---")
            st.subheader("Input Telemetry Echo")
            st.json(result['inputs'])
            
        with col2:
            st.plotly_chart(create_gauge(result['value'], result['info']), use_container_width=True)

    else:
        st.info("System is on standby. Please provide input telemetry via the sidebar to initiate analysis.", icon="📡")