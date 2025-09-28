# AURA: Air-Quality-Index Predictor

A Streamlit app for predicting Air Quality Index (AQI) from environmental telemetry (PM10, PM2.5, NO2, SO2, O3, Temperature, Humidity, WindSpeed).

## Important: Use a scikit-learn Pipeline for the saved model

To ensure correct and stable predictions, the saved artifact in `model/Air_quality_index.pkl` MUST be a scikit-learn Pipeline that encapsulates any preprocessing (e.g., scaling, encoding, feature selection) and the final regressor. Do NOT save just a bare regressor — otherwise, feature order/naming or preprocessing mismatches can produce incorrect or constant outputs.

- Input feature names and order must match between training and inference.
- If your training uses scaling/encoding/selection, include those steps in the Pipeline.
- The app will attempt to validate and align input features to the model's training features when available.

## Feature Specifications

**Exact Input Features (required in this order):**
- `PM10` (µg/m³) - Particulate Matter 10 concentration
- `PM2_5` (µg/m³) - Particulate Matter 2.5 concentration  
- `NO2` (µg/m³) - Nitrogen Dioxide concentration
- `SO2` (µg/m³) - Sulfur Dioxide concentration
- `O3` (µg/m³) - Ozone concentration
- `Temperature` (°C) - Average temperature
- `Humidity` (%) - Relative humidity
- `WindSpeed` (km/h) - Average wind speed

**Target Variable:**
- `AQI` - Air Quality Index value (continuous)

## Step-by-Step Model Retraining Instructions

### Step 1: Data Preparation

```python
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load your dataset
df = pd.read_csv('model/AQI_data.csv')

# Define exact feature names and target
num_features = ['PM10','PM2_5','NO2','SO2','O3','Temperature','Humidity','WindSpeed']
target_column = 'AQI'

# Extract features and target
X = df[num_features]
y = df[target_column]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
```

### Step 2: Create and Train Pipeline

```python
# Create preprocessing pipeline
preprocessor = ColumnTransformer([
    ('scale', StandardScaler(), num_features),
], remainder='drop')

# Create complete pipeline
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the pipeline
print("Training the pipeline...")
pipeline.fit(X_train, y_train)
print("Training completed!")
```

### Step 3: Evaluate Model

```python
# Make predictions
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)

# Calculate metrics
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"Training MAE: {train_mae:.2f}, R²: {train_r2:.3f}")
print(f"Test MAE: {test_mae:.2f}, R²: {test_r2:.3f}")
```

### Step 4: Export Pipeline

```python
# Export the complete pipeline (CRITICAL: Must be a Pipeline, not just the regressor!)
model_path = 'model/Air_quality_index.pkl'
joblib.dump(pipeline, model_path)
print(f"Pipeline saved to {model_path}")

# Verify the export
loaded_pipeline = joblib.load(model_path)
test_prediction = loaded_pipeline.predict(X_test[:1])
print(f"Test prediction from loaded pipeline: {test_prediction[0]:.2f}")
```

### Step 5: Test with Sample "Safe Zone" Values

```python
# Test prediction with good air quality values
safe_zone_data = pd.DataFrame({
    'PM10': [25.0],          # Good: < 50 µg/m³
    'PM2_5': [15.0],         # Good: < 25 µg/m³  
    'NO2': [20.0],           # Good: < 40 µg/m³
    'SO2': [10.0],           # Good: < 50 µg/m³
    'O3': [60.0],            # Moderate: 51-100 µg/m³
    'Temperature': [22.0],    # Comfortable temperature
    'Humidity': [45.0],      # Moderate humidity
    'WindSpeed': [12.0]      # Good wind dispersion
})

safe_zone_aqi = loaded_pipeline.predict(safe_zone_data)
print(f"Safe zone AQI prediction: {safe_zone_aqi[0]:.2f}")
print(f"Expected: AQI should be in 'Good' range (0-50) or 'Moderate' range (51-100)")
```

## Model Export Guidelines

Recommended directory layout:
```
model/
  ├─ AQI_data.csv
  ├─ Air_quality_index.ipynb
  └─ Air_quality_index.pkl  # export your trained Pipeline here
```

## Complete Training Example

Below is a complete example that you can run in your notebook:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib

# 1) Define features and target
num_features = ['PM10','PM2_5','NO2','SO2','O3','Temperature','Humidity','WindSpeed']

# 2) Load and prepare data
df = pd.read_csv('model/AQI_data.csv')
X = df[num_features]
y = df['AQI']  # Ensure this is your target column name
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3) Build preprocessing + model pipeline
preprocessor = ColumnTransformer([
    ('scale', StandardScaler(), num_features),
], remainder='drop')

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# 4) Train and export
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'model/Air_quality_index.pkl')

print("Pipeline training and export completed successfully!")
```

Notes:
- Ensure the target column name is correct when training.
- If you engineer additional features, ensure the same logic exists at inference (ideally inside the Pipeline).
- The feature names and order must match exactly between training and inference.

## App Diagnostics and Guidance

This app includes comprehensive diagnostics to help debug input/feature issues:

- **Input Validation**: Prints the input dictionary and DataFrame before prediction.
- **Feature Alignment**: Validates and aligns input features to the Pipeline's training feature names when available.
- **Degeneracy Detection**: Warns when different inputs repeatedly produce the exact same prediction. This often indicates an export/training issue — re-train and export the Pipeline.
- **Error Handling**: Provides clear error messages for common issues like missing model files or preprocessing errors.

## Running Locally

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your trained Pipeline is saved at `model/Air_quality_index.pkl`.

3. Launch the app:
```bash
streamlit run app.py
```

## Troubleshooting

**Common Issues and Solutions:**

- **Model file not found**: Ensure `model/Air_quality_index.pkl` exists.
- **Prediction fails with preprocessing error**: Export a Pipeline that includes all preprocessing steps, not just the bare regressor.
- **Constant/identical predictions for diverse inputs**: Re-train and export the Pipeline, and verify feature names/order match exactly.
- **Feature name mismatches**: Ensure the input features match exactly: `['PM10','PM2_5','NO2','SO2','O3','Temperature','Humidity','WindSpeed']`
- **Missing preprocessing steps**: The Pipeline must include StandardScaler or other preprocessing steps used during training.

**Verification Steps:**

1. Check if the model file exists: `os.path.exists('model/Air_quality_index.pkl')`
2. Verify the model is a Pipeline: `isinstance(model, sklearn.pipeline.Pipeline)`
3. Test with known input values to ensure reasonable output
4. Check feature names alignment in the app diagnostics section

**For Support:**

If issues persist, check the app's diagnostic output for detailed error messages and feature alignment information.
