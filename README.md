# AURA: Air-Quality-Index Predictor

A Streamlit app for predicting Air Quality Index (AQI) from environmental telemetry (PM10, PM2.5, NO2, SO2, O3, Temperature, Humidity, WindSpeed).

## Important: Use a scikit-learn Pipeline for the saved model

To ensure correct and stable predictions, the saved artifact in `model/Air_quality_index.pkl` MUST be a scikit-learn Pipeline that encapsulates any preprocessing (e.g., scaling, encoding, feature selection) and the final regressor. Do NOT save just a bare regressor — otherwise, feature order/naming or preprocessing mismatches can produce incorrect or constant outputs.

- Input feature names and order must match between training and inference.
- If your training uses scaling/encoding/selection, include those steps in the Pipeline.
- The app will attempt to validate and align input features to the model's training features when available.

## Model Export Guidelines

Recommended directory layout:

```
model/
  ├─ AQI_data.csv
  ├─ Air_quality_index.ipynb
  └─ Air_quality_index.pkl  # export your trained Pipeline here
```

### Example: Train and export a Pipeline

Below is a minimal example using a numeric feature pipeline with scaling and a random forest regressor. Adapt the estimator and preprocessing as needed.

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
# df = pd.read_csv('model/AQI_data.csv')
# X = df[num_features]
# y = df['AQI']  # replace with your target column name

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2) Build preprocessing + model pipeline
pre = ColumnTransformer([
    ('scale', StandardScaler(), num_features),
], remainder='drop')

pipe = Pipeline([
    ('pre', pre),
    ('model', RandomForestRegressor(random_state=42))
])

# 3) Train and export
# pipe.fit(X_train, y_train)
# joblib.dump(pipe, 'model/Air_quality_index.pkl')
```

Notes:
- Ensure the target column name is correct when training.
- If you engineer additional features, ensure the same logic exists at inference (ideally inside the Pipeline).

## App Diagnostics and Guidance

This app includes diagnostics to help debug input/feature issues:
- Prints the input dictionary and DataFrame before prediction.
- Validates and aligns input features to the Pipeline's training feature names when available.
- Warns when different inputs repeatedly produce the exact same prediction. This often indicates an export/training issue — re-train and export the Pipeline.

## Running Locally

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Ensure your trained Pipeline is saved at `model/Air_quality_index.pkl`.

3. Launch the app:
```
streamlit run app.py
```

## Troubleshooting

- Model file not found: Ensure `model/Air_quality_index.pkl` exists.
- Prediction fails with preprocessing error: Export a Pipeline that includes all preprocessing.
- Constant/identical predictions for diverse inputs: Re-train and export the Pipeline, and verify feature names/order.
