import os
from pathlib import Path

import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.linear_model import LinearRegression

# Create Flask application
app = Flask(__name__)
CORS(app)

# Locate the DA.xlsx file next to the project root
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT_DIR / "DA.xlsx"
if not DATA_FILE.exists():
    DATA_FILE = Path(__file__).resolve().parent / "DA.xlsx"

# Load Excel file into a pandas DataFrame
df = pd.read_excel(DATA_FILE)

# The year columns in the Excel file used for training the model
YEAR_COLS = [
    "2013-14", "2014-15", "2015-16", "2016-17",
    "2017-18", "2018-19", "2019-20", "2020-21",
    "2021-22", "2022-23", "2023-24", "2024-25"
]

# Numeric years used for the regression model
YEARS = np.array([
    2013, 2014, 2015, 2016,
    2017, 2018, 2019, 2020,
    2021, 2022, 2023, 2024
]).reshape(-1, 1)

# Clean commodity, variety, and category columns
df["Commodity"] = df["Commodity"].astype(str).str.strip()
df["Variety"] = df["Variety"].fillna("").astype(str).str.strip()
df["Category"] = df["Category"].fillna("Other Crops").astype(str).str.strip()

# Create unique DisplayName for each crop variety
df["DisplayName"] = df.apply(
    lambda r: f"{r['Commodity']} ({r['Variety']})" if r["Variety"] else r["Commodity"],
    axis=1
)

# Filter out rows with less than 2 valid (non-NaN) data points in YEAR_COLS, since linear regression requires at least 2 points
df = df[df[YEAR_COLS].notna().sum(axis=1) >= 2].reset_index(drop=True)


def get_crop_row(crop_name_or_display):
    """Return the row for a crop name or display name, case-insensitive."""
    match = df[df["DisplayName"].str.lower() == crop_name_or_display.lower()]
    if match.empty:
        match = df[df["Commodity"].str.lower() == crop_name_or_display.lower()]
    return match


def train_model_for_crop(crop_row):
    """Train and return a linear regression model along with the raw values."""
    raw_values = crop_row[YEAR_COLS].astype(float).values
    
    # Filter out NaNs for training
    valid_mask = ~np.isnan(raw_values)
    train_years = YEARS[valid_mask]
    train_values = raw_values[valid_mask]
    
    model = LinearRegression()
    model.fit(train_years, train_values)
    return model, raw_values


def predict_msp(crop_name, year):
    """Predict MSP for the given crop and year."""
    crop_row = get_crop_row(crop_name)
    if crop_row.empty:
        return None

    model, _ = train_model_for_crop(crop_row.iloc[0])
    prediction = model.predict([[year]])[0]
    return round(float(prediction), 2)


def forecast_crop(crop_name, start_year=2025, end_year=2029):
    """Forecast MSP values for a crop, returning historical values and model metrics."""
    crop_row = get_crop_row(crop_name)
    if crop_row.empty:
        return None

    row = crop_row.iloc[0]
    model, values = train_model_for_crop(row)

    # Historical data mapping
    historical = {}
    for col, val in zip(YEAR_COLS, values):
        historical[col] = round(float(val), 2) if not np.isnan(val) else None

    # Forecast data mapping: 2025 -> "2025-26" etc.
    forecast_data = {}
    for year in range(start_year, end_year + 1):
        pred = model.predict([[year]])[0]
        label = f"{year}-{(year+1)%100:02d}"
        forecast_data[label] = round(float(pred), 2)

    # Calculate model metrics
    # R^2 score based on valid historical data
    valid_mask = ~np.isnan(values)
    r2 = model.score(YEARS[valid_mask], values[valid_mask])
    slope = model.coef_[0]
    
    # Growth percentage rate relative to the last valid historical year
    non_nan_values = values[valid_mask]
    last_val = non_nan_values[-1] if len(non_nan_values) > 0 else 0
    growth_pct = (slope / last_val) * 100 if last_val > 0 else 0

    metrics = {
        "r2_score": round(float(r2), 4),
        "slope": round(float(slope), 2),
        "growth_rate_pct": round(float(growth_pct), 2),
        "historical_avg": round(float(np.mean(non_nan_values)), 2) if len(non_nan_values) > 0 else 0,
        "historical_min": round(float(np.min(non_nan_values)), 2) if len(non_nan_values) > 0 else 0,
        "historical_max": round(float(np.max(non_nan_values)), 2) if len(non_nan_values) > 0 else 0,
        "predicted_min": round(float(min(forecast_data.values())), 2) if forecast_data else 0,
        "predicted_max": round(float(max(forecast_data.values())), 2) if forecast_data else 0
    }

    return {
        "displayName": row["DisplayName"],
        "commodity": row["Commodity"],
        "variety": row["Variety"],
        "category": row["Category"],
        "historical": historical,
        "forecast": forecast_data,
        "metrics": metrics
    }


def best_crop_for_year(year):
    """Recommend the crop with the highest predicted MSP for a specific year."""
    best_crop_name = None
    best_value = -float("inf")
    best_commodity = None
    best_variety = None

    for _, row in df.iterrows():
        try:
            model, _ = train_model_for_crop(row)
            predicted_value = model.predict([[year]])[0]
            if predicted_value > best_value:
                best_value = predicted_value
                best_crop_name = row["DisplayName"]
                best_commodity = row["Commodity"]
                best_variety = row["Variety"]
        except Exception:
            continue

    return {
        "year": year,
        "best_crop": best_crop_name,
        "commodity": best_commodity,
        "variety": best_variety,
        "predicted_msp": round(float(best_value), 2) if best_crop_name else None
    }


@app.route("/")
def home():
    """Health check endpoint."""
    return jsonify({"message": "MSP Prediction API is running"})


@app.route("/crops")
def crops():
    """Return the list of all crop commodities and their details."""
    crop_list = []
    for _, row in df.iterrows():
        crop_list.append({
            "displayName": row["DisplayName"],
            "commodity": row["Commodity"],
            "variety": row["Variety"],
            "category": row["Category"]
        })
    return jsonify(crop_list)


@app.route("/predict")
def predict():
    """Return a single MSP prediction for crop and year."""
    crop = request.args.get("crop")
    year_str = request.args.get("year")

    if not crop or not year_str:
        return jsonify({"error": "Please provide crop and year query parameters."}), 400

    try:
        year = int(year_str)
    except ValueError:
        return jsonify({"error": "Year must be a valid integer."}), 400

    result = predict_msp(crop, year)
    if result is None:
        return jsonify({"error": "Crop not found."}), 404

    crop_row = get_crop_row(crop).iloc[0]
    year_display = f"{year}-{(year+1)%100:02d}"

    return jsonify({
        "displayName": crop_row["DisplayName"],
        "commodity": crop_row["Commodity"],
        "variety": crop_row["Variety"],
        "category": crop_row["Category"],
        "year": year,
        "year_display": year_display,
        "predicted_msp": result
    })


@app.route("/forecast")
def forecast():
    """Return forecasted MSP values for a crop, along with historical actuals and metrics."""
    crop = request.args.get("crop")
    if not crop:
        return jsonify({"error": "Please provide crop query parameter."}), 400

    result = forecast_crop(crop)
    if result is None:
        return jsonify({"error": "Crop not found."}), 404

    return jsonify(result)


@app.route("/bestcrop")
def bestcrop():
    """Return best crop recommendation for a given year."""
    year_str = request.args.get("year", "2025")
    try:
        year = int(year_str)
    except ValueError:
        return jsonify({"error": "Year must be a valid integer."}), 400

    result = best_crop_for_year(year)
    if result["best_crop"] is None:
        return jsonify({"error": "No valid crop predictions available."}), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)

