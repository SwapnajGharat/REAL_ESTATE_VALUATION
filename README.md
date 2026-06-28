# Real Estate Valuation Hub

An AI-powered property valuation tool built with Streamlit and scikit-learn. Input home specifications and neighborhood indicators to get an instant market value estimate from a trained Random Forest model.

---

## Project Structure

```
.
├── train.py                  # ML pipeline: trains and serializes the model
├── app.py                    # Streamlit application
├── requirements.txt          # Python dependencies
├── usa_housing_kaggle.csv    # Source dataset (300 records)
├── housing_scaler.pkl        # Generated after running train.py
└── housing_model.pkl         # Generated after running train.py
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

Run this once to generate the two `.pkl` files. Both files must be present before launching the app.

```bash
python train.py
```

Expected output:

```
--- Model Evaluation ---
Mean Absolute Error : $230,646.03
R-squared Score     : -0.1270

Assets saved: housing_scaler.pkl, housing_model.pkl
```

### 3. Launch the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Dataset

**File:** `usa_housing_kaggle.csv`  
**Records:** 300  
**Target:** `Price` (USD)

| Feature | Type | Description |
|---|---|---|
| `Bedrooms` | int | Number of bedrooms (1–5) |
| `Bathrooms` | float | Number of bathrooms (1.0–4.0) |
| `SquareFeet` | int | Interior area in square feet |
| `YearBuilt` | int | Year of construction |
| `GarageSpaces` | int | Number of garage car spaces (0–3) |
| `LotSize` | float | Lot area in acres |
| `ZipCode` | int | Postal code (used as geographic token) |
| `CrimeRate` | float | Local crime index (0 = safest, 100 = highest) |
| `SchoolRating` | int | Public school quality score (1–10) |

---

## ML Pipeline

`train.py` executes the following steps:

**Feature scaling** — `StandardScaler` normalizes all features to zero mean and unit variance. This prevents high-magnitude columns like `SquareFeet` or `ZipCode` from dominating the model.

**Model** — `RandomForestRegressor` with 200 trees and a max depth of 15. An ensemble approach is used to capture non-linear relationships between community indicators and price, such as how crime rate interacts differently with school rating at different price points.

**Serialization** — The fitted scaler and trained model are written to `housing_scaler.pkl` and `housing_model.pkl` using `pickle`. The app loads these at startup rather than retraining on every run.

---

## App Features

**Valuation panel** — Sliders and number inputs for all nine features. Clicking "Estimate Property Value" runs the input through the scaler and model and displays the predicted price alongside a summary of the submitted values.

**Dataset table** — The raw CSV is displayed at the bottom with formatted columns for browsing and comparison.

---

## Requirements

```
streamlit==1.35.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.0
```

Python 3.9 or higher is recommended.

---

## Notes

The dataset contains 300 records, which is small for a regression task of this complexity. The `ZipCode` column is treated as a raw integer rather than a categorical feature, which limits its predictive usefulness. Model accuracy can be improved by encoding zip codes (e.g., target encoding or one-hot), acquiring more data, or replacing the zip code with more granular location features such as latitude/longitude or county-level statistics.
