import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

df = pd.read_csv("usa_housing_kaggle.csv")

X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("--- Model Evaluation ---")
print(f"Mean Absolute Error : ${mae:,.2f}")
print(f"R-squared Score     : {r2:.4f}")

with open("housing_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("housing_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nAssets saved: housing_scaler.pkl, housing_model.pkl")
