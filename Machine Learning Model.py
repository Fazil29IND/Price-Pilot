import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from math import sqrt
import joblib

df = pd.read_csv("Price Pilot Dataset.csv")


# Drop leakage and irrelevant columns
drop_cols = [
    "pickup_address",
    "drop_address",
    "trip_duration_minutes",
    "drop_hour",
    "drop_minute",
    "pickup_address_encoded",   # target leakage
    "drop_address_encoded",     # target leakage
]
for col in drop_cols:
    if col in df.columns:
        df = df.drop(columns=[col])

X = df.drop("fare_price", axis=1)
y = df["fare_price"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(
    max_depth=5,
    learning_rate=0.05,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred  = model.predict(X_test)
train_rmse = sqrt(mean_squared_error(y_train, train_pred))
test_rmse  = sqrt(mean_squared_error(y_test,  test_pred))
train_r2   = r2_score(y_train, train_pred)
test_r2    = r2_score(y_test,  test_pred)

print("\nModel Results:")
print(f"Train RMSE : {train_rmse:.4f}")
print(f"Test  RMSE : {test_rmse:.4f}")
print(f"Train R²   : {train_r2:.4f}")
print(f"Test  R²   : {test_r2:.4f}")

joblib.dump(model, "model.pkl")