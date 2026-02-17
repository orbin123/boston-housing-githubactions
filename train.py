import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# Load Training Data 
X_train = pd.read_csv("data/train/X_train.csv")
y_train = pd.read_csv("data/train/y_train.csv")

# Convert y_train to 1D array (required for sklearn)
y_train = y_train.values.ravel()

# Create Model 
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# Train Model 
model.fit(X_train, y_train)

print("Model training completed!")

# Create models folder
os.makedirs("models", exist_ok=True)

# Save Model 
model_path = "models/housing_model.pkl"

joblib.dump(model, model_path)

print(f"Model saved at: {model_path}")