import os
import re
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# Load Training Data 
X_train = pd.read_csv("data/train/X_train.csv")
y_train = pd.read_csv("data/train/y_train.csv")
y_train = y_train.values.ravel()

# Create Model 
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# Train Model 
model.fit(X_train, y_train)

print("Model training completed!")

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# VERSION LOGIC

current_model_file = "models/current_model.txt"

# If file exists → read current version
if os.path.exists(current_model_file):

    with open(current_model_file, "r") as f:
        current_model_name = f.read().strip()

    # Extract number from model_vX.joblib
    match = re.search(r"model_v(\d+)\.joblib", current_model_name)

    if match:
        current_version = int(match.group(1))
        new_version = current_version + 1
    else:
        new_version = 1

else:
    # No version file → start from v1
    new_version = 1

# New model name
new_model_name = f"model_v{new_version}.joblib"
model_path = os.path.join("models", new_model_name)

# Save model
joblib.dump(model, model_path)

print(f"Model saved at: {model_path}")

# UPDATE ACTIVE MODEL

with open(current_model_file, "w") as f:
    f.write(new_model_name)

print(f"Active model updated to: {new_model_name}")