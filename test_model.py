import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load Test Data
X_test = pd.read_csv("data/test/X_test.csv")
y_test = pd.read_csv("data/test/y_test.csv")

# Convert y_test to 1D array
y_test = y_test.values.ravel()

# Load Trained Model
model = joblib.load("models/housing_model.pkl")

# Make Predictions
y_pred = model.predict(X_test)

# Calculate Metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# Assertions with small tolerance
assert mse < 18, "MSE is too high"
assert mae < 3.0, "MAE is too high"

print("Model performance test passed!")