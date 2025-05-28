import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load trained LSTM model
model = load_model("bus_ridership_model.keras")

# Load scalers (feature scaler and target scaler trained on original passenger count)
feature_scaler = joblib.load("feature_scaler.pkl")
target_scaler = joblib.load("target_scaler.pkl")  # Ensure this was fitted on the uncleaned dataset

# Example real-time input
real_time_input = pd.DataFrame({
    "Timestamp": ["2025-04-15 11:00:00"],
    "Day": [1],
    "Weekday/Weekend": ["Weekend"],
    "Holiday": [1],
    "Special Event": [1],
    "Route": [2]
})

# Convert timestamp to datetime and extract features
real_time_input["Timestamp"] = pd.to_datetime(real_time_input["Timestamp"])
real_time_input["DayOfYear"] = real_time_input["Timestamp"].dt.dayofyear
real_time_input["Hour"] = real_time_input["Timestamp"].dt.hour
real_time_input["Weekday/Weekend"] = real_time_input["Weekday/Weekend"].map({"Weekday": 0, "Weekend": 1})

# Select relevant features
features = ["Hour", "Day", "DayOfYear", "Weekday/Weekend", "Holiday", "Special Event", "Route"]

# Normalize input features using the saved feature scaler
X_real_time = feature_scaler.transform(real_time_input[features])

# Create LSTM input sequence (simulating past data points)
X_real_time_seq = np.tile(X_real_time, (3, 1)).reshape(1, 3, X_real_time.shape[1])

# Predict passenger count
predicted_passenger_count_scaled = model.predict(X_real_time_seq)

# Convert back to real passenger count using the correct target scaler
real_passenger_count = target_scaler.inverse_transform(predicted_passenger_count_scaled)

# Print final passenger count prediction
print(f"✅ Real Passenger Count Prediction: {real_passenger_count[0][0]:.2f}")
