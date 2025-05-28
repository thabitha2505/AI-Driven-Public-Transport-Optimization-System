import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load trained LSTM model
model = load_model("bus_ridership_model.keras")

# Load scalers
feature_scaler = joblib.load("feature_scaler.pkl")
target_scaler = joblib.load("target_scaler.pkl")  # Load the target scaler

# Example real-time input
real_time_input = pd.DataFrame({
    "Timestamp": ["2025-04-15 14:00:00"],
    "Day": ["Tuesday"],
    "Weekday/Weekend": ["Weekday"],
    "Holiday": [0],
    "Special Event": [0],  # Changed to 0 for consistency
    "Route": [3]
})

# Convert timestamp to datetime and extract features
real_time_input["Timestamp"] = pd.to_datetime(real_time_input["Timestamp"])
real_time_input["DayOfYear"] = real_time_input["Timestamp"].dt.dayofyear
real_time_input["Hour"] = real_time_input["Timestamp"].dt.hour

# Convert categorical features using same mappings as training
day_mapping = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, 
    "Friday": 4, "Saturday": 5, "Sunday": 6
}
real_time_input["Day"] = real_time_input["Day"].map(day_mapping)
real_time_input["Weekday/Weekend"] = real_time_input["Weekday/Weekend"].map({"Weekday": 0, "Weekend": 1})

# Select relevant features
features = ["Hour", "Day", "DayOfYear", "Weekday/Weekend", "Holiday", "Special Event", "Route"]

# Normalize input features using the saved feature scaler
X_real_time = feature_scaler.transform(real_time_input[features])

# Print scaled input for debugging
print("✅ Scaled input features:", X_real_time)

# Ensure input is shaped correctly for LSTM
X_real_time_seq = np.tile(X_real_time, (3, 1)).reshape(1, 3, X_real_time.shape[1])

# Predict passenger count
predicted_passenger_count_scaled = model.predict(X_real_time_seq)

# Debugging: Print raw model output before inverse scaling
print(f"🔹 Raw model output (before scaling): {predicted_passenger_count_scaled}")

# ✅ Correct inverse transform to get real passenger count
real_passenger_count = target_scaler.inverse_transform(predicted_passenger_count_scaled.reshape(-1, 1))

# Print final passenger count prediction
print(f"✅ Real Passenger Count Prediction: {real_passenger_count[0][0]:.2f}")
df_original = pd.read_csv("coimbatore_bus_ridership_cleaned.csv")

# Get min and max passenger count before scaling
min_passengers = df_original["Passenger Count"].min()
max_passengers = df_original["Passenger Count"].max()

# Debug: Print min and max values
print(f"🔍 Min Passenger Count: {min_passengers}, Max Passenger Count: {max_passengers}")

# Manually rescale the predicted value
real_passenger_count = predicted_passenger_count_scaled * (max_passengers - min_passengers) + min_passengers

# Print final rescaled prediction
print(f"✅ Corrected Passenger Count Prediction: {real_passenger_count[0][0]:.2f}")

