import time
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Load trained model & scalers
model = load_model("bus_ridership_model.keras")
feature_scaler = joblib.load("feature_scaler.pkl")
target_scaler = joblib.load("target_scaler.pkl")

# Load historical dataset & external data
df = pd.read_csv("coimbatore_bus_ridership.csv")
holidays_df = pd.read_csv("full_holidays_2025.csv")
special_events_df = pd.read_csv("special_events_2025.csv")

# Convert date columns to datetime format
holidays_df["Date"] = pd.to_datetime(holidays_df["Date"])
special_events_df["Date"] = pd.to_datetime(special_events_df["Date"])

# Define functions to check holidays and special events
def is_holiday(date):
    return 1 if date in holidays_df["Date"].values else 0

def special_event_factor(date):
    return 1 if date in special_events_df["Date"].values else 0

# Ensure categorical "Route" column is converted to numeric
le = LabelEncoder()
df["Route"] = le.fit_transform(df["Route"])

# Preprocessing function
def preprocess_data(data):
    data["Timestamp"] = pd.to_datetime(data["Timestamp"])
    data["DayOfYear"] = data["Timestamp"].dt.dayofyear
    data["Hour"] = data["Timestamp"].dt.hour
    data["Weekday/Weekend"] = data["Timestamp"].dt.weekday.apply(lambda x: 0 if x < 5 else 1)
    data["Day"] = data["Timestamp"].dt.dayofweek
    data["Holiday"] = data["Timestamp"].apply(is_holiday)
    data["Special Event"] = data["Timestamp"].apply(special_event_factor)
    return feature_scaler.transform(data[["Hour", "Day", "DayOfYear", "Weekday/Weekend", "Holiday", "Special Event", "Route"]])

# Function to create sequences for LSTM training
def create_sequences(X, y, time_steps=3):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i: i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# Main loop to update model every hour
while True:
    print("\n🔄 Checking for new data for all routes...")
    now = pd.Timestamp.now()
    day_of_year = now.dayofyear
    hour = now.hour
    weekday_weekend = 0 if now.weekday() < 5 else 1
    day = now.dayofweek
    holiday = is_holiday(now)
    special_event = special_event_factor(now)

    unique_routes = df["Route"].unique()
    new_data_list = []

    for route in unique_routes:
        real_time_input = pd.DataFrame({
            "Timestamp": [now],
            "DayOfYear": [day_of_year],
            "Hour": [hour],
            "Weekday/Weekend": [weekday_weekend],
            "Day": [day],
            "Holiday": [holiday],
            "Special Event": [special_event],
            "Route": [route]
        })

        # Preprocess input
        X_real_time = preprocess_data(real_time_input)
        X_real_time_seq = np.tile(X_real_time, (3, 1)).reshape(1, 3, X_real_time.shape[1])
        predicted_scaled = model.predict(X_real_time_seq)
        predicted_count = target_scaler.inverse_transform(predicted_scaled)[0][0]

        print(f"✅ Route {route} - Predicted Passenger Count: {predicted_count:.2f}")

        # Append new data
        new_data_list.append({
            "Timestamp": now,
            "DayOfYear": day_of_year,
            "Hour": hour,
            "Weekday/Weekend": weekday_weekend,
            "Day": day,
            "Holiday": holiday,
            "Special Event": special_event,
            "Route": route,
            "Passenger Count": predicted_count
        })

    # Update dataset with new predictions
    new_data_df = pd.DataFrame(new_data_list)
    df = pd.concat([df, new_data_df], ignore_index=True)

    # Fix "Day" column issues
    df["Day"] = pd.to_numeric(df["Day"], errors="coerce")
    df["Day"].fillna(df["Day"].mode()[0], inplace=True)
    df["Day"] = df["Day"].astype(int)

    # Fix categorical columns
    df["Weekday/Weekend"] = df["Weekday/Weekend"].replace({"Weekday": 0, "Weekend": 1})
    df["Weekday/Weekend"] = pd.to_numeric(df["Weekday/Weekend"], errors="coerce")
    df["Weekday/Weekend"].fillna(0, inplace=True)
    df["Weekday/Weekend"] = df["Weekday/Weekend"].astype(int)

    df["Holiday"] = df["Holiday"].replace({"Yes": 1, "No": 0})
    df["Holiday"] = df["Holiday"].fillna(0).astype(int)
    df["Special Event"] = df["Special Event"].replace({"Yes": 1, "No": 0})
    df["Special Event"] = df["Special Event"].fillna(0).astype(int)    

    # Check for NaN values and clean data
    if df.isna().any().any():
        print("⚠️ Data contains NaN values, cleaning up...")
        df = df.dropna()

    # Ensure no NaN after scaling
    X_scaled = feature_scaler.fit_transform(df[["Hour", "Day", "DayOfYear", "Weekday/Weekend", "Holiday", "Special Event", "Route"]])
    y_scaled = target_scaler.fit_transform(df["Passenger Count"].values.reshape(-1, 1))

    if np.isnan(X_scaled).any() or np.isnan(y_scaled).any():
        print("⚠️ NaN values found after scaling!")
        X_scaled = np.nan_to_num(X_scaled)  # Replace NaNs with zeros if necessary
        y_scaled = np.nan_to_num(y_scaled)  # Replace NaNs with zeros if necessary

    # Check the range of target values
    print("Target range:", np.min(y_scaled), np.max(y_scaled))

    # Create sequences
    X_seq, y_seq = create_sequences(X_scaled, y_scaled)

    # Build and train the model
    new_model = Sequential([
        tf.keras.Input(shape=(X_seq.shape[1], X_seq.shape[2])),  # First layer input shape
        LSTM(64, activation='relu', return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])

    new_model.compile(optimizer='adam', loss='mse')
    new_model.fit(X_seq, y_seq, epochs=10, batch_size=16, verbose=1)

    # Save the retrained model
    new_model.save("bus_ridership_model.keras")
    print("✅ Model retrained and saved.")

    # Wait 1 hour before next update
    print("🕒 Sleeping for 1 hour...")
    time.sleep(3600)
