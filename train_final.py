import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load cleaned dataset for feature extraction
df = pd.read_csv("coimbatore_bus_ridership_cleaned.csv")

# Load original dataset for correct target scaling
df_original = pd.read_csv("coimbatore_bus_ridership.csv")

# Convert Timestamp to datetime format
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df["DayOfYear"] = df["Timestamp"].dt.dayofyear
df["Hour"] = df["Timestamp"].dt.hour

# Convert categorical column
df["Weekday/Weekend"] = df["Weekday/Weekend"].map({"Weekday": 0, "Weekend": 1})

# Select features and target
features = ["Hour", "Day", "DayOfYear", "Weekday/Weekend", "Holiday", "Special Event", "Route"]
target = "Passenger Count"

# Normalize feature data using MinMaxScaler
feature_scaler = MinMaxScaler()
X_scaled = feature_scaler.fit_transform(df[features])

# Normalize target variable using the original (uncleaned) dataset
target_scaler = MinMaxScaler()
y_scaled = target_scaler.fit_transform(df_original[target].values.reshape(-1, 1))

# Save scalers for later use in prediction
joblib.dump(feature_scaler, "feature_scaler.pkl")
joblib.dump(target_scaler, "target_scaler.pkl")

# Prepare sequences for LSTM input
def create_sequences(X, y, time_steps=3):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i: i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y_scaled)

# Split into train and test sets
train_size = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:train_size], X_seq[train_size:]
y_train, y_test = y_seq[:train_size], y_seq[train_size:]

# Build LSTM Model
model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

# Compile Model
model.compile(optimizer='adam', loss='mse')

# Train Model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=16)

# Save Model
model.save("bus_ridership_model.keras")
print("✅ Model training complete and saved as 'bus_ridership_model.keras'")
