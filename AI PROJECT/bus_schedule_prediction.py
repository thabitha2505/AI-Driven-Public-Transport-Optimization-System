import joblib
import numpy as np
import pandas as pd
import torch
from tensorflow.keras.models import load_model
import torch.optim as optim
from collections import deque

# Load trained LSTM model for passenger count prediction
lstm_model = load_model("bus_ridership_model.keras")

# Load DQN model
class DQN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 20)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Load the saved DQN model
dqn_model = DQN(input_dim=7, output_dim=20)  # Assuming 7 features, 14 actions (7 routes, increase or decrease frequency)
dqn_model.load_state_dict(torch.load("dqn_bus_scheduler.pth"))
dqn_model.eval()  # Set model to evaluation mode

# Load scalers
feature_scaler = joblib.load("feature_scaler.pkl")
target_scaler = joblib.load("target_scaler.pkl")  # Load the target scaler

# Example real-time input
real_time_input = pd.DataFrame({
    "Timestamp": ["2025-04-01 14:00:00"],
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

# Predict passenger count using LSTM model
predicted_passenger_count_scaled = lstm_model.predict(X_real_time_seq)

# Correct inverse scaling to get real passenger count
real_passenger_count = target_scaler.inverse_transform(predicted_passenger_count_scaled.reshape(-1, 1))

# Print final passenger count prediction
print(f"✅ Real Passenger Count Prediction: {real_passenger_count[0][0]:.2f}")

# Use DQN model to decide bus scheduling actions
state = torch.FloatTensor(X_real_time.flatten()).unsqueeze(0)  # Flatten the state for input to DQN model
action_values = dqn_model(state)  # Get Q-values for all possible actions
print(action_values)
# Choose action with highest Q-value (greedy policy)
action = torch.argmax(action_values).item()

# Map action to route and frequency change
route_idx = action // 2
frequency_change = 1 if action % 2 == 0 else -1  # 0 for increase, 1 for decrease

# Output the bus scheduling decision
print(f"🚌 Bus Scheduling Decision:")
print(f"Route: {route_idx} (Routes are 0-indexed)")
print(f"Action: {'Increase' if frequency_change == 1 else 'Decrease'} bus frequency")
