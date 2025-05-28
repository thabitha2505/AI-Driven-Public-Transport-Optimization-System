import streamlit as st
import pandas as pd
import numpy as np
import datetime
import joblib
import torch
import torch.nn as nn
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder


# --- Fake Auth ---
users = {"admin": "password123"}
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    
    st.title("🔐 Login to Bus Scheduler Dashboard")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if users.get(username) == password:
            st.session_state.logged_in = True
        else:
            st.error("Invalid username or password")
    st.stop()

# --- Streamlit Config ---
st.set_page_config(page_title="Bus Scheduling Dashboard", layout="wide")
st.title("🚌 Coimbatore Bus Demand & Dynamic Scheduling")

# --- Load Models & Scalers ---
lstm_model = load_model("bus_ridership_model.keras")
feature_scaler = joblib.load("feature_scaler.pkl")
target_scaler = joblib.load("target_scaler.pkl")

# --- DQN Model Definition ---
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# --- Load and preprocess ridership data ---
df = pd.read_csv("coimbatore_bus_ridership.csv")
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['DayOfYear'] = df['Timestamp'].dt.dayofyear

categorical_cols = ['Day', 'Weekday/Weekend', 'Holiday', 'Special Event', 'Route']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Mapping routes to actual names
route_names = [
    "Ukkadam ↔ Pollachi",
    "Singanallur ↔ Peelamedu",
    "Ukkadam ↔ Kuniamuthur",
    "Gandhipuram ↔ Saravanampatti",
    "Ukkadam ↔ Townhall",
    "Gandhipuram ↔ Ukkadam",
    "Gandhipuram ↔ Kovaipudur",
    "Gandhipuram ↔ Peelamedu",
    "Gandhipuram ↔ Singanallur",
    "Gandhipuram ↔ Railway Station"
]

routes = list(range(10))
action_space = len(routes) * 2

# Route-specific bus capacities and current bus counts
bus_info = {
    0: {"capacity": 40, "buses": 2},
    1: {"capacity": 35, "buses": 1},
    2: {"capacity": 40, "buses": 2},
    3: {"capacity": 30, "buses": 2},
    4: {"capacity": 25, "buses": 2},
    5: {"capacity": 30, "buses": 2},
    6: {"capacity": 40, "buses": 2},
    7: {"capacity": 25, "buses": 2},
    8: {"capacity": 40, "buses": 1},
    9: {"capacity": 25, "buses": 1}
}

# --- Load trained DQN model ---
dqn_model = DQN(input_dim=7, output_dim=action_space)
dqn_model.load_state_dict(torch.load("dqn_bus_scheduler.pth"))
dqn_model.eval()

# --- Load holidays and special events ---
holiday_df = pd.read_csv("full_holidays_2025.csv", parse_dates=["Date"])
holiday_df["Date"] = holiday_df["Date"].dt.date
holiday_dates = set(holiday_df["Date"])

event_df = pd.read_csv("special_events_2025.csv", parse_dates=["Date"])
event_df["Date"] = event_df["Date"].dt.date
event_dates = set(event_df["Date"])

# --- Predict demand using LSTM ---
def predict_passenger_count(row):
    features = ['Hour', 'Day', 'DayOfYear', 'Weekday/Weekend', 'Holiday', 'Special Event', 'Route']
    row_df = pd.DataFrame([[row[feature] for feature in features]], columns=features)
    X_scaled = feature_scaler.transform(row_df)
    X_seq = np.tile(X_scaled, (3, 1)).reshape(1, 3, len(features))
    y_pred_scaled = lstm_model.predict(X_seq)
    return target_scaler.inverse_transform(y_pred_scaled)[0][0]

# --- RL action from DQN ---
def get_rl_action(scaled_state):
    state_tensor = torch.FloatTensor(scaled_state)
    q_values = dqn_model(state_tensor)
    action_idx = torch.argmax(q_values).item()
    return action_idx

def decode_action(action_idx):
    route_idx = action_idx // 2
    return route_idx

# --- Main Display ---
selected_date = st.date_input("Select Date", datetime.date.today())
selected_hour = st.slider("Select Hour", min_value=0, max_value=23, step=1)
current_time = datetime.datetime.combine(selected_date, datetime.time(selected_hour))

holiday_name = holiday_df.loc[holiday_df["Date"] == current_time.date(), "Holiday Name"]
event_name = event_df.loc[event_df["Date"] == current_time.date(), "Event Name"]

if not holiday_name.empty:
    st.info(f"📅 Holiday: {holiday_name.values[0]}")
if not event_name.empty:
    st.success(f"🎉 Special Event: {event_name.values[0]}")

st.subheader(f"📊 Demand Forecast & RL Scheduling for {current_time.strftime('%Y-%m-%d %H:00')}")

table_data = []

for route in routes:
    row = {
        'Hour': current_time.hour,
        'Day': current_time.weekday(),
        'DayOfYear': current_time.timetuple().tm_yday,
        'Weekday/Weekend': 1 if current_time.weekday() < 5 else 0,
        'Holiday': 1 if current_time.date() in holiday_dates else 0,
        'Special Event': 1 if current_time.date() in event_dates else 0,
        'Route': route
    }

    predicted = predict_passenger_count(row)

    features = ['Hour', 'Day', 'DayOfYear', 'Weekday/Weekend', 'Holiday', 'Special Event', 'Route']
    row_df = pd.DataFrame([[row[feature] for feature in features]], columns=features)
    X_scaled = feature_scaler.transform(row_df)

    action_idx = get_rl_action(X_scaled[0])
    route_rl = decode_action(action_idx)

    bus_capacity = bus_info[route]['capacity']
    current_buses = bus_info[route]['buses']
    current_total_capacity = bus_capacity * current_buses
    adjusted_capacity = current_total_capacity + 30

    if adjusted_capacity < predicted:
        rl_decision = "⬆️ Increase Buses"
    elif adjusted_capacity > predicted + 30:
        rl_decision = "⬇️ Decrease Buses"
    else:
        rl_decision = "✅ Remain Same"

    table_data.append({
        "Route": route_names[route],
        "Predicted Demand": int(predicted),
        "RL Decision": rl_decision
    })

st.dataframe(pd.DataFrame(table_data), use_container_width=True)
# --- 📈 Historical Demand Trends ---
st.subheader("📈 Historical Demand Trends")
selected_route_name = st.selectbox("Select Route for Trend", route_names)
selected_route_index = route_names.index(selected_route_name)
trend_df = df[df['Route'] == selected_route_index].copy()
trend_df = trend_df.set_index('Timestamp').resample('D').mean(numeric_only=True)
st.line_chart(trend_df['Passenger Count'] if 'Passenger Count' in trend_df else trend_df.iloc[:, 0])



# --- 🧪 Manual Bus Allocation Simulator ---
st.subheader("🧪 Manual Bus Allocation Simulator")
manual_alloc_data = []
for route in routes:
    manual_buses = st.slider(f"{route_names[route]}", 0, 10, bus_info[route]['buses'], key=f"manual_route_{route}")
    capacity = manual_buses * bus_info[route]['capacity']
    predicted_demand = int([row["Predicted Demand"] for row in table_data if row["Route"] == route_names[route]][0])
    status = "✅ Adequate" if capacity >= predicted_demand else "⚠️ Insufficient"
    manual_alloc_data.append({
        "Route": route_names[route],
        "Manual Buses": manual_buses,
        "Total Capacity": capacity,
        "Predicted Demand": predicted_demand,
        "Status": status
    })

st.dataframe(pd.DataFrame(manual_alloc_data), use_container_width=True)

# --- 📄 Downloadable Report ---
st.subheader("📄 Downloadable Reports")
csv = pd.DataFrame(table_data).to_csv(index=False)
st.download_button("Download Current Forecast & Decision (CSV)", data=csv, file_name="bus_forecast_report.csv", mime='text/csv')

# --- 💬 Feedback Mechanism ---
st.subheader("💬 Feedback")
feedback_text = st.text_area("Share your feedback or report an issue:")
if st.button("Submit Feedback"):
    with open("user_feedback.txt", "a") as f:
        f.write(f"{datetime.datetime.now()}: {feedback_text}\n")
    st.success("✅ Feedback submitted. Thank you!")

