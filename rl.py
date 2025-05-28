import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import gym
import random
from tensorflow.keras.models import load_model
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder

# Load trained LSTM model
lstm_model = load_model("bus_ridership_model.keras")
feature_scaler = joblib.load("feature_scaler.pkl")
target_scaler = joblib.load("target_scaler.pkl")

# Load historical ridership data
df = pd.read_csv("coimbatore_bus_ridership.csv")

# Convert Timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Extract numerical features
df['DayOfYear'] = df['Timestamp'].dt.dayofyear

# List of categorical columns to encode
categorical_cols = ['Day', 'Weekday/Weekend', 'Holiday', 'Special Event', 'Route']

# Encode categorical columns using LabelEncoder
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # Convert to numerical labels
    label_encoders[col] = le  # Store encoder for future use

from sklearn.preprocessing import MinMaxScaler

feature_scaler = MinMaxScaler()
df_scaled = feature_scaler.fit_transform(df[['Hour', 'Day', 'DayOfYear', 'Weekday/Weekend', 'Holiday', 'Special Event', 'Route']]).copy()


# Define RL environment
class BusSchedulingEnv(gym.Env):
    def __init__(self):
        super(BusSchedulingEnv, self).__init__()
    
        self.routes = df['Route'].unique()
        self.action_space = gym.spaces.Discrete(len(self.routes) * 2)  # Increase/decrease frequency per route
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32)  # Scaled state
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return self._get_state()

    def step(self, action):
        if self.current_step >= len(df) - 1:  # Ensure we don't go out of bounds
            return np.zeros(self.observation_space.shape), 0, True, {}  

        route_idx = action // 2
        change = 1 if action % 2 == 0 else -1  # Increase or decrease bus frequency
        reward = self._calculate_reward(route_idx, change)
        
        self.current_step += 1
        done = self.current_step >= len(df) - 1  # End episode if we reach the last row

        return self._get_state(), reward, done, {}



    def _get_state(self):
        if self.current_step >= len(df):  # Prevent out-of-bounds error
            return np.zeros(self.observation_space.shape)  

        sample = df.iloc[self.current_step][['Hour', 'Day', 'DayOfYear', 'Weekday/Weekend', 'Holiday', 'Special Event', 'Route']]
        
        # Convert sample to DataFrame
        sample_df = pd.DataFrame([sample])

        # Encode categorical features safely
        for col in categorical_cols:
            if sample_df[col].values[0] not in label_encoders[col].classes_:
                label_encoders[col].classes_ = np.append(label_encoders[col].classes_, sample_df[col].values[0])
            sample_df[col] = label_encoders[col].transform(sample_df[col])

        # Scale the sample
        sample_scaled = feature_scaler.transform(sample_df)
        return sample_scaled[0]




    def _calculate_reward(self, route_idx, change):
        # Convert real-time input to DataFrame
        real_time_input_df = pd.DataFrame([df.iloc[self.current_step][['Hour', 'Day', 'DayOfYear', 'Weekday/Weekend', 'Holiday', 'Special Event', 'Route']]])
        
        # Ensure categorical encoding
        for col in categorical_cols:
            if real_time_input_df[col].values[0] not in label_encoders[col].classes_:
                label_encoders[col].classes_ = np.append(label_encoders[col].classes_, real_time_input_df[col].values[0])
            real_time_input_df[col] = label_encoders[col].transform(real_time_input_df[col])

        # Scale features
        X_real_time = feature_scaler.transform(real_time_input_df)
        print("Scaled Input:", X_real_time)

        # Reshape to match LSTM input (1, 3, 7)
        X_real_time_seq = np.tile(X_real_time, (3, 1)).reshape(1, 3, X_real_time.shape[1])
        print("LSTM Input Shape:", X_real_time_seq.shape)

        # Predict passenger demand
        predicted_demand = target_scaler.inverse_transform(lstm_model.predict(X_real_time_seq))[0][0]
        print(lstm_model.summary)
        # 🔍 Debug: Print predictions  
        print(f"Predicted demand: {predicted_demand}, Action change: {change}")

        # Reward logic  
        base_penalty = -1  # Small penalty per step to encourage faster solutions  

        if predicted_demand > 50 and change > 0:  
            return 10  # ✅ Reward for adding a bus when demand is high  
        elif predicted_demand < 20 and change < 0:  
            return 10  # ✅ Reward for reducing a bus when demand is low  
        elif predicted_demand > 50 and change < 0:  
            return -10  # ❌ Penalty for removing buses when demand is high  
        elif predicted_demand < 20 and change > 0:  
            return -10  # ❌ Penalty for adding buses when demand is low  
        else:  
            return base_penalty  # ✅ Small penalty for neutral situations  



# Define DQN Model
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

# Training DQN Model
def train_dqn(env, episodes=500, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.99, lr=0.001):
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    model = DQN(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    memory = deque(maxlen=2000)
    
    for episode in range(episodes):
        state = env.reset()
        state = torch.FloatTensor(state)
        total_reward = 0
        done = False
        step_count = 0  # Track steps per episode
        
        while not done:
            step_count += 1
            if step_count > 200:  # Emergency exit to prevent infinite loop
                print(f"⚠️ Stopping Episode {episode} early (stuck in infinite loop)")
                break  

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = model(state)
                action = torch.argmax(q_values).item()
            
            next_state, reward, done, _ = env.step(action)
            next_state = torch.FloatTensor(next_state)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
        
            print(f"✅ Episode {episode} finished in {step_count} steps, Total Reward: {total_reward}")

            
            if len(memory) > 32:
                batch = random.sample(memory, 32)
                for s, a, r, s_next, d in batch:
                    target = r + (gamma * torch.max(model(s_next)).item() * (1 - d))    
                    q_values = model(s)
                    loss = criterion(q_values[a], torch.tensor(target))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if episode % 50 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")
    
    torch.save(model.state_dict(), "dqn_bus_scheduler.pth")
    print("✅ RL Model Training Complete and Saved!")

# Train the model
env = BusSchedulingEnv()
train_dqn(env)