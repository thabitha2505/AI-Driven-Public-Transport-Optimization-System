# 🚌 AI-Driven-Public-Transport-Optimization-System
Developed an AI-Driven Public Transport Optimization System for Coimbatore that predicts hourly passenger demand on various city routes using a trained LSTM model and recommends real-time bus frequency adjustments using a Reinforcement Learning (RL) agent. The system displays routes, predicted demand, and decisions such as increase, decrease, or retain the same bus frequency for a selected date and hour. The results are fetched in under 300 ms, ensuring a seamless user experience. Historical demand trends can be viewed by selecting a route, and users can download detailed reports for further analysis. Additionally, a feedback mechanism is provided, and all feedback entries are stored in a separate text file for administrative review. Both the LSTM and RL models are trained and saved as .keras files for efficient deployment.

## 🔍 Project Overview

This AI system aims to reduce bus overcrowding and underutilization by forecasting demand and adjusting the frequency of buses dynamically. It helps transport authorities make data-driven decisions by analyzing historical and real-time passenger data.

## ✅ Key Features

- **LSTM-Based Demand Forecasting**: Predicts hourly passenger demand for each route.
- **RL-Based Scheduling (DQN)**: Decides whether to increase, decrease, or maintain bus frequency based on predicted demand.
- **Interactive Website**:
  - Select Date and Hour to view predictions.
  - Displays each route, its predicted demand, and scheduling decision.
  - View **Historical Demand Trends** per route.
  - **Downloadable Reports** and **Feedback** options (stored in text file).
- **Pretrained Models**: Both LSTM and RL models are trained and stored as `.h5` Keras files.
- **Fast Response Time**: Average response time after dateand hour selection is ~263ms (inference + fetch).

## 🚀 Tech Stack

- **Frontend**: Streamlit
- **Backend/AI**: Python, Keras, NumPy, Pandas
- **Models**:
  - LSTM – Passenger demand predictor
  - DQN – RL scheduler for dynamic frequency control
- **Deployment**: Hosted using Streamlit Sharing / Render / Local server

## 📊 Sample Output

| Route                       | Predicted Demand | RL Decision    |
|------------------------------|--------------|-------------------|
| Ukkadam ↔ Pollachi           | 31           | 🔽 Decrease Buses |
| Gandhipuram ↔ Saravanampatti | 32           | 🔽 Increase Buses |
**---**
**---**

## 📝 Feedback and Reports

- Users can give feedback on scheduling decisions, which is saved in plain `.txt` file.
- Reports are downloadable in `.csv` format for offline analysis.

## 📈 Performance

- Average data loading and result rendering time: **~263ms**
- Model inference and fetch time is optimized for real-time use.

