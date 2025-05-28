import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Load dataset
df = pd.read_csv("coimbatore_bus_ridership.csv")

# Step 1: Handle Missing Values
df.dropna(inplace=True)  # Remove missing values

# Step 2: Convert Categorical Features
df["Holiday"] = df["Holiday"].map({"Yes": 1, "No": 0})
df["Special Event"] = df["Special Event"].map({"Yes": 1, "No": 0})

# Convert 'Day' to numerical format
day_mapping = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, 
    "Friday": 4, "Saturday": 5, "Sunday": 6
}
df["Day"] = df["Day"].map(day_mapping)

# Encode Route using Label Encoding
le = LabelEncoder()
df["Route"] = le.fit_transform(df["Route"])

# Step 3: Normalize Passenger Count
scaler = MinMaxScaler()
df["Passenger Count"] = scaler.fit_transform(df[["Passenger Count"]])

# Save cleaned dataset
df.to_csv("coimbatore_bus_ridership_cleaned.csv", index=False)

print("Data cleaning and preprocessing completed!")
