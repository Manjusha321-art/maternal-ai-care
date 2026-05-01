import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
data = pd.read_csv("data.csv")

# Convert target
data['RiskLevel'] = data['RiskLevel'].map({
    'low risk': 0,
    'mid risk': 1,
    'high risk': 1
})

X = data.drop('RiskLevel', axis=1)
y = data['RiskLevel']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("Model saved successfully")