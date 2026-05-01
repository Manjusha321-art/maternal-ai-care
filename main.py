import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load dataset
data = pd.read_csv("data.csv")

# Convert target
data['RiskLevel'] = data['RiskLevel'].map({
    'low risk': 0,
    'mid risk': 1,
    'high risk': 1
})

# Features and target
X = data.drop('RiskLevel', axis=1)
y = data['RiskLevel']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# ----------- Prediction Function -----------

def predict_risk(age, sys_bp, dia_bp, bs, temp, heart_rate):
    input_data = [[age, sys_bp, dia_bp, bs, temp, heart_rate]]
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        return "High Risk"
    else:
        return "Low Risk"

# ----------- Test Prediction -----------

result = predict_risk(30, 140, 90, 10.0, 98.0, 80)
print("\nSample Prediction:", result)