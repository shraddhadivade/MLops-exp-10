
from flask import Flask, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Sample dataset
data = {
    'Study_Hours': [2, 4, 5, 6, 8, 9, 3, 7],
    'Attendance': [50, 60, 70, 80, 90, 95, 55, 85],
    'Result': [0, 0, 1, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[['Study_Hours', 'Attendance']]
y = df['Result']

# Train Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Home API
@app.route('/')
def home():
    return jsonify({
        "message": "Student Result Prediction API is running",
        "status": "success"
    })

# Health check API
@app.route('/health')
def health():
    return jsonify({
        "status": "healthy"
    })

# Prediction API
@app.route('/predict')
def predict():
    student = pd.DataFrame({
        'Study_Hours': [6],
        'Attendance': [85]
    })

    prediction = model.predict(student)

    result = "Pass" if prediction[0] == 1 else "Fail"

    return jsonify({
        "prediction": result
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
