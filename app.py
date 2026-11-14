
from flask import Flask, jsonify, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# 1. Sample Dataset
data = {
    "Study_Hours": [
        2, 4, 5, 6, 8, 9, 3, 7, 1, 10,
        4, 6, 8, 5, 9, 2, 7, 3, 10, 6
    ],
    "Attendance": [
        50, 60, 70, 80, 90, 95, 55, 85, 30, 98,
        65, 75, 92, 72, 96, 40, 88, 58, 99, 82
    ],
    "Result": [
        0, 0, 1, 1, 1, 1, 0, 1, 0, 1,
        0, 1, 1, 1, 1, 0, 1, 0, 1, 1
    ]
}

df = pd.DataFrame(data)

# 2. Features and Target
X = df[["Study_Hours", "Attendance"]]
y = df["Result"]

# 3. Train Random Forest Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

print("Random Forest Model Trained Successfully")

# 4. Home Route
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Student Result Prediction API",
        "status": "running"
    }), 200

# 5. Health Check
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model": "Random Forest"
    }), 200

# 6. Prediction API
@app.route("/predict", methods=["GET", "POST"])
def predict():

    try:
        study_hours = 6
        attendance = 85

        if request.method == "POST":
            input_data = request.get_json(silent=True) or {}

            study_hours = float(
                input_data.get("Study_Hours", 6)
            )

            attendance = float(
                input_data.get("Attendance", 85)
            )

        if study_hours < 0 or attendance < 0 or attendance > 100:
            return jsonify({
                "status": "error",
                "message": "Invalid input values"
            }), 400

        student = pd.DataFrame({
            "Study_Hours": [study_hours],
            "Attendance": [attendance]
        })

        prediction = model.predict(student)[0]

        result = "Pass" if prediction == 1 else "Fail"

        return jsonify({
            "status": "success",
            "Study_Hours": study_hours,
            "Attendance": attendance,
            "prediction": result
        }), 200

    except (ValueError, TypeError):
        return jsonify({
            "status": "error",
            "message": "Invalid numeric input"
        }), 400

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# 7. Run Application
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
