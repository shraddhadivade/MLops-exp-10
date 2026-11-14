
from flask import Flask, jsonify, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# --------------------------------
# 1. Create Sample Dataset
# --------------------------------

data = {
    'Study_Hours': [2, 4, 5, 6, 8, 9, 3, 7, 1, 10,
                    4, 6, 8, 5, 9, 2, 7, 3, 10, 6],

    'Attendance': [50, 60, 70, 80, 90, 95, 55, 85, 30, 98,
                   65, 75, 92, 72, 96, 40, 88, 58, 99, 82],

    'Result': [0, 0, 1, 1, 1, 1, 0, 1, 0, 1,
               0, 1, 1, 1, 1, 0, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

# --------------------------------
# 2. Prepare Features and Target
# --------------------------------

X = df[['Study_Hours', 'Attendance']]
y = df['Result']

# --------------------------------
# 3. Train Random Forest Model
# --------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

print("Random Forest Model Trained Successfully")

# --------------------------------
# 4. Home API
# --------------------------------

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Student Result Prediction API",
        "status": "running"
    }), 200

# --------------------------------
# 5. API Health Check
# --------------------------------

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model": "Random Forest",
        "message": "API is working successfully"
    }), 200

# --------------------------------
# 6. ML Prediction API
# --------------------------------

@app.route('/predict', methods=['GET', 'POST'])
def predict():

    try:
        # Default prediction values
        study_hours = 6
        attendance = 85

        # Read custom JSON input for POST requests
        if request.method == 'POST':
            input_data = request.get_json(silent=True) or {}

            study_hours = float(
                input_data.get('Study_Hours', study_hours)
            )

            attendance = float(
                input_data.get('Attendance', attendance)
            )

        # Validate input
        if study_hours < 0 or attendance < 0 or attendance > 100:
            return jsonify({
                "error": "Invalid input values"
            }), 400

        # Prepare input for prediction
        student = pd.DataFrame({
            'Study_Hours': [study_hours],
            'Attendance': [attendance]
        })

        # Predict student result
        prediction = model.predict(student)

        result = "Pass" if prediction[0] == 1 else "Fail"

        return jsonify({
            "status": "success",
            "Study_Hours": study_hours,
            "Attendance": attendance,
            "prediction": result
        }), 200

    except (ValueError, TypeError):
        return jsonify({
            "error": "Please provide valid numeric inputs"
        }), 400

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

# --------------------------------
# 7. Run Flask Application
# --------------------------------

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False
    )
