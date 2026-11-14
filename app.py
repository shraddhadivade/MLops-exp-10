
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create sample dataset
data = {
    'Study_Hours': [2, 4, 5, 6, 8, 9, 3, 7],
    'Attendance': [50, 60, 70, 80, 90, 95, 55, 85],
    'Result': [0, 0, 1, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[['Study_Hours', 'Attendance']]
y = df['Result']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Student Result Prediction")
print("Model Accuracy:", accuracy)

# Predict result for a new student
new_student = pd.DataFrame({
    'Study_Hours': [6],
    'Attendance': [85]
})

prediction = model.predict(new_student)

if prediction[0] == 1:
    print("Predicted Result: Pass")
else:
    print("Predicted Result: Fail")
