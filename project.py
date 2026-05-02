import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Create dataset
data = {
    'study_hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'attendance': [50, 60, 65, 70, 75, 80, 85, 90],
    'previous_marks': [40, 45, 50, 55, 60, 65, 70, 80],
    'performance': [0, 0, 0, 1, 1, 1, 1, 1]  # 0 = Poor, 1 = Good
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features and labels
X = df[['study_hours', 'attendance', 'previous_marks']]
y = df['performance']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Take user input
study_hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance (%): "))
previous_marks = float(input("Enter previous marks: "))

# Predict
prediction = model.predict([[study_hours, attendance, previous_marks]])

# Output result
if prediction[0] == 1:
    print("Predicted Performance: Good")
else:
    print("Predicted Performance: Poor")
