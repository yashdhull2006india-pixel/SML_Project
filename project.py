import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load dataset
df = pd.read_csv("Semester_Risk_Predictor_Dataset.csv")

# 2. Inspect and prepare data
print(df.head())
print(df.info())
print(df.isnull().sum())

# 3. Visualize academic indicators
sns.scatterplot(
    data=df,
    x="attendance_pct",
    y="current_sem_sgpa",
    hue="risk_level"
)
plt.title("Attendance vs Semester SGPA")
plt.show()

# 4. Select input features and target
features = [
    "attendance_pct",
    "study_hours_per_day",
    "assignment_completion_pct",
    "internal_marks_out_of_40",
    "previous_sgpa",
    "quiz_average_pct",
    "sleep_hours_per_day",
    "backlogs"
]

X = df[features]
y = df["risk_level"]

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Train classification model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# 7. Predict risk categories
y_pred = model.predict(X_test)

# 8. Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

sns.heatmap(
    confusion_matrix(
        y_test, y_pred,
        labels=["LOW", "MEDIUM", "HIGH"]
    ),
    annot=True,
    fmt="d",
    xticklabels=["LOW", "MEDIUM", "HIGH"],
    yticklabels=["LOW", "MEDIUM", "HIGH"]
)
plt.xlabel("Predicted Risk")
plt.ylabel("Actual Risk")
plt.title("Confusion Matrix")
plt.show()

# 9. Predict risk for a sample student
sample = pd.DataFrame([{
    "attendance_pct": 72,
    "study_hours_per_day": 2.5,
    "assignment_completion_pct": 80,
    "internal_marks_out_of_40": 24,
    "previous_sgpa": 7.1,
    "quiz_average_pct": 65,
    "sleep_hours_per_day": 6.5,
    "backlogs": 1
}])

print("Predicted Academic Risk:", model.predict(sample)[0])