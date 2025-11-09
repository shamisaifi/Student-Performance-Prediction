import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("student-data.csv", sep=";")

le = LabelEncoder()
data['Gender'] = le.fit_transform(data['sex'])

data['StudyHours'] = data['studytime']

data['PreviousScores'] = (data['G1'] + data['G2']) / 2

data['Participation'] = data['activities'].map({"yes": "High", "no": "Low"})

data['Performance'] = np.where(data['G3'] >= 10, "Pass", "Fail")

# Encode categorical columns
data['Participation'] = le.fit_transform(data['Participation'])  
data['Performance'] = le.fit_transform(data['Performance'])

# Select features
X = data[['Gender', 'StudyHours', 'PreviousScores', 'Participation']]
y = data['Performance']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)


results = pd.DataFrame(X_test, columns=['Gender', 'StudyHours', 'PreviousScores', 'Participation'])
results['Actual'] = y_test.values
results['Predicted'] = y_pred

results['Actual'] = results['Actual'].map({0: "Fail", 1: "Pass"})
results['Predicted'] = results['Predicted'].map({0: "Fail", 1: "Pass"})

print("\n🔹 Sample Predictions:\n")
print(results.head(10))



accuracy = accuracy_score(y_test, y_pred)

print("🔹 Model Accuracy:", accuracy)
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))

# Feature importance visualization
plt.figure(figsize=(6, 4))
sns.barplot(x=model.feature_importances_, y=X.columns)
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in Student Performance Prediction")
plt.show()
