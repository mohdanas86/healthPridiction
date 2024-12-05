# # Import libraries
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# import pickle

# # Load dataset (replace with your dataset path)
# data = pd.read_csv("./diabetes.csv")

# # Select features and target
# X = data[['Age', 'BMI', 'Glucose']]  # Example columns
# y = data['Outcome']  # 0 = No Diabetes, 1 = Diabetes

# # Split dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train model
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # Test accuracy
# y_pred = model.predict(X_test)
# print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# # Save model
# with open("health_model.pkl", "wb") as f:
#     pickle.dump(model, f)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load dataset (replace with your dataset path)
data = pd.read_csv("./diabetes.csv")  # Ensure this file exists in the current directory

# Select features and target
X = data[['Age', 'BMI', 'Glucose']]  # Adjust these based on your dataset
y = data['Outcome']  # 0 = No Diabetes, 1 = Diabetes

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model accuracy
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save the trained model as 'health_risk_model.pkl'
with open("health_risk_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as 'health_risk_model.pkl'")

