# train.py
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Load and split the data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate the model
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
print(f"Accuracy: {accuracy:.2f}")

# Save the model to file
os.makedirs("output", exist_ok=True)
joblib.dump(model, "output/model.pkl")
