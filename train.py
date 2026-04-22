from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(max_depth=3,criterion="entropy",random_state=42)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

with open("model_v1.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved")
print("this is version 2")