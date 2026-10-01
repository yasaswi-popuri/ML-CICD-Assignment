from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
def train_and_evaluate():
    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Model: Random Forest Classifier")
    print(f"Dataset: Iris")
    print(f"Model Accuracy: {acc:.4f}")
    with open("accuracy.txt", "w") as f:
        f.write(f"{acc * 100:.2f}% ({acc:.4f})")
    return acc
if __name__ == "__main__":
    train_and_evaluate()
