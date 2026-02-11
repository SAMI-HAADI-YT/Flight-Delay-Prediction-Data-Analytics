import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def run_classification():
    df = pd.read_csv("outputs/cleaned_data.csv")

    X = df[[
        "DEPARTURE_DELAY",
        "DISTANCE",
        "WEATHER_DELAY",
        "AIRLINE_DELAY",
        "AIR_SYSTEM_DELAY",
        "LATE_AIRCRAFT_DELAY"
    ]]

    y = df["DELAYED"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Logistic Regression
    log_model = LogisticRegression(max_iter=2000)
    log_model.fit(X_train, y_train)
    log_pred = log_model.predict(X_test)
    log_acc = accuracy_score(y_test, log_pred)

    # kNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    knn_acc = accuracy_score(y_test, knn_pred)

    with open("outputs/model_results.txt", "a") as f:
        f.write(f"\nLogistic Regression Accuracy: {log_acc}\n")
        f.write(f"kNN Accuracy: {knn_acc}\n")

    print("Classification Completed")
    print("Logistic Accuracy:", log_acc)
    print("kNN Accuracy:", knn_acc)

if __name__ == "__main__":
    run_classification()
