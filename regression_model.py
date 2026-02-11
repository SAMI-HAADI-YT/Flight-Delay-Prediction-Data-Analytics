import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def run_regression():
    df = pd.read_csv("outputs/cleaned_data.csv")

    X = df[[
        "DEPARTURE_DELAY",
        "DISTANCE",
        "WEATHER_DELAY",
        "AIRLINE_DELAY",
        "AIR_SYSTEM_DELAY",
        "LATE_AIRCRAFT_DELAY"
    ]]

    y = df["ARRIVAL_DELAY"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    with open("outputs/model_results.txt", "w") as f:
        f.write("Linear Regression Results\n")
        f.write(f"MSE: {mse}\n")
        f.write(f"R2 Score: {r2}\n")

    print("Regression Completed")
    print("MSE:", mse)
    print("R2:", r2)

if __name__ == "__main__":
    run_regression()
