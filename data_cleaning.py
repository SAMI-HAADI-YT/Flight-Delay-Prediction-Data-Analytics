import pandas as pd

def clean_data():
    print("Loading dataset...")

    df = pd.read_csv("data/flights.csv", low_memory=False)

    print("Original Shape:", df.shape)

    # Keep only useful columns
    df = df[[
        "DEPARTURE_DELAY",
        "ARRIVAL_DELAY",
        "DISTANCE",
        "WEATHER_DELAY",
        "AIRLINE_DELAY",
        "AIR_SYSTEM_DELAY",
        "LATE_AIRCRAFT_DELAY",
        "CANCELLED"
    ]]

    # Remove cancelled flights
    df = df[df["CANCELLED"] == 0]

    # Drop missing values
    df = df.dropna()

    # Create Delay Category (Classification target)
    df["DELAYED"] = df["ARRIVAL_DELAY"].apply(lambda x: 1 if x > 15 else 0)

    # Sample for faster ML training
    df = df.sample(n=200000, random_state=42)

    df.to_csv("outputs/cleaned_data.csv", index=False)

    print("Cleaned Shape:", df.shape)
    print("Data cleaning completed successfully!")

if __name__ == "__main__":
    clean_data()
