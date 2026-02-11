import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

def run_clustering():
    os.makedirs("outputs/plots", exist_ok=True)

    df = pd.read_csv("outputs/cleaned_data.csv")

    X = df[["DEPARTURE_DELAY", "DISTANCE"]]

    kmeans = KMeans(n_clusters=3, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X)

    plt.figure()
    plt.scatter(df["DEPARTURE_DELAY"], df["DISTANCE"], c=df["Cluster"])
    plt.xlabel("Departure Delay")
    plt.ylabel("Distance")
    plt.title("Flight Delay Clusters")
    plt.savefig("outputs/plots/clusters.png")

    print("Clustering Completed")

if __name__ == "__main__":
    run_clustering()
