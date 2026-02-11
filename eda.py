import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda():
    os.makedirs("outputs/plots", exist_ok=True)

    df = pd.read_csv("outputs/cleaned_data.csv")

    print(df.describe())

    # Histogram
    plt.figure()
    sns.histplot(df["ARRIVAL_DELAY"], bins=50)
    plt.title("Arrival Delay Distribution")
    plt.savefig("outputs/plots/arrival_delay_hist.png")

    # Boxplot
    plt.figure()
    sns.boxplot(x=df["ARRIVAL_DELAY"])
    plt.title("Arrival Delay Boxplot")
    plt.savefig("outputs/plots/arrival_delay_box.png")

    # Correlation heatmap
    plt.figure()
    sns.heatmap(df.corr(), annot=True)
    plt.title("Correlation Heatmap")
    plt.savefig("outputs/plots/correlation.png")

    print("EDA Completed!")

if __name__ == "__main__":
    perform_eda()
