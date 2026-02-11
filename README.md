FlightDelayAnalytics/
│
├── data/
│   └── flights.csv
├── src/
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── regression_model.py
│   ├── classification_model.py
│   ├── clustering.py
│
├── outputs/
│   ├── cleaned_data.csv
│   ├── model_results.txt
│   └── plots/
│
├── requirements.txt
└── README.md



Run the Scripts (In Order)

⚠️ Run from the project root folder (FlightDelayAnalytics)

Step 1 – Data Cleaning
python src/data_cleaning.py


✔ Creates cleaned_flights.csv

Step 2 – Exploratory Data Analysis
python src/eda.py


✔ Generates visualizations
✔ Shows delay patterns

Step 3 – Hypothesis Testing
python src/hypothesis_testing.py


✔ Performs statistical tests

Step 4 – Regression Model
python src/regression_model.py


✔ Predicts Arrival Delay
✔ Displays R² Score & MSE

Step 5 – Classification Model
python src/classification_model.py


✔ Classifies flights as Delayed / On-Time
✔ Shows Accuracy & Confusion Matrix

Step 6 – Clustering
python src/clustering.py


✔ Groups flights into delay patterns using k-Means

🛠 Common Errors & Fixes
❌ ModuleNotFoundError

Install missing libraries:

pip install library-name

❌ FileNotFoundError

Make sure:

flights.csv


is inside the data/ folder.
