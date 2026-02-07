🌫️ Air Quality Index (AQI) Prediction – Jaipur

Machine Learning project to predict daily Air Quality Index (AQI) using pollutant concentration data.

This project focuses on building a reliable regression pipeline using real-world noisy environmental data and evaluating model performance under practical constraints.

📌 Problem Statement

Air pollution varies due to multiple pollutants and seasonal patterns.
The goal is to predict the AQI of a given day from measured pollutant levels.

📂 Dataset

Daily city-wise pollution records

Features include: PM2.5, PM10, NO, NO2, NOx, CO, SO2, O3 etc.

Filtered to Jaipur for city-level modeling.

🧹 Data Preprocessing

Real-world data is messy. The following steps were applied:

Converted Date to datetime

Removed highly sparse / less useful columns (Xylene, Benzene, Toluene, NH3)

Dropped rows missing both PM2.5 and PM10

Filled pollutant gaps using median imputation

Removed rows where AQI was unavailable

Built a city-specific dataset

🧠 Feature / Target

Features (X):
Pollutant concentrations.

Target (y):
AQI value.

Non-numeric and identifier columns like City, Date, AQI bucket were excluded.

✂️ Train Test Strategy

Dataset was split into:

80% training

20% testing

to ensure unbiased evaluation on unseen data.

🤖 Model Used

Random Forest Regressor.

Why?

Handles nonlinear relationships

Works well with mixed distributions

Robust to noise and outliers

📏 Evaluation Metrics

MAE → average prediction error

MSE → penalizes large mistakes

R² → how well variability is explained

Results
MAE ≈ 14
R² ≈ 0.80–0.87


The model captures general AQI movement effectively, while extreme pollution spikes remain harder to predict due to missing external factors like weather and events.

📈 Visualization

Actual vs Predicted AQI comparison shows strong alignment during normal ranges with mild underestimation during extreme peaks.

<img width="1494" height="903" alt="image" src="https://github.com/user-attachments/assets/079ab5ac-6301-4932-91e5-b1eec0aa7e55" />


🚧 Limitations

The model uses only pollutant data.

It does not include:

meteorological parameters

wind patterns

traffic / industrial activity

sudden environmental events

Hence extreme variations may not be fully captured.

🛠️ Tech Stack

Python

Pandas, NumPy

Scikit-learn

Matplotlib / Seaborn

🎯 Key Learning Outcomes

Handling missing data in real datasets

Feature engineering

Regression modeling

Avoiding data leakage

Model evaluation & interpretation
