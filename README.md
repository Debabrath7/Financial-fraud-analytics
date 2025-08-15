Fraud Detection Project

Detect fraudulent financial transactions using machine learning on large-scale datasets.

Project Overview

Fraudulent transactions pose a huge financial risk to companies. This project focuses on:

Cleaning and preprocessing a 6M+ transaction dataset

Analyzing transaction patterns to detect anomalies

Building predictive models to classify fraudulent transactions

Evaluating models using metrics suited for imbalanced data

Dataset

Size: 6,000,000+ rows

Columns: Transaction details including amount, type, timestamp, user info

Target: isFraud (0 = legitimate, 1 = fraudulent)


Installation & Setup

Clone the repository:

git clone https://github.com/<your-username>/fraud-detection-project.git
cd fraud-detection-project


Create a virtual environment:

python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows


Install dependencies:

pip install -r requirements.txt


Place your dataset in the data/ folder.

Usage

Exploratory Data Analysis & Preprocessing:

jupyter notebook notebooks/fraud_detection.ipynb


Train the Model:

python src/train_model.py


Load and Predict:

import joblib
model = joblib.load("models/fraud_model.pkl")
predictions = model.predict(new_transaction_data)

Key Features

Transaction amount, type, and timestamp

User behavior and anomaly patterns

Feature engineering for predictive modeling

Scalable to millions of transactions

Model

Algorithm: Random Forest Classifier

Evaluation Metrics:

Accuracy

Precision, Recall, F1-score

ROC-AUC (for imbalanced data)

Results
Metric	Score
Accuracy	0.98
Precision	0.92
Recall	0.87
F1-Score	0.89
ROC-AUC	0.95

The model balances detecting fraud while minimizing false positives.

Future Work

Implement deep learning for improved accuracy

Real-time transaction fraud detection

Automated feature selection and hyperparameter tuning

Deploy as API for financial applications

References

Scikit-learn documentation: https://scikit-learn.org

Fraud detection research papers and best practices

License

MIT License
