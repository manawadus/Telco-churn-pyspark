# Telecom Customer Churn Prediction (PySpark + XGBoost)

This project develops a machine learning pipeline to predict customer churn in a telecom dataset using PySpark. It includes large scale data processing, exploratory data analysis (EDA) feature engineering, advanced model evaluation, and model explainability using SHAP.

The solution moves beyond traditional classification by incorporating threshold optimisation and business-aligned evaluation, ensuring the model is suitable for real world decision making.

---

## Project Structure

```
telco-churn-pyspark-xgboost/
│
├── notebooks/
│   └── telco_churn_pyspark_xgboost.ipynb
│
├── data/
│   └── dataset instructions
│
├── README.md
└── requirements.txt
```

---

## Technologies Used

- PySpark (data processing & feature engineering)
- Pandas exploratory data analysis (EDA)
- Scikit-learn (model evaluation & metrics)
- XGBoost (gradient boosting model)
- Random Forest (baseline model)
- SHAP  (Explainable AI)
- Matplotlib / Seaborn (visualisation)

---

## Pipeline Overview
1. Data Cleaning & Preprocessing
- Handled missing values and corrected data types (e.g., TotalCharges)
- Filtered invalid records (e.g., zero tenure)
Exploratory Data Analysis (EDA)
Analysed churn distribution and class imbalance
Examined relationships between churn and key features (tenure, contract type, charges)
Feature Engineering (PySpark ML Pipeline)
Categorical encoding (StringIndexer + OneHotEncoder)
Feature vector assembly (VectorAssembler)
Scalable pipeline for production-style processing
Model Training
Random Forest (baseline)
XGBoost with hyperparameter tuning (GridSearchCV)
Class imbalance handled using scale_pos_weight
Advanced Model Evaluation
Fine threshold tuning (0.01 granularity)
Metrics:
Accuracy
Precision
Recall
F1-score
ROC-AUC
PR-AUC (critical for imbalanced data)
Fair comparison by evaluating both models at optimal thresholds
Explainability
SHAP used to interpret model predictions
Feature importance derived from actual engineered feature names

---

## Key Insights

* Customers with **low tenure** are more likely to churn
* **Month-to-month contracts** significantly increase churn risk
* **Higher monthly charges** are associated with higher churn

---

## Results

| Model         | Accuracy | F1 Score | ROC AUC |
| ------------- | -------- | -------- | ------- |
| Random Forest | 0.80     | 0.79     | 0.86    |
| XGBoost       | 0.76     | 0.64     | 0.86    |

---

## Business Impact

This solution helps telecom companies:

* Identify high-risk customers
* Improve retention strategies
* Make data-driven decisions

---

## 📎 Dataset

Download from:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---

## 👤 Author

Suresh Manawadu
MSc Data Science, Coventry University
