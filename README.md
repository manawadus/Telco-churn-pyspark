# Telecom Customer Churn Prediction (PySpark + XGBoost)

This project develops a machine learning pipeline to predict customer churn in a telecom dataset using PySpark. It includes exploratory data analysis (EDA), feature engineering, model comparison (Random Forest vs XGBoost), and model explainability using SHAP.

The project simulates a real world telecom analytics pipeline, combining large-scale data processing, machine learning, and business focused insights.

# 📊 Telecom Customer Churn Prediction using PySpark & XGBoost

## 🚀 Overview

This project predicts customer churn in a telecom company using PySpark and machine learning techniques. It simulates a real-world analytics workflow including data preprocessing, feature engineering, model building, and explainability.

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

* PySpark
* Pandas
* Scikit-learn
* XGBoost
* SHAP (Explainable AI)
* Matplotlib / Seaborn

---

## Key Steps

1. Data Cleaning & Preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Model Training (Random Forest, XGBoost)
5. Model Evaluation (Accuracy, F1, ROC)
6. Explainability using SHAP

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
MSc Data Science (Distinction), Coventry University
