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
2. Exploratory Data Analysis (EDA)
- Analysed churn distribution and class imbalance
- Examined relationships between churn and key features (tenure, contract type, charges)
3. Feature Engineering (PySpark ML Pipeline)
- Categorical encoding (StringIndexer + OneHotEncoder)
- Feature vector assembly (VectorAssembler)
- Scalable pipeline for production-style processing
4. Model Training
- Random Forest (baseline)
- XGBoost with hyperparameter tuning (GridSearchCV)
- Class imbalance handled using scale_pos_weight
5. Advanced Model Evaluation
- Fine threshold tuning (0.01 granularity)
- Metrics:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - ROC-AUC
   - PR-AUC (critical for imbalanced data)
- Fair comparison by evaluating both models at optimal thresholds
6. Explainability
- SHAP used to interpret model predictions
- Feature importance derived from actual engineered feature names

---

## Model Results (Tuned Thresholds)

| Model                 | Accuracy | Precision | Recall   | F1       | ROC AUC | PR AUC    |
| --------------------- | -------- | --------- | -------- | -------- | ------- | --------- |
| Random Forest (Tuned) | 0.80     | **0.61**  | 0.70     | **0.65** | 0.856   | 0.677     |
| XGBoost (Tuned)       | 0.77     | 0.54      | **0.82** | 0.65     | 0.856   | **0.684** |


---

## Key Insights
- Customers with low tenure are more likely to churn
- Month-to-month contracts significantly increase churn risk
- Higher monthly charges are associated with higher churn
- Dataset is imbalanced, requiring specialised evaluation (PR-AUC, Recall)

---

## Model Behaviour
- Random Forest
   - Higher accuracy and precision
   - More conservative predictions (fewer false positives)
- XGBoost
   - Significantly higher recall (0.82)
   - Better at identifying churners
   - Slightly higher PR-AUC → better minority class performance

---

## Business Impact

In telecom churn prediction, the cost of missing a churner (false negative) is significantly higher than incorrectly targeting a non-churner (false positive).

- Random Forest
   - Provides stable predictions
   - Reduces unnecessary retention actions
   - However, misses a larger number of potential churners
- XGBoost
   - Detects a higher proportion of churners (~82% recall)
   - Enables earlier and more effective retention strategies
   - Accepts a higher number of false positives as a trade-off
     
---
## Final Recommendation:
XGBoost is the preferred model for this use case, as it aligns better with business objectives by maximising churn detection and reducing revenue loss.

---

## Dataset

Download from:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---

## Author

Suresh Manawadu
MSc Data Science, Coventry University

## Related Project: Dashboard

This machine learning pipeline is connected to an interactive Power BI dashboard that visualises churn predictions and business insights.

View Dashboard Repository:  
https://github.com/manawadus/your-dashboard-repo-name

### Dashboard Features
- Customer churn risk segmentation
- KPI monitoring (churn rate, retention trends)
- High-risk customer identification
- Business insights for decision-making

This demonstrates how machine learning outputs are transformed into actionable business intelligence.
