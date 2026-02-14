# Bank Marketing Campaign - ML Classification Models

## Problem Statement
The goal is to predict whether a client will subscribe to a term deposit (yes/no) based on direct marketing campaigns (phone calls) of a Portuguese banking institution. This is a binary classification problem that helps banks optimize their marketing strategies by identifying potential clients who are more likely to subscribe to term deposits.

## Dataset Description

**Dataset Name:** Bank Marketing Dataset  
**Source:** UCI Machine Learning Repository  
**Link:** https://archive.ics.uci.edu/dataset/222/bank+marketing

### Dataset Characteristics:
- **Number of Instances:** 41,188
- **Number of Features:** 20 (after excluding target variable)
- **Target Variable:** `y` - has the client subscribed to a term deposit? (binary: 'yes', 'no')
- **Class Distribution:** Imbalanced (approximately 11% positive class)

### Feature Description:

**Bank Client Data:**
1. `age` - Age of the client (numeric)
2. `job` - Type of job (categorical)
3. `marital` - Marital status (categorical)
4. `education` - Education level (categorical)
5. `default` - Has credit in default? (categorical)
6. `housing` - Has housing loan? (categorical)
7. `loan` - Has personal loan? (categorical)

**Campaign Data:**
8. `contact` - Contact communication type (categorical)
9. `month` - Last contact month of year (categorical)
10. `day_of_week` - Last contact day of the week (categorical)
11. `duration` - Last contact duration in seconds (numeric)
12. `campaign` - Number of contacts performed during this campaign (numeric)
13. `pdays` - Number of days since client was last contacted (numeric)
14. `previous` - Number of contacts performed before this campaign (numeric)
15. `poutcome` - Outcome of previous marketing campaign (categorical)

**Social and Economic Context:**
16. `emp.var.rate` - Employment variation rate (numeric)
17. `cons.price.idx` - Consumer price index (numeric)
18. `cons.conf.idx` - Consumer confidence index (numeric)
19. `euribor3m` - Euribor 3 month rate (numeric)
20. `nr.employed` - Number of employees (numeric)

## Models Used

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.9139 | 0.9370 | 0.7002 | 0.4127 | 0.5193 | 0.4956 |
| Decision Tree | 0.8956 | 0.7530 | 0.5344 | 0.5690 | 0.5511 | 0.4925 |
| K-Nearest Neighbor | 0.9053 | 0.8617 | 0.6267 | 0.3944 | 0.4841 | 0.4491 |
| Naive Bayes | 0.8536 | 0.8606 | 0.4024 | 0.6175 | 0.4872 | 0.4189 |
| Random Forest (Ensemble) | 0.9204 | 0.9491 | 0.6889 | 0.5345 | 0.6019 | 0.5640 |
| XGBoost (Ensemble) | 0.9167 | 0.9495 | 0.6505 | 0.5636 | 0.6039 | 0.5595 |

### Model Performance Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Best precision (70.02%). Excellent at minimizing false positives. Good baseline performance with interpretable linear model, suitable for understanding feature importance and providing business insights. |
| Decision Tree | Moderate performance (89.56% accuracy). Provides highly interpretable decision rules that can be easily visualized. May suffer from overfitting but useful for understanding decision boundaries. |
| K-Nearest Neighbor | Good accuracy (90.53%) but lower recall (39.44%). Instance-based learning approach, performance sensitive to feature scaling. Computationally expensive for large datasets during prediction. |
| Naive Bayes | Best recall (61.75%), excellent for identifying positive cases. Fast probabilistic classifier with independence assumption. Lower precision but useful when catching all positive cases is critical. |
| Random Forest (Ensemble) | Best overall model with highest accuracy (92.04%) and MCC (56.40%). Ensemble method provides robust and balanced performance across all metrics. Handles non-linear relationships well and resistant to overfitting. |
| XGBoost (Ensemble) | Highest AUC (94.95%) and F1 score (60.39%). Advanced gradient boosting algorithm excels at balancing precision and recall. Best for ranking predictions and handling imbalanced data. Recommended for deployment. |

## Deployment

**Live Streamlit App:** [Add your deployed Streamlit app URL here]

**GitHub Repository:** https://github.com/2025ab05206/ML_assignment

## Author

**Name:** Srineveda R S  
**BITS ID:** 2025AB05206  
**Course:** M.Tech (AIML)  
**Assignment:** Machine Learning Assignment 2  
**Submission Date:** February 15, 2026
