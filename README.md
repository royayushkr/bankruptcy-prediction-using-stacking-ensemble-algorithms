# Bankruptcy Prediction: 1st Place Solution

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-green)
![LightGBM](https://img.shields.io/badge/LightGBM-Enabled-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

## Executive Summary
This repository contains the code and documentation for our **1st Rank** solution in the Bankruptcy Prediction competition. The goal was to predict firm bankruptcy using noisy, high-dimensional financial data.

Our solution employs a **3-stage stacking ensemble** that combines Gradient Boosting Machines (XGBoost, LightGBM) Algorithms and Hill Climbing optimization. The final model achieved a Cross-Validation AUC of **0.9528**, significantly outperforming standard baselines.


## The Challenge
The dataset presented several hurdles common in financial distress prediction:
* **High Dimensionality:** 64 raw features hiding complex signals.
* **Noisy Data:** Massive missing values (`NaN`/`Inf`) and non-linear relationships.
* **Class Imbalance:** Only ~4.7% of firms in the training set were bankrupt.


## Solution Architecture

### 1. Feature Engineering: "Brute Force" Generation
We recognized that raw financial ratios were insufficient. We generated **16,448 synthetic features** by performing pairwise arithmetic operations (Addition, Subtraction, Multiplication, Division) on all 64 original features.

### 2. Advanced Feature Selection (RFECV)
Training on 16k features is inefficient and prone to overfitting. We used **Recursive Feature Elimination with Cross-Validation (RFECV)** accelerated by an **NVIDIA A100 GPU**.
* **Scout Model:** A fast XGBoost model ranked features by Information Gain.
* **Optimization:** We reduced the feature space from 16,448 to an optimal **3,600 features**, retaining 98% of predictive power.



### 3. Dual Imputation Strategy
Different models handle missing math differently. We used two distinct strategies:
* **Tree Models (XGB/LGB):** Imputed with `-999`.Trees isolate this value into a separate branch.
* **Evolutionary/Linear Models:** Imputed with `Median`.This prevents massive outliers from destroying linear relationships (e.g., Feature A + Feature B).

### 4. The Stacking Ensemble
Our final pipeline consisted of three levels:

| Level | Description | Details |
| :--- | :--- | :--- |
| **Level 1** | **Base Learners** | **12 XGBoost Models** (Seed diversities) + **8 LightGBM Models**. Trained using 5-Fold Stratified CV. |
| **Level 2** | **Evolutionary** | Evolutionary XGBoost (Genetic Algorithm wrapper) to evolve features over 100 rounds. |
| **Level 3** | **Meta-Learner** | **Hill Climbing (Weighted Average)**. This outperformed Logistic Regression and Bayesian Ridge stackers. |



## 📊 Performance Results

We benchmarked various architectures during development:

| Model | CV AUC Score | Notes |
| :--- | :--- | :--- |
| Logistic Regression (Baseline) | ~0.8500 | Simple linear baseline. |
| Single XGBoost | ~0.9100 | High variance. |
| RFECV Optimized XGB | ~0.9382 | Top 3600 features selected. |
| **Final Ensemble (Hill Climbing)** | **0.9528** | **Winning Solution**. |



## Installation & Usage

### Prerequisites
* Python 3.8+
* NVIDIA GPU (Recommended for RFECV and XGBoost training)

### Libraries
```bash
pip install pandas numpy xgboost lightgbm scikit-learn matplotlib
# Optional: catboost
pip install catboost
```

### Running the Solution
1.  **Data Placement:** Ensure `bankruptcy_Train.csv` and `bankruptcy_Test_X.csv` are in the project root or drive path.
2.  **Execution:** Run the Jupyter Notebook `3600_12_8_HC_A100.ipynb`.
    * *Note:* The notebook detects GPU availability automatically.
3.  **Output:** The script generates `submission2.csv` containing the predicted probabilities.


## File Structure
```text
.
├── 3600_12_8_HC_A100.ipynb    # Main training and inference pipeline
├── teamSHIV_presentation.pdf  # Executive summary slides
├── bankruptcy_Train.csv       # Training data (Not included in repo)
├── bankruptcy_Test_X.csv      # Testing data (Not included in repo)
└── README.md                  # Project documentation
```


