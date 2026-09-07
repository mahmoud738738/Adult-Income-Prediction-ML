# Adult Census Income Prediction

## Project Overview
This capstone project aims to predict whether an individual's annual income exceeds  based on census demographic data. It is a binary classification problem that deals with class imbalance, missing values, and high-cardinality categorical features.

## Workflow and Methodology

1. Exploratory Data Analysis (EDA):
   - Investigated target class imbalance (76% vs 24%).
   - Analyzed numerical and categorical feature distributions.
   - Identified missing values and skewed features.

2. Data Cleaning & Preprocessing:
   - Dropped a small percentage (7.4%) of rows with missing values to avoid mode imputation bias.
   - Removed redundant and non-predictive features.
   - Handled high-cardinality in 'native-country' by bucketing into 'United-States' vs 'Other'.
   - Applied One-Hot Encoding to nominal variables and mapped the target variable to binary integers.

3. Data Splitting & Leakage Prevention:
   - Split the data into 80% training and 20% testing sets using stratified sampling to maintain class proportions.
   - Applied StandardScaler strictly on the training set to prevent data leakage, then transformed the test set.

4. Advanced Modeling (AutoML):
   - Established baselines using standard Scikit-Learn models.
   - Utilized FLAML (Fast and Lightweight AutoML) to perform automated hyperparameter tuning for robust algorithms, including Logistic Regression, Random Forest, and LightGBM.

## Key Results
Due to the imbalanced nature of the dataset, F1-Score (for the positive class >50K) and ROC-AUC were prioritized over raw accuracy. 

After running AutoML tuning, the models achieved the following performance on the test set:
- LightGBM: Achieved the best overall balance with the highest F1-Score and ROC-AUC.
- Random Forest: Demonstrated strong performance and high precision.
- Logistic Regression & Decision Tree: Served as solid baselines but were outperformed by the ensemble methods.

Detailed performance metrics, confusion matrices, and ROC curves are fully documented and visualized within the Jupyter Notebook.

## Technologies Used
- Python (Pandas, NumPy)
- Data Visualization (Matplotlib, Seaborn)
- Machine Learning (Scikit-Learn)
- Automated Machine Learning (FLAML)
