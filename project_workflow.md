# Parkinson's Disease Detection from Voice Data

## Project Overview

**Objective**: Develop a classification model to predict whether a person has Parkinson's disease using vocal biomarkers extracted from sustained phonations. Early diagnosis may help patients get better care and treatment.
**Dataset**: UCI Parkinson's Dataset

## Environment & Setup

### Initial Environment Activation

```
conda activate university-machine-learning-project-parkinsons-voice-v3
```

### Project Roadmap

1. Data Collection
1.1 Download the data with wget if it is not already downloaded
1.2 Validate with head command

2. Environment Setup
2.1 Create conda environment from env.yaml if it is not already created:
  ```conda env create -f env.yaml```
2.2 Activate environment if it is not already activated:
  ```conda activate university-machine-learning-project-parkinsons-voice-v3```

3. Exploratory Data Analysis (EDA)
3.1 Load data/parkinsons.data with correct headers
3.2 Inspect shape, info, missing values, duplicates
3.3 Analyze target distribution and class-imbalance ratio (≈3:1)
3.4 Perform Mann–Whitney U tests
3.5 Calculate Cohen’s d for all significant features
3.6 Conduct post-hoc statistical power analysis using the observed Cohen’s d
3.7 Plot histograms, box plots, density plots for the just a few key points
3.8 Draw correlation heatmap for all numerical features with target variable

4. Feature Engineering
4.1 Compute jitter_stability_ratio:
  ```df['jitter_stability_ratio'] = df['MDVP:Jitter(%)'] / (df['MDVP:Jitter(Abs)'] + 1e-8)```
4.2 Compute shimmer_composite:
  ```df['shimmer_composite'] = (df['MDVP:Shimmer'] + df['Shimmer:APQ3'] + df['Shimmer:APQ5']) / 3```
4.3 Compute voice_quality_index:
  ```df['voice_quality_index'] = df['HNR'] / (df['NHR'] + 1)```
4.4 Compute frequency_range:
  ```df['frequency_range'] = df['MDVP:Fhi(Hz)'] - df['MDVP:Flo(Hz)']```
4.5 Compute frequency_cv:
  ```df['frequency_cv'] = df['frequency_range'] / df['MDVP:Fo(Hz)']```
4.6 Save the feature engineered data

5. Data Preprocessing
5.1 Load the feature engineered data
5.2 Split data stratified by status and by name for LOSO CV
5.3 Scale features with RobustScaler to handle outliers
5.4 Balance classes with SMOTE to achieve 1:1 ratio
5.5 Combine statistical and RF-based methods into consensus set
5.6 Apply PCA to retain 95% variance
5.7 Create dataset variants: Baseline, SMOTE, Feature-Selected, SMOTE+Feature, PCA
5.8 Save the datasets

6. Model Training
6.1 Load the datasets
6.2 Define algorithms: Random Forest, Support Vector Machine (SVM), Logistic Regression, XGBoost, K-Nearest Neighbors (KNN), Decision Tree
6.3 Set up hyperparameter grids for GridSearch
6.4 Train all model-dataset combos
6.5 Track F1, accuracy, precision, recall, ROC-AUC, balanced accuracy
6.6 Identify top 5 models by validation F1
6.7 Document feature importance rankings for tree-based models
6.8 Save top 5 models for evaluation

7. Model Evaluation & Explainability
7.1 Evaluate top 5 models on held-out test set
7.2 Calculate comprehensive performance metrics: F1-score, accuracy, precision, recall, sensitivity, specificity, ROC-AUC
7.3 Generate confusion matrices for clinical interpretation
7.4 Assess generalization by comparing validation vs. test performance gaps
7.5 Identify final model
7.6 Explain global feature importance with SHAP values
7.7 Document clinical performance interpretation: sensitivity, specificity, false positive/negative rates
7.8 Create model comparison summary table with all key metrics
7.9 Save final trained model

Critical Success Factors
- Data Quality Assurance: Verify zero missing values and no duplicates before analysis
- Stratified Sampling: Maintain class distribution proportions across train/validation/test splits
- Robust Scaling: Use RobustScaler specifically for medical data with outliers and diverse scales
- Class Balance Strategy: Implement SMOTE rather than undersampling to preserve information
- Feature Selection Consensus: Combine statistical and tree-based importance for robust feature selection
- Cross-Validation Rigor: Use 5-fold stratified CV with consistent random seeds for reproducibility
- Clinical Metric Focus: Prioritize sensitivity for screening applications over overall accuracy
- Generalization Testing: Always validate on completely held-out test set to assess real-world performance
- Model Interpretability: Document feature importance and clinical significance of voice biomarkers
- Performance Documentation: Record all metrics, not just F1-score, for comprehensive evaluation
- Apply feature-selection method for robustness
- Ensure reproducibility via fixed random seeds

Common Pitfalls to Avoid
- Data Leakage: Never apply preprocessing (scaling, SMOTE) before train/test split
- Metric Misinterpretation: Don't rely solely on accuracy with imbalanced data (75.4% vs 24.6%)
- Overfitting Indicators: Watch for large validation-test performance gaps (>0.05 F1-score difference)
- Scale Sensitivity: Don't use StandardScaler or MinMaxScaler with medical data containing outliers
- Feature Selection Bias: Avoid selecting features based on single method; use consensus approaches
- Cross-Validation Errors: Don't use regular CV; always use stratified CV to maintain class proportions
- Clinical Context Loss: Don't optimize for F1-score alone; consider sensitivity/specificity trade-offs for medical screening
- Hyperparameter Overfitting: Avoid excessive hyperparameter tuning on validation set without final test evaluation
- Reproducibility Issues: Always set random seeds for train/test splits, SMOTE, and model training
- False Performance Claims: Don't report validation scores as final performance; always use held-out test set

