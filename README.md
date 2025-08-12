## Parkinson's Disease Detection from Voice Data

A machine learning pipeline to detect Parkinson's disease using vocal biomarkers from sustained phonations. The project emphasizes clinical interpretability, rigorous statistics, and reproducible evaluation.

- **Best model**: Support Vector Machine (SVM) on baseline features
- **F1-score**: 0.966 • **Accuracy**: 0.948 • **Sensitivity (Recall)**: 0.977 • **Specificity**: 0.857
- **Dataset**: UCI Parkinson's dataset (195 samples, 24 vocal features)

## Why this matters
Early, non-invasive screening using voice data can help triage patients for specialist assessment. The pipeline prioritizes sensitivity to minimize missed Parkinson's cases while keeping specificity clinically acceptable.

## Project structure
- `data/raw/`: Original dataset files (`parkinsons.data`, `parkinsons.names`)
- `data/processed/`: Engineered features, split datasets, PCA variants
- `notebooks/`: Analysis notebook (`parkinson.ipynb`)
- `models/`: Trained models (including `final_model.pkl`)
- `figures/`: Plots (confusion matrices, SHAP, correlations, distributions)
- `results/`: Metrics, SHAP values, statistical analyses
- `PROJECT_REPORT.md`: Comprehensive report with methods, results, and clinical interpretation

## Setup
1. Install Conda (or Mambaforge/Miniconda).
2. Create and activate the environment:
   - `conda env create -f env.yaml`
   - `conda activate university-machine-learning-project-parkinsons-voice-v3`

## Quick start
- Open and run the notebook: `notebooks/parkinson.ipynb`
- Outputs (metrics, figures, models) are written to `results/`, `figures/`, and `models/`

## Data
- Source: [UCI Machine Learning Repository — Parkinson's Data Set](https://archive.ics.uci.edu/ml/datasets/parkinsons)
- 195 samples, 24 vocal biomarkers, target `status` (1=Parkinson's, 0=healthy)
- Class imbalance: 75.4% Parkinson's vs 24.6% healthy (~3:1)

## Methodology (high-level)
- **Statistical rigor**: Mann–Whitney U tests with FDR correction, Cohen's d, post-hoc power analysis
- **Feature engineering**: Five domain-informed composites (e.g., `jitter_stability_ratio`, `shimmer_composite`, `voice_quality_index`, `frequency_range`, `frequency_cv`)
- **Preprocessing**: Leave-One-Subject-Out (LOSO) CV, RobustScaler, SMOTE class balancing, consensus feature selection, PCA (95% variance)
- **Models evaluated**: Random Forest, SVM, Logistic Regression, KNN, Decision Tree, XGBoost
- **Hyperparameters**: Grid search with 5-fold stratified CV (primary metric: F1-score)

## Results
- Top 5 (validation) models:

  | Rank | Algorithm | Dataset | F1-Score | Accuracy | Precision | Recall |
  |------|-----------|---------|----------|----------|-----------|--------|
  | 1 | SVM | baseline | 0.966 | 0.948 | 0.956 | 0.977 |
  | 2 | SVM | smote | 0.966 | 0.948 | 0.956 | 0.977 |
  | 3 | XGBoost | baseline | 0.957 | 0.931 | 0.917 | 1.000 |
  | 4 | RandomForest | smote | 0.956 | 0.931 | 0.935 | 0.977 |
  | 5 | XGBoost | smote | 0.956 | 0.931 | 0.935 | 0.977 |

- Best model (SVM + baseline features): ROC-AUC 0.964, Sensitivity 97.7%, Specificity 85.7%

### Explainability
- SHAP analysis highlights nonlinear dynamics and jitter features as driving factors:
  - Top features: `D2`, `spread2`, `Jitter:DDP`, `MDVP:RAP`, `MDVP:Fhi(Hz)`, `spread1`, `DFA`, `frequency_cv`, `MDVP:Jitter(%)`, `jitter_stability_ratio`
- See figures for global and local explanations:
  - `figures/shap_summary_bar.png`
  - `figures/shap_summary_detailed.png`
  - `figures/shap_local_healthy.png`, `figures/shap_local_parkinson.png`

### Key figures
- Class imbalance: `figures/target_distribution.png`
- Correlations: `figures/correlation_heatmap.png`
- Discriminative feature distributions: `figures/key_features_distributions.png`
- Confusion matrices: see `figures/confusion_matrix_*.png`

## Reproducing the pipeline
- Use the notebook `notebooks/parkinson.ipynb` to:
  - Load raw data, validate quality (missing/duplicates)
  - Run EDA (non-parametric tests, effect sizes, power)
  - Engineer composite features and save `data/processed/parkinsons_engineered.csv`
  - Preprocess: LOSO-aware splits, Robust scaling, SMOTE, consensus selection, PCA
  - Train models across dataset variants; grid search with fixed seeds
  - Evaluate and export metrics to `results/`
  - Generate SHAP explanations and figures

## Using the trained model
Load `models/final_model.pkl` and predict on a pandas DataFrame with the same feature schema as the chosen dataset variant (baseline for the best model):

```python
import pickle
import pandas as pd

with open('models/final_model.pkl', 'rb') as f:
    model = pickle.load(f)

# X should contain the baseline feature set (after the same preprocessing used in training)
X = pd.read_csv('data/processed/X_test_baseline.csv')
proba = model.predict_proba(X)[:, 1]
preds = (proba >= 0.5).astype(int)
print(pd.Series(preds).value_counts())
```

Note: Ensure you apply the identical preprocessing pipeline used during training. When in doubt, use the prepared `data/processed/X_*_baseline.csv` files.

## Clinical framing
- Sensitivity is prioritized for screening to avoid missed cases
- Specificity remains acceptable for minimizing unnecessary referrals

## Reproducibility and safeguards
- Fixed random seeds across splits, SMOTE, and training
- Avoids data leakage by applying preprocessing only after train/test splits
- LOSO cross-validation to prevent subject overlap

## Acknowledgements
- Dataset: [UCI Machine Learning Repository — Parkinson's Data Set](https://archive.ics.uci.edu/ml/datasets/parkinsons)
