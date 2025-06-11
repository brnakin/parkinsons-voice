# Parkinson's Disease Classification Using Voice Biomarkers

A machine learning system for early detection of Parkinson's disease through voice analysis, achieving 95.7% sensitivity suitable for clinical screening applications.

## 🎯 Project Overview

### Problem Statement

Parkinson's Disease affects over 10 million people worldwide, with diagnosis often delayed due to subtle early symptoms and reliance on subjective clinical assessments. This project develops an automated voice-based screening system that can detect Parkinson's disease using objective biomarkers extracted from simple voice recordings.

### Solution

An end-to-end machine learning pipeline that analyzes voice characteristics to distinguish between healthy individuals and those with Parkinson's disease, enabling non-invasive, cost-effective screening suitable for primary care settings.

## 📊 Dataset

**Oxford Parkinson's Disease Detection Dataset**
- **Source**: University of Oxford + National Centre for Voice and Speech
- **Size**: 195 voice recordings from 31 individuals
- **Features**: 23 numerical voice measurements covering:
  - Fundamental frequency measures
  - Jitter and shimmer (voice stability indicators)
  - Noise ratios and voice quality measures
  - Nonlinear dynamics and complexity metrics
- **Class Distribution**: 147 Parkinson's cases (75.4%) vs 48 healthy cases (24.6%)

## 🛠️ Techniques & Methodology

### Data Preprocessing
- **Feature Scaling**: RobustScaler for outlier resistance
- **Class Balancing**: SMOTE (Synthetic Minority Oversampling Technique)
- **Feature Engineering**: Statistical significance testing and Random Forest importance
- **Dimensionality Reduction**: Principal Component Analysis (95% variance retention)

### Machine Learning Approach
- **Algorithms Evaluated**: 11 ML algorithms including XGBoost, Random Forest, Gradient Boosting, SVM, Logistic Regression, and KNN
- **Model Selection**: Systematic evaluation of 55 model-dataset combinations
- **Validation**: 5-fold stratified cross-validation with held-out test set
- **Optimization**: Grid search and randomized search for hyperparameter tuning

### Performance Metrics
- Primary: F1-Score (optimal for imbalanced medical data)
- Clinical: Sensitivity, Specificity, Precision, ROC-AUC
- Validation: Balanced accuracy and generalization assessment

## 🏆 Key Findings

### Model Performance
- **Best Model**: XGBoost Classifier on baseline dataset
- **Test F1 Score**: 91.7%
- **Sensitivity**: 95.7% (excellent for screening - only 4.3% of Parkinson's patients missed)
- **Precision**: 88.0% (high confidence in positive predictions)
- **Specificity**: 57.1% (moderate false positive rate)
- **ROC-AUC**: 92.6% (strong discriminative ability)

### Voice Biomarker Insights
- **Fundamental Frequency**: Parkinson's patients show significantly lower voice pitch
- **Voice Stability**: Increased jitter and shimmer indicating frequency/amplitude instability
- **Voice Quality**: Reduced harmonic-to-noise ratio in Parkinson's patients
- **Complexity**: Higher nonlinear dynamics and irregular voice patterns

### Statistical Validation
- All key voice features showed highly significant differences (p < 0.001) between groups
- Large effect sizes (Cohen's d > 0.4) confirming strong discriminative power
- Robust performance across multiple validation strategies

## 💡 Key Recommendations

### Clinical Implementation
1. **Primary Screening Tool**: Deploy as first-line screening in primary care settings
2. **Confirmatory Testing Required**: Follow positive results with comprehensive neurological assessment
3. **Quality Control**: Implement standardized recording protocols and audio quality validation
4. **Provider Training**: Educate healthcare workers on appropriate tool use and result interpretation

### Technical Deployment
1. **Production Model**: Use XGBoost baseline model for optimal generalization
2. **Monitoring**: Implement continuous performance monitoring and model updates
3. **API Development**: Create RESTful API for integration with existing healthcare systems
4. **Threshold Optimization**: Adjust decision boundaries based on clinical priorities

### Future Enhancements
1. **Population Validation**: Test across diverse demographics and recording conditions
2. **Feature Expansion**: Incorporate additional voice biomarkers and acoustic features
3. **Multi-modal Integration**: Combine with other biomarkers (movement, cognitive tests)
4. **Real-time Processing**: Develop mobile applications for point-of-care screening

## 🏥 Clinical Impact

### Benefits
- **Non-invasive Testing**: Simple voice recording procedure
- **Cost-effective Screening**: Reduces expensive diagnostic workups
- **Accessibility**: Deployable in remote and resource-limited settings
- **Early Intervention**: Enables earlier treatment initiation and better outcomes
- **Standardization**: Consistent screening methodology across healthcare providers

### Risk Mitigation
- **False Positive Management**: 43% false positive rate requires follow-up protocols
- **Clinical Guidelines**: Clear protocols for result interpretation and patient management
- **Continuous Validation**: Regular model updates with new clinical data
- **Technology Standards**: Standardized recording equipment and quality metrics

## 📁 Repository Structure

```
├── notebooks/
│   ├── parkinson-eda.ipynb                    # Exploratory data analysis
│   ├── parkinson-preprocessing.ipynb          # Data preprocessing pipeline
│   └── parkinson-training.ipynb              # Model training and evaluation
├── reports/
│   ├── EDA_Report_Parkinsons_Disease.md
│   ├── Data_Preprocessing_Report_Parkinsons_Disease.md
│   ├── Model_Training_Report_Parkinsons_Disease.md
│   └── Ultimate_Project_Report_Parkinsons_Classification.md
├── data/
│   ├── parkinsons.data                       # Oxford dataset
│   └── parkinsons.name
└── env.yaml                                 # Environment configuration
```

## 📈 Performance Summary

| Metric | Value | Clinical Interpretation |
|--------|-------|------------------------|
| **Sensitivity** | 95.7% | Excellent screening capability |
| **Specificity** | 57.1% | Moderate, requires confirmation |
| **Precision** | 88.0% | High confidence in positive cases |
| **F1-Score** | 91.7% | Balanced overall performance |
| **ROC-AUC** | 92.6% | Strong discriminative ability |

## 🤝 Contributing

Contributions are welcome! Please read the contributing guidelines and submit pull requests for any improvements.

---

**Status**: ✅ Complete - Ready for clinical validation and deployment  
**Version**: 1.0  
**Last Updated**: 11/06/2025