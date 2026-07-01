# QSkill AI/ML Internship Project

Domain: **Artificial Intelligence & Machine Learning**
Duration: 1st June 2026 – 1st July 2026

Two of the three offered tasks were completed end-to-end, from data loading through model evaluation.

## Tasks Completed

### 1. Iris Flower Classification (`iris_classification.py`)
Classifies iris flowers into Setosa, Versicolor, and Virginica using sepal/petal measurements.
- **Dataset:** Classic Iris dataset (scikit-learn / UCI)
- **Models:** Logistic Regression, K-Nearest Neighbors, Decision Tree
- **Best result:** 93.3% accuracy

### 2. Spam Mail Detector (`spam_mail_detector.py`)
Classifies SMS messages as spam or ham using TF-IDF text features.
- **Dataset:** [SMS Spam Collection](https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv) (UCI), included as `sms.tsv`
- **Models:** Naive Bayes, Logistic Regression
- **Best result:** 97.2% accuracy, 100% precision

## Repository Structure

```
├── iris_classification.py          # Task 1 source code
├── spam_mail_detector.py           # Task 2 source code
├── sms.tsv                         # Spam dataset used by Task 2
├── QSkill_AI_ML_Internship_Report.docx   # Full write-up with charts & results
└── outputs/                        # Generated charts and result tables
    ├── iris_pairplot.png
    ├── iris_boxplots.png
    ├── iris_correlation.png
    ├── iris_confusion_matrix.png
    ├── iris_model_comparison.png / .csv
    ├── spam_class_distribution.png
    ├── spam_message_length.png
    ├── spam_confusion_matrix.png
    ├── spam_roc_curve.png
    └── spam_model_comparison.png / .csv
```

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn

python iris_classification.py
python spam_mail_detector.py
```

Both scripts print evaluation metrics to the console and save all charts to `outputs/`.

## Results Summary

| Task | Best Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| Iris Classification | Logistic Regression | 0.9333 | 0.9333 | 0.9333 | 0.9333 |
| Spam Mail Detector | Logistic Regression | 0.9722 | 1.0000 | 0.7919 | 0.8839 |

See `QSkill_AI_ML_Internship_Report.docx` for the full report with methodology, visualizations, and discussion.
