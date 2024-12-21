# Fraud Detection Project

## Repository Overview
This repository contains two notebooks demonstrating the application of supervised and unsupervised learning techniques for detecting fraudulent transactions in a credit card dataset.

---

## Notebooks

### 1. **Supervised Learning**
- **Objective:** Train supervised models to classify fraudulent and non-fraudulent transactions.
- **Models Used:**
  - Logistic Regression
  - Random Forest
  - XGBoost
- **Highlights:**
  - Data preprocessing including normalization and handling class imbalance using SMOTE.
  - Evaluation metrics: Precision, Recall, F1-Score, ROC-AUC.
  - Insights into the strengths and limitations of supervised models.

### 2. **Unsupervised Learning**
- **Objective:** Detect anomalies (fraudulent transactions) using unsupervised techniques.
- **Models Used:**
  - Isolation Forest
  - Autoencoder
  - Hybrid Model (Combining Isolation Forest and Autoencoder)
- **Highlights:**
  - Reconstruction error analysis for Autoencoders.
  - Visualization of anomalies and reconstruction errors.
  - Comparative analysis of Isolation Forest, Autoencoder, and Hybrid approaches.

---

## Key Features
- **Dataset:** Kaggle's Credit Card Fraud Detection dataset.
  - Highly imbalanced with only ~0.17% fraudulent transactions.
  - Features `V1` to `V28` are anonymized and PCA-transformed.

- **Metrics Used:**
  - Precision: Percentage of true positives among predicted positives.
  - Recall: Percentage of actual positives detected.
  - F1-Score: Harmonic mean of Precision and Recall.
  - ROC and Precision-Recall curves.

- **Visualizations:**
  - Reconstruction Error Distribution
  - Precision-Recall Curves
  - Comparative performance metrics for models.

---

## Repository Structure
```
|-- supervised_learning.ipynb  # Notebook for supervised learning techniques
|-- unsupervised_learning.ipynb  # Notebook for unsupervised learning techniques
|-- README.md  # This documentation file
```

---

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/fraud-detection.git
   ```
2. Install required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow xgboost
   ```
3. Open the notebooks in Jupyter or Google Colab and run the cells sequentially.

---

## Results Summary
| Model              | Precision | Recall | F1-Score |
|--------------------|-----------|--------|----------|
| Logistic Regression (Supervised) | 0.95      | 0.90   | 0.92     |
| XGBoost (Supervised)             | 0.98      | 0.93   | 0.95     |
| Isolation Forest (Unsupervised)  | 0.37      | 0.21   | 0.27     |
| Autoencoder (Unsupervised)       | 0.03      | 0.89   | 0.06     |
| Hybrid Model (Unsupervised)      | 0.03      | 0.89   | 0.06     |

---

## Future Work
- Experiment with other unsupervised models like DBSCAN or Variational Autoencoders.
- Explore semi-supervised approaches to label data and train a classifier.
- Improve precision for unsupervised models by fine-tuning thresholds.
- Add explainability tools like SHAP for feature importance analysis.

---

## Acknowledgments
This project is based on the Kaggle Credit Card Fraud Detection dataset. Special thanks to the contributors for making this data publicly available.

---

## License
This repository is licensed under the MIT License.
