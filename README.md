# HeartDisease-ML-Predicition
This project aims to build a machine learning model that predicts the presence of heart disease in patients based on medical attributes. It utilizes various classification algorithms and evaluates them based on accuracy, precision, recall, and F1-score.

## Dataset
The dataset contains several patient features including:
- Age, Sex, Chest Pain Type (cp)
- Resting Blood Pressure (trestbps)
- Serum Cholesterol (chol)
- Fasting Blood Sugar (fbs)
- Resting ECG results (restecg)
- Max heart rate (thalach)
- Exercise-induced angina (exang)
- ST depression (oldpeak)
- Slope of the peak exercise ST segment (slope)
- Number of major vessels (ca)
- Thalassemia (thal)
- Target (0 = No heart disease, 1 = Heart disease)

## Models Used
- Logistic Regression
- Random Forest Classifier
## Preprocessing
- Label encoding of categorical features
- Standard scaling for numerical features
- Train-test split with stratification
- Performance evaluated using:
  - Confusion Matrix
  - Classification Report
  - ROC-AUC Score

## Results
Best performance was achieved using:
- **Random Forest** giving the highest accuracy and precision.

