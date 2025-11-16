# 🚨 **Credit Card Fraud Detection using Machine Learning**

A machine learning project to detect fraudulent credit card transactions using classification models and anomaly detection techniques. This project uses a highly imbalanced real-world dataset to build, evaluate, and analyze fraud detection performance.

---

## 📌 **Overview**

Credit card fraud is a major risk in digital payments. This project builds a machine learning pipeline to classify transactions as:

* **1 → Fraudulent**
* **0 → Normal**

The dataset is extremely imbalanced (fraud cases are <1%), making this a challenge in real-world fraud detection.

This project includes:

* Data preprocessing
* Feature scaling
* Train-test split
* SVM / Random Forest model training
* Confusion matrix
* Classification report
* Fraud detection results & insights

---

## 📂 **Project Structure**

```
├── creditcard.csv           # Dataset
├── fraud_detection.ipynb    # Notebook with full code
├── README.md                # Project documentation
└── requirements.txt         # Dependencies
```

---

## 🧠 **Machine Learning Models Used**

* **Support Vector Machine (SVC)**
* **Random Forest Classifier** *(recommended for imbalanced data)*

To handle dataset imbalance, the model uses:

* Class weights (class_weight="balanced")
* Feature scaling (StandardScaler)

---

## 🛠️ **Technologies & Libraries**

* Python
* Pandas
* NumPy
* Scikit-learn
* Seaborn
* Matplotlib

---

## 📊 **Dataset Information**

* **Source:** Kaggle – Credit Card Fraud Detection Dataset
* **Rows:** 284,807
* **Fraud cases (Class = 1):** 492
* **Normal cases (Class = 0):** 284,315
* **Type:** Highly imbalanced dataset
* **Features:** Time, Amount, V1–V28 (PCA-transformed)

---

## 🚀 **Model Training Code (Summary)**

```python
X = df.drop('Class', axis=1)
y = df['Class']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model_svc = SVC(class_weight='balanced')
model_svc.fit(X_train, y_train)

y_pred = model_svc.predict(X_test)
```

---

## 📈 **Evaluation Metrics**

The model is evaluated using:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **Confusion Matrix**

Fraud detection focuses heavily on:

* **Recall** (catching fraud cases)
* **Precision** (avoiding false alarms)

---

## 📉 **Confusion Matrix Example**

```
                Predicted Fraud   Predicted Normal
Actual Fraud          TP                 FN
Actual Normal         FP                 TN
```

A heatmap visualization is included in the notebook.

---

## 🧾 **Results (Expected)**

Because the dataset is imbalanced:

* Accuracy is high (98–99%)
* But detecting fraud (class 1) is more important
* Random Forest with class weights improves fraud detection significantly

---

## 📦 **Installation**

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## ▶️ **How to Run**

```bash
python fraud_detection.py
```

or open the notebook:

```
fraud_detection.ipynb
```

---

## 📌 **Future Improvements**

* Add SMOTE oversampling
* Use XGBoost for higher recall
* Include anomaly detection models
* Build a Streamlit dashboard

---
