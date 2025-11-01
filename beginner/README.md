# 🟢 Beginner Track: Machine Learning Heart Attack Risk Predictor

Welcome to the **Beginner Track** of the **CardioSentinel: Predicting Heart Attack Risk from Lifestyle and Clinical Data** project! 💓  

This track is designed for learners who are comfortable with Python and have a basic understanding of data preprocessing and supervised learning. You’ll be guided through the entire **machine learning pipeline** — from exploring the data and training models to deploying your prediction app.

---

## 🎯 Project Objective

Your goal is to build a **machine learning model** that predicts whether a patient is **at risk of a heart attack** (`Heart Attack Risk = 1`) or **not at risk** (`Heart Attack Risk = 0`) using the Heart Attack Risk Prediction Dataset.  

You will:
- Perform exploratory data analysis (EDA) to uncover trends in clinical and lifestyle factors  
- Engineer and preprocess features for modeling  
- Train and evaluate multiple ML models  
- Deploy a Streamlit web app that predicts heart attack risk based on user inputs  

---

## 🩺 Business Context

You’re a **Data Scientist at CardioSentinel HealthTech**, a company pioneering data-driven solutions for heart disease prevention.  

The company wants to provide hospitals, doctors, and insurance firms with a **risk prediction tool** that uses patient data to flag high-risk individuals early.  

Your model’s insights will help healthcare professionals:
- Detect risk before a heart event occurs  
- Recommend personalized interventions  
- Improve preventive healthcare outcomes  

---

## 📊 Dataset Overview

**Dataset:** Heart Attack Risk Prediction Dataset  
**Link:** [Kaggle Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/heart-attack-prediction-dataset)  
**Records:** 8,763 patient entries  

---

## 🗓️ Project Workflow

| Phase | Core Tasks | Deliverables | Duration |
|-------|-------------|--------------|-----------|
| **1 · Setup + EDA** | Load and explore dataset, handle missing values, visualize correlations, encode categories | EDA report with insights and charts | **Week 1** |
| **2 · Feature Engineering** | Scale/normalize features, manage class imbalance, split into train/test sets | Clean and preprocessed dataset | **Week 2** |
| **3 · Model Development** | Train Logistic Regression, Random Forest, and XGBoost models | Model comparison report with evaluation metrics | **Week 3** |
| **4 · Optimization & Evaluation** | Fine-tune best model, analyze feature importances, cross-validate | Final model + performance summary | **Week 4** |
| **5 · Deployment** | Build a Streamlit app to serve predictions | Deployed Streamlit app | **Week 5** |

---

## 🧭 Phase 1: Setup & Exploratory Data Analysis (Week 1)

**Tasks:**
- Clean and preprocess missing or inconsistent values  
- Encode categorical variables (e.g., Diet, Smoking, Alcohol Consumption)  
- Visualize distributions and correlations  

---

## ⚙️ Phase 2: Feature Engineering (Week 2)

**Tasks:**
- Create new features such as:
  - `BMI_Category` (underweight, normal, overweight, obese)  
  - `Risk_Index` = (Cholesterol + BloodPressure) / ExerciseHours  
- Normalize or standardize numeric columns  
- Handle class imbalance (e.g., using SMOTE or class weighting)  
- Split data into **train/test** sets (e.g., 80/20 ratio)

**Goal:**  
Prepare a balanced, clean dataset ready for modeling.

---

## 🧪 Phase 3: Model Development (Week 3)

**Tasks:**
- Train multiple models:
  - Logistic Regression  
  - Random Forest  
  - XGBoost  
- Compare performance using:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-score**
  - **ROC-AUC**

- Use **MLflow** to log all experiments (parameters, metrics, and models).  

**Goal:**  
Select the model with the best overall performance and interpret its feature importances.

---

## 🚀 Phase 4: Model Optimization & Evaluation (Week 4)

**Tasks:**
- Fine-tune hyperparameters using GridSearchCV or RandomizedSearchCV  
- Check for overfitting and improve generalization  
- Evaluate using confusion matrix and ROC curve  
- Rank top 5 most important features influencing heart attack risk  

**Deliverables:**
- Optimized model  
- Evaluation report with visualizations  

---

## 💻 Phase 5: Model Deployment (Week 5)

**Tasks:**
- Build a **Streamlit app** that allows user input for all patient features  
- Display:
  - Predicted Risk: *At Risk / Not at Risk*  
  - Probability score  
  - Top 3 contributing features (from feature importance)  
- Deploy app on **Streamlit Community Cloud**  

**Optional Enhancements:**
- Add visual feedback (e.g., risk meter gauge)  
- Save prediction history to CSV for audit tracking  

---

## 🧰 Technical Requirements

| Category | Libraries |
|-----------|------------|
| **Data Handling & Visualization** | `pandas`, `numpy`, `matplotlib`, `seaborn` |
| **Machine Learning** | `scikit-learn`, `xgboost`, `mlflow` |
| **Deployment** | `streamlit` |

---

## 🧾 Deliverables Summary

By the end of this track, you should have:
1. **Cleaned & analyzed dataset** (EDA notebook)  
2. **Trained models** (with logged experiments)  
3. **Optimized final model** (with evaluation report)  
4. **Deployed Streamlit app** for public access  

---

## 🧑‍💻 Submission Guidelines

- Save all notebooks under your team folder inside:  
  `beginner/submissions/team-members/your-name/`
- Include:
  - `EDA.ipynb`
  - `model_training.ipynb`
  - `app.py`
  - `REPORT.md` (summary of findings)
- Community contributors can submit under `beginner/submissions/community-contributions/`
