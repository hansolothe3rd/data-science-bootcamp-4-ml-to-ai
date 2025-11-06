# 🧠 Data Science Bootcamp – Repo 4: From ML to AI

Welcome to the fourth stage of your 100-day data science journey!  
This repo transitions from traditional machine learning to **AI and deep learning concepts**, introducing you to neural networks, optimization, and real-world applications.

---

## 🗓️ Days Overview

| Day | Topic | Focus |
|-----|-------|--------|
| 31 | Gradient Descent & Optimization | How models learn through iterative improvement |
| 32 | Feature Engineering Deep Dive | Handling outliers, skewness, and transformations |
| 33 | Regularization (L1/L2/Ridge/Lasso) | Preventing overfitting |
| 34 | Decision Trees & Random Forests | Non-linear ML models |
| 35 | Gradient Boosting (XGBoost, LightGBM) | Advanced ensemble methods |
| 36 | Neural Network Basics | Intro to perceptrons and activation functions |
| 37 | Building an ANN with TensorFlow/Keras | First deep learning model |
| 38 | Improving Neural Nets | Dropout, batch norm, tuning |
| 39 | CNNs (Image Classification) | Convolutional Neural Networks intro |
| 40 | Mini Capstone Project | Build an end-to-end AI model project |

---

## Day 31 – Gradient Descent & Optimization

**Focus:** Understanding how models learn by minimizing loss functions.  
**Topics Covered:**
- Concept and math of gradient descent  
- Learning rate and convergence  
- Manual implementation with Python  
- Mini project: Linear Regression from scratch  

**Skills Gained:**
- Mathematical intuition for optimization  
- Parameter tuning practice  
- Implementing ML fundamentals manually

---

## Day 32 – Feature Engineering Deep Dive
**Focus:** Handling outliers, skewness, transformations, and scaling features.

**Mini Project:**  
- Dataset: California Housing  
- Steps:
  1. Detect and cap outliers (IQR method)  
  2. Fix skewed features (log transformation)  
  3. Scale numeric columns (StandardScaler)  
  4. Encode categorical features (one-hot)  
  5. Build preprocessing pipeline  
  6. Train Linear Regression and Random Forest models  
  7. Evaluate and visualize feature importances

**Key Takeaways:**  
- Clean, well-transformed features improve model performance  
- Scaling helps optimization  
- Pipelines make preprocessing reproducible

---

## 🧠 Day 33 — Data Preprocessing with scikit-learn

**Focus:** Cleaning and preparing data before modeling.  
**Key Tools:** `SimpleImputer`, `OneHotEncoder`, `StandardScaler`, `ColumnTransformer`, `Pipeline`

### 📚 What I Learned
- Handle missing values using mean or most frequent imputation  
- Encode categorical variables with one-hot encoding  
- Standardize numeric features for consistent scaling  
- Combine steps into a preprocessing pipeline for cleaner workflows  

### 🧩 Mini Project
Preprocessed the **Titanic dataset** — imputing missing data, encoding categorical variables, scaling features, and splitting into train/test sets for modeling.

---

## 🌳 Day 34 — Decision Trees & Random Forests

**Focus:** Understanding tree-based models for non-linear decision making.  
**Key Tools:** `scikit-learn`, `pandas`, `matplotlib`, `seaborn`

### 📚 What I Learned
- How **Decision Trees** split data using **Gini impurity** and **entropy**  
- Visualized trees and understood how splits affect bias and variance  
- Learned how **Random Forests** improve accuracy by averaging multiple trees  
- Compared model performance between Decision Trees and Random Forests  
- Tuned hyperparameters like `max_depth`, `n_estimators`, and `min_samples_split`

### 🧩 Mini Project
Built a **Customer Churn Predictor** using Random Forests:  
- Preprocessed categorical and numeric data  
- Trained and tuned models using cross-validation  
- Evaluated accuracy, precision, recall, and feature importance  

---

## 🚀 Day 35 — Gradient Boosting (XGBoost & LightGBM)

**Focus:** Mastering advanced ensemble methods for high-performance ML.  
**Key Tools:** `XGBoost`, `LightGBM`, `scikit-learn`, `matplotlib`

### 📚 What I Learned
- How **Gradient Boosting** builds models sequentially to reduce errors  
- The difference between **Bagging (Random Forests)** and **Boosting (XGBoost)**  
- Tuned hyperparameters like:
  - `learning_rate`
  - `n_estimators`
  - `max_depth`
  - `subsample`
- Evaluated performance using metrics like accuracy, AUC, and confusion matrices  
- Compared **XGBoost** and **LightGBM** efficiency and accuracy  

### 🧩 Mini Project
Built a **Loan Default Prediction Model**:
- Cleaned and encoded financial dataset features  
- Trained models using XGBoost and LightGBM  
- Tuned hyperparameters with grid search  
- Visualized **feature importance** and **ROC curves**

---

**Next:** Day 36 — *Neural Network Basics* 🧠 — diving into perceptrons and activation functions.


## 🧩 Skills You’ll Learn
- Model optimization and tuning  
- Deep learning with TensorFlow/Keras  
- Advanced feature engineering  
- Bias-variance tradeoff  
- Image and structured data modeling  
- Practical ML deployment techniques

---

## ⚙️ Setup
```bash
git clone https://github.com/yourusername/data-science-bootcamp-4-ml-to-ai.git
cd data-science-bootcamp-4-ml-to-ai
