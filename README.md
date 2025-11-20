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

## 🧠 Day 36 — Neural Network Basics

**Focus:** Understanding how neural networks learn through layers, weights, and activations.  
**Key Tools:** `NumPy`, `TensorFlow`, `Keras`, `matplotlib`

### 📚 What I Learned
- The structure of a **Perceptron** — inputs, weights, bias, activation  
- Difference between **linear** and **non-linear** activation functions  
- How forward propagation computes predictions  
- How backpropagation adjusts weights through gradients  
- Used TensorFlow/Keras to build a simple feedforward neural network

### 🧩 Mini Project
Built a **Binary Classifier** to predict whether a student passes or fails based on study hours and sleep:
- Implemented a single-layer perceptron in NumPy  
- Recreated it using **Keras Sequential API**  
- Visualized the decision boundary and model performance


## 🧩 Skills You’ll Learn
- Model optimization and tuning  
- Deep learning with TensorFlow/Keras  
- Advanced feature engineering  
- Bias-variance tradeoff  
- Image and structured data modeling  
- Practical ML deployment techniques

---

# 🧠 Day 37 – Building an ANN with TensorFlow/Keras

## 📌 Overview
Today we expand on neural networks by building a **deeper artificial neural network (ANN)** with multiple hidden layers using TensorFlow/Keras.

---

## 🔹 Learning Objectives
- Understand the **Sequential model** in Keras  
- Add multiple **hidden layers**  
- Use different **activation functions**  
- Train and evaluate your ANN  
- Visualize training metrics like **loss** and **accuracy**

---

## 🧩 Key Concepts

### ANN Components
- **Input layer** – receives features  
- **Hidden layers** – learn patterns from data  
- **Output layer** – predicts target values  

### Activation Functions
- **ReLU** – common in hidden layers  
- **Sigmoid** – good for binary outputs  
- **Softmax** – for multi-class outputs  

### Architecture Example
- Input → Hidden layer 1 (8 neurons, ReLU)  
- Hidden layer 2 (4 neurons, ReLU)  
- Output layer (1 neuron, Sigmoid)  

### Training Notes
- **Loss function:** binary_crossentropy  
- **Optimizer:** Adam  
- **Epochs:** 100+  
- **Batch size:** 32  

---

## 🛠 Notebook Tasks
1. Build a 2-hidden-layer ANN using TensorFlow/Keras  
2. Train the network for 100 epochs on a small dataset  
3. Track **loss** and **accuracy**  
4. Plot training curves  
5. Write a 2–4 sentence interpretation of results  

---

## 🧪 Mini Project
- Build your own **2-hidden-layer ANN**  
- Experiment with **4–10 neurons per layer**  
- Try different **activation functions** (ReLU, Tanh, Sigmoid)  
- Train for 100+ epochs  
- Plot **loss and accuracy curves**  
- Write a brief **analysis of network performance**  

---

# 🧠 Day 38 – Improving Neural Networks

## 📌 Overview
Today we focus on improving your artificial neural networks (ANNs) using techniques like **dropout, batch normalization, and hyperparameter tuning**. These methods help prevent overfitting and make your model generalize better.

---

## 🔹 Learning Objectives
- Understand **overfitting** vs **underfitting**  
- Implement **dropout layers**  
- Apply **batch normalization**  
- Tune hyperparameters (neurons, activation functions, learning rate)  
- Evaluate model performance and improvements

---

## 🧩 Key Concepts

### Overfitting vs Underfitting
- **Overfitting:** model performs well on training data but poorly on new data  
- **Underfitting:** model fails to learn patterns, low performance overall  

### Dropout
- Randomly disables neurons during training  
- Prevents over-reliance on specific nodes  
- Typical dropout rate: 0.2–0.5  

### Batch Normalization
- Normalizes inputs to each layer  
- Speeds up training and stabilizes learning  

### Hyperparameter Tuning
- Adjust:  
  - Number of hidden layers and neurons  
  - Activation functions  
  - Learning rate  
  - Batch size and epochs  

---

## 🛠 Notebook Tasks
1. Add **dropout** to your existing ANN  
2. Apply **batch normalization** to hidden layers  
3. Tune hyperparameters and re-train  
4. Compare **loss and accuracy curves** with original model  
5. Write a short interpretation (2–4 sentences)  

---

## 🧪 Mini Project
- Take your **Day 37 ANN**  
- Add **dropout** (0.2–0.5) to hidden layers  
- Add **batch normalization**  
- Train for 100+ epochs  
- Plot **loss and accuracy curves**  
- Experiment with:  
  - Different numbers of neurons per layer (4–10)  
  - Different activation functions (ReLU, Tanh, Sigmoid)  
  - Different learning rates (0.001–0.01)  
- Write a brief analysis of performance improvements

---

# 🧠 Day 39 — Convolutional Neural Networks (CNNs)

Today you step into **computer vision** with **Convolutional Neural Networks (CNNs)** — the models behind image classification, object detection, and modern AI vision systems.  
This lesson introduces convolutions, filters, pooling, and how CNNs extract hierarchical features from images.

---

## 📌 Learning Objectives

By the end of Day 39, you will be able to:

- Explain convolutions, filters, feature maps  
- Understand stride, padding, and pooling layers  
- Build a CNN using TensorFlow/Keras  
- Train a CNN on Fashion-MNIST  
- Visualize training curves and predictions  
- Improve model performance with deeper layers or augmentation  

---

## 📦 1. Imports & Dataset

You will use **Fashion-MNIST**, a 28×28 grayscale clothing dataset, perfect for CNN training.

---

## 🧵 2. Data Preparation

- Normalize pixel values  
- Reshape images to (28, 28, 1)  
- One-hot encode labels  

This setup matches the input shape Keras expects for Conv2D.

---

## 👁 3. Visualizing Images

Preview a batch of clothing items to understand the classification challenge.

---

## 🧩 4. Build Your First CNN

Your basic CNN includes:

- **Conv2D(32)** → extract edges/textures  
- **MaxPool2D** → downsample  
- **Conv2D(64)** → deeper features  
- **Flatten → Dense → Dropout → Dense** output layer  

This is the classic architecture for early CNN tasks.

---

## ⚙️ 5. Compile

Use:

- **Adam** optimizer  
- **Categorical Crossentropy** loss  
- Accuracy metric  

---

## 🚀 6. Train the Model

Train for 10 epochs with a validation split.

Track:

- Training vs validation accuracy  
- Training vs validation loss  

---

## 📈 7. Plot Results

Visualize overfitting, stability, and general learning progression.

---

## 🧪 8. Evaluate on Test Data

Measure model performance on unseen data and print the final accuracy.

---

## 🔍 9. Predictions & Errors

Identify misclassified images to see where your CNN struggles.

---

## 🎯 Mini-Project: Improve the CNN

Improve accuracy by **5–10%** through at least one of the following:

### 🔧 Option A — More Filters  
Increase representational capacity.

### 🧱 Option B — Add Another Conv Block  
Build deeper feature hierarchies.

### 🛡 Option C — Increase Dropout  
Regularize and reduce overfitting.

### 🌪 Option D — Data Augmentation  
Generate transformed samples:
- rotation  
- zoom  
- width/height shifts  

Retrain with augmented batches to see performance gains.

---

## 🎉 End of Day 39

You now understand how CNNs extract visual features and build image classifiers.

Tomorrow: **Day 40 Mini Capstone Project — Build an End-to-End Vision Model**.


## ⚙️ Setup
- Install TensorFlow:  
```bash
pip install tensorflow


## ⚙️ Setup
- Install TensorFlow:  
```bash
pip install tensorflow


## ⚙️ Setup
```bash
git clone https://github.com/yourusername/data-science-bootcamp-4-ml-to-ai.git
cd data-science-bootcamp-4-ml-to-ai
