# 🏠 California Housing Price Predictor

A production-ready **machine learning project** that predicts median housing prices in California using demographic and geographic data.  
Developed with **Python**, **scikit-learn**, and **Streamlit**, the project demonstrates model training, evaluation, and explainability.

---

## 🚀 Overview

This project estimates **California house prices** based on features such as:
- Median income
- House age
- Average rooms per household
- Population
- Geographic coordinates (latitude, longitude)

It implements multiple regression algorithms and evaluates their performance using standard metrics (RMSE, MAE, R²).

---

## ⚙️ Model Performance Summary

| Model | PCA | Tuning | RMSE | MAE | R² Score | Performance |
|--------|-----|---------|------|------|-----------|--------------|
| Linear Regression | ✅ | ✗ | 0.8770 | 0.6559 | 0.4216 | ⚠️ Moderate |
| Random Forest | ✅ | ✗ | **0.7284** | **0.5139** | **0.6010** | ✅ Good |
| Gradient Boosting | ✅ | ✅ | 0.7296 | 0.5183 | 0.5997 | ✅ Good |

> **Best Model:** Random Forest — explains ~60% of variance in housing prices with lowest error.

---

## 🧩 Project Structure

california-housing-ml/
│
├── src/
│ ├── data.py # Data ingestion and splitting
│ ├── features.py # Feature engineering and scaling
│ ├── models.py # ML model definitions
│ ├── train.py # Training and saving models
│ ├── evaluate.py # Metrics and evaluation
│ └── explain.py # SHAP-based model explainability
│
├── config/
│ └── config.yaml # Model configuration
│
├── artifacts/ # Trained models and outputs
├── app.py # Streamlit web interface
├── requirements.txt # Dependencies
└── README.md


---

## 🧠 Key Features

- **Multiple Algorithms:** Linear Regression, Random Forest, Gradient Boosting  
- **Feature Engineering:** PCA and scaling for optimal performance  
- **Performance Tracking:** RMSE, MAE, R² metrics  
- **Explainability:** SHAP plots for feature importance  
- **Interactive Web UI:** Built with Streamlit for training and predictions  

---

## 📈 Results Interpretation

- **R² ≈ 0.60** → Model explains 60% of price variability  
- **MAE ≈ 0.51** → Average prediction error ≈ $51,000  
- **Random Forest** offered best balance between accuracy and speed  

---

## 💡 Future Enhancements

- Integrate **MLflow** for experiment tracking  
- Add **FastAPI** for REST API deployment  
- Extend model with **neural network baseline**  
- Automate **hyperparameter optimization** with Optuna  

---

## 🧰 Technologies Used

- **Python 3.8+**
- **scikit-learn**
- **pandas**, **numpy**
- **matplotlib**
- **Streamlit**
- **SHAP**
