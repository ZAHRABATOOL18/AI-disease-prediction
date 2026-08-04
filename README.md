# AI-disease-prediction
🩺 AI Disease Prediction System
An end-to-end Machine Learning application that predicts the likelihood of multiple diseases using structured medical datasets. This project integrates data preprocessing, model training, hyperparameter tuning, performance evaluation, visualization, and an interactive Streamlit web application into a single workflow.
📌 Project Overview
The system predicts diseases using supervised machine learning algorithms trained on real-world medical datasets. It evaluates and compares multiple models to identify the best-performing classifier for each disease based on evaluation metrics.
Supported diseases include:
Breast Cancer
Heart Disease
Diabetes
The project follows a complete machine learning pipeline from data preprocessing to deployment.
✨ Features
🔐 User Registration & Login System
🤖 AI-powered Disease Prediction
💬 Interactive Medical Chatbot
📊 Dashboard for Model Performance
📈 Hyperparameter Tuning using GridSearchCV
📉 ROC Curve Visualization
📊 Confusion Matrix Generation
📌 Feature Importance Analysis
📄 Automatic Classification Reports
💾 Trained Model Saving using Pickle
🌐 Interactive Streamlit Web Interface
🧠 Machine Learning Models
The project compares multiple classification algorithms:
Logistic Regression
Support Vector Machine (RBF Kernel)
Random Forest Classifier
XGBoost (with Gradient Boosting fallback)
Hyperparameter tuning is performed to optimize model performance and improve prediction accuracy.
📊 Performance Evaluation
Models are evaluated using:
Accuracy
Precision
Recall
F1-Score
ROC-AUC Score
Confusion Matrix
ROC Curves
Classification Reports
Feature Importance Visualization
Training Time Comparison
The best-performing model for each dataset is automatically selected based on the F1-Score.
📂 Datasets
The application is trained using publicly available medical datasets:
Breast Cancer Wisconsin Dataset
UCI Heart Disease Dataset
Pima Indians Diabetes Dataset
🛠 Technologies Used
Programming Language
Python
Machine Learning
Scikit-learn
XGBoost
NumPy
Pandas
Data Visualization
Matplotlib
Seaborn
Web Application
Streamlit
Model Persistence
Pickle
👩‍💻 Author
Zahra Batool
Software Engineering Student | Machine Learning Enthusiast | Python Developer
