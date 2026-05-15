# AI510-Team-Project

# Part 1  Data Cleaning & Preprocessing

### Dataset
- Original dataset: customer_support_dataset.csv
- Cleaned dataset: cleaned_customer_support_dataset.csv

### What I did
- Removed missing values
- Standardized text fields
- Cleaned inconsistent data
- Prepared dataset for model training

### File
- preprocessing/data_cleaning.py: script for cleaning dataset

- ## What I did

### Part 2 — ML Model Development
- Built intent classification model (27 intents)
- Used TF-IDF vectorizer + Logistic Regression pipeline
- 5-fold cross-validation on training set
- Test Accuracy: 99.57% | F1 Score: 0.9957

### Flask API Deployment  
- Built REST API with /predict and /health endpoints
- Created chat UI (ShopBot) served at /
- Model loads automatically on startup
- Dockerfile included for containerization

## Files added
- models/model_training.py
- app.py
- Dockerfile
- requirements.txt
