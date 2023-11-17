# 88192025_Churning_Customers



---

# Customer Churn Prediction

This repository contains code for predicting customer churn in a telecommunications dataset. The code uses various machine learning techniques to preprocess the data, build models, and evaluate their performance.

## Dataset Overview

The dataset (`CustomerChurn_dataset.csv`) contains information about customers, including features like contract details, services subscribed, and customer churn status.

## Steps Overview

### 1. Data Loading and Exploration
- Read the dataset using Pandas and display the first few rows.
- Visualize the distribution of the 'Churn' variable and churn rates based on senior citizenship using seaborn countplots.
- Conduct a chi-square test for 'SeniorCitizen' and 'Churn' correlation.

### 2. Data Preprocessing
- Drop unnecessary columns like 'customerID'.
- Convert categorical data into numerical labels using label encoding.
- Separate features and target variable ('Churn') for modeling.

### 3. Feature Importance
- Train a Random Forest classifier and determine the top 10 important features.
- Visualize feature importances using a horizontal bar plot.
- Create a new dataset containing only the top 10 features.
- Scale these features using StandardScaler.

### 4. Model Building and Tuning
- Define an MLP model using Keras.
- Split the dataset into training and testing sets, scale the features.
- Perform a grid search with cross-validation to tune hyperparameters for the MLP classifier.

### 5. Model Evaluation
- Evaluate the best-tuned MLP model on the test set.
- Compute metrics like accuracy, precision, recall, F1 score, ROC AUC score, and AUC score for model assessment.

### 6. Model Saving
- Save the trained best MLP model and the scaler object using joblib for future use or deployment.

## File Structure
- `CustomerChurn_dataset.csv`: Dataset file.
- `best_mlp_model.pkl`: Saved best-tuned MLP model.
- `scaler.pkl`: Saved scaler object for feature scaling.

---

