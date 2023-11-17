# -*- coding: utf-8 -*-
"""CChurning.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rj8XTXIGp4ZPFSRx8WCp88aaPge8BSBy
"""



!pip install scikeras

"""Import of necessary libraries for the customer churning task."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
import torch
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from scikeras.wrappers import KerasClassifier
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.model_selection import GridSearchCV

from google.colab import drive
drive.mount('/content/drive')

"""Read the dataset."""

telecom_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/CustomerChurn_dataset.csv')

telecom_df

"""Displays the first 5 parts of the dataset."""

telecom_df.head()

"""This block of code below visualizes the distribution of the 'Churn' variable in the 'telecom_df' dataset using a count plot."""

plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=telecom_df)
plt.title('Churn Distribution')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.show()

"""Visualizinging churn rates in a telecommunications dataset based on senior citizenship."""

plt.figure(figsize=(6, 4))
sns.countplot(x='SeniorCitizen', hue='Churn', data=telecom_df)
plt.title('Senior Citizen and Churn')
plt.xlabel('Senior Citizen')
plt.ylabel('Count')
plt.show()

"""
 Chi-square test for SeniorCitizen and Churn"""

from scipy.stats import chi2_contingency

senior_churn_crosstab = pd.crosstab(telecom_df['SeniorCitizen'], telecom_df['Churn'])
chi2, p, _, _ = chi2_contingency(senior_churn_crosstab)
print(f"Chi-square p-value for SeniorCitizen and Churn: {p}")

"""Code to drop customerID column from the dataset."""

columns_to_drop =['customerID']
telecom_df = telecom_df.drop(columns = columns_to_drop, axis= 1)

"""The code converts categorical data in the `telecom_df` DataFrame into numerical labels using label encoding."""

categorical_columns = telecom_df.select_dtypes(include=['object']).columns.tolist()

label_encoder = LabelEncoder()

for col in categorical_columns:
    telecom_df[col] = label_encoder.fit_transform(telecom_df[col])

telecom_df.head()

"""The code separates a dataset (`telecom_df`) into a set of features (`X`) excluding the 'Churn' column, and a target variable (`Y`) representing the 'Churn' column specifically for analytical purposes."""

X = telecom_df.drop('Churn',axis=1)
Y = telecom_df['Churn']

"""This code  is training a Random Forest classifier using the features (X) and target variable (Y) to create a predictive model."""

rf_classifier = RandomForestClassifier()

rf_classifier.fit(X, Y)

"""This code identifies and displays the top 10 most important features along with their importances by a trained Random Forest classifier."""

feature_importances = rf_classifier.feature_importances_


importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})


importance_df = importance_df.sort_values('Importance', ascending=False)

top_10_features = importance_df['Feature'].head(10).values

top_10_importances = importance_df.head(10)
print(top_10_importances)

"""This code generates a horizontal bar plot that shows the 10 feature importances aiding in visualizing the importance of different features in a model or dataset."""

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=top_10_importances)
plt.title('Top 10 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

"""This code selects the 10 important features from a dataframe storing feature importances and creates a new dataset (X_top_10) containing only these selected features."""

top_10_features = importance_df['Feature'].head(10).tolist()


X_top_10 = X[top_10_features]

"""This code scales the features in X_top_10 using StandardScaler to ensure each feature has a mean of 0 and a standard deviation of 1, preparing the data for machine learning models sensitive to feature scaling."""

scaler = StandardScaler()


X_scaled = scaler.fit_transform(X_top_10)

X_scaled = pd.DataFrame(X_scaled, columns=X_top_10.columns)

"""This code defines a function to create a multi-layer perceptron model using Keras, allowing for customization of optimizer, loss function, and metrics, and then creates a Keras classifier using this defined model architecture."""

def create_mlp_model(input_shape=X_train.shape[1], optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']):
    inputs = Input(shape=(input_shape,))
    hidden1 = Dense(128, activation='relu')(inputs)
    hidden2 = Dense(64, activation='relu')(hidden1)
    output = Dense(1, activation='sigmoid')(hidden2)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model

mlp_classifier = KerasClassifier(build_fn=create_mlp_model, verbose=0)

"""This code splits the dataset into training and testing sets, scaling the features (X) and assigning corresponding target variables (Y), printing the shapes of the resulting train-test split datasets."""

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)


print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of Y_train:", Y_train.shape)
print("Shape of Y_test:", Y_test.shape)

"""This code defines a parameter grid specifying different values for the number of epochs and batch sizes to explore during hyperparameter tuning for the machine learning model."""

param_grid = {
    'epochs': [10, 20, 30],
    'batch_size': [16, 32, 64],

"""This code performs a grid search with cross-validation to systematically explore a set of hyperparameters for the MLP classifier to determine the best combination for model performance based on accuracy."""

grid_search = GridSearchCV(estimator=mlp_classifier, param_grid=param_grid, cv=3, scoring='accuracy')

grid_search.fit(X_train, Y_train)



"""This code displays the best parameters and their corresponding best score achieved through a grid search performed in machine learning model tuning."""

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Score:", best_score)

"""This code evaluates the performance of the best-tuned MLP model on a test set, computing various classification metrics like accuracy, precision, recall, F1 score, ROC AUC score, and AUC score for model assessment."""

best_mlp_model = grid_search.best_estimator_

Y_pred = best_mlp_model.predict(X_test)


accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred)
roc_auc = roc_auc_score(Y_test, Y_pred)
auc_score = roc_auc_score(Y_test, Y_pred)



print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)

"""This code saves the trained best MLP model and a scaler object into separate files using joblib for deployment.

"""

import joblib

joblib.dump(best_mlp_model, 'best_mlp_model.pkl')

joblib.dump(scaler, 'scaler.pkl')

