import streamlit as st
import joblib
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense

def create_mlp_model(input_shape):
    inputs = Input(shape=(input_shape,))
    hidden1 = Dense(64, activation='relu')(inputs)
    hidden2 = Dense(32, activation='relu')(hidden1)
    hidden3 = Dense(16, activation='relu')(hidden2)
    output = Dense(1, activation='sigmoid')(hidden3)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

model = joblib.load('best_mlp_model.pkl')
scaler = joblib.load('scaler.pkl')

def preprocess_input(tenure, MonthlyCharges, TotalCharges, Contract, PaymentMethod, OnlineSecurity, TechSupport, InternetService, gender, OnlineBackup):
    tenure = int(tenure)
    MonthlyCharges = float(MonthlyCharges)
    TotalCharges = float(TotalCharges)
    
    contract_mapping = {'Month-to-month': 0, 'One year': 1, 'Two years': 2}
    payment_method_mapping = {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3}
    yes_no_mapping = {'No': 0, 'Yes': 1}
    internet_service_mapping = {'DSL': 0, 'Fiber optic': 1}
    gender_mapping = {'Male': 0, 'Female': 1}
    
    Contract = contract_mapping.get(Contract, 0)
    PaymentMethod = payment_method_mapping.get(PaymentMethod, 0)
    OnlineSecurity = yes_no_mapping.get(OnlineSecurity, 0)
    TechSupport = yes_no_mapping.get(TechSupport, 0)
    InternetService = internet_service_mapping.get(InternetService, 0)
    gender = gender_mapping.get(gender, 0)
    OnlineBackup = yes_no_mapping.get(OnlineBackup, 0)

    user_input = pd.DataFrame({
        'Monthly Charges': [MonthlyCharges],
        'Tenure': [tenure],
        'Total Charges': [TotalCharges],
        'Contract': [Contract],
        'Payment Method': [PaymentMethod],
        'Tech Support': [TechSupport],
        'Online Security': [OnlineSecurity],
        'Internet Service': [InternetService],
        'gender': [gender],
        'Online Backup': [OnlineBackup]
    })

    return user_input.values

def main():
    st.title('Churn Prediction')
    st.write("Enter customer details on the left for churn prediction.")

    tenure = st.sidebar.text_input('Tenure', '24')
    MonthlyCharges = st.sidebar.text_input('Monthly Charges', '50.0')
    TotalCharges = st.sidebar.text_input('Total Charges', '2000.0')
    Contract = st.sidebar.radio('Contract', ['Month-to-month', 'One year', 'Two years'])
    PaymentMethod = st.sidebar.radio('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    OnlineSecurity = st.sidebar.radio('Online Security', ['No', 'Yes'])
    TechSupport = st.sidebar.radio('Tech Support', ['No', 'Yes'])
    InternetService = st.sidebar.radio('Internet Service', ['DSL', 'Fiber optic'])
    gender = st.sidebar.radio('Gender', ['Male', 'Female'])
    OnlineBackup = st.sidebar.radio('Online Backup', ['Yes', 'No'])

    user_input_scaled = scaler.transform(preprocess_input(tenure, MonthlyCharges, TotalCharges, Contract, PaymentMethod, OnlineSecurity, TechSupport, InternetService, gender, OnlineBackup))
    prediction = model.predict(user_input_scaled)

    st.subheader('Prediction')

    prediction = prediction[0] if isinstance(prediction, (list, np.ndarray)) else prediction
    is_churn = prediction >= 0.5
    
    confidence = abs(prediction - 0.5) + 0.5 if is_churn else 1 - abs(prediction - 0.5)
    choice = "Likely to churn" if is_churn else "Not likely to churn"

    st.write(f'Confidence: {confidence:.2f}')
    st.write(f'Decision: {choice}')

if _name_ == '_main_':
    main()
