# Importing necessary libraries
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense

# Defining the MLP model  
def create_mlp_model(input_shape):
    inputs = Input(shape=(input_shape,))
    hidden1 = Dense(64, activation='relu')(inputs)
    hidden2 = Dense(32, activation='relu')(hidden1)
    hidden3 = Dense(16, activation='relu')(hidden2)
    output = Dense(1, activation='sigmoid')(hidden3)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Loading the trained machine learning model and scaler 
optimized_model = joblib.load('best_mlp_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to preprocess user input 
def preprocess_input(tenure, MonthlyCharges, TotalCharges, Contract, PaymentMethod, OnlineSecurity, TechSupport, InternetService, gender, OnlineBackup):
    tenure = int(tenure) 
    MonthlyCharges = float(MonthlyCharges) 
    TotalCharges = float(TotalCharges)  
    Contract = 0 if Contract == 'Month-to-month' else 1 if Contract == 'One year' else 2
    PaymentMethod = 0 if PaymentMethod == 'Electronic check' else 1 if PaymentMethod == 'Mailed check' else 2 if PaymentMethod == 'Bank transfer (automatic)' else 3
    OnlineSecurity = 1 if OnlineSecurity == 'Yes' else 0
    TechSupport = 1 if TechSupport == 'Yes' else 0
    InternetService = 0 if InternetService == 'DSL' else 1  
    gender = 0 if gender == 'Male' else 1
    OnlineBackup = 0 if OnlineBackup == 'Yes' else 1

    user_input = pd.DataFrame({
        'MonthlyCharges': [MonthlyCharges],
        'tenure': [tenure],
        'TotalCharges': [TotalCharges],
        'Contract': [Contract],
        'PaymentMethod': [PaymentMethod],
        'TechSupport': [TechSupport],
        'OnlineSecurity': [OnlineSecurity],
        'InternetService': [InternetService],
        'gender': [gender],
        'OnlineBackup': [OnlineBackup]
    })

    return user_input.values  

# Main function for Streamlit app
def main():
    st.set_page_config(
        page_title='Customer Churn Prediction',
        page_icon='🔄', 
        layout='wide',
        initial_sidebar_state='expanded'
    )

    st.title('Customer Churn Prediction 🔄')
    st.write(
        """
        Welcome to the Customer Churn Prediction app! 🔄
        Enter the customer details on the left, and we'll predict the likelihood of churn.
        """
    )

    # Input features in the sidebar
    st.sidebar.header('Customer Details  :)')

    # Sidebar input fields for user input
    tenure = st.sidebar.text_input('Tenure', '24')  
    MonthlyCharges = st.sidebar.text_input('Monthly Charges', '50.0')  
    TotalCharges = st.sidebar.text_input('Total Charges', '2000.0')  
    Contract = st.sidebar.selectbox('Contract', ['Month-to-month', 'One year', 'Two years'])
    PaymentMethod = st.sidebar.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    OnlineSecurity = st.sidebar.selectbox('Online Security', ['No', 'Yes'])
    TechSupport = st.sidebar.selectbox('Tech Support', ['No', 'Yes'])
    InternetService = st.sidebar.selectbox('Internet Service', ['DSL', 'Fiber optic'])
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    OnlineBackup = st.sidebar.selectbox('Online Backup', ['Yes', 'No'])

    if st.sidebar.button('Predict'):
        # Preprocessing the user input (scaling) and making predictions
        user_input_scaled = scaler.transform(preprocess_input(tenure, MonthlyCharges, TotalCharges, Contract, PaymentMethod, OnlineSecurity, TechSupport, InternetService, gender, OnlineBackup))

        prediction = optimized_model.predict(user_input_scaled)

        st.subheader('Prediction')
        predicted_likelihood = prediction[0] if isinstance(prediction, (list, np.ndarray)) else prediction

        if predicted_likelihood >= 0.5:
            confidence_factor = predicted_likelihood
            churn_decision = "Yes, customer is likely to churn"
        else:
            confidence_factor = 1 - predicted_likelihood
            churn_decision = "No, customer is not likely to churn"

        st.write(f'The predicted likelihood of churn (Confidence Factor) is: {confidence_factor:.2f}')
        st.write(f'Decision: {churn_decision}')

# Run the Streamlit app
if __name__ == '__main__':
    main()
