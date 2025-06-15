
import streamlit as st
import pandas as pd
import joblib

# Load the trained Random Forest model pipeline
# Ensure the model file is in the same directory or provide the correct path
try:
    model_pipeline = joblib.load("heart_disease_model.pkl")
except FileNotFoundError:
    st.error("Model file (heart_disease_model.pkl) not found. Please ensure it is in the correct directory.")
    st.stop()

# Define the input features based on the model training
# These should match the columns used during training (excluding the target)
feature_order = [
    'age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol',
    'fasting blood sugar', 'resting ecg', 'max heart rate', 'exercise angina',
    'oldpeak', 'ST slope'
]

# Streamlit App Interface
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.title("Heart Disease Prediction App")
st.markdown("""
This app uses a Random Forest model to predict the likelihood of heart disease based on patient data.
Fill in the details below to get a prediction.
""")

st.sidebar.header("Patient Data Input")

# Create input fields in the sidebar
def user_input_features():
    age = st.sidebar.slider("Age (years)", 28, 77, 54)
    sex = st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.sidebar.selectbox("Chest Pain Type", options=[1, 2, 3, 4],
                              format_func=lambda x: {1: "Typical Angina", 2: "Atypical Angina",
                                                     3: "Non-Anginal Pain", 4: "Asymptomatic"}.get(x))
    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 130)
    chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 0, 603, 240) # Adjusted max based on potential outliers or data range
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1],
                               format_func=lambda x: "False" if x == 0 else "True")
    restecg = st.sidebar.selectbox("Resting Electrocardiogram Results", options=[0, 1, 2],
                                   format_func=lambda x: {0: "Normal", 1: "ST-T Wave Abnormality",
                                                          2: "Probable/Definite LV Hypertrophy"}.get(x))
    thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 60, 202, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", options=[0, 1],
                                 format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak = st.sidebar.number_input("Oldpeak (ST depression induced by exercise relative to rest)", min_value=-3.0, max_value=7.0, value=1.0, step=0.1)
    slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment", options=[1, 2, 3],
                                 format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}.get(x))

    data = {
        'age': age,
        'sex': sex,
        'chest pain type': cp,
        'resting bp s': trestbps,
        'cholesterol': chol,
        'fasting blood sugar': fbs,
        'resting ecg': restecg,
        'max heart rate': thalach,
        'exercise angina': exang,
        'oldpeak': oldpeak,
        'ST slope': slope
    }
    features = pd.DataFrame(data, index=[0])
    # Reorder columns to match the training order
    features = features[feature_order]
    return features

input_df = user_input_features()

st.subheader("Patient Data Summary")
st.write(input_df)

# Prediction
if st.button("Predict Heart Disease Status"):
    try:
        prediction = model_pipeline.predict(input_df)
        prediction_proba = model_pipeline.predict_proba(input_df)

        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error("Prediction: **Heart Disease Detected**")
        else:
            st.success("Prediction: **No Heart Disease Detected**")

        st.subheader("Prediction Probability")
        st.write(f"Probability of No Heart Disease: {prediction_proba[0][0]:.2f}")
        st.write(f"Probability of Heart Disease: {prediction_proba[0][1]:.2f}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("""
*Disclaimer: This prediction is based on a machine learning model and should not replace professional medical advice.*
""")

