import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load and preprocess dataset
@st.cache_data
def load_data():
    data = pd.read_csv("adult.csv")
    data.replace("?", np.nan, inplace=True)
    data.dropna(inplace=True)

    # Drop less relevant columns
    data.drop(columns=["fnlwgt"], inplace=True)

    # Encode categorical columns
    label_encoders = {}
    for column in data.select_dtypes(include="object").columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    return data, label_encoders

def train_model(data):
    X = data.drop(columns=["income"])
    y = data["income"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    return model, scaler, acc

# Main Streamlit app
st.title("Employee Salary Category Predictor")
st.write("Upload your employee attributes to predict if income is >50K or <=50K.")

data, encoders = load_data()
model, scaler, accuracy = train_model(data)

st.success(f"Model trained with accuracy: {accuracy * 100:.2f}%")

# Input form
with st.form("prediction_form"):
    st.header("Enter Employee Details")

    age = st.slider("Age", 17, 90, 30)
    workclass = st.selectbox("Workclass", encoders["workclass"].classes_)
    education = st.selectbox("Education", encoders["education"].classes_)
    education_num = st.slider("Education Number", 1, 16, 9)
    marital_status = st.selectbox("Marital Status", encoders["marital-status"].classes_)
    occupation = st.selectbox("Occupation", encoders["occupation"].classes_)
    relationship = st.selectbox("Relationship", encoders["relationship"].classes_)
    race = st.selectbox("Race", encoders["race"].classes_)
    gender = st.selectbox("Gender", encoders["gender"].classes_)
    capital_gain = st.number_input("Capital Gain", value=0)
    capital_loss = st.number_input("Capital Loss", value=0)
    hours_per_week = st.slider("Hours per week", 1, 100, 40)
    native_country = st.selectbox("Native Country", encoders["native-country"].classes_)

    submit = st.form_submit_button("Predict")

    if submit:
        input_data = pd.DataFrame([[
            age,
            encoders["workclass"].transform([workclass])[0],
            encoders["education"].transform([education])[0],
            education_num,
            encoders["marital-status"].transform([marital_status])[0],
            encoders["occupation"].transform([occupation])[0],
            encoders["relationship"].transform([relationship])[0],
            encoders["race"].transform([race])[0],
            encoders["gender"].transform([gender])[0],
            capital_gain,
            capital_loss,
            hours_per_week,
            encoders["native-country"].transform([native_country])[0],
        ]], columns=[
            "age", "workclass", "education", "educational-num", "marital-status", "occupation",
            "relationship", "race", "gender", "capital-gain", "capital-loss", "hours-per-week",
            "native-country"
        ])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        label = encoders["income"].inverse_transform([prediction])[0]

        st.subheader("Prediction")
        st.write(f"The employee is likely to earn: **{label}**")
