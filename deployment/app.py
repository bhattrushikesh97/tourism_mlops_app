
# ============================================
# IMPORTS
# ============================================
import streamlit as st
import pandas as pd
import joblib
import os
from huggingface_hub import hf_hub_download

# ============================================
# PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
# ============================================
st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="✈️",
    layout="wide"
)

# ============================================
# CONFIGURATION
# ============================================
MODEL_REPO = "bhattrushikesh97/tourism-mlops-app"
MODEL_FILE = "Tourism_Package_Prediction.joblib"
HF_TOKEN = os.getenv("HF_TOKEN")  # Optional (only if private repo)

# ============================================
# LOAD MODEL (CACHED)
# ============================================
@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            token=HF_TOKEN
        )
        return joblib.load(model_path)
    except Exception as e:
        return str(e)

model = load_model()

# ============================================
# UI HEADER
# ============================================
st.title("✈️ Tourism Package Prediction App")

st.markdown("""
This app predicts whether a customer is likely to purchase a tourism package
based on demographic and behavioral features.
""")

# ============================================
# ERROR HANDLING
# ============================================
if isinstance(model, str):
    st.error(f"❌ Model loading failed: {model}")
    st.stop()

# ============================================
# INPUT FORM
# ============================================
st.header("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", 18, 100, 30)
    TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    DurationOfPitch = st.number_input("Duration Of Pitch", 1, 60, 10)
    Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    NumberOfPersonVisiting = st.number_input("Persons Visiting", 1, 10, 2)
    NumberOfFollowups = st.number_input("Followups", 0, 10, 2)

with col2:
    ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    PreferredPropertyStar = st.selectbox("Property Star", [3, 4, 5])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    NumberOfTrips = st.number_input("Trips per Year", 0, 20, 2)
    Passport = st.selectbox("Passport", [0, 1])
    PitchSatisfactionScore = st.selectbox("Satisfaction Score", [1, 2, 3, 4, 5])
    OwnCar = st.selectbox("Own Car", [0, 1])
    NumberOfChildrenVisiting = st.number_input("Children Visiting", 0, 5, 0)
    Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    MonthlyIncome = st.number_input("Monthly Income", 1000, 500000, 30000)

# ============================================
# CREATE INPUT DATAFRAME
# ============================================
input_df = pd.DataFrame({
    "Age": [Age],
    "TypeofContact": [TypeofContact],
    "CityTier": [CityTier],
    "DurationOfPitch": [DurationOfPitch],
    "Occupation": [Occupation],
    "Gender": [Gender],
    "NumberOfPersonVisiting": [NumberOfPersonVisiting],
    "NumberOfFollowups": [NumberOfFollowups],
    "ProductPitched": [ProductPitched],
    "PreferredPropertyStar": [PreferredPropertyStar],
    "MaritalStatus": [MaritalStatus],
    "NumberOfTrips": [NumberOfTrips],
    "Passport": [Passport],
    "PitchSatisfactionScore": [PitchSatisfactionScore],
    "OwnCar": [OwnCar],
    "NumberOfChildrenVisiting": [NumberOfChildrenVisiting],
    "Designation": [Designation],
    "MonthlyIncome": [MonthlyIncome]
})

# ============================================
# PREDICTION BUTTON
# ============================================
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.subheader("Prediction Result")
        st.write(f"📊 Probability of Purchase: **{probability:.2f}**")

        if prediction == 1:
            st.success("✅ Customer is likely to purchase the package")
        else:
            st.error("❌ Customer is NOT likely to purchase the package")

    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")
