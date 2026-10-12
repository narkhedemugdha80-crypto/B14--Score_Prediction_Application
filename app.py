import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern layout and card shadows
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .card {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4F46E5;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #4338CA;
        color: white;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Load the trained KNN model safely
@st.cache_resource
def load_model():
  try:
    with open("knn.pkl", "rb") as f:
      model = pickle.load(f)
    return model
  except Exception as e:
    st.error(f"Error loading model file: {e}")
    return None


model = load_model()

# App Header
st.markdown(
    """
    <div class="card" style="text-align: center;">
        <h2>🎓 Student Performance Prediction App</h2>
        <p style="color: #6B7280;">Enter student details below to predict performance metrics using a K-Nearest Neighbors model.</p>
    </div>
""",
    unsafe_allow_html=True,
)

if model is not None:
  with st.form("prediction_form"):
    st.markdown("### 📝 Input Features")

    col1, col2, col3 = st.columns(3)

    with col1:
      age = st.number_input("Age", min_value=15, max_value=60, value=20)
      gender = st.selectbox("Gender", ["Male", "Female", "Other"])
      course = st.selectbox(
          "Course",
          ["Engineering", "Business", "Arts", "Science", "Medicine", "Other"],
      )
      study_hours = st.slider(
          "Study Hours (per day)", 0.0, 16.0, 4.0, step=0.5
      )

    with col2:
      class_attendance = st.slider(
          "Class Attendance (%)", 0.0, 100.0, 75.0, step=1.0
      )
      internet_access = st.selectbox("Internet Access", ["Yes", "No"])
      sleep_hours = st.slider("Sleep Hours (per night)", 0.0, 12.0, 7.0, step=0.5)
      sleep_quality = st.selectbox("Sleep Quality", ["Poor", "Average", "Good"])

    with col3:
      study_method = st.selectbox(
          "Study Method", ["Self-Study", "Group", "Online", "Mixed"]
      )
      facility_rating = st.slider("Facility Rating (1-5)", 1, 5, 3)
      exam_difficulty = st.slider("Exam Difficulty (1-5)", 1, 5, 3)

    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.form_submit_button("Predict Performance")

  if submit_button:
    # Note: Depending on how your original model was trained, categorical variables
    # may need encoding (e.g., Label Encoding or One-Hot Encoding).
    # Adjust the mapping below to match your training pipeline if necessary:

    # Example placeholder numeric conversion for categorical fields:
    gender_val = 1 if gender == "Male" else (0 if gender == "Female" else 2)
    internet_val = 1 if internet_access == "Yes" else 0

    # Map categorical strings to dummy or integer values matching your training data:
    # If your model expects encoded categorical inputs, update these arrays accordingly.
    features = np.array([[age, gender_val, 0, study_hours, class_attendance, internet_val, 
                          sleep_hours, 1, 0, facility_rating, exam_difficulty]])

    try:
      prediction = model.predict(features)
      st.markdown(
          f"""
            <div class="card" style="background-color: #EEF2FF; text-align: center; border: 1px solid #C7D2FE;">
                <h3 style="color: #3730A3;">Prediction Result</h3>
                <p style="font-size: 1.5rem; font-weight: bold; color: #4F46E5;">{prediction[0]:.2f}</p>
            </div>
            """,
          unsafe_allow_html=True,
      )
    except Exception as e:
      st.error(
          "Prediction failed. Ensure that categorical variables match the exact"
          f" format/encoding expected by the model. Details: {e}"
      )
else:
  st.warning(
      "Please place your `knn.pkl` file in the same directory as `app.py`."
  )
