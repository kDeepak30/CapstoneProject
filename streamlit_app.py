import os
from io import BytesIO

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Must be called before any other Streamlit commands
st.set_page_config(layout="wide", page_title="AI in Mental Health")

# --- Base paths for repo-relative artifacts ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_FILENAME = "new_dt_xgb_voting_classifier.pkl"
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)

DATA_FILENAME = "Mental Health Dataset.csv"
DATA_PATH = os.path.join(DATA_DIR, DATA_FILENAME)

# --- Load the saved model (which is a Pipeline) ---
@st.cache_resource
def load_model(model_path: str = MODEL_PATH):
    """Load the trained model from the models/ directory."""
    if not os.path.exists(model_path):
        st.error(
            f"Model file not found at: {model_path}. "
            "Place the trained model file in the 'models' directory."
        )
        return None
    try:
        loaded_model = joblib.load(model_path)
        return loaded_model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


loaded_model = load_model()

if loaded_model is None:
    # Stop the app if model failed to load
    st.stop()

# --- Load country list from dataset (fallback to default if missing) ---
try:
    original_df = pd.read_csv(DATA_PATH)
    unique_countries_from_df = original_df["Country"].dropna().unique().tolist()
    unique_countries_from_df.sort()
except FileNotFoundError:
    # Fallback if the dataset is not found (e.g., in deployed environment)
    st.warning(
        f"Dataset file not found at: {DATA_PATH}. "
        "Using a default list of countries instead."
    )
    unique_countries_from_df = [
        "United States",
        "Canada",
        "United Kingdom",
        "Australia",
        "Other",
    ]

# --- Define the input features ---
input_features_ordered = [
    "Age",
    "gender",
    "Country",
    "sleep hours (per day)",
    "working hours (per day)",
    "work pressure (1-5)",
]

# --- Define choices for dropdowns ---
feature_choices = {
    "gender": ["Male", "Female"],
    "Country": unique_countries_from_df,
}

st.title("AI in Mental Health")
st.markdown(
    "Predict potential mental health conditions based on various factors "
    "and receive personalized recommendations."
)

# --- Streamlit Input Widgets ---
with st.sidebar:
    st.header("User Input")
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    gender = st.selectbox("Gender", options=feature_choices["gender"])
    country = st.selectbox("Country", options=feature_choices["Country"])
    sleep_hours = st.number_input(
        "Sleep Hours (per day)", min_value=0, max_value=24, value=8
    )
    working_hours = st.number_input(
        "Working Hours (per day)", min_value=0, max_value=24, value=8
    )
    work_pressure = st.number_input(
        "Work Pressure (1-5)", min_value=1, max_value=5, value=3
    )

if st.sidebar.button("Predict Mental Health Risk"):
    # --- Simple heuristic-based prediction logic ---
    num_bad_conditions = 0

    if sleep_hours < 6:
        num_bad_conditions += 1
    if working_hours > 7:
        num_bad_conditions += 1
    if work_pressure > 3:
        num_bad_conditions += 1

    if num_bad_conditions == 0:
        predicted_yes_count = 3
    elif num_bad_conditions == 1:
        predicted_yes_count = 5
    elif num_bad_conditions == 2:
        predicted_yes_count = 8
    else:  # num_bad_conditions == 3
        predicted_yes_count = 10

    treatment_needed = "Yes" if predicted_yes_count >= 6 else "No"

    # --- Generate Suggestions ---
    suggestions = []
    if treatment_needed == "Yes":
        suggestions.append("Consider seeking professional mental health support.")
    if sleep_hours < 6:
        suggestions.append(
            "Improve your sleep hygiene (e.g., consistent bedtime, dark room, "
            "no screens before bed)."
        )
    if working_hours > 7:
        suggestions.append(
            "Focus on maintaining a healthy work-life balance and avoid overworking."
        )
    if work_pressure > 3:
        suggestions.append(
            "Explore stress management techniques like mindfulness, exercise, or hobbies."
        )
    if not suggestions:
        suggestions.append("Keep up your good habits for mental well-being!")

    suggestions_text = "\nRecommendations:\n" + "\n".join(
        [f"- {s}" for s in suggestions]
    )

    # --- Streamlit Output ---
    st.header("Prediction Results")
    st.write(
        f"Predicted Mental Health Risk Score (Yes_Count): "
        f"**{predicted_yes_count}**"
    )
    st.write(f"Treatment Needed: **{treatment_needed}**")
    st.markdown(suggestions_text)

    # --- Generate PDF Report ---
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.drawString(100, 750, "Mental Health Risk Prediction Report")
    c.drawString(100, 730, "-----------------------------------")
    y_position = 710
    c.drawString(100, y_position, f"Age: {age}")
    y_position -= 20
    c.drawString(100, y_position, f"Gender: {gender}")
    y_position -= 20
    c.drawString(100, y_position, f"Country: {country}")
    y_position -= 20
    c.drawString(100, y_position, f"Sleep Hours (per day): {sleep_hours}")
    y_position -= 20
    c.drawString(100, y_position, f"Working Hours (per day): {working_hours}")
    y_position -= 20
    c.drawString(100, y_position, f"Work Pressure (1-5): {work_pressure}")
    y_position -= 20
    c.drawString(
        100,
        y_position,
        f"Predicted Mental Health Risk Score (Yes_Count): {predicted_yes_count}",
    )
    y_position -= 20
    c.drawString(100, y_position, f"Treatment Needed: {treatment_needed}")
    y_position -= 30  # Add space before suggestions
    c.drawString(100, y_position, "Recommendations:")
    y_position -= 20
    for s in suggestions:
        c.drawString(100, y_position, f"- {s}")
        y_position -= 20
    c.save()
    pdf_bytes = buf.getvalue()
    buf.close()

    st.download_button(
        label="Download PDF Report",
        data=pdf_bytes,
        file_name="mental_health_report.pdf",
        mime="application/pdf",
    )

    # --- Generate Contextual Plot ---
    plt.figure(figsize=(8, 6))
    synthetic_ages = np.random.randint(18, 66, size=100)
    synthetic_yes_counts = np.random.randint(0, 10, size=100)
    plt.scatter(
        synthetic_ages,
        synthetic_yes_counts,
        alpha=0.5,
        label="Synthetic Data (Age vs. Yes_Count)",
    )

    plt.scatter(
        age,
        predicted_yes_count,
        color="red",
        s=100,
        zorder=5,
        label="Current Prediction",
    )
    plt.xlabel("Age")
    plt.ylabel("Yes_Count")
    plt.title("Age vs. Mental Health Risk Score")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)