from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student_mental_health.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_optimized_model.joblib"
PREPROCESSOR_PATH = PROJECT_ROOT / "data" / "processed" / "preprocessor.joblib"
ENCODER_PATH = PROJECT_ROOT / "data" / "processed" / "target_label_encoder.joblib"


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    for path in [MODEL_PATH, PREPROCESSOR_PATH, ENCODER_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
    return (
        joblib.load(MODEL_PATH),
        joblib.load(PREPROCESSOR_PATH),
        joblib.load(ENCODER_PATH),
    )


st.set_page_config(
    page_title="Mental health risk prediction",
    page_icon="🧠",
    layout="wide",
)

st.title("Student mental health risk prediction")
st.warning(
    "This application is educational only. It is not a diagnostic, "
    "clinical screening, disciplinary, or student selection tool."
)

try:
    data = load_data()
    model, preprocessor, target_encoder = load_model()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()


st.header("Model performance")
metrics_path = PROJECT_ROOT / "results" / "optimized_model_test_results.csv"
confusion_path = (
    PROJECT_ROOT / "results" / "logistic_regression_optimized_confusion_matrix.csv"
)

if metrics_path.exists():
    metrics = pd.read_csv(metrics_path)
    st.dataframe(metrics, use_container_width=True, hide_index=True)
else:
    st.info("No optimized model metrics file was found.")

if confusion_path.exists():
    st.subheader("Confusion matrix")
    confusion_matrix = pd.read_csv(confusion_path, index_col=0)
    st.dataframe(confusion_matrix, use_container_width=True)
else:
    st.info("No confusion matrix file was found.")


st.header("Test a new student")
st.write("Complete the form to obtain a technical model prediction.")

with st.form("prediction_form"):
    age = st.number_input("Age", min_value=15, max_value=100, value=21)
    gender = st.selectbox("Gender", sorted(data["Gender"].dropna().unique()))
    year_of_study = st.number_input(
        "Year of study",
        min_value=1,
        max_value=10,
        value=2,
    )
    sleep_hours = st.number_input(
        "Sleep hours",
        min_value=0.0,
        max_value=24.0,
        value=7.0,
    )
    study_hours = st.number_input(
        "Study hours per week",
        min_value=0.0,
        max_value=168.0,
        value=25.0,
    )
    gpa = st.number_input("GPA", min_value=0.0, max_value=4.0, value=3.0)
    social_media_hours = st.number_input(
        "Social media hours per day",
        min_value=0.0,
        max_value=24.0,
        value=3.0,
    )
    screen_time_hours = st.number_input(
        "Screen time hours per day",
        min_value=0.0,
        max_value=24.0,
        value=6.0,
    )
    physical_activity = st.selectbox(
        "Physical activity",
        sorted(data["Physical_Activity"].dropna().unique()),
    )
    financial_stress = st.number_input(
        "Financial stress",
        min_value=1.0,
        max_value=10.0,
        value=5.0,
    )
    family_support = st.number_input(
        "Family support",
        min_value=1.0,
        max_value=10.0,
        value=5.0,
    )
    attendance_rate = st.number_input(
        "Attendance rate",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
    )
    part_time_job = st.selectbox(
        "Part-time job",
        sorted(data["Part_Time_Job"].dropna().unique()),
    )
    diet_quality = st.selectbox(
        "Diet quality",
        sorted(data["Diet_Quality"].dropna().unique()),
    )
    previous_history = st.selectbox(
        "Previous mental health history",
        sorted(data["Previous_Mental_Health_History"].dropna().unique()),
    )
    submitted = st.form_submit_button("Predict risk level")

if submitted:
    input_data = pd.DataFrame(
        [
            {
                "Age": age,
                "Gender": gender,
                "Year_of_Study": year_of_study,
                "Sleep_Hours": sleep_hours,
                "Study_Hours_Per_Week": study_hours,
                "GPA": gpa,
                "Social_Media_Hours": social_media_hours,
                "Screen_Time_Hours": screen_time_hours,
                "Physical_Activity": physical_activity,
                "Financial_Stress": financial_stress,
                "Family_Support": family_support,
                "Attendance_Rate": attendance_rate,
                "Part_Time_Job": part_time_job,
                "Diet_Quality": diet_quality,
                "Previous_Mental_Health_History": previous_history,
            }
        ]
    )
    processed_data = preprocessor.transform(input_data)
    prediction = model.predict(processed_data)[0]
    risk_level = target_encoder.inverse_transform([prediction])[0]
    st.success(f"Predicted risk level: **{risk_level}**")