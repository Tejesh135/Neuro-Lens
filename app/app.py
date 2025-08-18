import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import google.generativeai as genai

# --- Load model and preprocessor ---
model = load_model('mlp_model.h5')
preprocessor = joblib.load('preprocessor.save')

# --- Configure Gemini API key ---
genai.configure(api_key="AIzaSyCeKFco-Crl5XaOvMkNPIpy7_DIRLa5nmw")

st.set_page_config(page_title="Student Depression Prediction & Chatbot")

tab1, tab2 = st.tabs(["🧠 Depression Predictor", "💬 Help Chatbot"])

with tab1:
    st.title("🧠 Student Depression Prediction")

    with st.form("input_form"):
        st.header("👤 Demographic Information")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.number_input("Age", min_value=10.0, max_value=80.0, value=25.0, step=0.1)
        city = st.text_input("City")
        profession = st.selectbox("Profession", ["Student", "Working", "Other"])

        st.header("📚 Academic or Professional Life")
        academic_pressure = st.number_input("Academic Pressure (0–10)", min_value=0.0, max_value=10.0, value=5.0)
        work_pressure = st.number_input("Work Pressure (0–10)", min_value=0.0, max_value=10.0, value=0.0)
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5, step=0.01)
        study_satisfaction = st.number_input("Study Satisfaction (0–10)", min_value=0.0, max_value=10.0, value=5.0)
        job_satisfaction = st.number_input("Job Satisfaction (0–10)", min_value=0.0, max_value=10.0, value=0.0)
        degree = st.text_input("Degree (e.g., B.Tech, M.Tech, etc.)")

        st.header("🛌 Lifestyle & Well-being")
        sleep_duration = st.selectbox("Sleep Duration", ["<5 hours", "5-6 hours", "6-7 hours", "7+ hours"])
        dietary_habits = st.text_input("Dietary Habits (e.g., Vegetarian, Non-Vegetarian, etc.)")
        suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["Yes", "No"])
        work_study_hours = st.number_input("Work/Study Hours/day", min_value=0.0, max_value=24.0, value=4.0, step=0.1)
        financial_stress = st.number_input("Financial Stress (0–10)", min_value=0.0, max_value=10.0, value=1.0)
        family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_dict = {
            "Gender": gender,
            "Age": age,
            "City": city,
            "Profession": profession,
            "Academic Pressure": academic_pressure,
            "Work Pressure": work_pressure,
            "CGPA": cgpa,
            "Study Satisfaction": study_satisfaction,
            "Job Satisfaction": job_satisfaction,
            "Sleep Duration": sleep_duration,
            "Dietary Habits": dietary_habits,
            "Degree": degree,
            "Have you ever had suicidal thoughts ?": suicidal_thoughts,
            "Work/Study Hours": work_study_hours,
            "Financial Stress": financial_stress,
            "Family History of Mental Illness": family_history
        }

        input_df = pd.DataFrame([input_dict])

        X_processed = preprocessor.transform(input_df)
        proba = float(model.predict(X_processed)[0, 0])
        prediction = int(proba > 0.5)

        st.success(f"Prediction: {'Depression' if prediction else 'No Depression'} (Probability: {proba:.2f})")
        if prediction:
            st.info("This indicates possible depression risk. Please consider consulting a professional or use the chatbot for support.")
        else:
            st.info("No indication of depression risk detected. Stay healthy!")

with tab2:
    st.title("💬 Well-being Chatbot (Gemini)")
    st.write("Chat with our AI Assistant for advice, suggestions, and emotional support.")

    # Store chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Type your message here...")

    if st.button("Send"):
        if user_input:
            st.session_state.chat_history.append(("You", user_input))
            model = genai.GenerativeModel(model_name="gemini-2.5-flash")
            response = model.generate_content(user_input)
            bot_reply = response.text
            st.session_state.chat_history.append(("Gemini", bot_reply))

    # Display chat history
    for sender, text in st.session_state.chat_history:
        if sender == "You":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Gemini:** {text}")

    st.info("💡 Note: This chatbot is for supportive conversation and suggestions only. If you feel very distressed or need urgent help, please consult a professional or call a mental health helpline.")