import streamlit as st

with st.form("my_form"):
    # Your input fields here
    name = st.text_input("Name")
    age = st.number_input("Age", 0, 120)
    
    # This line creates the submit button
    submitted = st.form_submit_button("Submit")
    
    # Code to run after pressing submit
    if submitted:
        st.success(f"Hello {name}, you are {age} years old.")