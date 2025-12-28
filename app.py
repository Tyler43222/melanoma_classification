import streamlit as st
from predict import analyze_image

st.title("Skin Scan")

uploaded_file = st.file_uploader("Upload a file", type=["png", "jpg", "jpeg"])

if st.button("Analyze"):
    if uploaded_file:
        result = analyze_image(uploaded_file)
        st.write(result)
    else:
        st.warning("Please upload a file first")