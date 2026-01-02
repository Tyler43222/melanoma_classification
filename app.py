import streamlit as st
import io
from predict import analyze_image, guided_gradcam_png

st.title("Skin Scan")
st.write("")
st.markdown("For accurate results make sure skinspot is: (1) large in frame, (2) in-focus, (3) brightly lit.<br>Best if taken from a high-resolution digital camera", unsafe_allow_html=True)
st.write("")
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
st.write("")

if st.button("Analyze"):
    if uploaded_file:
        try:
            image_bytes = uploaded_file.getvalue()
            result = analyze_image(io.BytesIO(image_bytes))
            st.write("")
            st.write("*These results are not a diagnosis*")
            st.markdown(f"<h3>RESULTS: Benign: {result[0]}%, Malignant: {result[1]}%</h3>", unsafe_allow_html=True)
            st.write("")

            # Generate and display the Grad-CAM image
            st.image(guided_gradcam_png(image_bytes), caption="Grad-CAM gradient visualizing the regions most influential in the model’s prediction", width='stretch')

        except Exception as exc:
            st.error(str(exc))
    else:
        st.warning("Please upload a file first")