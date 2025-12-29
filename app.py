import streamlit as st
import io

from predict import analyze_image, guided_gradcam_png

st.title("Skin Scan")

uploaded_file = st.file_uploader("Upload a file", type=["png", "jpg", "jpeg"])

if st.button("Analyze"):
    if uploaded_file:
        try:
            image_bytes = uploaded_file.getvalue()
            result = analyze_image(io.BytesIO(image_bytes))
            st.write(result)

            # Generate the Guided Grad-CAM image
            gradcam_image = guided_gradcam_png(image_bytes)

            # Display the image with a larger width
            st.image(gradcam_image, caption="Guided Grad-CAM", use_column_width=True)
        except Exception as exc:
            st.error(str(exc))
    else:
        st.warning("Please upload a file first")