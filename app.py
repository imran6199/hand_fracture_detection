import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# --- Streamlit Page Config ---
st.set_page_config(page_title="Hand Fracture Detection Using YOLO", page_icon="🩻", layout="centered")
st.title("🩻 Hand Fracture Detection App")
st.write("Upload an X-ray image to detect fractures using your trained YOLO model.")

# --- Load the YOLO model ---
@st.cache_resource
def load_model():
    model_path = "best.pt"  # Make sure your model file is in the same folder
    model = YOLO(model_path)
    return model

model = load_model()

# --- File Uploader ---
uploaded_file = st.file_uploader("📤 Upload an X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_image_path = tmp_file.name

    # --- Run YOLO prediction ---
    with st.spinner("🔍 Detecting fractures..."):
        results = model.predict(source=temp_image_path, conf=0.25, save=False, show=False)
        result_image = results[0].plot()  # returns numpy array with boxes

    # --- Display Result ---
    st.image(result_image, caption="🩻 Detected Fractures", use_container_width=True)

    # --- Option to Download ---
    result_image_pil = Image.fromarray(result_image)
    output_path = "fracture_result.jpg"
    result_image_pil.save(output_path)

    with open(output_path, "rb") as f:
        st.download_button("💾 Download Result", f, file_name="fracture_result.jpg", mime="image/jpeg")
