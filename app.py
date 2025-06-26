# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image

# # ─── 1. LOAD THE MODEL ONCE (cached) ───────────────────────────────────────────
# @st.cache_resource(show_spinner="Loading model…")
# def load_trained_model():
#     return tf.keras.models.load_model("malaria_cnn.h5")  # or "malaria_cnn.h5"

# model = load_trained_model()

# # ─── 2. STREAMLIT UI ───────────────────────────────────────────────────────────
# st.set_page_config(page_title="Malaria Cell Diagnosis", layout="centered")
# st.title("🩸 Malaria Cell Image Diagnosis (CNN)")

# uploaded_file = st.file_uploader(
#     "Upload a *single* cell image (JPG/PNG, RGB).", type=["jpg", "jpeg", "png"]
# )

# if uploaded_file is not None:
#     img = Image.open(uploaded_file).convert("RGB")
#     st.image(img, caption="Your image", use_column_width=True)


#     img_resized = img.resize((64, 64))    # Resize to (64, 64)   
#     img_arr = np.asarray(img_resized, dtype=np.float32) / 255.0 # Normalize to [0, 1]
#     img_arr = np.expand_dims(img_arr, axis=0)    # shape (1, 64, 64, 3)


#     if st.button("🔍 Predict"):
#         prob = model.predict(img_arr)[0][0]
#         st.write(f"predicted probability: {prob:.3f}")
#         label = "Parasitized" if prob >= 0.5 else "Uninfected"
#         st.markdown(
#             f"### **Result:** {label}  \n"
#             f"Confidence: `{prob:.3f}`"
#         )
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
@st.cache_resource(show_spinner="Loading model …")
def load_model():
    model = tf.keras.models.load_model("malaria_cnn.h5", compile=False)
    return model

model = load_model()

# model.input_shape -> (None, H, W, C)
_, H, W, C = model.input_shape
TARGET_SIZE = (W, H)        

# Build the Streamlit interface
st.set_page_config(page_title="Malaria Cell Diagnosis", layout="centered")
st.title("🩸 Malaria Cell Image Diagnosis")

st.caption(f"Model expects **{H}×{W}×{C}** images (in RGB).")

uploaded_file = st.file_uploader(
    "Upload a single cell image (JPG / PNG)", type=("jpg", "jpeg", "png")
)

# Handle the upload, preprocessing, and prediction
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)

    # ----- PRE-PROCESS ---------------------------------------------------------
    img_resized = img.resize(TARGET_SIZE)              
    arr = np.asarray(img_resized, dtype=np.float32) / 255.0

    if C == 1:  # rare, but handle grayscale models
        arr = arr.mean(axis=-1, keepdims=True)

    arr = np.expand_dims(arr, axis=0)                 

    # ----- PREDICT -------------------------------------------------------------
    if st.button("🔍 Predict"):
        prob = float(model.predict(arr, verbose=0)[0][0])           # scalar
        label = "Parasitized" if prob < 0.5 else "Uninfected"

        st.markdown(f"### Result: **{label}**")

        