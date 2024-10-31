import streamlit as st
from tensorflow.keras.applications import EfficientNetV2L
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the EfficientNetV2L model with pretrained weights
try:
    model = EfficientNetV2L(weights='imagenet')
except Exception as e:
    st.error(f"Error loading the model: {str(e)}")

# Streamlit app title
st.title("Image Classification with EfficientNetV2L")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for the model
    img = img.resize((480, 480))  # Resize to 480x480 for EfficientNetV2L
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Button to predict the image class
    if st.button("Predict"):
        # Predict and decode the results
        preds = model.predict(img_array)
        predictions = decode_predictions(preds, top=3)[0]

        # Display predictions
        st.write("Top Predictions:")
        for i, (imagenet_id, label, score) in enumerate(predictions):
            st.write(f"{i+1}: {label} ({score*100:.2f}%)")
