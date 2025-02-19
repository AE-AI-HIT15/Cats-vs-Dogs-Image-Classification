import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model('models/1.0.0.h5')

# Define a function to classify the image
def classify_image(image):
    image = image.resize((150, 150))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return prediction

# Streamlit app
st.title("Cat vs Dog Image Classification")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.')
    st.write("")
    

    prediction = classify_image(image)
    confidence = max(prediction[0])

    if confidence < 0.9:
        st.write("Could not identify the image with sufficient confidence.")
    else:
        if np.argmax(prediction) == 0:
            st.write(f"This is a Cat with {confidence * 100:.2f}% confidence.")
        else:
            st.write(f"This is a Dog with {confidence * 100:.2f}% confidence.")