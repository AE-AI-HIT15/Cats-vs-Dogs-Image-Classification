import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

model = load_model('D:/Demo_Cat-Dog/cat_dog_model.h5')


def prepare_image(img):
    img = img.resize((150, 150)) 
    img_array = np.array(img) / 255.0 
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array


st.title("Cat vs Dog Image Classifier")


st.write("Upload an image of a cat or dog to predict the class.")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    img = Image.open(uploaded_file)
    
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = prepare_image(img)
    
    result = model.predict(img_array)
    
    predicted_class = np.argmax(result, axis=1)[0]
    classes = ['cat', 'dog']  
    predicted_label = classes[predicted_class]
    
    confidence = result[0][predicted_class] * 100 

    if confidence >= 90:
        st.write(f"Prediction: {predicted_label}")
        st.write(f"Confidence: {confidence:.2f}%")
    else:
        st.write(f"Prediction confidence is too low: {confidence:.2f}%. Try another image.")