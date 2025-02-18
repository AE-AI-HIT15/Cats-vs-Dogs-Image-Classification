from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI()

model = load_model('models/cat_dog_model.h5')  


def prepare_image(img_bytes):
    img = Image.open(BytesIO(img_bytes)) 
    img = img.resize((150, 150))  
    img_array = np.array(img) / 255.0 
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img_array = prepare_image(img_bytes)

    result = model.predict(img_array)
    
    predicted_class = np.argmax(result, axis=1)[0]
    classes = ['cat', 'dog']
    predicted_label = classes[predicted_class]

    return {"prediction": predicted_label, "confidence": float(result[0][predicted_class])}