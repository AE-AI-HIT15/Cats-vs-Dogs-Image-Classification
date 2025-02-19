import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

model = load_model('models/1.0.0.h5')


new_data_dir = 'data/train_incorrect'


def load_incorrect_data(data_dir):
    datagen = ImageDataGenerator(rescale=1./255)
    new_data_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(150, 150), 
        batch_size=32,
        class_mode='categorical'  
    )
    return new_data_generator


new_data_generator = load_incorrect_data(new_data_dir)


early_stopping = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True, verbose=1)
learning_rate_reduction = ReduceLROnPlateau(monitor='loss', patience=2, factor=0.5, verbose=1)

model.fit(new_data_generator, epochs=10, callbacks=[early_stopping, learning_rate_reduction])


model.save('models/1.1.0.h5')
print("done 1.1.0.h5")
