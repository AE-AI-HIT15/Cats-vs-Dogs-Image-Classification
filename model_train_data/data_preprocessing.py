from tensorflow.keras.preprocessing.image import ImageDataGenerator

def prepare_data(image_size, batch_size):
    train_datagen = ImageDataGenerator(rescale=1./255, 
                                       shear_range=0.2, 
                                       zoom_range=0.2, 
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        'data/train',  
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical'  
    )

    val_generator = test_datagen.flow_from_directory(
        'data/validation', 
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    return train_generator, val_generator
