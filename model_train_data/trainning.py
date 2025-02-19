from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

def train_model(model, train_generator, val_generator, epochs=30):
    # Callback
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                                 patience=2,
                                                 factor=0.5,
                                                 min_lr=0.00001,
                                                 verbose=1)

    early_stopping = EarlyStopping(monitor='val_loss', 
                                   patience=3, 
                                   restore_best_weights=True, 
                                   verbose=1)

    # Biên dịch mô hình
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    # Huấn luyện mô hình
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=[early_stopping, learning_rate_reduction]
    )

    return model, history

