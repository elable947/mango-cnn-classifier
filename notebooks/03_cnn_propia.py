import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")

@app.cell
def __():
    import os
    from pathlib import Path
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    return (
        Conv2D,
        Dense,
        Dropout,
        EarlyStopping,
        Flatten,
        ImageDataGenerator,
        MaxPooling2D,
        ModelCheckpoint,
        Path,
        ReduceLROnPlateau,
        Sequential,
        os,
        plt,
        tf,
    )


@app.cell
def __(ImageDataGenerator):
    # Preparar generadores (reutilizando lógica de fase 2)
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        'data/val',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    return BATCH_SIZE, IMG_SIZE, train_datagen, train_generator, val_datagen, val_generator


@app.cell
def __(Conv2D, Dense, Dropout, Flatten, IMG_SIZE, MaxPooling2D, Sequential):
    # 3.2 Definir arquitectura
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        MaxPooling2D(2, 2),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5), # Regularización para evitar overfitting
        Dense(5, activation='softmax') # 5 salidas para los 5 tipos
    ])
    model.summary()
    return model,


@app.cell
def __(model):
    # 3.3 Compilar modelo
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return


@app.cell
def __(EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, model, train_generator, val_generator):
    # 3.4 Entrenar con Callbacks
    # 3.5 Guardar mejor modelo en models/cnn_propia.keras
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint('models/cnn_propia.keras', monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
    ]

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=50,
        callbacks=callbacks
    )
    return callbacks, history


@app.cell
def __(history, plt):
    # 3.6 Graficar curvas de entrenamiento
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    return acc, epochs_range, loss, val_acc, val_loss


if __name__ == "__main__":
    app.run()
