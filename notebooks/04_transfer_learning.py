import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")

@app.cell
def __():
    import os
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    return (
        Dense,
        Dropout,
        EarlyStopping,
        GlobalAveragePooling2D,
        ImageDataGenerator,
        Model,
        ModelCheckpoint,
        ReduceLROnPlateau,
        ResNet50,
        Sequential,
        os,
        plt,
        tf,
    )


@app.cell
def __(ImageDataGenerator, tf):
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32

    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)

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
def __(Dense, Dropout, GlobalAveragePooling2D, IMG_SIZE, Model, ResNet50):
    # 4.2 Cargar ResNet50 preentrenado (ImageNet)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    # 4.3 Congelar capas base
    base_model.trainable = False
    
    # 4.4 Añadir cabezal propio
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(5, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.summary()
    return base_model, model, predictions, x


@app.cell
def __(model):
    # Compilar modelo con Transfer Learning
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return


@app.cell
def __(EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, model, train_generator, val_generator):
    # 4.6 Entrenar y guardar
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint('models/resnet50_mango.keras', monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
    ]

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=30, # Menos épocas iniciales para Transfer Learning
        callbacks=callbacks
    )
    return callbacks, history


@app.cell
def __(base_model, model, train_generator, val_generator):
    # 4.5 Fine-tuning (opcional)
    # Descongelar las últimas capas del modelo base
    base_model.trainable = True
    for layer in base_model.layers[:-20]: # Descongelar solo las últimas 20 capas
        layer.trainable = False
        
    from tensorflow.keras.optimizers import Adam
    model.compile(
        optimizer=Adam(learning_rate=1e-5), # learning rate muy bajo
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    callbacks_ft = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint('models/resnet50_mango.keras', monitor='val_loss', save_best_only=True, verbose=1),
    ]
    
    history_ft = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=15,
        callbacks=callbacks_ft
    )
    return Adam, callbacks_ft, history_ft, layer


@app.cell
def __(history, plt):
    # Graficar curvas
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
    plt.title('Training and Validation Accuracy (Initial)')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss (Initial)')
    plt.show()
    return acc, epochs_range, loss, val_acc, val_loss


if __name__ == "__main__":
    app.run()
