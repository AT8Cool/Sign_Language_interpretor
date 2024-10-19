import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os


def train_model(dataset_dir='dataset'):
    # Image data preprocessing with augmentation
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_gen = datagen.flow_from_directory(
        dataset_dir, target_size=(64, 64), batch_size=32, subset='training'
    )
    val_gen = datagen.flow_from_directory(
        dataset_dir, target_size=(64, 64), batch_size=32, subset='validation'
    )

    # Build the model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),  # Dropout layer to prevent overfitting
        Dense(train_gen.num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Model checkpoint to save the best model
    model_checkpoint = ModelCheckpoint('best_sign_language_model.keras', save_best_only=True, monitor='val_loss')

    # Train the model
    model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=[early_stopping, model_checkpoint])


if __name__ == "__main__":
    train_model()
