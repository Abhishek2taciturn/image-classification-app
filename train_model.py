import tensorflow as tf
from tensorflow.keras import layers, models
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import json

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32  # Increased batch size for faster training
EPOCHS = 10  # Reduced epochs

def create_data_generator(data_dir, batch_size=32, validation_split=0.2):
    # Optimized data augmentation for faster training
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=validation_split
    )

    # Only rescaling for validation
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )

    # Training generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode='sparse',
        subset='training',
        shuffle=True
    )

    # Validation generator
    validation_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation',
        shuffle=False
    )

    return train_generator, validation_generator

def create_model(num_classes):
    model = models.Sequential([
        # Input layer
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Global Average Pooling
        layers.GlobalAveragePooling2D(),
        
        # Dense Layers
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

def main():
    # Set up data generators
    data_dir = 'archive/data'
    train_generator, validation_generator = create_data_generator(data_dir, BATCH_SIZE)
    
    # Get number of classes
    num_classes = len(train_generator.class_indices)
    
    # Save class names
    class_names = list(train_generator.class_indices.keys())
    np.save('class_names.npy', class_names)
    
    # Create and compile model
    model = create_model(num_classes)
    
    # Use a higher learning rate for faster training
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=500,
        decay_rate=0.9,
        staircase=True
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Optimized callbacks for faster training
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=2,
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]
    
    # Calculate steps per epoch
    steps_per_epoch = train_generator.samples // BATCH_SIZE
    validation_steps = validation_generator.samples // BATCH_SIZE
    
    # Train model
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks
    )
    
    # Save the final metrics
    final_metrics = {
        'accuracy': str(history.history['accuracy'][-1]),
        'val_accuracy': str(history.history['val_accuracy'][-1]),
        'loss': str(history.history['loss'][-1]),
        'val_loss': str(history.history['val_loss'][-1])
    }

    # Save metrics to JSON file
    with open('model_metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=4)

    # Save the model
    model.save('model.h5')
    
    print("Training completed. Model and metrics saved successfully!")
    print("Class names saved as 'class_names.npy'")

if __name__ == "__main__":
    main() 