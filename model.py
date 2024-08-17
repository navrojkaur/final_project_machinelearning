import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Dataset preprocessing and loading (e.g., CIFAR-10)
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

# Defining the CNN model
model = models.Sequential([
    layers.Conv2D(64, (4, 4), activation='relu', padding='same', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(64, (4, 4), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.2),

    # Second Conv Block
    layers.Conv2D(128, (4, 4), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (4, 4), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    # Third Conv Block
    layers.Conv2D(128, (4, 4), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (4, 4), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.35),

    # Flattening Layer
    layers.Flatten(),

    # Dense Layers
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    # Output Layer
    layers.Dense(10, activation='softmax')
])

# Compilation of the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, save_format='keras')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

# Model Training
history = model.fit(train_images, train_labels,
                    epochs=20,
                    validation_data=(test_images, test_labels),
                    callbacks=[early_stopping, model_checkpoint, reduce_lr])

# Save the final model
model.save('final_model.keras')