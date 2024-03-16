# Importing Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, LeakyReLU, GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from keras.layers import BatchNormalization
from tempfile import TemporaryFile
import tensorflow.keras.regularizers as regularizers
import matplotlib.patches as patches
from google.colab import drive
import time

LOG_DIR = f"{int(time.time())}"

# Rest of your code...


# Dataset directory
dataset_dir = "dataset_aircraft"
root_path = "dataset_aircraft/"
RESIZE_VALUE = 130

# Image processing and data augmentation
def image_process():
    # Apply data augmentation techniques
    datagen = ImageDataGenerator(
        rotation_range=360,
        width_shift_range=0,
        height_shift_range=0,
        shear_range=0,
        zoom_range=0.0,
        horizontal_flip=True,
        fill_mode='nearest')
    
    # Process each image in the dataset
    for type in aircraft_types:
        selected_path = root_path + type
        for img_ in os.listdir(selected_path):
            img = load_img(selected_path + img_)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            
            i = 0
            for batch in datagen.flow(x, batch_size=1,
                                      save_to_dir=root_path + type, save_prefix=img_, save_format='jpeg'):
                i += 1
                if i > 20:
                    break

# Load and process dataset
def load_and_process_dataset():
    # Read aircraft types from file
    aircraft_types = []
    types_path = "dataset_aircraft/TYPE-NAMES.txt"
    f = open(types_path, "r")
    for types in f.read().split():
        aircraft_types.append(types.replace('"',''))
    f.close()
    
    # Load images and their corresponding labels
    training_data = []
    for type in aircraft_types:
        selected_path = root_path + type
        class_num = aircraft_types.index(type)
        for img in os.listdir(selected_path):
            try:
                image = Image.open(os.path.join(selected_path, img))
                img_array = np.array(image)
                aircraft_img = cv2.resize(img_array, (RESIZE_VALUE, RESIZE_VALUE))
                training_data.append([aircraft_img, class_num])
            except Exception as e:
                print("error:", selected_path, img)
    
    # Shuffle the data
    random.shuffle(training_data)
    
    # Save the processed data to a file
    np.savez_compressed(root_path + 'primary_training_dataset.npz', a=training_data)

# Load training data
def load_training_data():
    loaded = np.load(root_path + "primary_training_dataset.npz", allow_pickle=True)
    training_data = loaded['a']
    for x in range(len(training_data)):
        if training_data[x][0].shape[2] == 4:
            training_data[x][0] = cv2.cvtColor(training_data[x][0], cv2.COLOR_BGRA2BGR)
    return training_data

# Split the dataset into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), padding='same', input_shape=(RESIZE_VALUE, RESIZE_VALUE, 3)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Conv2D(32, (3, 3), padding='same'),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), padding='same'),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Conv2D(64, (3, 3), padding='same'),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), padding='same'),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Conv2D(128, (3, 3), padding='same'),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    MaxPooling2D((2, 2)),
    
    Conv2D(256, (3, 3), padding='same'),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Conv2D(256, (3, 3), padding='same'),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    MaxPooling2D((2, 2)),
    
    GlobalAveragePooling2D(),
    Dense(512),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.5),
    Dense(len(aircraft_types), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define a checkpoint to save the best model
checkpoint_path = "best_model_checkpoint.h5"
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# Train the model with validation data
batch_size = 32
epochs = 10
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val), callbacks=[checkpoint])

# Load the best model
model.load_weights(checkpoint_path)

# Evaluate the best model on the test data
loss, accuracy = model.evaluate(x_val, y_val)
print('Validation loss:', loss)
print('Validation accuracy:', accuracy)

# Plot training history
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), history.history['accuracy'], label='Training Accuracy')
plt.plot(range(1, epochs + 1), history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), history.history['loss'], label='Training Loss')
plt.plot(range(1, epochs + 1), history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()
