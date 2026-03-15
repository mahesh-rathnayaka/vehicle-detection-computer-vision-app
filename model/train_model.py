import sys
import os

sys.path.append(os.path.abspath("../"))

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from preprocessing.preprocess import create_dataset

vehicle_path = "../dataset/vehicles"
non_vehicle_path = "../dataset/non-vehicles"

X, y = create_dataset(vehicle_path, non_vehicle_path)

X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(128,activation='relu'))

model.add(layers.Dense(1,activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_test,y_test)
)

model.save("../saved_model/vehicle_model.h5")