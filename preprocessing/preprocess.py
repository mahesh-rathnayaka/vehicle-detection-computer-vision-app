import cv2
import os
import numpy as np

IMG_SIZE = 64

def load_images(folder):

    images = []

    for root, dirs, files in os.walk(folder):

        for file in files:

            path = os.path.join(root, file)

            img = cv2.imread(path)

            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            images.append(img)

    return images


def create_dataset(vehicle_path, non_vehicle_path):

    vehicles = load_images(vehicle_path)

    non_vehicles = load_images(non_vehicle_path)

    X = np.array(vehicles + non_vehicles)

    y = np.array([1]*len(vehicles) + [0]*len(non_vehicles))

    return X, y