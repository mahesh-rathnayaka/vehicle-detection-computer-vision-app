import cv2
import numpy as np

def extract_features(image):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, 50, 150)

    edges = edges / 255.0

    return edges