import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("../saved_model/vehicle_model.h5")

IMG_SIZE = 64

cap = cv2.VideoCapture("C:/Users/usr/Desktop/vehicle-detection-project/detection/test_video.mp4")

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    height, width, _ = frame.shape

    roi = frame[height//2:height, :]

    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))

    img = img / 255.0

    img = np.reshape(img,(1,64,64,3))

    prediction = model.predict(img)
    print(prediction)

    if prediction > 0.5:

        cv2.putText(
            frame,
            "Vehicle Detected",
            (50,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

    cv2.imshow("Vehicle Detection",frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()