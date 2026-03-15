import cv2

cap = cv2.VideoCapture("C:/Users/usr/Desktop/vehicle-detection-project/detection/test_video.mp4")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Video Test", frame)

    if cv2.waitKey(25) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()