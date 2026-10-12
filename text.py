import cv2

# Replace with your ESP32-CAM IP address
url = "http://192.168.1.100:81/stream"

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Cannot connect to ESP32-CAM")
    exit()

print("Connected to ESP32-CAM")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("ESP32-CAM Stream", frame)

    # Press ESC to quit
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()