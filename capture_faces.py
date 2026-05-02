import cv2
import os

# Create dataset folder if it doesn't exist
if not os.path.exists("dataset"):
    os.makedirs("dataset")

# Ask for user name
name = input("Enter your name: ")
user_path = os.path.join("dataset", name)
if not os.path.exists(user_path):
    os.makedirs(user_path)

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break   # <-- properly indented now

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face = gray[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(user_path, f"{count}.jpg"), face)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Capturing Faces", frame)

    # Stop after 50 images or if ESC is pressed
    if cv2.waitKey(1) == 27 or count >= 50:
        break

cap.release()
cv2.destroyAllWindows()
print(f"Saved {count} images to {user_path}")
