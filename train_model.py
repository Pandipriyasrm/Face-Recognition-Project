import cv2
import os
import numpy as np

# Path to dataset
dataset_path = "dataset"

faces = []
labels = []
label_map = {}
current_label = 0

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     'haarcascade_frontalface_default.xml')

# Loop through dataset folders
for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person_name

    for image_name in os.listdir(person_path):
        img_path = os.path.join(person_path, image_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        faces.append(img)
        labels.append(current_label)

    current_label += 1

# Train LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

# Save model and labels
recognizer.save("face_model.yml")
np.save("labels.npy", label_map)

print("Training complete. Model saved as face_model.yml and labels.npy")
