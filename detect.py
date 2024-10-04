import cv2
import dlib
import numpy as np
import os

def load_clusters(base_path):
    clusters = []
    for folder in os.listdir(base_path):
        cluster_path = os.path.join(base_path, folder)
        images = []
        for filename in os.listdir(cluster_path):
            img = cv2.imread(os.path.join(cluster_path, filename))
            if img is not None:
                images.append(img)
        clusters.append((folder, images))
    return clusters

def extract_face_descriptor(image):
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

    dets = detector(image, 1)
    if len(dets) == 0:
        return None
    shape = shape_predictor(image, dets[0])
    face_descriptor = face_recognizer.compute_face_descriptor(image, shape)
    return face_descriptor

def find_matching_folder(photo, clusters, threshold=0.6):
    descriptor = extract_face_descriptor(photo)
    if descriptor is None:
        return "Unknown"

    for folder, images in clusters:
        for img in images:
            distance = np.linalg.norm(np.array(descriptor) - np.array(extract_face_descriptor(img)))
            if distance < threshold:
                return folder
    return "Unknown"

def main():
    base_path = 'clusters_folder'
    clusters = load_clusters(base_path)

    photo_path = 'selfie.jpg'
    photo = cv2.imread(photo_path)

    if photo is None:
        print("Could not load the photo.")
        return
    matching_folder = find_matching_folder(photo, clusters)
    print(f"The matching folder is: {matching_folder}")

if __name__ == "__main__":
    main()
