import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.cluster import DBSCAN
from mtcnn.mtcnn import MTCNN
from scipy.spatial.distance import cdist

# Load FaceNet model
model = load_model('facenet_keras.h5')
model.save('facenet_tf')
facenet_model = load_model('facenet_tf')
# Load MTCNN detector
detector = MTCNN()


def extract_face(filename, required_size=(160, 160)):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(image)
    x, y, width, height = results[0]['box']

    face = image[y:y + height, x:x + width]
    face = cv2.resize(face, required_size)
    return face


def get_embedding(face):
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    samples = np.expand_dims(face, axis=0)
    embedding = facenet_model.predict(samples)
    return embedding[0]


# Folder containing images
folder = 'images/'

# Extract faces and embeddings
faces = []
embeddings = []

for filename in os.listdir(folder):
    face = extract_face(os.path.join(folder, filename))
    embedding = get_embedding(face)
    faces.append(face)
    embeddings.append(embedding)

# Cluster using DBSCAN
clust = DBSCAN(eps=0.5, min_samples=2, metric='euclidean')
labels = clust.fit_predict(embeddings)

# Create folders and save images
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

for i in range(num_clusters):
    os.makedirs(f'cluster_{i}')

for i in range(len(faces)):
    if labels[i] == -1:
        continue
    cv2.imwrite(f'cluster_{labels[i]}/image{i}.jpg', cv2.cvtColor(faces[i], cv2.COLOR_RGB2BGR))

print(f'Found {num_clusters} clusters')