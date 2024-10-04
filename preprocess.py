import cv2
import os
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.cluster import DBSCAN

# Function to load and align faces using MTCNN
def load_and_align_faces(folder, min_face_size=20):
    images = []
    mtcnn = MTCNN(min_face_size=min_face_size)
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = mtcnn(img_rgb)
            if faces is not None:
                images.extend(faces)
    return images

# Function to extract face embeddings using FaceNet
# Function to extract face embeddings using FaceNet
def extract_face_embeddings(face_images):
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    face_embeddings = []
    for face in face_images:
        face = cv2.resize(np.array(face), (160, 160))  # Convert to NumPy array before resizing
        face = (face - 127.5) / 128.0  # Normalize
        with torch.no_grad():
            face_embedding = resnet(torch.unsqueeze(face, 0))
        face_embeddings.append(face_embedding.numpy())
    return face_embeddings


# Function to cluster faces using DBSCAN
def cluster_faces(face_embeddings, eps=0.5, min_samples=5):
    X = np.concatenate(face_embeddings)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = dbscan.fit_predict(X)
    return labels

def main():
    folder = 'images'
    face_images = load_and_align_faces(folder)
    face_embeddings = extract_face_embeddings(face_images)
    labels = cluster_faces(face_embeddings)

    # Create clusters
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(face_images[i])

    # Save clustered faces
    for cluster_label, cluster_faces in clusters.items():
        os.makedirs(f'cluster_{cluster_label}')
        for i, face in enumerate(cluster_faces):
            cv2.imwrite(f'cluster_{cluster_label}/image_{i}.jpg', face)

if __name__ == "__main__":
    main()
