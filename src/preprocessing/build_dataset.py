import os
import cv2
import numpy as np
import shutil
from sklearn.cluster import KMeans

base_raw = "data/raw"
base_out = "data/processed/severity"

sources = [
    "train/Normal",
    "train/Earthquake",
    "train/Fire",
    "train/Flood",
    "test/Normal",
    "test/Earthquake",
    "test/Fire",
    "test/Flood",
    "aider/collapsed_building",
    "aider/fire",
    "aider/flooded_areas",
    "aider/normal",
]

features = []
paths = []


def extract(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size
    variance = np.var(gray)
    mean = np.mean(gray)
    return [edge_density, variance, mean]


for src in sources:
    folder = os.path.join(base_raw, src)
    if not os.path.exists(folder):
        continue
    for file in os.listdir(folder):
        p = os.path.join(folder, file)
        img = cv2.imread(p)
        if img is None:
            continue
        f = extract(img)
        features.append(f)
        paths.append(p)

features = np.array(features)

kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(features)

cluster_scores = []
for i in range(4):
    cluster_scores.append(np.mean(features[labels == i][:, 0]))

order = np.argsort(cluster_scores)

mapping = {
    order[0]: "no_damage",
    order[1]: "minor",
    order[2]: "major",
    order[3]: "destroyed",
}

for i in range(len(paths)):
    label = mapping[labels[i]]
    dst = os.path.join(base_out, label, os.path.basename(paths[i]))
    shutil.copy(paths[i], dst)
