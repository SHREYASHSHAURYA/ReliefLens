import os
import cv2
import numpy as np

base_in = "data/processed/severity"
base_out = "data/processed/final"

labels = ["no_damage", "minor", "major", "destroyed"]

size = 128

for label in labels:
    in_path = os.path.join(base_in, label)
    out_path = os.path.join(base_out, label)
    os.makedirs(out_path, exist_ok=True)
    for file in os.listdir(in_path):
        p = os.path.join(in_path, file)
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.resize(img, (size, size))
        img = img / 255.0
        np.save(os.path.join(out_path, file.split(".")[0] + ".npy"), img)
