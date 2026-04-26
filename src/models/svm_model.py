import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

base = "data/processed/final"

labels = ["no_damage", "minor", "major", "destroyed"]

X = []
y = []

limit = 2000


def extract(x):
    gray = np.mean(x, axis=2)
    return [np.mean(gray), np.var(gray), np.max(gray), np.min(gray)]


for idx, label in enumerate(labels):
    folder = os.path.join(base, label)
    count = 0
    for file in os.listdir(folder):
        x = np.load(os.path.join(folder, file))
        f = extract(x)
        X.append(f)
        y.append(idx)
        count += 1
        if count >= limit:
            break

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = SVC(kernel="linear")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
