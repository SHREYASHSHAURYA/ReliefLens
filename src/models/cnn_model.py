import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

base = "data/processed/final"

labels = ["no_damage", "minor", "major", "destroyed"]

X = []
y = []

limit = 800

for idx, label in enumerate(labels):
    folder = os.path.join(base, label)
    count = 0
    for file in os.listdir(folder):
        x = np.load(os.path.join(folder, file))
        X.append(x)
        y.append(idx)
        count += 1
        if count >= limit:
            break

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(128, 128, 3), include_top=False, weights="imagenet"
)

base_model.trainable = True

model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(128, 128, 3)),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(4, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)
