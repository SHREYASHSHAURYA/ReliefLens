import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


LABELS = ["no_damage", "minor", "major", "destroyed"]
SEVERE_CLASS_IDS = {2, 3}
IMG_SIZE = (128, 128)
DEFAULT_THRESHOLD = 0.5


@st.cache_resource
def load_model(model_path: str):
    return tf.keras.models.load_model(model_path)


def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def threshold_predict(probabilities: np.ndarray, severe_threshold: float) -> tuple[int, float]:
    severe_score = float(probabilities[2] + probabilities[3])
    if severe_score >= severe_threshold:
        cls = int(np.argmax(probabilities[2:4]) + 2)
    else:
        cls = int(np.argmax(probabilities[0:2]))
    return cls, severe_score


def read_threshold(metrics_path: Path) -> float:
    if not metrics_path.exists():
        return DEFAULT_THRESHOLD
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        return float(payload.get("best_severe_threshold", DEFAULT_THRESHOLD))
    except Exception:
        return DEFAULT_THRESHOLD


def main() -> None:
    st.set_page_config(page_title="ReliefLens Damage Assessment", layout="wide")

    st.title("ReliefLens: Disaster Damage Assessment")
    st.write("Upload an image to classify damage severity and view severe-risk confidence.")

    model_path = st.sidebar.text_input("Model path", value="outputs/cnn_full/best_model.keras")
    metrics_path = st.sidebar.text_input("Metrics path", value="outputs/cnn_full/metrics.json")

    threshold = read_threshold(Path(metrics_path))
    custom_threshold = st.sidebar.slider("Severe threshold", 0.2, 0.95, float(threshold), 0.01)

    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload an image to run inference.")
        return

    image = Image.open(uploaded)
    st.image(image, caption="Input Image", use_column_width=True)

    if not Path(model_path).exists():
        st.error(f"Model not found at: {model_path}")
        return

    model = load_model(model_path)
    x = preprocess_image(image)
    probs = model.predict(x, verbose=0)[0]

    pred_class, severe_score = threshold_predict(probs, custom_threshold)
    pred_name = LABELS[pred_class]
    pred_conf = float(probs[pred_class])

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Class", pred_name)
    c2.metric("Class Confidence", f"{pred_conf:.3f}")
    c3.metric("Severe Score", f"{severe_score:.3f}")

    st.subheader("Class Probabilities")
    prob_table = {
        "class": LABELS,
        "probability": [float(x) for x in probs],
    }
    st.dataframe(prob_table, use_container_width=True)

    if pred_class in SEVERE_CLASS_IDS:
        st.warning("Potentially severe damage detected. Prioritize human review.")
    elif severe_score > custom_threshold * 0.85:
        st.info("Borderline severe-risk signal. Consider secondary review.")
    else:
        st.success("Low severe-risk prediction.")


if __name__ == "__main__":
    main()
