import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps
from torchvision import transforms

# ---------------------------------
# Make project root importable
# ---------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.model import DigitCNN
from src.config import DEVICE, CHECKPOINT_DIR

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 MNIST Handwritten Digit Recognition")
st.write(
    "Upload a handwritten digit image (0–9). The app converts it into MNIST format before prediction."
)

# ---------------------------------
# Preprocessing Function
# ---------------------------------
def preprocess(image):

    # Convert to grayscale
    image = ImageOps.grayscale(image)

    # PIL -> NumPy
    img = np.array(image)

    # Invert colors (MNIST = white digit)
    img = 255 - img

    # Binary threshold
    _, img = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)

    # Find digit
    coords = cv2.findNonZero(img)

    if coords is None:
        raise ValueError("No digit detected.")

    x, y, w, h = cv2.boundingRect(coords)

    digit = img[y:y+h, x:x+w]

    # Keep aspect ratio
    if h > w:
        new_h = 20
        new_w = max(1, int(w * 20 / h))
    else:
        new_w = 20
        new_h = max(1, int(h * 20 / w))

    digit = cv2.resize(digit, (new_w, new_h))

    # Create 28x28 canvas
    canvas = np.zeros((28, 28), dtype=np.uint8)

    start_x = (28 - new_w) // 2
    start_y = (28 - new_h) // 2

    canvas[
        start_y:start_y + new_h,
        start_x:start_x + new_w
    ] = digit

    tensor = transforms.ToTensor()(canvas)

    tensor = transforms.Normalize(
        (0.1307,),
        (0.3081,)
    )(tensor)

    return tensor, canvas


# ---------------------------------
# Load Model
# ---------------------------------
@st.cache_resource
def load_model():

    model = DigitCNN().to(DEVICE)

    model.load_state_dict(
        torch.load(
            CHECKPOINT_DIR / "best_model.pth",
            map_location=DEVICE
        )
    )

    model.eval()

    return model


model = load_model()

# ---------------------------------
# Upload Section
# ---------------------------------
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Original Image")
    st.image(image, width=180)

    if st.button("🚀 Predict", use_container_width=True):

        try:

            tensor, processed = preprocess(image)

            img = tensor.unsqueeze(0).to(DEVICE)

            with torch.no_grad():

                outputs = model(img)

                probabilities = torch.softmax(outputs, dim=1)

                confidence, prediction = torch.max(
                    probabilities,
                    1
                )

            probs = probabilities.squeeze().cpu().numpy()

            pred_digit = int(prediction.item())

            confidence_percent = confidence.item() * 100

            # ---------------------------------
            # Show Processed Image
            # ---------------------------------
            st.divider()

            st.subheader("28×28 Image Used by CNN")

            st.image(
                processed,
                width=180,
                clamp=True
            )

            st.caption(
                "This is the exact image the CNN receives after preprocessing."
            )

            # ---------------------------------
            # Prediction Result
            # ---------------------------------
            st.divider()

            st.subheader("🎯 Prediction Result")

            col1, col2 = st.columns([1, 2])

            with col1:

                st.markdown(
                    f"""
                    <div style="
                        background:#2563EB;
                        color:white;
                        border-radius:12px;
                        text-align:center;
                        padding:18px;">
                        <h1 style="font-size:56px;margin:0;">
                            {pred_digit}
                        </h1>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:

                st.metric(
                    "Confidence",
                    f"{confidence_percent:.2f}%"
                )

                if confidence_percent >= 98:
                    st.success("Very High Confidence")
                elif confidence_percent >= 90:
                    st.info("High Confidence")
                else:
                    st.warning("Low Confidence")

            # ---------------------------------
            # Top 3 Predictions
            # ---------------------------------
            st.divider()

            st.subheader("🏆 Top 3 Predictions")

            sorted_probs = sorted(
                enumerate(probs),
                key=lambda x: x[1],
                reverse=True
            )

            medals = ["🥇", "🥈", "🥉"]

            for i, (digit, prob) in enumerate(sorted_probs[:3]):

                st.write(
                    f"{medals[i]} **Digit {digit}** — {prob*100:.2f}%"
                )

                st.progress(float(prob))

            # ---------------------------------
            # Full Probability Distribution
            # ---------------------------------
            with st.expander("Show all class probabilities"):

                for digit, prob in sorted_probs:

                    st.write(
                        f"Digit {digit}: {prob*100:.2f}%"
                    )

                    st.progress(float(prob))

            st.divider()

            st.caption("Powered by PyTorch CNN • MNIST Dataset")

        except Exception as e:

            st.error(f"Prediction failed: {e}")