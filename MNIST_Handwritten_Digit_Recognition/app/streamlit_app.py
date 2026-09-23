
# ==========================================
# MNIST VISION AI - STREAMLIT APPLICATION
# ==========================================

from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn

from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms


# ==========================================
# PROJECT PATH
# ==========================================

ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT_DIR / "models" / "best_digit_cnn.pth"


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="MNIST Vision AI",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown(
    """
    <style>

    /* ======================================
       MAIN BACKGROUND
       ====================================== */

    .stApp {
        background-color: #0f172a;
    }


    /* ======================================
       MAIN CONTENT
       ====================================== */

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ======================================
       ALL NORMAL TEXT
       ====================================== */

    .stApp p,
    .stApp label,
    .stApp span,
    .stApp div {
        color: #f8fafc;
    }


    /* ======================================
       HEADINGS
       ====================================== */

    h1,
    h2,
    h3,
    h4 {
        color: #ffffff !important;
    }


    /* ======================================
       PROJECT TITLE
       ====================================== */

    .project-title {
        color: #ffffff !important;

        font-size: 2.8rem;

        font-weight: 800;

        margin-bottom: 0.3rem;
    }


    /* ======================================
       SUBTITLE
       ====================================== */

    .project-subtitle {
        color: #e2e8f0 !important;

        font-size: 1.05rem;

        margin-bottom: 2rem;
    }


    /* ======================================
       SIDEBAR
       ====================================== */

    section[data-testid="stSidebar"] {
        background-color: #111827;
    }


    section[data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }


    /* ======================================
       FILE UPLOADER
       ====================================== */

    [data-testid="stFileUploader"] {
        background-color: #1e293b;

        border-radius: 14px;

        padding: 10px;
    }


    [data-testid="stFileUploader"] * {
        color: #ffffff !important;
    }


    /* ======================================
       INFO BOX
       ====================================== */

    [data-testid="stAlert"] {
        color: #ffffff !important;
    }


    [data-testid="stAlert"] * {
        color: #ffffff !important;
    }


    /* ======================================
       METRICS
       ====================================== */

    [data-testid="stMetric"] {
        background-color: #1e293b;

        border: 1px solid #475569;

        border-radius: 14px;

        padding: 1rem;
    }


    [data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
    }


    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }


    /* ======================================
       EXPANDER
       ====================================== */

    [data-testid="stExpander"] {
        background-color: #1e293b;

        border: 1px solid #475569;

        border-radius: 14px;
    }


    [data-testid="stExpander"] * {
        color: #f8fafc !important;
    }


    /* ======================================
       FOOTER
       ====================================== */

    .footer {
        color: #cbd5e1 !important;

        text-align: center;

        padding-top: 2rem;

        font-size: 0.9rem;
    }

    /* Make upload area text black */
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] section div,
    [data-testid="stFileUploader"] section span {
        color: black !important;
    }

    /* Make all upload-related text black */ 
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] p, 
    [data-testid="stFileUploader"] span {
    color: black !important; }
    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ==========================================
# CNN MODEL
# ==========================================

class DigitCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64 * 7 * 7,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                10
            )
        )


    def forward(self, x):

        x = self.features(x)

        return self.classifier(x)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    model = DigitCNN().to(device)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model.eval()

    return model


model = load_model()


# ==========================================
# MNIST TRANSFORM
# ==========================================

mnist_transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        (0.1307,),
        (0.3081,)
    )
])


# ==========================================
# IMAGE PREPARATION
# ==========================================

def prepare_image(image):

    # --------------------------------------
    # Convert to grayscale
    # --------------------------------------

    image = image.convert("L")


    # --------------------------------------
    # Improve contrast
    # --------------------------------------

    image = ImageOps.autocontrast(image)


    # --------------------------------------
    # Convert to NumPy
    # --------------------------------------

    array = np.array(image)


    # --------------------------------------
    # Estimate background using corners
    # --------------------------------------

    height, width = array.shape

    corner_size = max(
        1,
        min(height, width) // 20
    )


    corners = np.concatenate(
        [
            array[
                :corner_size,
                :corner_size
            ].flatten(),

            array[
                :corner_size,
                -corner_size:
            ].flatten(),

            array[
                -corner_size:,
                :corner_size
            ].flatten(),

            array[
                -corner_size:,
                -corner_size:
            ].flatten()
        ]
    )


    background_mean = corners.mean()


    # --------------------------------------
    # Invert bright-background images
    # --------------------------------------

    if background_mean > 127:

        image = ImageOps.invert(image)


    # --------------------------------------
    # Detect digit
    # --------------------------------------

    array = np.array(image)

    foreground = array > 30


    if foreground.any():

        coords = np.argwhere(
            foreground
        )


        y_min, x_min = coords.min(
            axis=0
        )


        y_max, x_max = coords.max(
            axis=0
        )


        # Add small padding

        padding = max(
            2,
            int(
                0.08 *
                max(
                    x_max - x_min + 1,
                    y_max - y_min + 1
                )
            )
        )


        y_min = max(
            0,
            y_min - padding
        )


        x_min = max(
            0,
            x_min - padding
        )


        y_max = min(
            array.shape[0] - 1,
            y_max + padding
        )


        x_max = min(
            array.shape[1] - 1,
            x_max + padding
        )


        image = image.crop(
            (
                x_min,
                y_min,
                x_max + 1,
                y_max + 1
            )
        )


    # --------------------------------------
    # Make image square
    # --------------------------------------

    width, height = image.size

    side = max(
        width,
        height
    )


    canvas = Image.new(
        "L",
        (
            side,
            side
        ),
        color=0
    )


    x_offset = (
        side - width
    ) // 2


    y_offset = (
        side - height
    ) // 2


    canvas.paste(
        image,
        (
            x_offset,
            y_offset
        )
    )


    # --------------------------------------
    # Resize to MNIST 28x28
    # --------------------------------------

    image = canvas.resize(
        (28, 28),
        Image.Resampling.LANCZOS
    )


    # --------------------------------------
    # Slight smoothing
    # --------------------------------------

    image = image.filter(
        ImageFilter.GaussianBlur(
            radius=0.2
        )
    )


    return image


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_digit(image):

    processed_image = prepare_image(
        image
    )


    tensor = mnist_transform(
        processed_image
    )


    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)


    with torch.inference_mode():

        outputs = model(
            tensor
        )


        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        predicted_digit = (
            probabilities
            .argmax(dim=1)
            .item()
        )


        confidence = (
            probabilities
            .max(dim=1)
            .values
            .item()
        )


    return (
        processed_image,
        predicted_digit,
        confidence,
        probabilities
    )


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown(
        "## MNIST Vision AI"
    )


    st.write(
        "Handwritten Digit Classifier"
    )


    st.divider()


    st.markdown(
        "### Model"
    )


    st.write(
        "Architecture: CNN"
    )


    st.write(
        "Input: 28 × 28"
    )


    st.write(
        "Channels: 1"
    )


    st.write(
        "Classes: 10"
    )


    st.write(
        f"Device: {device}"
    )


    st.divider()


    st.markdown(
        "### Supported Images"
    )


    st.write(
        "PNG"
    )

    st.write(
        "JPG"
    )

    st.write(
        "JPEG"
    )


    st.caption(
        "Color images are automatically "
        "converted to grayscale."
    )


# ==========================================
# MAIN TITLE
# ==========================================

st.markdown(
    '<div class="project-title">'
    'MNIST Vision AI'
    '</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="project-subtitle">'
    'Handwritten Digit Classification '
    'powered by a PyTorch Convolutional '
    'Neural Network.'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================
# MAIN COLUMNS
# ==========================================

left_col, right_col = st.columns(
    [1, 1],
    gap="large"
)


# ==========================================
# UPLOAD SECTION
# ==========================================

with left_col:

    st.markdown(
        "## Upload Your Digit"
    )


    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "png",
            "jpg",
            "jpeg"
        ]
    )


    st.info(
        "The image is automatically "
        "converted to grayscale and "
        "prepared for the CNN."
    )


# ==========================================
# SHOW ORIGINAL IMAGE
# ==========================================

if uploaded_file is not None:

    original_image = Image.open(
        uploaded_file
    )


    with left_col:

        st.image(
            original_image,
            caption="Uploaded Image",
            width=300
        )


# ==========================================
# PREDICTION BUTTON
# ==========================================

with right_col:

    st.markdown(
        "## Prediction"
    )


    predict_button = st.button(
        "Predict Digit",
        use_container_width=True,
        type="primary"
    )


# ==========================================
# RUN PREDICTION
# ==========================================

if (
    uploaded_file is not None
    and predict_button
):

    with st.spinner(
        "Analyzing your image..."
    ):

        (
            processed_image,
            predicted_digit,
            confidence,
            probabilities
        ) = predict_digit(
            original_image
        )


    with right_col:

        # ----------------------------------
        # PROCESSED IMAGE
        # ----------------------------------

        st.image(
            processed_image,
            caption="Image Sent to CNN",
            width=220
        )


        # ==================================
        # PURE STREAMLIT RESULT
        # ==================================

        st.markdown(
            "### Prediction Result"
        )


        # Green native Streamlit box

        st.success(
            "Prediction completed successfully."
        )


        # Large native Streamlit text

        st.markdown(
            f"# {predicted_digit}"
        )


        st.markdown(
            f"**Predicted Digit: {predicted_digit}**"
        )


        st.markdown(
            f"**Confidence: {confidence * 100:.2f}%**"
        )


        st.progress(
            float(confidence)
        )


        # ==================================
        # CLASS PROBABILITIES
        # ==================================

        st.markdown(
            "### Class Probabilities"
        )


        probability_array = (
            probabilities[0]
            .detach()
            .cpu()
            .numpy()
        )


        for digit, probability in enumerate(
            probability_array
        ):

            st.write(
                f"Digit **{digit}** — "
                f"{probability * 100:.2f}%"
            )


            st.progress(
                float(probability)
            )


# ==========================================
# IMAGE PREPARATION GUIDE
# ==========================================

st.divider()


with st.expander(
    "How to prepare an image for the model"
):

    st.markdown(
        "### Recommended Image"
    )


    st.write(
        "For reliable predictions:"
    )


    st.write(
        "One digit only"
    )


    st.write(
        "Clearly visible handwriting"
    )


    st.write(
        "Minimal background noise"
    )


    st.write(
        "Reasonably centered digit"
    )


    st.divider()


    st.markdown(
        "### Gemini / Image Generator Prompt"
    )


    st.write(
        "Use this prompt to create a "
        "test image:"
    )


    st.code(
        """
Create a single handwritten digit "7",
centered in the image, using simple
MNIST-style handwriting.

Use a plain black background and a
clearly visible white digit.

No other objects, no text, no shadows,
no decorations, square image.
        """.strip(),
        language="text"
    )


    st.caption(
        "Replace 7 with any digit from 0 to 9."
    )


# ==========================================
# ABOUT MODEL
# ==========================================

with st.expander(
    "About the CNN model"
):

    st.markdown(
        "### Architecture"
    )


    st.write(
        "PyTorch Convolutional Neural Network"
    )


    st.markdown(
        "### Input"
    )


    st.code(
        "1 × 28 × 28"
    )


    st.markdown(
        "### Feature Extraction"
    )


    st.write(
        "Conv2D → BatchNorm → ReLU → "
        "MaxPool → Conv2D → BatchNorm → "
        "ReLU → MaxPool"
    )


    st.markdown(
        "### Classifier"
    )


    st.write(
        "Flatten → Linear → ReLU → "
        "Dropout → Linear"
    )


    st.markdown(
        "### Output"
    )


    st.write(
        "10 classes representing digits "
        "0 through 9."
    )


# ==========================================
# FOOTER
# ==========================================

st.markdown(
    '<div class="footer">'
    'MNIST Vision AI · PyTorch CNN'
    '</div>',
    unsafe_allow_html=True
)
