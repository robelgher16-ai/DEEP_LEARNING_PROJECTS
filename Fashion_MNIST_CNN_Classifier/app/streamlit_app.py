# ==========================================
# Fashion MNIST CNN Classifier
# Streamlit Frontend (Render API Version)
# ==========================================

import streamlit as st
import requests
from PIL import Image

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Fashion MNIST CNN",
    page_icon="👕",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
.stApp{
    background: linear-gradient(135deg,#eef6ff,#f8fbff);
}

.hero{
    background: linear-gradient(135deg,#2563eb,#7c3aed);
    padding:28px;
    border-radius:18px;
    color:white;
    text-align:center;
    margin-bottom:20px;
}

.hero h1{
    margin:0;
    font-size:38px;
}

.card{
    background:white;
    padding:18px;
    border-radius:16px;
    box-shadow:0 6px 18px rgba(0,0,0,.08);
    border-left:6px solid #2563eb;
}

.result{
    background:linear-gradient(135deg,#10b981,#059669);
    color:white;
    padding:22px;
    border-radius:16px;
    text-align:center;
}

section[data-testid="stSidebar"]{
    background:#172554;
}

section[data-testid="stSidebar"] *{
    color:white;
}

[data-testid="stFileUploader"]{
    border:2px dashed #2563eb;
    border-radius:14px;
    background:white;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# RENDER API
# ==========================================

API_URL = "https://fashion-mnist-cnn-api.onrender.com/predict"

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("Fashion MNIST CNN")

st.sidebar.markdown("### Dataset")
st.sidebar.write("• 70,000 Images")
st.sidebar.write("• 28×28 Grayscale")
st.sidebar.write("• 10 Classes")

st.sidebar.markdown("---")

st.sidebar.markdown("### Architecture")
st.sidebar.write("Conv → BatchNorm")
st.sidebar.write("ReLU → MaxPool")
st.sidebar.write("Dropout")
st.sidebar.write("Fully Connected")

st.sidebar.markdown("---")
st.sidebar.success("Render API Connected")

# ==========================================
# HERO
# ==========================================

st.markdown("""
<div class="hero">
<h1>Fashion MNIST CNN Classifier</h1>
<p>Professional Deep Learning Image Classification using PyTorch, FastAPI & Streamlit</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# INFO CARDS
# ==========================================

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="card">
    <h3>Dataset</h3>
    <h2>70,000</h2>
    <p>Fashion-MNIST Images</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="card">
    <h3>Classes</h3>
    <h2>10</h2>
    <p>Clothing Categories</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="card">
    <h3>Backend</h3>
    <h2>FastAPI</h2>
    <p>Hosted on Render</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ==========================================
# FILE UPLOADER
# ==========================================

uploaded = st.file_uploader(
    "Upload a Fashion-MNIST style clothing image",
    type=["png", "jpg", "jpeg"]
)

if uploaded is not None:

    image = Image.open(uploaded).convert("L")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.image(image.resize((28, 28)),
                 caption="28×28 Preview",
                 use_container_width=True)

    st.write("")

    with st.spinner("Predicting..."):

        try:

            response = requests.post(
                API_URL,
                files={
                    "file": (
                        uploaded.name,
                        uploaded.getvalue(),
                        uploaded.type
                    )
                },
                timeout=60
            )

            response.raise_for_status()

            result = response.json()

            st.markdown(f"""
            <div class="result">
                <p>Predicted Class</p>
                <h2>{result['prediction']}</h2>
                <h3>{result['confidence']:.2f}% Confidence</h3>
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            st.subheader("Top 3 Predictions")

            for item in result["top3"]:
                st.write(f"**{item['class']}**")
                st.progress(item["confidence"] / 100)
                st.caption(f"{item['confidence']:.2f}%")

        except requests.exceptions.RequestException:

            st.error("""
The prediction server is unavailable.

Render free services sleep after inactivity.
Please wait **30–60 seconds** and try again.
""")

# ==========================================
# GEMINI TEST PROMPTS
# ==========================================

with st.expander("🧪 Hidden AI Test Prompts"):

    st.markdown("Generate Fashion-MNIST style images for testing.")

    sneaker_prompt = """
Generate a Fashion-MNIST style sneaker image.

Requirements:
- Single sneaker
- Side view
- Centered
- Black background
- White object
- Grayscale only
- High contrast
- 28×28 appearance
- No text
- No logo
"""

    st.code(sneaker_prompt)

    st.info("""
Replace **Sneaker** with:

• T-shirt/Top
• Trouser
• Pullover
• Dress
• Coat
• Sandal
• Shirt
• Sneaker
• Bag
• Ankle Boot
""")

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")
st.caption("Developed by Robel Gebregziabher • Deep Learning Portfolio Project")