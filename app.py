"""
GreenScreen Studio: Interactive Web Application.
Built with Streamlit for real-time virtual background replacement and chroma keying.
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from greenscreen.chroma_key import ChromaKeyer
from greenscreen.segmentation import SegmentationEngine
from greenscreen.backgrounds import BackgroundManager
from greenscreen.pipeline import VirtualBackgroundPipeline, PipelineMode, BackgroundType

st.set_page_config(
    page_title="GreenScreen Virtual Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎬 GreenScreen Virtual Studio")
st.caption("AI-Powered & Chroma-Key Virtual Background Studio with Real-Time Edge Feathering & Bokeh Blur.")

# Presets directory
FILES_DIR = os.path.join(os.path.dirname(__file__), "files")
preset_images = {
    "Lush Greenery": os.path.join(FILES_DIR, "greenery.jpeg"),
    "Retro Mario": os.path.join(FILES_DIR, "mario.jpeg"),
    "Warm Wood Studio": os.path.join(FILES_DIR, "wood.jpg"),
    "Minimalist White": os.path.join(FILES_DIR, "white_bg.jpg"),
}

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Studio Controls")

    # Engine selection
    engine_choice = st.radio(
        "Segmentation Engine",
        ["🤖 AI Selfie Segmentation", "🟩 Classic Chroma Key"],
        index=0
    )

    feather = st.slider("Edge Feathering (Smoothing)", min_value=1, max_value=21, value=7, step=2)

    # Chroma key fine tuning
    if "Chroma Key" in engine_choice:
        st.subheader("Chroma Tuning")
        preset_color = st.selectbox("Key Color Preset", ["Green Screen", "Blue Screen", "Custom HSV Range"])
        if preset_color == "Green Screen":
            h_low, h_high = 35, 85
            s_low, v_low = 80, 50
        elif preset_color == "Blue Screen":
            h_low, h_high = 100, 135
            s_low, v_low = 100, 50
        else:
            h_low, h_high = st.slider("Hue Range (0-179)", 0, 179, (35, 85))
            s_low = st.slider("Min Saturation", 0, 255, 80)
            v_low = st.slider("Min Value / Brightness", 0, 255, 50)

        lower_hsv = np.array([h_low, s_low, v_low], dtype=np.uint8)
        upper_hsv = np.array([h_high, 255, 255], dtype=np.uint8)
        chroma_keyer = ChromaKeyer(lower_hsv, upper_hsv, feather_radius=feather)
    else:
        chroma_keyer = ChromaKeyer(feather_radius=feather)

    segmenter = SegmentationEngine(feather_radius=feather)

    st.markdown("---")
    st.subheader("🖼️ Background Selection")
    bg_mode = st.radio("Background Type", ["Preset Image", "Custom Upload", "🌫️ Bokeh Blur", "🎨 Solid Color"])

    selected_bg_bgr = None
    blur_ksize = 35
    solid_bgr = (255, 255, 255)

    if bg_mode == "Preset Image":
        preset_name = st.selectbox("Choose Preset", list(preset_images.keys()))
        path = preset_images[preset_name]
        if os.path.exists(path):
            selected_bg_bgr = cv2.imread(path)
            st.image(Image.open(path), caption=preset_name, width=180)
    elif bg_mode == "Custom Upload":
        custom_bg_file = st.file_uploader("Upload Background Image", type=["jpg", "jpeg", "png", "webp"])
        if custom_bg_file:
            pil_bg = Image.open(custom_bg_file).convert("RGB")
            selected_bg_bgr = cv2.cvtColor(np.array(pil_bg), cv2.COLOR_RGB2BGR)
            st.image(pil_bg, caption="Uploaded Background", width=180)
    elif bg_mode == "🌫️ Bokeh Blur":
        blur_ksize = st.slider("Blur Intensity", min_value=11, max_value=81, value=35, step=2)
    elif bg_mode == "🎨 Solid Color":
        color_hex = st.color_picker("Pick Backdrop Color", "#1E88E5")
        # Convert hex to BGR
        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)
        solid_bgr = (b, g, r)

# Configure pipeline
mode = PipelineMode.AI_SEGMENTATION if "AI" in engine_choice else PipelineMode.CHROMA_KEY
pipeline = VirtualBackgroundPipeline(mode=mode, chroma_keyer=chroma_keyer, segmenter=segmenter)

if bg_mode in ("Preset Image", "Custom Upload") and selected_bg_bgr is not None:
    pipeline.set_background_image(selected_bg_bgr)
elif bg_mode == "🌫️ Bokeh Blur":
    pipeline.set_background_blur(blur_ksize)
elif bg_mode == "🎨 Solid Color":
    pipeline.set_background_solid(solid_bgr)

# Main area
input_tab1, input_tab2 = st.tabs(["📷 Live Camera Snapshot", "📁 Upload Image / Synthetic Test"])

input_frame_bgr = None

with input_tab1:
    st.write("Capture a frame from your webcam:")
    camera_photo = st.camera_input("Webcam Capture")
    if camera_photo is not None:
        pil_img = Image.open(camera_photo).convert("RGB")
        input_frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

with input_tab2:
    uploaded_source = st.file_uploader("Upload Image to apply Virtual Background", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_source is not None:
        pil_img = Image.open(uploaded_source).convert("RGB")
        input_frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        if st.button("🧪 Generate Synthetic Chroma Screen Test Frame"):
            # Create a 640x480 test image with green background and a center person-like oval
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            test_frame[:] = (0, 220, 0)  # Pure green BGR
            # Draw subject
            cv2.circle(test_frame, (320, 180), 70, (180, 150, 120), -1)  # Head
            cv2.ellipse(test_frame, (320, 360), (120, 150), 0, 0, 360, (220, 100, 50), -1)  # Torso
            input_frame_bgr = test_frame

if input_frame_bgr is not None:
    st.markdown("### 🎞️ Composition Result")
    # Process
    output_bgr = pipeline.process_frame(input_frame_bgr)

    # Convert to RGB for Streamlit
    orig_rgb = cv2.cvtColor(input_frame_bgr, cv2.COLOR_BGR2RGB)
    out_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Input")
        st.image(orig_rgb, use_container_width=True)

    with col2:
        st.subheader("Virtual Background Output")
        st.image(out_rgb, use_container_width=True)

    # Download button
    pil_out = Image.fromarray(out_rgb)
    import io
    buf = io.BytesIO()
    pil_out.save(buf, format="PNG")
    st.download_button(
        label="💾 Download Composite Image (PNG)",
        data=buf.getvalue(),
        file_name="greenscreen_output.png",
        mime="image/png"
    )
else:
    st.info("💡 Capture a webcam photo or upload an image above to see the virtual background magic in action!")
