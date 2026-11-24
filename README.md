# 🎬 GreenScreen Virtual Studio

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/Interface-Streamlit-red.svg)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.x%20%7C%204.x-5C3EE8.svg)](https://opencv.org)
[![Tests: Passing](https://img.shields.io/badge/Tests-5%20Passed-brightgreen.svg)]()

> Real-time virtual background studio combining **AI selfie segmentation** and **high-precision HSV chroma keying** with soft edge feathering, bokeh background blurring, and custom virtual backdrops.

---

## 🌟 Key Features

- **🤖 Dual Segmentation Engines**:
  - **AI Selfie Segmentation**: Deep learning-based human extraction requiring no physical green screen.
  - **HSV Chroma Keying**: Tunable color-space keyer with custom color pickers, tolerance margins, and noise filtering.
- **🌫️ Virtual Bokeh & Custom Backdrops**:
  - Background Blur (depth-of-field simulation similar to Zoom / MS Teams).
  - Preset Virtual Studios (Lush Greenery, Retro Mario, Warm Wood, Modern White).
  - Custom image file uploads with automatic aspect-ratio preservation.
  - Solid color virtual backdrops.
- **✨ Edge Refinement & Anti-Aliasing**:
  - Morphological opening/closing for removing camera noise and speckles.
  - Gaussian edge feathering for smooth human-to-background transitions without harsh cutouts.
- **💻 Interactive Streamlit Studio**:
  - Live webcam preview and snapshot recording.
  - Real-time parameter sliders (feathering, hue, saturation, blur intensity).
  - Side-by-side comparison and instant PNG download.
- **⚡ Unified CLI**:
  - Process webcam streams, video clips, or static photos with a single command.

---

## 📐 Pipeline Architecture

```
+-------------------------------------------------------------+
|                      Camera / Image Input                   |
+------------------------------+------------------------------+
                               |
        +----------------------+----------------------+
        |                                             |
+-------v-------+                             +-------v-------+
|  Chroma Keyer |                             | AI Segmenter  |
|  - HSV InRange|                             |  - Deep Mask  |
|  - Morphology |                             |  - Soft Edges |
+-------+-------+                             +-------+-------+
        |                                             |
        +----------------------+----------------------+
                               |
                       [ Alpha Mask α ]
                               |
+------------------------------v------------------------------+
|                    Background Synthesis                     |
|         (Image Preset / Upload / Bokeh Blur / Solid)        |
+------------------------------+------------------------------+
                               |
+------------------------------v------------------------------+
|                   Alpha Blending Engine                     |
|           Result = α · Foreground + (1 - α) · Background    |
+-------------------------------------------------------------+
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/ketan-rmcf123/green-screen.git
cd green-screen
pip install -r requirements.txt
```

### 2. Launch Interactive Web App

```bash
streamlit run app.py
```
Open `http://localhost:8501` to use the interactive studio with your webcam or test images.

### 3. Command Line Interface (CLI)

```bash
# 1. Run live webcam with virtual background image
python -m greenscreen.cli --input 0 --bg files/wood.jpg

# 2. Run live webcam with virtual bokeh blur (Zoom/Teams style)
python -m greenscreen.cli --input 0 --bg blur --blur 35

# 3. Process a static photo and save the result
python -m greenscreen.cli --input my_photo.jpg --bg files/greenery.jpeg --output result.png
```

---

## 🧪 Testing

Run the automated test suite:

```bash
python -m unittest discover -s tests
# or using pytest
pytest tests/
```

All 5 unit tests validate chroma keying, background aspect ratio scaling, alpha blending math, and pipeline execution.

---

## 📁 Repository Structure

```
green-screen/
├── app.py                      # Interactive Streamlit Web Studio
├── pyproject.toml              # Modern package manifest
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git exclusions
├── greenscreen/                # Core virtual background engine
│   ├── __init__.py
│   ├── chroma_key.py           # Precision HSV chroma keyer
│   ├── segmentation.py         # AI selfie segmenter & edge smoother
│   ├── backgrounds.py          # Background management & alpha compositing
│   ├── pipeline.py             # Unified virtual background pipeline
│   └── cli.py                  # CLI runner
├── files/                      # Preset virtual backgrounds
│   ├── greenery.jpeg
│   ├── mario.jpeg
│   ├── white_bg.jpg
│   └── wood.jpg
├── code/                       # Original reference scripts
│   ├── capturescreen.py
│   ├── configuration.py
│   └── mediapipe_test.py
└── tests/
    ├── __init__.py
    └── test_greenscreen.py     # Unit test suite
```

---

## 📜 License

MIT License. Free for personal, educational, and commercial use.
