"""
GreenScreen CLI Runner.
"""

import argparse
import sys
import os
import cv2
import numpy as np

from .pipeline import VirtualBackgroundPipeline, PipelineMode, BackgroundType
from .chroma_key import ChromaKeyer
from .segmentation import SegmentationEngine
from .backgrounds import BackgroundManager


def main():
    parser = argparse.ArgumentParser(
        prog="greenscreen",
        description="GreenScreen: Modern Real-Time Virtual Background Studio"
    )
    parser.add_argument("--mode", choices=["chroma", "ai"], default="chroma", help="Segmentation mode")
    parser.add_argument("--input", default="0", help="Webcam index (0) or path to input image/video")
    parser.add_argument("--bg", default=None, help="Path to background image or 'blur'")
    parser.add_argument("--output", default=None, help="Path to save output image or video")
    parser.add_argument("--blur", type=int, default=35, help="Blur radius if using background blur")

    args = parser.parse_args()

    mode = PipelineMode.AI_SEGMENTATION if args.mode == "ai" else PipelineMode.CHROMA_KEY
    pipeline = VirtualBackgroundPipeline(mode=mode)

    if args.bg == "blur" or args.bg is None:
        pipeline.set_background_blur(args.blur)
    elif os.path.exists(args.bg):
        bg_img = cv2.imread(args.bg)
        if bg_img is not None:
            pipeline.set_background_image(bg_img)
        else:
            print(f"[!] Warning: Could not read background {args.bg}, defaulting to blur.")
            pipeline.set_background_blur(args.blur)

    # Check if input is a static image
    if os.path.isfile(args.input) and args.input.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        frame = cv2.imread(args.input)
        if frame is None:
            print(f"[-] Failed to load image {args.input}")
            sys.exit(1)
        res = pipeline.process_frame(frame)
        if args.output:
            cv2.imwrite(args.output, res)
            print(f"[+] Saved output to {args.output}")
        else:
            cv2.imshow("GreenScreen Result (Press Any Key to Exit)", res)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # Video or Webcam
    cam_id = int(args.input) if args.input.isdigit() else args.input
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[-] Failed to open video source: {args.input}")
        sys.exit(1)

    print("[+] Running GreenScreen. Press 'q' in the preview window to exit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        output = pipeline.process_frame(frame)
        cv2.imshow("GreenScreen Studio Preview ('q' to quit)", output)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
