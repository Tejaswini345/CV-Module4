#!/usr/bin/env python3
"""
============================================================================
app.py — CSc 8830 Module 4 web application

Runs your ACTUAL, already-tested segmentation scripts
(rgb_human_segmentation.py, thermal_human_segmentation.py) and the ACTUAL
comparison metrics (compare_with_sam2.py) behind a simple Flask web page.

HOW TO RUN
----------------------------------------------------------------------------
    pip install -r requirements.txt
    python3 app.py

Then open http://localhost:5000 in your browser.
============================================================================
"""

import base64
import glob
import os

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, send_from_directory

from rgb_human_segmentation import segment_human_rgb
from thermal_human_segmentation import segment_human_thermal
from compare_with_sam2 import load_binary_mask, iou_dice, boundary_f_score

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")


def find_asset(base_name: str):
    """Look for assets/<base_name>.<ext> across common extensions.
    Returns the filename (not full path) or None if not found."""
    for ext in ("jpg", "jpeg", "png"):
        path = os.path.join(ASSETS_DIR, f"{base_name}.{ext}")
        if os.path.isfile(path):
            return f"{base_name}.{ext}"
    return None


def discover_samples():
    """Real filesystem scan (this is Python, so — unlike a static site —
    we can actually list what's in the folder) for sample1, sample2, ...
    stopping at the first missing rgb image."""
    samples = []
    n = 1
    while True:
        rgb = find_asset(f"sample{n}_rgb")
        if not rgb:
            break
        samples.append({
            "id": n,
            "label": f"Sample {n}",
            "rgb": rgb,
            "thermal": find_asset(f"sample{n}_thermal"),
            "sam2_rgb": find_asset(f"sample{n}_sam2_rgb_mask"),
            "sam2_thermal": find_asset(f"sample{n}_sam2_thermal_mask"),
        })
        n += 1
    return samples


def encode_png(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Failed to encode image.")
    return "data:image/png;base64," + base64.b64encode(buf).decode("utf-8")


@app.route("/")
def index():
    return render_template("index.html", samples=discover_samples())


@app.route("/assets/<path:filename>")
def assets(filename):
    return send_from_directory(ASSETS_DIR, filename)


@app.route("/api/process/<int:sample_id>/<modality>")
def process(sample_id, modality):
    if modality not in ("rgb", "thermal"):
        return jsonify({"error": "modality must be 'rgb' or 'thermal'"}), 400

    samples = {s["id"]: s for s in discover_samples()}
    sample = samples.get(sample_id)
    if not sample:
        return jsonify({"error": f"sample {sample_id} not found"}), 404

    image_filename = sample["rgb"] if modality == "rgb" else sample["thermal"]
    if not image_filename:
        return jsonify({"error": f"No {modality} image found for sample {sample_id}. "
                                  f"Expected assets/sample{sample_id}_{modality}.jpg (or .png)."}), 404

    image = cv2.imread(os.path.join(ASSETS_DIR, image_filename))
    if image is None:
        return jsonify({"error": f"Could not read {image_filename}"}), 500

    # GrabCut (used by segment_human_rgb) is a multi-iteration global
    # optimizer whose cost grows with pixel count -- a full-resolution phone
    # photo can take a long time. Downscale large images for this interactive
    # demo so the page stays responsive; your offline script run on the
    # original full-resolution image is still what should be reported as
    # your primary result.
    MAX_DIM = 900
    h, w = image.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    try:
        if modality == "rgb":
            mask, saliency, bbox, contour = segment_human_rgb(image)
            extra_info = {"bbox": [int(v) for v in bbox]}
        else:
            mask, contour, bbox, thresh_val, polarity = segment_human_thermal(image)
            extra_info = {"bbox": [int(v) for v in bbox], "otsu_threshold": float(thresh_val), "polarity": polarity}
    except Exception as e:
        return jsonify({"error": f"Segmentation failed: {e}"}), 500

    overlay = image.copy()
    cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)

    result = {
        "original": encode_png(image),
        "boundary": encode_png(overlay),
        "mask": encode_png(mask),
        **extra_info,
    }

    sam2_filename = sample["sam2_rgb"] if modality == "rgb" else sample["sam2_thermal"]
    if sam2_filename:
        classical_binary = (mask > 0).astype(np.uint8)
        sam2_binary = load_binary_mask(os.path.join(ASSETS_DIR, sam2_filename))
        if sam2_binary.shape != classical_binary.shape:
            sam2_binary = cv2.resize(
                sam2_binary, (classical_binary.shape[1], classical_binary.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        iou, dice = iou_dice(classical_binary, sam2_binary)
        bf = boundary_f_score(classical_binary, sam2_binary)

        compare_vis = image.copy()
        c_contours, _ = cv2.findContours(classical_binary * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        s_contours, _ = cv2.findContours(sam2_binary * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(compare_vis, c_contours, -1, (0, 255, 0), 2)   # yours = green
        cv2.drawContours(compare_vis, s_contours, -1, (0, 0, 255), 2)   # SAM2 = red

        result.update({
            "compare": encode_png(compare_vis),
            "iou": round(float(iou), 4),
            "dice": round(float(dice), 4),
            "boundary_f": round(float(bf), 4),
        })
    else:
        result["sam2_missing"] = f"No SAM2 mask found. Expected assets/sample{sample_id}_sam2_{modality}_mask.png"

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
