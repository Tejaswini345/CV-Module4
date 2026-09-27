#!/usr/bin/env python3
"""
============================================================================
thermal_human_segmentation.py
CSc 8830 Computer Vision — Module 4, Question 2

WHAT THIS SCRIPT DOES
----------------------------------------------------------------------------
Finds the exact pixel boundary of a human in a thermal/IR image using only
classical (non-ML, non-deep-learning) OpenCV / NumPy operations. Thermal
imagery of people (e.g. FLIR-style frames, see the Roboflow thermal dataset
blog) has a much simpler statistical structure than RGB: a human body is
usually a single compact blob that is radiometrically hotter (brighter)
than most of the background, so a global/adaptive intensity threshold is
enough to get a very clean boundary, refined with morphology.

USAGE
----------------------------------------------------------------------------
    python3 thermal_human_segmentation.py --input person_thermal.jpg --output out_thermal
    python3 thermal_human_segmentation.py --input frame.png --output out_thermal --min-area 500

    Required libs: opencv-python, numpy

OUTPUT
----------------------------------------------------------------------------
    <output>_mask.png       binary (0/255) segmentation mask
    <output>_boundary.png   original image with the extracted boundary
                             contour drawn in green
    stdout: chosen polarity, threshold value, bounding box, contour count
============================================================================
"""

import argparse
import os
import sys

import cv2
import numpy as np


def _blob_quality(binary: np.ndarray) -> float:
    """Simple, classifier-free heuristic score for 'does this binary mask
    look like a single compact human blob?' Higher is better. Rewards a
    large, solid, roughly-portrait-shaped single component and penalizes
    fragmentation (many tiny separate blobs, e.g. thresholded background
    clutter)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return -1.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = np.argmax(areas)
    largest_area = areas[largest]
    total_fg = areas.sum()
    solidity = largest_area / max(total_fg, 1)  # 1.0 = one clean blob

    x, y, w, h, _ = stats[1 + largest]
    extent = largest_area / max(w * h, 1)  # filled fraction of its own bbox
    aspect = h / max(w, 1)  # humans are usually taller than wide (standing/sitting)
    aspect_score = 1.0 - min(abs(aspect - 1.8) / 1.8, 1.0)

    area_frac = largest_area / binary.size
    size_score = 1.0 - min(abs(area_frac - 0.12) / 0.5, 1.0)  # not tiny, not whole frame

    return 0.4 * solidity + 0.25 * extent + 0.2 * aspect_score + 0.15 * size_score


def segment_human_thermal(image: np.ndarray, min_area: int = 200):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # ---- Stage B: Otsu threshold (gives the split value once) ----
    thresh_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def clean(binary):
        m = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
        return m

    # ---- Stage C: try both polarities, keep the better-scoring blob ----
    bright_fg = clean(cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)[1])
    dark_fg = clean(cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)[1])

    bright_score = _blob_quality(bright_fg)
    dark_score = _blob_quality(dark_fg)
    polarity = "bright=hot=foreground" if bright_score >= dark_score else "dark=hot=foreground"
    binary_fg = bright_fg if bright_score >= dark_score else dark_fg

    # ---- Stage D: keep the largest component above min_area ----
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary_fg, connectivity=8)
    if n <= 1:
        raise RuntimeError("No foreground blob found in thermal image.")
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + np.argmax(areas)
    if areas.max() < min_area:
        raise RuntimeError(f"Largest blob ({areas.max()}px) below --min-area {min_area}.")
    largest_mask = np.where(labels == idx, 255, 0).astype(np.uint8)

    contours, _ = cv2.findContours(largest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    final_mask = np.zeros_like(largest_mask)
    cv2.drawContours(final_mask, [contour], -1, 255, thickness=cv2.FILLED)

    bbox = cv2.boundingRect(contour)
    return final_mask, contour, bbox, thresh_val, polarity


def main():
    parser = argparse.ArgumentParser(description="Classical thermal human boundary segmentation.")
    parser.add_argument("--input", required=True, help="Path to input thermal image.")
    parser.add_argument("--output", default="out_thermal", help="Output file prefix.")
    parser.add_argument("--min-area", type=int, default=200, help="Minimum blob area in pixels.")
    args = parser.parse_args()

    image = cv2.imread(args.input)
    if image is None:
        sys.exit(f"Could not read image: {args.input}")

    mask, contour, bbox, thresh_val, polarity = segment_human_thermal(image, args.min_area)

    overlay = image.copy()
    cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)
    cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 1)

    out_paths = [f"{args.output}_mask.png", f"{args.output}_boundary.png"]
    out_images = [mask, overlay]
    for path, img in zip(out_paths, out_images):
        out_dir = os.path.dirname(path)
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        ok = cv2.imwrite(path, img)
        if not ok:
            sys.exit(f"ERROR: could not write '{path}'. Check that --output "
                      f"doesn't end in a slash/backslash and that the folder exists.")

    print(f"Otsu threshold value: {thresh_val:.1f}")
    print(f"Chosen polarity: {polarity}")
    print(f"Bounding box (x,y,w,h): {bbox}")
    print(f"Contour points: {len(contour)}")
    print(f"Saved: {args.output}_mask.png, {args.output}_boundary.png")


if __name__ == "__main__":
    main()
