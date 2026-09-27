#!/usr/bin/env python3
"""
============================================================================
batch_convert_masks.py
CSc 8830 Computer Vision — Module 4

WHAT THIS DOES
----------------------------------------------------------------------------
Scans assets/ for every sample's raw SAM2 overlay export (the blue-tinted
image straight off sam2.metademolab.com) and converts each one into a
clean binary (black/white) mask, by diffing it against the matching
original image. Does this for BOTH rgb and thermal, for every sample it
finds — one run instead of doing this 8 times by hand.

BEFORE RUNNING
----------------------------------------------------------------------------
Your assets/ folder needs, for each sample N:
    sampleN_rgb.jpg                    <- original RGB photo
    sampleN_sam2_rgb_mask_raw.png      <- raw SAM2 overlay export for the RGB image
    sampleN_thermal.jpg                <- original thermal image
    sampleN_sam2_thermal_mask_raw.png  <- raw SAM2 overlay export for the thermal image

(Extensions .jpg/.jpeg/.png are all accepted for every one of these.)

This script produces, for each pair found:
    sampleN_sam2_rgb_mask.png       <- clean binary mask (what app.py needs)
    sampleN_sam2_thermal_mask.png
    sampleN_sam2_rgb_diff_debug.png       <- heatmap, for sanity-checking
    sampleN_sam2_thermal_diff_debug.png

USAGE
----------------------------------------------------------------------------
    python3 batch_convert_masks.py
    python3 batch_convert_masks.py --threshold 50   (raise/lower sensitivity)

    Required libs: opencv-python, numpy (already installed for the Flask app)

TUNING NOTE (read this if a mask covers way more than the person)
----------------------------------------------------------------------------
SAM2 demo exports frequently come back at a noticeably LOWER resolution
than your original photo (we've seen a 1536x2816 original paired with a
627x1189 export — more than 2x smaller per side). Resizing that back up
introduces mild blur/aliasing across the ENTIRE frame, which raises the
measured diff of even completely untouched background pixels to roughly
20-35. If --threshold is set anywhere near that range, large chunks of
background get misclassified as "changed" and merged into the mask
(commonly seen as a spurious full-width strip along a table/floor, or a
blob around any small logo/watermark in the corner). The default
--threshold below (50) sits comfortably above that noise floor and below
the true signal (a real SAM2 recolor typically diffs 80+), based on
measurements across the sample set — but you can always raise it further
via --threshold if a specific image still bleeds into the background.
Check the *_diff_debug.png heatmap: background should look solidly black;
if it looks hazy grey instead, raise --threshold above that haze level.
============================================================================
"""

import argparse
import os
import sys

import cv2
import numpy as np

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
EXTENSIONS = ["jpg", "jpeg", "png"]


def find_asset(base_name: str):
    for ext in EXTENSIONS:
        path = os.path.join(ASSETS_DIR, f"{base_name}.{ext}")
        if os.path.isfile(path):
            return path
    return None


def overlay_to_binary_mask(original: np.ndarray, overlay: np.ndarray, threshold: int):
    if original.shape[:2] != overlay.shape[:2]:
        # SAM2 demo exports often come back at a much LOWER resolution than
        # the original photo. Upscaling with INTER_AREA (meant for
        # *shrinking* an image) introduces visible blur/aliasing across the
        # WHOLE frame, which raises the diff of even totally untouched
        # background pixels enough to trip a low --threshold. Use a proper
        # upscaling interpolation when the overlay is the smaller image,
        # and only use INTER_AREA when actually shrinking it down.
        interp = cv2.INTER_AREA if (overlay.shape[0] >= original.shape[0]) else cv2.INTER_CUBIC
        overlay = cv2.resize(overlay, (original.shape[1], original.shape[0]), interpolation=interp)

    diff = np.sqrt(np.sum((original.astype(np.float32) - overlay.astype(np.float32)) ** 2, axis=2))
    diff_vis = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # NOTE: diff can range 0..~441 for a 3-channel image, so casting straight
    # to uint8 before thresholding wraps values above 255 back around to
    # small numbers (like an odometer rolling over) -- silently misclassifying
    # strongly-changed pixels as unchanged. Compare the float distance
    # directly instead of casting it down first.
    mask = np.where(diff > threshold, 255, 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, diff_vis
    largest = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [largest], -1, 255, thickness=cv2.FILLED)
    return clean_mask, diff_vis


def process_one(sample_id: int, modality: str, threshold: int) -> str:
    original_path = find_asset(f"sample{sample_id}_{modality}")
    raw_path = find_asset(f"sample{sample_id}_sam2_{modality}_mask_raw")

    if not original_path or not raw_path:
        return "skip"

    original = cv2.imread(original_path)
    overlay = cv2.imread(raw_path)
    if original is None or overlay is None:
        return f"error: could not read image for sample{sample_id} {modality}"

    mask, diff_vis = overlay_to_binary_mask(original, overlay, threshold)
    if mask is None:
        return f"error: no foreground region detected for sample{sample_id} {modality} (try --threshold lower)"

    mask_out = os.path.join(ASSETS_DIR, f"sample{sample_id}_sam2_{modality}_mask.png")
    debug_out = os.path.join(ASSETS_DIR, f"sample{sample_id}_sam2_{modality}_diff_debug.png")
    cv2.imwrite(mask_out, mask)
    cv2.imwrite(debug_out, diff_vis)

    fg_frac = float(np.mean(mask > 0)) * 100
    flag = "  <-- looks too large, try raising --threshold" if fg_frac > 60 else ""
    return f"OK  -> {os.path.basename(mask_out)}  ({fg_frac:.1f}% of frame is foreground){flag}"


def main():
    parser = argparse.ArgumentParser(description="Batch-convert all SAM2 raw overlays in assets/ into binary masks.")
    parser.add_argument("--threshold", type=int, default=50,
                         help="Color-change sensitivity (default 50 -- see the TUNING NOTE at the top of this "
                              "file if a mask still looks too large).")
    parser.add_argument("--max-samples", type=int, default=25, help="Highest sample number to check for.")
    args = parser.parse_args()

    if not os.path.isdir(ASSETS_DIR):
        sys.exit(f"assets/ folder not found at {ASSETS_DIR}")

    print(f"Scanning {ASSETS_DIR} for sample1..sample{args.max_samples} (threshold={args.threshold})\n")

    found_any = False
    for n in range(1, args.max_samples + 1):
        for modality in ("rgb", "thermal"):
            result = process_one(n, modality, args.threshold)
            if result == "skip":
                continue
            found_any = True
            print(f"sample{n} {modality:8s} : {result}")

    if not found_any:
        print("No matching sampleN_<modality>.* + sampleN_sam2_<modality>_mask_raw.* pairs found.")
        print("Check your filenames match the pattern described at the top of this script.")
    else:
        print("\nDone. Check the *_diff_debug.png files if any mask looks wrong -- background should look "
              "solidly black; a hazy grey background means --threshold needs to go higher.")


if __name__ == "__main__":
    main()