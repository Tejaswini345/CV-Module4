#!/usr/bin/env python3
"""
============================================================================
rgb_human_segmentation.py
CSc 8830 Computer Vision — Module 4, Question 1

WHAT THIS SCRIPT DOES
----------------------------------------------------------------------------
Finds the exact pixel boundary of a human in a regular RGB image using only
classical (non-ML, non-deep-learning) OpenCV / NumPy operations:

USAGE
----------------------------------------------------------------------------
    python3 rgb_human_segmentation.py --input person.jpg --output out_rgb
    python3 rgb_human_segmentation.py --input person.jpg --output out_rgb \
            --margin 15 --iters 5

    Required libs: opencv-python, numpy   (pip install opencv-python numpy)

OUTPUT
----------------------------------------------------------------------------
    <output>_mask.png       binary (0/255) segmentation mask
    <output>_boundary.png   original image with the extracted boundary
                             contour drawn in green
    <output>_saliency.png   intermediate spectral-residual saliency map
                             (for the report / sanity-checking)
    stdout: bounding box, contour point count, foreground pixel count
============================================================================
"""

import argparse
import os
import sys

import cv2
import numpy as np


def spectral_residual_saliency(gray: np.ndarray, resize_to: int = 128) -> np.ndarray:
    """Hou & Zhang (2007) spectral-residual saliency map, computed with
    cv2.dft (pure frequency-domain signal processing, no learning)."""
    h, w = gray.shape
    small = cv2.resize(gray, (resize_to, resize_to), interpolation=cv2.INTER_AREA)
    small = small.astype(np.float32)

    dft = cv2.dft(small, flags=cv2.DFT_COMPLEX_OUTPUT)
    real, imag = dft[:, :, 0], dft[:, :, 1]
    magnitude = cv2.magnitude(real, imag)
    phase = cv2.phase(real, imag)

    log_amplitude = np.log(magnitude + 1e-8)
    # "expected" spectrum = local average of the log spectrum (3x3 box filter)
    avg_log_amplitude = cv2.blur(log_amplitude, (3, 3))
    spectral_residual = log_amplitude - avg_log_amplitude

    # Reconstruct with the residual amplitude but the ORIGINAL phase.
    real2 = np.exp(spectral_residual) * np.cos(phase)
    imag2 = np.exp(spectral_residual) * np.sin(phase)
    combined = cv2.merge([real2, imag2])
    inv = cv2.idft(combined)
    inv_mag = cv2.magnitude(inv[:, :, 0], inv[:, :, 1])

    saliency = cv2.GaussianBlur(inv_mag ** 2, (9, 9), 2.5)
    saliency = cv2.resize(saliency, (w, h), interpolation=cv2.INTER_CUBIC)
    saliency = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return saliency


def largest_component_bbox(binary: np.ndarray, margin: int = 10):
    """Return (x, y, w, h) bounding box of the largest connected component,
    expanded by `margin` px and clipped to image bounds."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        h, w = binary.shape
        return 0, 0, w, h
    # label 0 is background; pick the largest non-background area
    idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x, y, w, h, _ = stats[idx]
    H, W = binary.shape
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(W, x + w + margin), min(H, y + h + margin)
    return x0, y0, x1 - x0, y1 - y0


def center_bias_prior(h: int, w: int, sigma_frac: float = 0.35) -> np.ndarray:
    """Classical center-surround prior (Judd et al.): photographed subjects
    are usually placed near the frame center, so weight saliency by a
    broad Gaussian centered on the image. Pure geometry, no learning."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h / 2.0, w / 2.0
    sy, sx = sigma_frac * h, sigma_frac * w
    prior = np.exp(-(((xx - cx) ** 2) / (2 * sx ** 2) + ((yy - cy) ** 2) / (2 * sy ** 2)))
    return cv2.normalize(prior, None, 0, 1, cv2.NORM_MINMAX)


def segment_human_rgb(image: np.ndarray, margin: int = 10, grabcut_iters: int = 5, use_center_bias: bool = True):
    """Full Stage A->D pipeline. Returns (final_mask, saliency_map, bbox, contour)."""
    H, W = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # ---- Stage A: frequency-domain saliency, re-weighted by a center prior.
    # Raw spectral-residual saliency responds to ANY high-frequency clutter
    # (a busy background, foliage, a crowd through a window) which can
    # outscore a smooth, well-exposed human subject. Multiplying by a
    # center-bias prior keeps the frequency-domain signal but stops
    # background clutter near the image edges from winning. ----
    raw_saliency = spectral_residual_saliency(gray)
    if use_center_bias:
        prior = center_bias_prior(H, W)
        saliency = (raw_saliency.astype(np.float32) * prior)
        saliency = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        saliency = raw_saliency

    # ---- Stage B: Otsu threshold + morphology -> rough foreground blob ----
    _, rough_mask = cv2.threshold(saliency, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    rough_mask = cv2.morphologyEx(rough_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    rough_mask = cv2.morphologyEx(rough_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    bbox = largest_component_bbox(rough_mask, margin=margin)

    # Keep only the largest saliency component (drop small distractors),
    # then build a *probable*-foreground region by dilating it generously
    # so GrabCut has room to grow into the rest of the body (saliency
    # typically only lights up high-contrast parts like the head/edges).
    n, labels, stats, _ = cv2.connectedComponentsWithStats(rough_mask, connectivity=8)
    if n > 1:
        idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        core = np.where(labels == idx, 255, 0).astype(np.uint8)
    else:
        core = rough_mask

    sure_fg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    probable_fg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 121))  # tall, human-shaped growth
    sure_fg = cv2.erode(core, sure_fg_kernel, iterations=1)
    probable_fg = cv2.dilate(core, probable_fg_kernel, iterations=1)

    # ---- Stage C: GrabCut refinement, seeded with a MASK (sure/probable
    # fg from saliency, sure bg = a thin image border) instead of just a
    # rectangle -- more robust than a rect alone when the salient blob
    # doesn't cover the whole body.
    # Default EVERYTHING to sure background, then open up only a generous
    # region around the detected person as "probable" territory GrabCut is
    # allowed to explore. Without this cap, a smooth/uniform backdrop (e.g.
    # a studio wall) has near-zero smoothness cost, so GrabCut's graph-cut
    # can "leak" across the whole flat region once it touches the subject's
    # edge — producing a large disconnected blob far from the actual body.
    bx, by, bw, bh = bbox
    search_margin_x = int(bw * 1.0)
    search_margin_y = int(bh * 1.0)
    sx0, sy0 = max(0, bx - search_margin_x), max(0, by - search_margin_y)
    sx1, sy1 = min(W, bx + bw + search_margin_x), min(H, by + bh + search_margin_y)

    gc_mask = np.full((H, W), cv2.GC_BGD, np.uint8)
    gc_mask[sy0:sy1, sx0:sx1] = cv2.GC_PR_BGD
    gc_mask[probable_fg > 0] = cv2.GC_PR_FGD
    gc_mask[sure_fg > 0] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, gc_mask, None, bgd_model, fgd_model,
                grabcut_iters, cv2.GC_INIT_WITH_MASK)
    binary_fg = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # ---- Stage D: clean up + keep largest contour as the exact boundary ----
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_fg = cv2.morphologyEx(binary_fg, cv2.MORPH_OPEN, kernel2)
    binary_fg = cv2.morphologyEx(binary_fg, cv2.MORPH_CLOSE, kernel2)

    contours, _ = cv2.findContours(binary_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours or cv2.contourArea(max(contours, key=cv2.contourArea)) < 30:
        # Fallback: GrabCut with a plain rectangle seed (classic OpenCV
        # tutorial approach) if the mask-seeded run collapsed to nothing.
        gc_mask2 = np.zeros((H, W), np.uint8)
        cv2.grabCut(image, gc_mask2, bbox, bgd_model, fgd_model,
                    grabcut_iters, cv2.GC_INIT_WITH_RECT)
        binary_fg = np.where((gc_mask2 == cv2.GC_FGD) | (gc_mask2 == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        binary_fg = cv2.morphologyEx(binary_fg, cv2.MORPH_OPEN, kernel2)
        binary_fg = cv2.morphologyEx(binary_fg, cv2.MORPH_CLOSE, kernel2)
        contours, _ = cv2.findContours(binary_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            raise RuntimeError("No foreground contour found — try a different image or --margin.")

    contour = max(contours, key=cv2.contourArea)
    final_mask = np.zeros_like(binary_fg)
    cv2.drawContours(final_mask, [contour], -1, 255, thickness=cv2.FILLED)

    return final_mask, saliency, bbox, contour


def main():
    parser = argparse.ArgumentParser(description="Classical RGB human boundary segmentation.")
    parser.add_argument("--input", required=True, help="Path to input RGB image.")
    parser.add_argument("--output", default="out_rgb", help="Output file prefix.")
    parser.add_argument("--margin", type=int, default=10, help="Pixel margin around the GrabCut seed box.")
    parser.add_argument("--iters", type=int, default=5, help="GrabCut iterations.")
    parser.add_argument("--no-center-bias", action="store_true",
                         help="Disable the center-bias prior (use if your subject is off-center / near an edge).")
    args = parser.parse_args()

    image = cv2.imread(args.input)
    if image is None:
        sys.exit(f"Could not read image: {args.input}")

    mask, saliency, bbox, contour = segment_human_rgb(
        image, args.margin, args.iters, use_center_bias=not args.no_center_bias)

    overlay = image.copy()
    cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)
    cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 1)

    out_paths = [f"{args.output}_mask.png", f"{args.output}_boundary.png", f"{args.output}_saliency.png"]
    out_images = [mask, overlay, saliency]
    for path, img in zip(out_paths, out_images):
        out_dir = os.path.dirname(path)
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        ok = cv2.imwrite(path, img)
        if not ok:
            sys.exit(f"ERROR: could not write '{path}'. Check that --output "
                      f"doesn't end in a slash/backslash and that the folder exists.")

    print(f"Seed bounding box (x,y,w,h): {bbox}")
    print(f"Contour points: {len(contour)}")
    print(f"Foreground pixels: {int(np.sum(mask > 0))} / {mask.size}")
    print(f"Saved: {args.output}_mask.png, {args.output}_boundary.png, {args.output}_saliency.png")


if __name__ == "__main__":
    main()
