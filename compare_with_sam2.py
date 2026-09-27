#!/usr/bin/env python3
"""
============================================================================
compare_with_sam2.py
CSc 8830 Computer Vision — Module 4, Questions 1 & 2 (SAM2 comparison)

WHAT THIS SCRIPT DOES
----------------------------------------------------------------------------
Quantitatively compares a classical-CV segmentation mask (produced by
rgb_human_segmentation.py or thermal_human_segmentation.py) against a
SAM2 (Segment Anything Model 2, https://ai.meta.com/research/sam2/) mask
for the SAME image, using three standard region/boundary metrics:

    IoU (Jaccard)   = |A ∩ B| / |A ∪ B|
    Dice (F1)       = 2|A ∩ B| / (|A| + |B|)
    Boundary F-score = harmonic mean of boundary precision/recall within a
                        small tolerance band (here 2 px), i.e. how well the
                        two CONTOURS line up, not just the filled regions.

WHY SAM2 ITSELF IS NOT RUN INSIDE THIS SCRIPT
----------------------------------------------------------------------------
SAM2 is a transformer-based foundation model (Hiera image encoder + a
memory-attention mask decoder, hundreds of millions of parameters). Running
it requires: (1) `pip install sam2` + `torch`, (2) downloading a >150 MB
checkpoint from Meta's servers, and (3) ideally a GPU. Restricted/offline
grading containers (and this assignment's classical-CV scripts) deliberately
have none of that, and the assignment explicitly forbids ML/DL code in the
segmentation scripts themselves. So SAM2 inference is done ONCE, separately
(Colab, a local GPU machine, or the official web demo at
https://sam2.metademolab.com), its output mask is saved as a PNG, and THIS
script only loads the two PNGs and computes metrics — no model weights,
no torch import, no network access needed here.

HOW TO GENERATE THE SAM2 MASK (run this separately, not part of the
classical pipeline — shown here as reference/documentation):
----------------------------------------------------------------------------
    # pip install sam2 torch torchvision
    # from sam2.build_sam import build_sam2
    # from sam2.sam2_image_predictor import SAM2ImagePredictor
    # import cv2, numpy as np
    #
    # sam2_model = build_sam2("sam2_hiera_l.yaml", "sam2_hiera_large.pt", device="cuda")
    # predictor = SAM2ImagePredictor(sam2_model)
    # image = cv2.cvtColor(cv2.imread("person.jpg"), cv2.COLOR_BGR2RGB)
    # predictor.set_image(image)
    # # reuse the SAME seed bounding box the classical script printed, for
    # # an apples-to-apples prompt:
    # box = np.array([x, y, x + w, y + h])
    # masks, scores, _ = predictor.predict(box=box, multimask_output=False)
    # cv2.imwrite("sam2_mask.png", (masks[0] * 255).astype("uint8"))

USAGE
----------------------------------------------------------------------------
    python3 compare_with_sam2.py --classical out_rgb_mask.png --sam2 sam2_mask.png \
        --image person.jpg --output comparison.png

OUTPUT
----------------------------------------------------------------------------
    Prints IoU, Dice, boundary F-score to stdout.
    Saves a side-by-side visualization (classical boundary in green, SAM2
    boundary in red, overlap region shaded) to --output if provided.
============================================================================
"""

import argparse
import sys

import cv2
import numpy as np


def load_binary_mask(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        sys.exit(f"Could not read mask: {path}")
    return (m > 127).astype(np.uint8)


def iou_dice(a: np.ndarray, b: np.ndarray):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    iou = inter / union if union else 0.0
    dice = 2 * inter / (a.sum() + b.sum()) if (a.sum() + b.sum()) else 0.0
    return iou, dice


def boundary_f_score(a: np.ndarray, b: np.ndarray, tolerance: int = 2):
    """Boundary precision/recall within `tolerance` px, averaged into an F1."""
    def boundary(mask):
        edges = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)
        return edges > 0

    ba, bb = boundary(a), boundary(b)
    if not ba.any() or not bb.any():
        return 0.0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance + 1, 2 * tolerance + 1))
    ba_dilated = cv2.dilate(ba.astype(np.uint8), kernel).astype(bool)
    bb_dilated = cv2.dilate(bb.astype(np.uint8), kernel).astype(bool)

    precision = np.logical_and(ba, bb_dilated).sum() / max(ba.sum(), 1)
    recall = np.logical_and(bb, ba_dilated).sum() / max(bb.sum(), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main():
    parser = argparse.ArgumentParser(description="Compare a classical mask against a SAM2 mask.")
    parser.add_argument("--classical", required=True, help="Binary mask PNG from the classical script.")
    parser.add_argument("--sam2", required=True, help="Binary mask PNG produced by SAM2 (generated separately).")
    parser.add_argument("--image", help="Original image, for the visualization overlay.")
    parser.add_argument("--output", help="Where to save the side-by-side comparison PNG.")
    args = parser.parse_args()

    classical = load_binary_mask(args.classical)
    sam2 = load_binary_mask(args.sam2)
    if classical.shape != sam2.shape:
        sam2 = cv2.resize(sam2, (classical.shape[1], classical.shape[0]), interpolation=cv2.INTER_NEAREST)

    iou, dice = iou_dice(classical, sam2)
    bf = boundary_f_score(classical, sam2)

    print(f"IoU (Jaccard):     {iou:.4f}")
    print(f"Dice (F1) overlap: {dice:.4f}")
    print(f"Boundary F-score:  {bf:.4f}  (2px tolerance)")

    if args.output:
        base = cv2.imread(args.image) if args.image else cv2.cvtColor(classical * 255, cv2.COLOR_GRAY2BGR)
        vis = base.copy()
        c_contours, _ = cv2.findContours(classical * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        s_contours, _ = cv2.findContours(sam2 * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, c_contours, -1, (0, 255, 0), 2)   # classical = green
        cv2.drawContours(vis, s_contours, -1, (0, 0, 255), 2)   # SAM2 = red
        cv2.putText(vis, f"IoU={iou:.3f} Dice={dice:.3f} BF={bf:.3f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(args.output, vis)
        print(f"Saved visualization: {args.output}")


if __name__ == "__main__":
    main()
