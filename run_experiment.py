"""
run_experiment.py

Full reproducible experiment supporting Section 8.5-8.7 of the paper:

    1. Loads a clean reference image (ground truth).
    2. Injects salt-and-pepper noise at a KNOWN rate, with a fixed
       random seed (full reproducibility).
    3. Runs the primal-topology detection and filtering pipeline
       (primal_noise_detection.py).
    4. Evaluates:
         - detection accuracy (precision / recall / F1) against the
           known ground-truth noise mask;
         - restoration quality (PSNR / SSIM) against the clean image.
    5. Compares against classical global median filtering (3x3, 5x5)
       and a hybrid variant (primal-topology detection + median
       replacement), reproducing Table 1 and Table 2 of the paper.

Requirements
------------
    pip install opencv-python numpy scipy scikit-image

Usage
-----
    python run_experiment.py --image lena_clean_rgb.png

Reproducibility
----------------
All randomness (noise injection) is controlled by a fixed seed
(default: 42). Running this script with the same image and the same
parameters will always reproduce exactly the numbers reported in the
paper.
"""

import argparse
import cv2
import numpy as np
from scipy.ndimage import median_filter
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from primal_noise_detection import (
    calculate_f_R,
    detect_noise,
    construct_primal,
    filter_image,
    pixel_indices_from_mask,
)


# ============================================================
# Controlled noise injection (ground truth)
# ============================================================

def add_salt_and_pepper_noise(img, noise_ratio, seed):
    """
    Corrupts `img` with salt-and-pepper noise at a known rate.
    Returns the noisy image together with the exact ground-truth
    corruption mask (True where a pixel was corrupted).
    """
    rng = np.random.default_rng(seed)
    noisy_img = img.copy()
    h, w = img.shape
    num_pixels = h * w
    num_corrupt = int(num_pixels * noise_ratio)

    flat_indices = rng.choice(num_pixels, size=num_corrupt, replace=False)
    gt_mask = np.zeros((h, w), dtype=bool)

    salt_indices = flat_indices[: num_corrupt // 2]
    pepper_indices = flat_indices[num_corrupt // 2:]

    flat_noisy = noisy_img.flatten()
    flat_mask = gt_mask.flatten()

    flat_noisy[salt_indices] = 255
    flat_noisy[pepper_indices] = 0
    flat_mask[salt_indices] = True
    flat_mask[pepper_indices] = True

    return flat_noisy.reshape(h, w), flat_mask.reshape(h, w)


# ============================================================
# Evaluation metrics
# ============================================================

def evaluate_quality(clean_img, test_img):
    return {
        "psnr": psnr(clean_img, test_img, data_range=255),
        "ssim": ssim(clean_img, test_img, data_range=255),
    }


def evaluate_detection(gt_mask, detected_mask):
    gt = gt_mask.flatten().astype(bool)
    det = detected_mask.flatten().astype(bool)

    tp = int(np.sum(gt & det))
    fp = int(np.sum(~gt & det))
    fn = int(np.sum(gt & ~det))

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="lena_clean_rgb.png",
                         help="Path to a clean reference image.")
    parser.add_argument("--size", type=int, default=256,
                         help="Image is resized to size x size.")
    parser.add_argument("--theta", type=float, default=50.0,
                         help="Similarity threshold theta.")
    parser.add_argument("--noise_ratio", type=float, default=0.05,
                         help="Fraction of pixels corrupted (0-1).")
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed for noise injection.")
    parser.add_argument("--outdir", default=".",
                         help="Directory to save output images.")
    args = parser.parse_args()

    # ---- Load clean reference image ----
    rgb = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    rgb = cv2.resize(rgb, (args.size, args.size), interpolation=cv2.INTER_AREA)
    clean_img = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    h, w = clean_img.shape
    total_pixels = h * w

    # ---- Step 1: inject known noise ----
    noisy_img, gt_mask = add_salt_and_pepper_noise(
        clean_img, args.noise_ratio, args.seed
    )

    # ---- Step 2: primal-topology detection (Theorem 8.7) ----
    f_R, difference, detected_mask = detect_noise(noisy_img, args.theta)
    noise_pixels = pixel_indices_from_mask(detected_mask, w)
    U, N, A0 = construct_primal(total_pixels, noise_pixels)

    # ---- Step 3: primal-topology filtering ----
    primal_filtered = filter_image(noisy_img, f_R, detected_mask)

    # ---- Step 4: classical median filtering (comparison) ----
    median_3x3 = median_filter(noisy_img, size=3)
    median_5x5 = median_filter(noisy_img, size=5)

    # ---- Step 5: hybrid (primal detection + median replacement) ----
    hybrid_filtered = noisy_img.copy()
    hybrid_filtered[detected_mask] = median_3x3[detected_mask]

    # ---- Evaluation ----
    quality = {
        "Noisy (no filtering)": evaluate_quality(clean_img, noisy_img),
        "Primal-topology (selective, f_R)": evaluate_quality(clean_img, primal_filtered),
        "Median filter 3x3 (global)": evaluate_quality(clean_img, median_3x3),
        "Median filter 5x5 (global)": evaluate_quality(clean_img, median_5x5),
        "Hybrid (primal detection + median replace)": evaluate_quality(clean_img, hybrid_filtered),
    }
    detection = evaluate_detection(gt_mask, detected_mask)

    # ---- Report ----
    print("=" * 70)
    print("PRIMAL-TOPOLOGY NOISE DETECTION — REPRODUCIBLE EXPERIMENT")
    print("=" * 70)
    print(f"Image: {args.image}  |  size: {w}x{h}  |  |U| = {total_pixels}")
    print(f"theta = {args.theta}  |  injected noise ratio = {args.noise_ratio*100:.2f}%  "
          f"|  seed = {args.seed}")
    print(f"|N*| injected (ground truth) = {int(gt_mask.sum())}")
    print(f"|N|  detected                = {len(noise_pixels)}  "
          f"({len(noise_pixels)/total_pixels*100:.2f}%)")
    print(f"|A0| = {len(A0)}")
    print()
    print("--- Detection accuracy vs. ground truth ---")
    print(f"Precision={detection['precision']:.4f}  Recall={detection['recall']:.4f}  "
          f"F1={detection['f1']:.4f}  "
          f"(TP={detection['tp']}, FP={detection['fp']}, FN={detection['fn']})")
    print()
    print(f"{'Method':45s} {'PSNR (dB)':>10s} {'SSIM':>8s}")
    print("-" * 70)
    for name, m in quality.items():
        print(f"{name:45s} {m['psnr']:10.2f} {m['ssim']:8.4f}")

    # ---- Save output images ----
    cv2.imwrite(f"{args.outdir}/0_clean.png", clean_img)
    cv2.imwrite(f"{args.outdir}/1_noisy.png", noisy_img)
    cv2.imwrite(f"{args.outdir}/2_fR.png", np.clip(f_R, 0, 255).astype(np.uint8))
    cv2.imwrite(f"{args.outdir}/3_detected_mask.png", (detected_mask * 255).astype(np.uint8))
    cv2.imwrite(f"{args.outdir}/4_primal_filtered.png", primal_filtered)
    cv2.imwrite(f"{args.outdir}/5_median3.png", median_3x3)
    cv2.imwrite(f"{args.outdir}/6_median5.png", median_5x5)
    cv2.imwrite(f"{args.outdir}/7_hybrid.png", hybrid_filtered)


if __name__ == "__main__":
    main()
