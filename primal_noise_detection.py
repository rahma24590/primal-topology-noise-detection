"""
primal_noise_detection.py

Core implementation of the noise-detection construction described in
Section 8 (Theorem 8.7) of the paper "Topology Derived via Primal
Spaces and Its Application to Digital Image Processing".

This module contains ONLY the mathematical pipeline described in the
paper:

    1. f_R(p): local reference intensity (mean of the existing
       8-neighbourhood of p).
    2. N = {p in U : |f(p) - f_R(p)| > theta}: the noise set,
       exactly as defined in Theorem 8.7.
    3. A0 = U \\ N and Omega = Omega_eps(A0) = {B subseteq U :
       N not subseteq B}: the primal on U, following Corollary 3.12.
    4. Replacement of each detected pixel p in N by f_R(p), producing
       the filtered image.

The construction is entirely deterministic and closed-form; no
learned or data-driven component is used at any stage.
"""

import numpy as np


def calculate_f_R(img):
    """
    Local reference intensity f_R(p): mean of the existing
    8-neighbourhood of each pixel p in U (boundary pixels use
    only neighbours lying inside the image domain, via edge padding).

    Parameters
    ----------
    img : 2D numpy array (grayscale image)

    Returns
    -------
    f_R : 2D numpy array of the same shape, dtype float64
    """
    img = img.astype(np.float64)
    h, w = img.shape
    padded = np.pad(img, 1, mode="edge")

    acc = np.zeros_like(img, dtype=np.float64)
    count = np.zeros_like(img, dtype=np.float64)

    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            shifted = padded[1 + di: 1 + di + h, 1 + dj: 1 + dj + w]
            acc += shifted
            count += 1

    return acc / count


def detect_noise(img, theta):
    """
    Noise set N = {p in U : |f(p) - f_R(p)| > theta}, as in Theorem 8.7.

    Parameters
    ----------
    img   : 2D numpy array (grayscale image)
    theta : similarity threshold (float)

    Returns
    -------
    f_R        : local reference intensity map
    difference : |f(p) - f_R(p)| for every pixel
    noise_mask : boolean array, True where the pixel belongs to N
    """
    f = img.astype(np.float64)
    f_R = calculate_f_R(img)
    difference = np.abs(f - f_R)
    noise_mask = difference > theta
    return f_R, difference, noise_mask


def construct_primal(num_pixels, noise_pixels):
    """
    Constructs U, N, and A0 = U \\ N as finite sets of pixel indices
    (1-indexed, raster order), following the notation of Section 8.

    The primal itself,
        Omega = Omega_eps(A0) = {B subseteq U : N not subseteq B},
    is defined implicitly through N and is not enumerated explicitly
    (its cardinality is 2^|U| - 2^|A0|, generally too large to store).

    Returns
    -------
    U, N, A0 : Python sets of pixel indices
    """
    U = set(range(1, num_pixels + 1))
    N = set(noise_pixels)
    A0 = U - N
    return U, N, A0


def filter_image(img, f_R, noise_mask):
    """
    Replaces every detected noise pixel p in N by f_R(p), producing
    the filtered image (Algorithm 1, step 6).
    """
    filtered = img.copy()
    filtered[noise_mask] = np.clip(
        np.round(f_R[noise_mask]), 0, 255
    ).astype(np.uint8)
    return filtered


def pixel_indices_from_mask(mask, width):
    """
    Converts a boolean 2D mask into a sorted list of 1-indexed pixel
    numbers under raster ordering, matching the indexing convention
    U = {1, ..., h*w} used in the paper.
    """
    positions = np.argwhere(mask)
    return sorted(int(i * width + j + 1) for i, j in positions)
