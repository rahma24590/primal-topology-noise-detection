# Supplementary Code — Section 8: Noise Detection via Primal Topology

This directory contains the complete, reproducible implementation
supporting Section 8 of the paper *"Topology Derived via Primal
Spaces and Its Application to Digital Image Processing."*

## Files

- **`primal_noise_detection.py`**
  Core algorithm only: computes the local reference intensity
  $f_R(p)$, extracts the noise set $N = \{p \in U : |f(p)-f_R(p)|
  > \theta\}$ (Theorem 8.7), constructs $U$, $N$, $A_0 = U\setminus N$,
  and performs the pixel replacement described in Algorithm 1.
  This file contains nothing beyond what is stated in the paper.

- **`run_experiment.py`**
  The full experimental pipeline used to produce the numbers and
  figures reported in Sections 8.5–8.7: noise injection at a known
  rate, detection accuracy (precision/recall/F1) against the known
  ground truth, restoration quality (PSNR/SSIM), and comparison
  against classical global median filtering (3×3, 5×5) and a hybrid
  variant.

- **`requirements.txt`**
  Exact Python package list needed to run the code.

## How to reproduce the paper's results

```bash
pip install -r requirements.txt
python run_experiment.py --image lena_clean_rgb.png --theta 50 --noise_ratio 0.05 --seed 42
```

This reproduces exactly Table 1 and Table 2 of the paper. All
randomness (noise injection) is controlled by `--seed`; using the
same seed with the same image reproduces the reported numbers
exactly, since no other stochastic component is used anywhere in
the pipeline.

## Test image

The experiments use the standard $256\times256$ grayscale Lenna
test image, a widely used benchmark in the image-processing
literature (source: USC-SIPI Image Database,
https://sipi.usc.edu/database/). The image is not redistributed
in this repository; it can be obtained freely from the USC-SIPI
database or any standard image-processing test-image archive.

## Notes on reproducibility

- No machine-learning or data-driven component is used anywhere:
  every quantity (reference intensity, noise mask, primal $\Omega$,
  filtered output) is obtained via closed-form computation exactly
  as specified in the paper.
- Running the script with different `--theta` or `--noise_ratio`
  values reproduces the sensitivity analysis discussed in the paper.
