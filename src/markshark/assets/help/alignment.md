# Alignment
Alignment is the process of warping each scanned page so that it matches the template precisely. Once aligned, the software knows exactly where every bubble is and can measure how dark each one is.
---
## How Alignment Works
### Step 1: Finding the Markers (ArUco Detection)
Each bubble sheet template has small square barcodes printed in known positions. These are **ArUco markers** — like tiny QR codes that each have a unique ID. When a student's scanned page comes in, the software looks for these markers in the image. Because they have distinct black-and-white patterns, the computer can find them even if the page is slightly rotated, shifted, or wrinkled.
### Step 2: Matching Markers
The software knows exactly where each marker should be on a perfect template. It matches the markers it found in the student's scan with the expected positions by their ID numbers — marker #7 in the scan corresponds to marker #7 on the template. This gives a set of **point pairs**: where each marker is vs. where it should be.
### Step 3: Filtering Bad Matches (RANSAC)
Sometimes a marker gets smudged, partially covered by a student's writing, or misidentified. **RANSAC** (Random Sample Consensus) ignores these bad data points. It works by repeatedly picking a small random subset of marker pairs, computing a transformation from just those points, and checking how many other markers agree. The transformation that gets the most "votes" wins. Even if a couple of markers are wrong or missing, the good markers outvote the bad ones.
### Step 4: Warping the Image
Once the best transformation is determined (a combination of rotation, scaling, and shifting), the entire scanned page is warped so that it lines up with the template. Now every bubble sits exactly where the software expects it, and scoring can begin.
**In short:** find the barcodes, match them to the template, use RANSAC to ignore bad ones, warp the whole page into alignment.
---
## Alignment Methods
| Method | Description | When to use |
|--------|-------------|-------------|
| `auto` | Tries ArUco first, falls back to feature matching | Default, works for most cases |
| `fast` | Coarse-to-fine (72 DPI ORB + bubble grid refinement) | Fast, requires bubblemap |
| `slow` | Full-resolution ORB feature matching | When fast mode struggles |
| `aruco` | ArUco markers only, no feature matching fallback | Templates with reliable markers |
---
## Alignment Parameters
### General
- **Render DPI** — Resolution for rendering PDF pages to images. Higher DPI = more detail but slower. Default: 150.
- **First / Last page** — Process a subset of pages. Set to 0 for all pages.
### ArUco Marker Detection (Step 1)
- **Min markers required** — Minimum ArUco markers that must be found. If fewer are detected, alignment falls back to feature matching. Default: 4.
- **ArUco dictionary** — Which marker dictionary to look for. Must match what's printed on your template. Default: `DICT_4X4_50`.
### Feature Detection (Step 2, fallback)
These parameters control how the software finds visual features when ArUco markers aren't sufficient.
- **Tile grid X / Y** — The image is divided into a grid for uniform feature extraction. More tiles = more evenly distributed features. Default: 8x10.
- **Top-K per tile** — Best features kept from each tile. Higher = more data but slower. Default: 150.
- **ORB max features** — Total ORB features to extract across the whole image. Default: 3000.
- **ORB FAST threshold** — Lower = more features detected (including noise). Default: 12.
### Feature Matching (Step 3)
- **Lowe ratio test** — Rejects ambiguous matches. Lower = stricter, fewer but better matches. Default: 0.75.
- **Mutual check** — Only keep matches where both images agree. Default: on.
- **Use FLANN matcher** — FLANN is faster for large feature sets but less precise. Default: off.
- **Max matches** — Upper limit on matches to consider. Default: 5000.
### RANSAC & Homography (Step 4)
- **Estimator method** — `auto`, `ransac`, or `usac` (newer adaptive variant). Default: auto.
- **RANSAC threshold** — Max pixel error for an inlier. Lower = stricter geometric fit. Default: 3.0 px.
- **Max iterations** — More iterations = more likely to find the best fit, but slower. Default: 10000.
- **ECC refinement** — Fine-tunes alignment by comparing pixel intensities after the geometric transform. Default: on.
- **ECC pyramid levels** — Multi-scale levels for ECC refinement. Default: 4.
### Quality Checks (Step 5)
These thresholds flag pages as failed if the alignment quality is poor.
- **Fail median residual** — Page flagged if the median alignment error exceeds this. Default: 3.0 px.
- **Fail P95 residual** — Page flagged if the 95th-percentile error exceeds this. Default: 8.0 px.
- **Fail BR residual** — Page flagged if the bottom-right corner error exceeds this. Default: 8.0 px.
---
## Troubleshooting Alignment
### Pages failing alignment
- **Increase DPI** — Try 200 or 300 DPI for faint or low-quality scans.
- **Lower min markers** — If markers are partially obscured, try reducing from 4 to 3 or 2.
- **Check the template** — Use the Map Viewer to verify marker positions match the bubblemap.
- **Try "slow" method** — Full-resolution feature matching may work better for difficult scans.
### Slightly misaligned bubbles
- **Enable ECC refinement** — This fine-tunes the warp after the initial geometric alignment.
- **Increase RANSAC iterations** — Gives the algorithm more chances to find the best transform.
- **Lower RANSAC threshold** — Requires more precise marker matching (stricter).
### Alignment is very slow
- **Lower DPI** — 150 DPI is usually sufficient. Only increase if you have quality issues.
- **Use "fast" method** — Coarse-to-fine alignment is significantly faster.
- **Reduce max matches** — Fewer feature matches to process.
- **Process a subset** — Use first/last page to test with a few pages before running the full batch.
