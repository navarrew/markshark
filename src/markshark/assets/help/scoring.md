# Scoring
After alignment, MarkShark measures how dark each bubble is and decides which ones the student filled in. The scoring engine uses multiple techniques to handle variation in pencil darkness, paper quality, and scanning conditions.
---
## How Scoring Works
1. **Render** each aligned page at the configured DPI.
2. **Extract** the pixel region for each bubble using the bubblemap grid coordinates.
3. **Measure** the average darkness (fill score) of each bubble, as a percentage.
4. **Decide** which bubbles are "filled" using thresholds and calibration.
5. **Compare** to the answer key and compute scores.
6. **Flag** any rows with ambiguous results (blanks, multi-fills).
---
## Scoring Thresholds
### Min Fill (%)
The minimum fill score for a bubble to count as filled. A bubble with a fill score below this threshold is considered empty.
- **Higher** = requires darker marks (may miss light pencil marks)
- **Lower** = accepts lighter marks (may pick up stray marks or printing)
- **Default: 45%**
### Top-2 Ratio (%)
When the second-darkest bubble is more than this percentage of the darkest, the row is flagged as a potential multi-fill.
- Example: If the darkest bubble scores 80% and the second scores 70%, the ratio is 87.5%. If the threshold is 80%, this row is flagged.
- **Default: 80%**
### Min Top-2 Diff
Minimum absolute difference between the two darkest bubbles. Even if the ratio is below threshold, if the difference is small the row may still be flagged.
- **Default: 10 points**
### Fixed Threshold (Gray)
A global binarisation threshold for gray pixel values (0-255). Used as a baseline for determining mark darkness.
- **Default: 180**
---
## Calibration & Adaptive Scoring
### Auto-Calibrate Threshold
Automatically tunes the fill threshold for each page. This handles variation between pages caused by different pencil types, scan darkness, or paper quality.
- **Default: on**
### Verbose Threshold Diagnostics
Prints per-page threshold calibration details to the log. Useful for debugging threshold issues.
### Calibrate Background
Subtracts per-column background darkness to remove bias from printed letters (A, B, C, D, E) inside or near bubbles. Without this, columns with darker printed letters may produce higher fill scores.
- **Default: on**
### Background Percentile
The percentile used for background calculation. The 10th percentile is robust to noise — it represents the "normal" unfilled darkness for each column.
- **Default: 10.0**
### Adaptive Rescoring
When a row initially comes back as "blank" (no bubble meets the threshold), adaptive rescoring tries progressively lower thresholds to find a valid answer. This rescues light pencil marks that the normal threshold misses.
- **Default: on**
### Adaptive Max Adjustment
Maximum threshold reduction to try during adaptive rescoring, in steps of 10. For example, if set to 40, the engine tries thresholds lowered by 10, 20, 30, and 40.
- **Default: 40**
### Adaptive Min Above Floor
During adaptive rescoring, the winning bubble must be this many points above the lowest bubble in the row. This prevents accepting noise as an answer.
- **Default: 30**
---
## Annotation Options
### Annotate All Bubbles
Draw circles on every bubble in the scored PDF, not just the filled ones. Makes it easy to visually verify alignment and see what the scorer detected.
- **Default: on**
### Show % Fill Labels
Overlay the percentage fill score as text at each bubble position. Useful for diagnosing threshold issues.
- **Default: on**
---
## Flag Types
MarkShark flags rows that need human review:
| Flag | Meaning |
|------|---------|
| **Blank** | No bubble met the fill threshold for this question |
| **Multi** | Two or more bubbles appear filled (ambiguous) |
| **Absent** | Student ID found in roster but no scanned sheet detected |
| **Low confidence** | Fill scores are close to the threshold |
Flagged rows appear highlighted on the Review & Correct page, where you can manually correct them.
---
## Tips for Clean Scoring
- **Use #2 pencils** — They produce consistent, dark marks that score reliably.
- **Fill bubbles completely** — Partial fills may not meet the threshold.
- **Erase thoroughly** — Incomplete erasures can be detected as fills.
- **Scan at consistent settings** — Use the same scanner settings for all sheets in a batch.
- **Start with defaults** — The default thresholds work well for most scan qualities. Only adjust if you see systematic issues.
